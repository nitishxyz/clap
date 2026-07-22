import { afterEach, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { validateCacheTestAssets, type AssetConfiguration } from
  "./validate-cache-test-assets";

const roots: string[] = [];
afterEach(async () => Promise.all(roots.splice(0).map((root) =>
  rm(root, { recursive: true, force: true }))));

const hash = (value: string) => createHash("sha256").update(value).digest("hex");
const base = (): AssetConfiguration => ({ schemaVersion: 1, assets: {
  gguf: { required: false, pin: null }, mlx: { required: false, pin: null },
} });

describe("cache correctness asset validation", () => {
  test("explicitly skips when optional assets are absent", async () => {
    expect(await validateCacheTestAssets({ env: {}, config: base() })).toEqual({
      status: "skipped", assets: [], skipped: [
        { backend: "gguf", reason: "asset_unprovisioned" },
        { backend: "mlx", reason: "asset_unprovisioned" },
      ],
    });
  });

  test("required assets and unsupported fetch fail without pins", async () => {
    const required = base();
    required.assets.gguf.required = true;
    expect(validateCacheTestAssets({ env: { REQUIRE_ASSETS: "1" }, config: required }))
      .rejects.toThrow("required cache correctness assets are absent");
    expect(validateCacheTestAssets({ env: { FETCH: "1" }, config: base() }))
      .rejects.toThrow("unsupported until reviewed pins exist");
  });

  test("validates a pinned GGUF regular file and aliases", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-assets-"))); roots.push(root);
    const model = join(root, "model.gguf");
    await writeFile(model, "real fixture bytes");
    const config = base();
    config.assets.gguf.pin = { source: "fixture", architecture: "test-arch", revision: "rev-1",
      maxBytes: 100, sha256: hash("real fixture bytes") };
    const result = await validateCacheTestAssets({ env: {
      CLAP_CACHE_TEST_ASSET_ROOT: root, CLAP_CACHE_TEST_GGUF_MODEL: model,
    }, config });
    expect(result.status).toBe("ready");
    expect(result.assets[0]).toMatchObject({ backend: "gguf",
      architecture: "test-arch", revision: "rev-1" });
  });

  test("validates MLX required files and manifest", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-assets-"))); roots.push(root);
    const model = join(root, "mlx"); await mkdir(model);
    await writeFile(join(model, "config.json"), "{}");
    await writeFile(join(model, "weights.safetensors"), "weights");
    const requiredFiles = { "config.json": hash("{}"),
      "weights.safetensors": hash("weights") };
    const manifest = Object.entries(requiredFiles).map(([file, digest]) =>
      `${file}\0${digest}\n`).sort().join("");
    const config = base();
    config.assets.mlx.pin = { source: "fixture", architecture: "test-mlx", revision: "rev-2",
      maxBytes: 100, requiredFiles, manifestSha256: hash(manifest) };
    const result = await validateCacheTestAssets({ env: {
      CLAP_CACHE_TEST_ASSET_ROOT: root, CLAP_TEST_MLX_MODEL: model,
    }, config });
    expect(result.assets[0].backend).toBe("mlx");
  });

  test("rejects symlink escape checksum mismatch and unreviewed pins", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-assets-"))); roots.push(root);
    const outside = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-outside-"))); roots.push(outside);
    await writeFile(join(outside, "model.gguf"), "bytes");
    await symlink(join(outside, "model.gguf"), join(root, "escape.gguf"));
    const config = base();
    config.assets.gguf.pin = { source: "fixture", architecture: "test", revision: "rev", maxBytes: 100,
      sha256: hash("bytes") };
    expect(validateCacheTestAssets({ env: { CLAP_CACHE_TEST_ASSET_ROOT: root,
      CLAP_TEST_GGUF_MODEL: join(root, "escape.gguf") }, config }))
      .rejects.toThrow("escapes its canonical root");
    await writeFile(join(root, "model.gguf"), "wrong");
    expect(validateCacheTestAssets({ env: { CLAP_CACHE_TEST_ASSET_ROOT: root,
      CLAP_TEST_GGUF_MODEL: join(root, "model.gguf") }, config }))
      .rejects.toThrow("checksum mismatch");
    config.assets.gguf.pin = null;
    expect(validateCacheTestAssets({ env: { CLAP_CACHE_TEST_ASSET_ROOT: root,
      CLAP_TEST_GGUF_MODEL: join(root, "model.gguf") }, config }))
      .rejects.toThrow("no reviewed pin");
  });
});
