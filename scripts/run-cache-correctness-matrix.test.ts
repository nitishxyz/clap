import { afterEach, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runCacheCorrectnessMatrix, scenariosFor } from
  "./run-cache-correctness-matrix";
import type { AssetConfiguration } from "./validate-cache-test-assets";

const roots: string[] = [];
afterEach(async () => Promise.all(roots.splice(0).map((root) =>
  rm(root, { recursive: true, force: true }))));
const hash = (value: string) => createHash("sha256").update(value).digest("hex");

function config(checksum: string): AssetConfiguration {
  return { schemaVersion: 1, assets: {
    gguf: { required: false, pin: { architecture: "llama-test", revision: "pin-1",
      maxBytes: 100, sha256: checksum } },
    mlx: { required: false, pin: null },
  } };
}

describe("cache correctness Tier B matrix", () => {
  test("has backend-specific physical probe commands", () => {
    const plans = scenariosFor([
      { backend: "gguf", path: "/asset/a", architecture: "a", revision: "1" },
      { backend: "mlx", path: "/asset/b", architecture: "b", revision: "2" },
    ]);
    expect(plans.map((plan) => plan.command.at(-1))).toEqual([
      "runtime:llama:physical:test", "runtime:mlx:physical:test",
    ]);
    expect(plans[0].environment.CLAP_TEST_GGUF_MODEL).toBe("/asset/a");
    expect(plans[1].environment.CLAP_TEST_MLX_MODEL).toBe("/asset/b");
  });

  test("no assets produces an explicit successful skip", async () => {
    const report = await runCacheCorrectnessMatrix({ env: {}, config: {
      schemaVersion: 1, assets: { gguf: { required: false, pin: null },
        mlx: { required: false, pin: null } },
    } });
    expect(report.status).toBe("skipped");
    expect(report.backends.map((entry) => entry.status)).toEqual(["skipped", "skipped"]);
  });

  test("runs with isolated CLAP_HOME and reports only pinned metadata", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-matrix-"))); roots.push(root);
    const model = join(root, "model.gguf"); await writeFile(model, "bytes");
    let captured: Record<string, string> = {};
    const report = await runCacheCorrectnessMatrix({ env: {
      CLAP_CACHE_TEST_ASSET_ROOT: root, CLAP_CACHE_TEST_GGUF_MODEL: model,
      SERVICE_TOKEN: "must-not-propagate", CLAP_CACHE_TEST_TIMEOUT_MS: "1234",
    }, config: config(hash("bytes")), execute: async (_, env, timeout) => {
      captured = env;
      expect(timeout).toBe(1234);
      expect(env.CLAP_HOME).toContain("clap-cache-correctness-");
    } });
    expect(report.status).toBe("passed");
    expect(report.backends.find((entry) => entry.backend === "gguf")).toMatchObject({
      architecture: "llama-test", revision: "pin-1", status: "passed",
    });
    expect(JSON.stringify(report)).not.toContain(root);
    expect(captured.SERVICE_TOKEN).toBeUndefined();
  });

  test("failure report sanitizes filesystem paths", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-matrix-"))); roots.push(root);
    const model = join(root, "model.gguf"); await writeFile(model, "bytes");
    const report = await runCacheCorrectnessMatrix({ env: {
      CLAP_CACHE_TEST_ASSET_ROOT: root, CLAP_TEST_GGUF_MODEL: model,
    }, config: config(hash("bytes")), execute: async () => {
      throw new Error(`failed at ${model}`);
    } });
    expect(report.status).toBe("failed");
    expect(report.backends.find((entry) => entry.backend === "gguf")?.error)
      .toBe("failed at <path>");
  });
});
