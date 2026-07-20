import { mkdtemp, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { createNativeBundleManifest, ensureNativeArtifact, nativeBundleBuildId, sha256 } from "./native-bundle";

const bytes = (value: string) => new TextEncoder().encode(value);

describe("native bundle content addressing", () => {
  test("changing either worker changes the bundle build key", async () => {
    const manifest = await createNativeBundleManifest([
      { name: "clap-llama", data: bytes("llama-v1"), executable: true },
      { name: "clap-mlx", data: bytes("mlx-v1"), executable: true },
      { name: "mlx.metallib", data: bytes("metal-v1"), executable: false },
    ]);
    const llamaChanged = await createNativeBundleManifest([
      { name: "clap-llama", data: bytes("llama-v2"), executable: true },
      { name: "clap-mlx", data: bytes("mlx-v1"), executable: true },
      { name: "mlx.metallib", data: bytes("metal-v1"), executable: false },
    ]);
    const mlxChanged = await createNativeBundleManifest([
      { name: "clap-llama", data: bytes("llama-v1"), executable: true },
      { name: "clap-mlx", data: bytes("mlx-v2"), executable: true },
      { name: "mlx.metallib", data: bytes("metal-v1"), executable: false },
    ]);

    expect(await nativeBundleBuildId(llamaChanged)).not.toBe(await nativeBundleBuildId(manifest));
    expect(await nativeBundleBuildId(mlxChanged)).not.toBe(await nativeBundleBuildId(manifest));
  });

  test.each(["clap-llama", "clap-mlx"])("replaces stale same-size extracted %s", async (name) => {
    const directory = await mkdtemp(join(tmpdir(), "clap-native-bundle-"));
    const target = join(directory, name);
    await Bun.write(target, "old-worker");
    const source = new Blob(["new-worker"]);
    const expected = { size: source.size, sha256: await sha256(source), executable: true };

    expect(await ensureNativeArtifact(source, target, expected)).toBe("replaced");
    expect(await readFile(target, "utf8")).toBe("new-worker");
    expect((await stat(target)).mode & 0o111).not.toBe(0);
  });

  test("does not rewrite an extracted worker whose digest is correct", async () => {
    const directory = await mkdtemp(join(tmpdir(), "clap-native-bundle-"));
    const target = join(directory, "clap-mlx");
    const source = new Blob(["correct-worker"]);
    await Bun.write(target, source);
    const expected = { size: source.size, sha256: await sha256(source), executable: true };
    const before = await stat(target);

    expect(await ensureNativeArtifact(source, target, expected)).toBe("unchanged");
    expect((await stat(target)).mtimeMs).toBe(before.mtimeMs);
  });
});
