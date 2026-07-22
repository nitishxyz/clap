import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, mkdir, realpath, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createWorkerLaunchPaths,
  fingerprintModelPath,
  hashModelIdentity,
  resolveClapHome,
} from "./launch-paths";

const originalClapHome = process.env.CLAP_HOME;
afterEach(() => {
  if (originalClapHome === undefined) delete process.env.CLAP_HOME;
  else process.env.CLAP_HOME = originalClapHome;
});

describe("worker launch paths", () => {
  test("resolves CLAP_HOME at call time", () => {
    process.env.CLAP_HOME = "/tmp/clap-one";
    expect(resolveClapHome()).toBe("/tmp/clap-one");
    process.env.CLAP_HOME = "/tmp/clap-two";
    expect(resolveClapHome()).toBe("/tmp/clap-two");
  });

  test("fingerprints the canonical real path", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-paths-"));
    const model = join(root, "model");
    const alias = join(root, "alias");
    await mkdir(model);
    await symlink(model, alias);
    expect(await fingerprintModelPath(alias)).toBe(await fingerprintModelPath(await realpath(model)));
  });

  test("domain-separates every model identity component", () => {
    const base = { backend: "llama", modelId: "owner/model", revision: "main", modelPathFingerprint: "abc" };
    const hash = hashModelIdentity(base);
    expect(hashModelIdentity({ ...base, backend: "mlx" })).not.toBe(hash);
    expect(hashModelIdentity({ ...base, modelId: "owner/other" })).not.toBe(hash);
    expect(hashModelIdentity({ ...base, revision: "v2" })).not.toBe(hash);
    expect(hashModelIdentity({ ...base, modelPathFingerprint: "def" })).not.toBe(hash);
  });

  test("gives concurrent launches independent UUID paths", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-launches-"));
    const model = join(root, "model");
    await mkdir(model);
    process.env.CLAP_HOME = join(root, "home");
    const input = { backend: "llama", modelId: "model", modelPath: model };
    const paths = await Promise.all(Array.from({ length: 12 }, () => createWorkerLaunchPaths(input)));
    expect(new Set(paths.map((entry) => entry.launchId)).size).toBe(12);
    expect(new Set(paths.map((entry) => entry.stderrPath)).size).toBe(12);
    expect(new Set(paths.map((entry) => entry.metadataPath)).size).toBe(12);
    expect(new Set(paths.map((entry) => entry.modelHash)).size).toBe(1);
    expect(paths[0]?.stderrPath).toContain("/logs/workers/llama/");
  });
});
