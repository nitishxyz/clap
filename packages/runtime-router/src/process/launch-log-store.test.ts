import { describe, expect, test } from "bun:test";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LaunchLogStore, pruneLaunchLogs, writeLaunchMetadataAtomic } from "./launch-log-store";
import { WORKER_LAUNCH_METADATA_VERSION, type WorkerLaunchMetadata, type WorkerLaunchPaths } from "./types";

function metadata(id: string, endedAt?: string): WorkerLaunchMetadata {
  return {
    version: WORKER_LAUNCH_METADATA_VERSION,
    launchId: id,
    modelId: "model",
    modelPathFingerprint: "fingerprint",
    backend: "llama",
    pid: 123,
    command: ["worker", "--model", "model"],
    protocolVersion: "1",
    startedAt: "2026-01-01T00:00:00.000Z",
    endedAt,
  };
}

async function createLaunch(modelDirectory: string, id: string, endedAt?: string, logBytes = 1) {
  await mkdir(modelDirectory, { recursive: true });
  await writeFile(join(modelDirectory, `${id}.stderr.log`), "x".repeat(logBytes));
  await writeFile(join(modelDirectory, `${id}.json`), JSON.stringify(metadata(id, endedAt)));
}

async function exists(path: string): Promise<boolean> {
  return Bun.file(path).exists();
}

describe("launch log store", () => {
  test("writes a versioned sidecar atomically", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-log-"));
    const path = join(root, "nested", "launch.json");
    await writeLaunchMetadataAtomic(path, metadata("launch"));
    expect(JSON.parse(await readFile(path, "utf8")).version).toBe(1);
  });

  test("prunes finalized pairs by per-model count before backend bytes", async () => {
    const backend = await mkdtemp(join(tmpdir(), "clap-retain-"));
    const first = join(backend, "first");
    const second = join(backend, "second");
    for (let index = 0; index < 3; index++) {
      await createLaunch(first, `first-${index}`, `2026-01-0${index + 1}T00:00:00Z`, 20);
    }
    await createLaunch(second, "second-0", "2026-01-04T00:00:00Z", 20);
    await pruneLaunchLogs(backend, { maxLaunchesPerModel: 2, maxBytesPerBackend: Number.MAX_SAFE_INTEGER });
    expect(await exists(join(first, "first-0.json"))).toBe(false);
    expect(await exists(join(first, "first-0.stderr.log"))).toBe(false);
    expect(await exists(join(first, "first-1.json"))).toBe(true);

    const newestBytes = (await Bun.file(join(second, "second-0.json")).size) + 20;
    await pruneLaunchLogs(backend, { maxLaunchesPerModel: 20, maxBytesPerBackend: newestBytes });
    expect(await exists(join(first, "first-1.json"))).toBe(false);
    expect(await exists(join(first, "first-2.json"))).toBe(false);
    expect(await exists(join(second, "second-0.json"))).toBe(true);
  });

  test("never removes malformed, unfinished, or active launches", async () => {
    const backend = await mkdtemp(join(tmpdir(), "clap-protect-"));
    const modelDirectory = join(backend, "model");
    await createLaunch(modelDirectory, "unfinished", undefined, 10);
    await createLaunch(modelDirectory, "active", "2026-01-01T00:00:00Z", 10);
    await writeFile(join(modelDirectory, "malformed.json"), "not json");
    await writeFile(join(modelDirectory, "malformed.stderr.log"), "diagnostic");
    await writeFile(join(modelDirectory, "invalid.json"), JSON.stringify({
      version: WORKER_LAUNCH_METADATA_VERSION,
      endedAt: "2026-01-01T00:00:00Z",
    }));
    const paths = {
      metadataPath: join(modelDirectory, "active.json"),
      stderrPath: join(modelDirectory, "active.stderr.log"),
    } as WorkerLaunchPaths;
    const release = new LaunchLogStore().registerActive(paths);
    await pruneLaunchLogs(backend, { maxLaunchesPerModel: 0, maxBytesPerBackend: 0 });
    expect(await exists(paths.metadataPath)).toBe(true);
    expect(await exists(join(modelDirectory, "unfinished.json"))).toBe(true);
    expect(await exists(join(modelDirectory, "malformed.json"))).toBe(true);
    expect(await exists(join(modelDirectory, "invalid.json"))).toBe(true);
    release();
    await pruneLaunchLogs(backend, { maxLaunchesPerModel: 0, maxBytesPerBackend: 0 });
    expect(await exists(paths.metadataPath)).toBe(false);
  });

  test("serializes concurrent pruning and tolerates already deleted files", async () => {
    const backend = await mkdtemp(join(tmpdir(), "clap-delete-"));
    await createLaunch(join(backend, "model"), "old", "2026-01-01T00:00:00Z", 10);
    await Promise.all(Array.from({ length: 8 }, () =>
      pruneLaunchLogs(backend, { maxLaunchesPerModel: 0, maxBytesPerBackend: 0 })));
    expect(await exists(join(backend, "model", "old.json"))).toBe(false);
  });
});
