import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ResidentWorkerRegistry } from "../resident";
import { ResidentWorkerProcess } from "./resident-worker-process";
import { WorkerLaunchLogStore } from "./launch-log-store";
import type { WorkerLaunchContext, WorkerLaunchMetadata } from "./types";

const originalHome = process.env.CLAP_HOME;
const originalWorker = process.env.CLAP_LLAMA_WORKER;

afterEach(() => {
  if (originalHome === undefined) delete process.env.CLAP_HOME; else process.env.CLAP_HOME = originalHome;
  if (originalWorker === undefined) delete process.env.CLAP_LLAMA_WORKER; else process.env.CLAP_LLAMA_WORKER = originalWorker;
});

async function workerScript(root: string): Promise<string> {
  const path = join(root, "worker");
  await writeFile(path, `#!/usr/bin/env bun
console.error("launch-stderr:" + process.pid);
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {} }));
const decoder = new TextDecoder(); let buffer = ""; const sequence = new Map();
const send = (type, id, fields = {}) => { const next = sequence.get(id) ?? 0; sequence.set(id, next + 1); console.log(JSON.stringify({ protocol: 1, type, request_id: id, sequence: next, ...fields })); };
for await (const chunk of Bun.stdin.stream()) { buffer += decoder.decode(chunk, { stream: true }); let newline;
while ((newline = buffer.indexOf("\\n")) >= 0) { const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
const command = JSON.parse(raw); if (command.type === "shutdown") process.exit(0); send("accepted", command.request_id); send("started", command.request_id);
send("completed", command.request_id, { result: { kind: command.type === "load" ? "loaded" : "unloaded" } }); }}
`);
  await chmod(path, 0o755);
  return path;
}

async function crashOnceWorker(root: string): Promise<string> {
  const path = join(root, "crash-once-worker");
  const marker = join(root, "crashed-once");
  await writeFile(path, `#!/usr/bin/env bun
import { existsSync, writeFileSync } from "node:fs";
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {} }));
const decoder = new TextDecoder(); let buffer = ""; const sequence = new Map();
const send = (type, id, fields = {}) => { const next = sequence.get(id) ?? 0; sequence.set(id, next + 1); console.log(JSON.stringify({ protocol: 1, type, request_id: id, sequence: next, ...fields })); };
for await (const chunk of Bun.stdin.stream()) { buffer += decoder.decode(chunk, { stream: true }); let newline;
while ((newline = buffer.indexOf("\\n")) >= 0) { const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
const command = JSON.parse(raw); if (command.type === "shutdown") process.exit(0); send("accepted", command.request_id); send("started", command.request_id);
if (command.type === "generate" && !existsSync(${JSON.stringify(marker)})) { send("token", command.request_id, { text: "partial" }); writeFileSync(${JSON.stringify(marker)}, "1"); process.exit(9); }
send("completed", command.request_id, { result: { kind: command.type === "load" ? "loaded" : "unloaded" } }); }}
`);
  await chmod(path, 0o755);
  return path;
}

async function metadataFiles(home: string): Promise<string[]> {
  const glob = new Bun.Glob("logs/workers/**/*.json");
  return Array.fromAsync(glob.scan({ cwd: home, absolute: true }));
}

async function waitForFinalized(home: string, count: number): Promise<WorkerLaunchMetadata[]> {
  for (let attempt = 0; attempt < 100; attempt++) {
    const files = await metadataFiles(home);
    const items = await Promise.all(files.map(async (path) =>
      JSON.parse(await readFile(path, "utf8")) as WorkerLaunchMetadata));
    const finalized = items.filter((item) => item.endedAt);
    if (finalized.length >= count) return finalized;
    await Bun.sleep(10);
  }
  throw new Error(`Timed out waiting for ${count} finalized launches`);
}

describe.serial("resident worker per-launch logs", () => {
  test("isolates concurrent models and preserves repeated launch metadata and stderr", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-process-launch-"));
    const home = join(root, "home");
    process.env.CLAP_HOME = home;
    process.env.CLAP_LLAMA_WORKER = await workerScript(root);
    const firstModel = join(root, "first.gguf");
    const secondModel = join(root, "second.gguf");
    await Promise.all([writeFile(firstModel, "first"), writeFile(secondModel, "second")]);
    const registry = new ResidentWorkerRegistry();
    const first = registry.getOrCreate("first-key", "llama", firstModel, { modelId: "owner/first", revision: "r1" });
    const second = registry.getOrCreate("second-key", "llama", secondModel, { modelId: "owner/second", revision: "r2" });
    await Promise.all([first.load(), second.load()]);
    expect(first.info().launchId).not.toBe(second.info().launchId);
    expect(first.info().stderrLogPath).not.toBe(second.info().stderrLogPath);
    expect(first.info().launchMetadataPath).not.toBe(second.info().launchMetadataPath);
    await Promise.all([first.unload(), second.unload()]);
    const initial = await waitForFinalized(home, 2);
    expect(new Set(initial.map((item) => item.modelId))).toEqual(new Set(["owner/first", "owner/second"]));
    expect(initial.every((item) => item.pid > 0 && item.readyAt && item.startedAt && item.endedAt)).toBe(true);
    expect(initial.every((item) => item.crashClassification === "expected_exit")).toBe(true);

    await first.load();
    await first.unload();
    const repeated = await waitForFinalized(home, 3);
    expect(repeated.filter((item) => item.modelId === "owner/first")).toHaveLength(2);
    const files = await metadataFiles(home);
    const stderrPaths = files.map((path) => path.replace(/\.json$/, ".stderr.log"));
    expect(new Set(stderrPaths).size).toBe(3);
    for (const path of stderrPaths) expect(await readFile(path, "utf8")).toContain("launch-stderr:");
    registry.shutdownAll();
  });

  test("keeps the original request log link after a restart", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-process-restart-"));
    const home = join(root, "home");
    process.env.CLAP_HOME = home;
    process.env.CLAP_LLAMA_WORKER = await crashOnceWorker(root);
    const model = join(root, "model.gguf");
    await writeFile(model, "model");
    const registry = new ResidentWorkerRegistry();
    const worker = registry.getOrCreate("restart", "llama", model);
    let firstError: Error | undefined;
    try {
      await worker.chat({ model: "restart", messages: [{ role: "user", content: "crash" }], stream: false });
    } catch (error) {
      firstError = error as Error;
    }
    expect(firstError?.message).toContain(".stderr.log");
    const firstLaunchId = worker.info().launchId;
    expect(worker.info()).toMatchObject({ crashClassification: "decode" });
    await worker.load();
    expect(worker.info().launchId).not.toBe(firstLaunchId);
    expect(firstError?.message).toContain(firstLaunchId!);
    const finalized = await waitForFinalized(home, 1);
    expect(finalized[0]?.crashClassification).toBe("decode");
    await worker.unload();
    registry.shutdownAll();
  });

  test("sidecar update failures do not mask the original worker exit", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-sidecar-failure-"));
    process.env.CLAP_HOME = join(root, "home");
    process.env.CLAP_LLAMA_WORKER = await crashOnceWorker(root);
    const model = join(root, "model.gguf"); await writeFile(model, "model");
    class FailingUpdates extends WorkerLaunchLogStore {
      override async markReady(_context: WorkerLaunchContext): Promise<void> { throw new Error("sidecar ready failed"); }
      override async finalize(_context: WorkerLaunchContext): Promise<void> { throw new Error("sidecar finalize failed"); }
    }
    const worker = new ResidentWorkerProcess("sidecar", "llama", model, undefined, undefined,
      undefined, { modelId: "sidecar" }, new FailingUpdates());
    await expect(worker.chat({ model: "sidecar", messages: [{ role: "user", content: "crash" }], stream: false }))
      .rejects.toThrow("exited during request with code 9");
    expect(worker.info().stderrLogPath).toContain(worker.info().launchId!);
    await worker.shutdownAsync();
  });

  test("records spawn failure without creating a resident process", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-spawn-failure-"));
    const home = join(root, "home");
    process.env.CLAP_HOME = home;
    process.env.CLAP_LLAMA_WORKER = await workerScript(root);
    const model = join(root, "model.gguf"); await writeFile(model, "model");
    const original = new Error("synthetic spawn failure");
    const spawn = (() => { throw original; }) as typeof Bun.spawn;
    const worker = new ResidentWorkerProcess("spawn", "llama", model, undefined, undefined,
      undefined, { modelId: "spawn" }, new WorkerLaunchLogStore(), spawn);
    await expect(worker.load()).rejects.toBe(original);
    expect(worker.info().state).toBe("not_started");
    const metadata = await waitForFinalized(home, 1);
    expect(metadata[0]).toMatchObject({ exitStatus: null, crashClassification: "spawn_failure" });
  });
});
