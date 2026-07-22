import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ResidentWorkerProcess } from "./resident-worker-process";
import { testResidentChatOptions } from "../test-cache-identity";
import { ResidentWorkerRegistry } from "../resident";
import { ModelLifecycleManager } from "../lifecycle";
import type { ResolvedModel } from "@clap/models";
import type { WorkerLaunchMetadata } from "./types";

const originalHome = process.env.CLAP_HOME;
const originalWorker = process.env.CLAP_LLAMA_WORKER;
const originalTimeout = process.env.CLAP_WORKER_SHUTDOWN_TIMEOUT_MS;

afterEach(() => {
  for (const [name, value] of [["CLAP_HOME", originalHome], ["CLAP_LLAMA_WORKER", originalWorker],
    ["CLAP_WORKER_SHUTDOWN_TIMEOUT_MS", originalTimeout]] as const) {
    if (value === undefined) delete process.env[name]; else process.env[name] = value;
  }
});

async function lifecycleWorker(root: string): Promise<{ path: string; commands: string }> {
  const path = join(root, "worker");
  const commands = join(root, "commands.jsonl");
  await writeFile(path, `#!/usr/bin/env bun
import { appendFileSync } from "node:fs";
const commands = ${JSON.stringify(commands)}; const behavior = process.env.TEST_SHUTDOWN ?? "terminal";
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {} }));
const decoder = new TextDecoder(); let buffer = ""; const sequence = new Map();
const send = (type, id, fields = {}) => { const next = sequence.get(id) ?? 0; sequence.set(id, next + 1); console.log(JSON.stringify({ protocol: 1, type, request_id: id, sequence: next, ...fields })); };
for await (const chunk of Bun.stdin.stream()) { buffer += decoder.decode(chunk, { stream: true }); let newline;
while ((newline = buffer.indexOf("\\n")) >= 0) { const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
const command = JSON.parse(raw); appendFileSync(commands, JSON.stringify({ type: command.type, at: Date.now() }) + "\\n");
if (command.type === "shutdown") { if (behavior === "ignore") continue; send("accepted", command.request_id); send("started", command.request_id); send("completed", command.request_id, { result: { kind: "shutdown" } }); process.exit(0); }
send("accepted", command.request_id); send("started", command.request_id);
if (command.type === "load") await Bun.sleep(Number(process.env.TEST_LOAD_DELAY ?? 0));
send("completed", command.request_id, { result: { kind: command.type === "load" ? "loaded" : "unloaded" } }); }}
`);
  await chmod(path, 0o755);
  return { path, commands };
}

async function staleExitWorker(root: string): Promise<string> {
  const path = join(root, "stale-worker");
  const marker = join(root, "first-launch");
  await writeFile(path, `#!/usr/bin/env bun
import { existsSync, writeFileSync } from "node:fs"; const first = !existsSync(${JSON.stringify(marker)});
if (first) writeFileSync(${JSON.stringify(marker)}, "1");
process.on("SIGTERM", async () => { await Bun.sleep(250); process.exit(9); });
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {} }));
const decoder = new TextDecoder(); let buffer = ""; const sequence = new Map();
const send = (type, id, fields = {}) => { const next = sequence.get(id) ?? 0; sequence.set(id, next + 1); console.log(JSON.stringify({ protocol: 1, type, request_id: id, sequence: next, ...fields })); };
for await (const chunk of Bun.stdin.stream()) { buffer += decoder.decode(chunk, { stream: true }); let newline;
while ((newline = buffer.indexOf("\\n")) >= 0) { const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue; const command = JSON.parse(raw);
if (command.type === "shutdown") process.exit(0); send("accepted", command.request_id); send("started", command.request_id);
if (first && command.type === "generate") { console.log(JSON.stringify({ protocol: 1, type: "token", request_id: "unknown", sequence: 0, text: "bad" })); continue; }
send("completed", command.request_id, { result: { kind: command.type === "load" ? "loaded" : "unloaded" } }); }}
`);
  await chmod(path, 0o755);
  return path;
}

async function setup(behavior = "terminal", loadDelay = "0") {
  const root = await mkdtemp(join(tmpdir(), "clap-lifecycle-race-"));
  const home = join(root, "home");
  const fixture = await lifecycleWorker(root);
  const model = join(root, "model.gguf");
  await writeFile(model, "model");
  process.env.CLAP_HOME = home;
  process.env.CLAP_LLAMA_WORKER = fixture.path;
  const worker = new ResidentWorkerProcess("key", "llama", model, undefined,
    { TEST_SHUTDOWN: behavior, TEST_LOAD_DELAY: loadDelay });
  return { worker, home, commands: fixture.commands };
}

async function commandTypes(path: string): Promise<string[]> {
  try { return (await readFile(path, "utf8")).trim().split("\n").filter(Boolean)
    .map((line) => JSON.parse(line).type as string); } catch { return []; }
}

async function finalized(home: string): Promise<WorkerLaunchMetadata> {
  for (let attempt = 0; attempt < 100; attempt++) {
    const files = await Array.fromAsync(new Bun.Glob("logs/workers/**/*.json").scan({ cwd: home, absolute: true }));
    if (files[0]) {
      const item = JSON.parse(await readFile(files[0], "utf8")) as WorkerLaunchMetadata;
      if (item.endedAt) return item;
    }
    await Bun.sleep(10);
  }
  throw new Error("launch did not finalize");
}

describe.serial("resident worker lifecycle races", () => {
  test("single-flights concurrent load commands", async () => {
    const { worker, commands } = await setup("terminal", "50");
    await Promise.all([worker.load(), worker.load(), worker.load()]);
    expect((await commandTypes(commands)).filter((type) => type === "load")).toHaveLength(1);
    await worker.shutdownAsync();
  });

  test("load racing shutdown rejects without resurrecting the launch", async () => {
    const { worker } = await setup("terminal", "100");
    const load = worker.load();
    await Bun.sleep(20);
    const close = worker.shutdownAsync();
    await expect(load).rejects.toThrow("shut down");
    await close;
    expect(worker.info().state).toBe("not_started");
  });

  test("graceful shutdown finalizes cleanly and repeated shutdown shares close", async () => {
    const { worker, home } = await setup();
    await worker.load();
    const first = worker.shutdownAsync();
    const second = worker.shutdownAsync();
    expect(second).toBe(first);
    await first;
    expect((await finalized(home)).crashClassification).toBe("expected_exit");
  });

  test("forces a worker after shutdown timeout", async () => {
    const { worker, home } = await setup("ignore");
    process.env.CLAP_WORKER_SHUTDOWN_TIMEOUT_MS = "30";
    await worker.load();
    const started = Date.now();
    await worker.shutdownAsync();
    expect(Date.now() - started).toBeGreaterThanOrEqual(25);
    expect((await finalized(home)).crashClassification).toBe("expected_exit");
  });

  test("unload terminal precedes graceful shutdown", async () => {
    const { worker, commands } = await setup();
    await worker.load();
    await worker.unload();
    expect(await commandTypes(commands)).toEqual(["load", "unload", "shutdown"]);
  });

  test("a stale launch exit cannot clear its replacement", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-stale-exit-"));
    process.env.CLAP_HOME = join(root, "home");
    process.env.CLAP_LLAMA_WORKER = await staleExitWorker(root);
    const model = join(root, "model.gguf"); await writeFile(model, "model");
    const worker = new ResidentWorkerProcess("key", "llama", model);
    await worker.load();
    await expect(worker.chat({ model: "key", messages: [{ role: "user", content: "x" }], stream: false }, undefined, undefined, undefined, undefined, testResidentChatOptions))
      .rejects.toMatchObject({ code: "worker_protocol_error" });
    await worker.load();
    const replacement = worker.info().launchId;
    await Bun.sleep(350);
    expect(worker.info()).toMatchObject({ state: "resident", launchId: replacement });
    await worker.shutdownAsync();
  });

  test("registry shutdown and model expiry finalize their launches", async () => {
    const first = await setup();
    const shutdownRegistry = new ResidentWorkerRegistry();
    const shutdownWorker = shutdownRegistry.getOrCreate("shutdown", "llama", first.worker.modelPath);
    await shutdownWorker.load();
    await shutdownRegistry.shutdownAsync();
    expect((await finalized(first.home)).crashClassification).toBe("expected_exit");

    const second = await setup();
    const expiryRegistry = new ResidentWorkerRegistry();
    const expiryWorker = expiryRegistry.getOrCreate("expiry", "llama", second.worker.modelPath);
    await expiryWorker.load();
    let now = Date.now();
    const lifecycle = new ModelLifecycleManager(() => now, (entry) => expiryRegistry.shutdown(entry.key));
    const model: ResolvedModel = { id: "expiry", input: "expiry", backend: "llama", format: "gguf",
      modelPath: second.worker.modelPath, status: "available" };
    lifecycle.load(model, { keepAlive: "1s", worker: expiryWorker.info() });
    now += 1_001;
    expect(lifecycle.list()).toEqual([]);
    await expiryRegistry.shutdownAsync();
    expect((await finalized(second.home)).crashClassification).toBe("expected_exit");
  });
});
