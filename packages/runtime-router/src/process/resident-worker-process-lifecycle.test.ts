import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ResidentWorkerProcess } from "./resident-worker-process";
import { testResidentChatOptions } from "../test-cache-identity";
import { ResidentWorkerRegistry } from "../resident";
import { ModelLifecycleManager } from "../lifecycle";
import type { ResolvedModel } from "@clap/models";
import type { WorkerLaunchMetadata, WorkerLoadStateEvent } from "./types";
import type { LifecycleResidencySnapshot } from "../lifecycle";

class RegistryResidencyLifecycle {
  snapshots: LifecycleResidencySnapshot[] = [];
  transitions = new Map<string, string>();
  evictionGate?: Promise<void>;
  evicted: string[] = [];
  snapshotForResidency() { return this.snapshots; }
  async tryEvictIdle(snapshot: LifecycleResidencySnapshot) {
    await this.evictionGate;
    this.evicted.push(snapshot.key);
    this.snapshots = this.snapshots.filter((entry) => entry.key !== snapshot.key);
    return "evicted" as const;
  }
  setResidencyTransition(key: string, state: "starting" | "loading" | "closing") { this.transitions.set(key, state); }
  clearResidencyTransition(key: string) { this.transitions.delete(key); }
}

function configureAdmission(registry: ResidentWorkerRegistry, lifecycle = new RegistryResidencyLifecycle(),
  available = 700 * 1024 ** 2) {
  registry.memorySnapshot = async () => ({
    physicalMemoryBytes: 8 * 1024 ** 3,
    availableMemoryBytes: available,
    residentBytesByPid: new Map(),
  });
  registry.rssSampler = async () => 600 * 1024 ** 2;
  registry.configureResidency({
    lifecycle,
    osHeadroomBytes: 0,
    runtimeHeadroomBytes: 0,
    policy: { minimumHeadroomBytes: 0 },
    env: {},
  });
  return lifecycle;
}

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
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {}, structured_output: { json_object: "native", json_schema: "post_validate", post_validation: true, max_schema_bytes: 65536 } }));
const decoder = new TextDecoder(); let buffer = ""; const sequence = new Map();
const send = (type, id, fields = {}) => { const next = sequence.get(id) ?? 0; sequence.set(id, next + 1); console.log(JSON.stringify({ protocol: 1, type, request_id: id, sequence: next, ...fields })); };
for await (const chunk of Bun.stdin.stream()) { buffer += decoder.decode(chunk, { stream: true }); let newline;
while ((newline = buffer.indexOf("\\n")) >= 0) { const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
const command = JSON.parse(raw); appendFileSync(commands, JSON.stringify({ ...command, at: Date.now() }) + "\\n");
if (command.type === "shutdown") { if (behavior === "ignore") continue; send("accepted", command.request_id); send("started", command.request_id); send("completed", command.request_id, { result: { kind: "shutdown" } }); process.exit(0); }
send("accepted", command.request_id); send("started", command.request_id);
if (command.type === "load") await Bun.sleep(Number(process.env.TEST_LOAD_DELAY ?? 0));
const result = command.type === "load" ? { kind: "loaded" } : command.type === "generate"
  ? { kind: "generated", content: "{}" } : { kind: "unloaded" };
send("completed", command.request_id, { result }); }}
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
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {}, structured_output: first
  ? { json_object: "native", json_schema: "native", post_validation: false, max_schema_bytes: 32768 }
  : { json_object: "post_validate", json_schema: "unsupported", post_validation: true, max_schema_bytes: 1024 } }));
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

async function commands(path: string): Promise<Array<Record<string, unknown>>> {
  try { return (await readFile(path, "utf8")).trim().split("\n").filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>); } catch { return []; }
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
  test("reports starting, loading, resident, closing, and not-started transitions", async () => {
    const { worker } = await setup("terminal", "80");
    const events: WorkerLoadStateEvent[] = [];
    worker.onLoadState((event) => events.push(event));
    expect(worker.info()).toMatchObject({ state: "not_started", loadState: "not_started" });

    const load = worker.load();
    expect(worker.info().loadState).toBe("starting");
    await Bun.sleep(30);
    expect(worker.info()).toMatchObject({ state: "resident", loadState: "loading" });
    const pid = worker.info().pid;
    await load;
    expect(worker.info()).toMatchObject({
      pid, state: "resident", loadState: "resident",
      structuredOutputCapabilities: {
        json_object: "native", json_schema: "post_validate", post_validation: true, max_schema_bytes: 65_536,
      },
    });

    const close = worker.shutdownAsync();
    expect(worker.info()).toMatchObject({ pid, state: "resident", loadState: "closing" });
    await close;
    expect(worker.info()).toMatchObject({ state: "not_started", loadState: "not_started" });
    expect(worker.info().structuredOutputCapabilities).toBeUndefined();
    expect(events.map((event) => event.loadState)).toEqual([
      "starting", "loading", "resident", "closing", "not_started",
    ]);
    expect(events.find((event) => event.loadState === "closing")?.pid).toBe(pid);
  });

  test("observes RSS only while a stable process PID exists", async () => {
    const { worker } = await setup();
    expect(await worker.observeRss(async () => 123)).toEqual({
      source: "unavailable", bytes: null, basis: "not_observed",
    });
    await worker.load();
    const pid = worker.info().pid!;
    expect(await worker.observeRss(async (sampledPid) => sampledPid === pid ? 12_345 : null)).toEqual({
      source: "measured", bytes: 12_345, basis: "resident_rss",
    });
    expect(await worker.observeRss(async () => 0)).toEqual({
      source: "unavailable", bytes: null, basis: "not_reported",
    });
    await worker.shutdownAsync();
  });

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
    expect(worker.info().loadState).toBe("closing");
    await expect(load).rejects.toThrow("shut down");
    await close;
    expect(worker.info().state).toBe("not_started");
    expect(worker.info().loadState).toBe("not_started");
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

  test("injects normalized structured-output contracts and strips caller contracts", async () => {
    const { worker, commands: path } = await setup();
    await worker.chat({
      model: "key",
      messages: [{ role: "user", content: "x" }],
      stream: false,
      response_format: {
        type: "json_schema",
        constraint: "required",
        json_schema: { name: "answer", schema: { type: "object", required: ["answer"] } },
      },
      structured_output: { kind: "grammar", strength: "required", schema: { caller: true } },
    } as Parameters<ResidentWorkerProcess["chat"]>[0], undefined, undefined, undefined, undefined, {
      ...testResidentChatOptions,
      structuredOutput: {
        kind: "json_schema", strength: "best_effort", schema: { type: "object", required: ["answer"] },
      },
    });
    const generate = (await commands(path)).find((command) => command.type === "generate")!;
    expect(generate.structured_output).toEqual({
      kind: "json_schema", strength: "best_effort", schema: { type: "object", required: ["answer"] },
    });
    expect(generate.request).not.toHaveProperty("structured_output");
    expect(JSON.parse(generate.prompt as string)).not.toHaveProperty("structured_output");
    await worker.shutdownAsync();
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
    expect(worker.info().structuredOutputCapabilities?.json_schema).toBe("native");
    await expect(worker.chat({ model: "key", messages: [{ role: "user", content: "x" }], stream: false }, undefined, undefined, undefined, undefined, testResidentChatOptions))
      .rejects.toMatchObject({ code: "worker_protocol_error" });
    expect(worker.info().structuredOutputCapabilities).toBeUndefined();
    await worker.load();
    const replacement = worker.info().launchId;
    expect(worker.info().structuredOutputCapabilities).toEqual({
      json_object: "post_validate", json_schema: "unsupported", post_validation: true, max_schema_bytes: 1024,
    });
    await Bun.sleep(350);
    expect(worker.info()).toMatchObject({ state: "resident", launchId: replacement });
    expect(worker.info().structuredOutputCapabilities?.json_schema).toBe("unsupported");
    expect(worker.info().loadState).toBe("resident");
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
    const lifecycle = new ModelLifecycleManager(() => now, (entry) => expiryRegistry.shutdownAsync(entry.key));
    const model: ResolvedModel = { id: "expiry", input: "expiry", backend: "llama", format: "gguf",
      modelPath: second.worker.modelPath, status: "available" };
    lifecycle.load(model, { keepAlive: "1s", worker: expiryWorker.info() });
    now += 1_001;
    expect(lifecycle.list()).toEqual([]);
    await expiryRegistry.shutdownAsync();
    expect((await finalized(second.home)).crashClassification).toBe("expected_exit");
  });

  test("registry exposes injected measured RSS without inventing unavailable values", async () => {
    const fixture = await setup();
    const registry = new ResidentWorkerRegistry();
    expect(await registry.observeWorkerRss("missing")).toEqual({
      source: "unavailable", bytes: null, basis: "not_observed",
    });
    const worker = registry.getOrCreate("sample", "llama", fixture.worker.modelPath);
    expect(await registry.observeWorkerRss("sample")).toEqual({
      source: "unavailable", bytes: null, basis: "not_observed",
    });
    await worker.load();
    const pid = worker.info().pid!;
    registry.rssSampler = async (sampledPid) => sampledPid === pid ? 98_765 : undefined;
    expect(await registry.observeWorkerRss("sample")).toEqual({
      source: "measured", bytes: 98_765, basis: "resident_rss",
    });
    registry.rssSampler = async () => undefined;
    expect(await registry.observeWorkerRss("sample")).toEqual({
      source: "unavailable", bytes: null, basis: "not_reported",
    });
    await registry.shutdownAsync();
  });

  test("registry admission joins same-key loads under one visible reservation", async () => {
    const fixture = await setup("terminal", "80");
    const registry = new ResidentWorkerRegistry();
    configureAdmission(registry);
    const worker = registry.getOrCreate("admitted", "llama", fixture.worker.modelPath,
      { artifactBytes: 1_000, configuredContext: 1, kv: { bytesPerToken: 1 } });
    const first = worker.load();
    const second = worker.load();
    await Bun.sleep(20);
    expect(registry.residencyReservations()).toHaveLength(1);
    expect(await first).toEqual(await second);
    expect(worker.info().residency).toMatchObject({
      estimateBytes: 512 * 1024 ** 2 + 1_201,
      estimateSource: "architecture_metadata",
      observedRssBytes: 600 * 1024 ** 2,
      observedRssSource: "resident_rss",
      reservationBytes: 512 * 1024 ** 2 + 1_201,
      lastAdmissionReason: "within_budget",
      lastEvictionReason: null,
    });
    expect((await commandTypes(fixture.commands)).filter((type) => type === "load")).toHaveLength(1);
    expect(registry.residencyReservations()).toEqual([]);
    await registry.shutdownAsync();
  });

  test("registry serializes different physical loads and does not reuse a stale snapshot", async () => {
    const fixture = await setup("terminal", "60");
    const registry = new ResidentWorkerRegistry();
    let memoryCalls = 0;
    registry.memorySnapshot = async () => ({
      physicalMemoryBytes: 8 * 1024 ** 3,
      availableMemoryBytes: ++memoryCalls === 1 ? 700 * 1024 ** 2 : 100 * 1024 ** 2,
      residentBytesByPid: new Map(),
    });
    registry.configureResidency({ lifecycle: new RegistryResidencyLifecycle(), osHeadroomBytes: 0,
      runtimeHeadroomBytes: 0, policy: { minimumHeadroomBytes: 0 }, env: {} });
    const descriptor = { artifactBytes: 1_000, configuredContext: 1, kv: { bytesPerToken: 1 } };
    const first = registry.getOrCreate("first", "llama", fixture.worker.modelPath, descriptor).load();
    const second = registry.getOrCreate("second", "llama", fixture.worker.modelPath, descriptor).load();
    await first;
    await expect(second).rejects.toMatchObject({ code: "insufficient_model_memory" });
    expect(memoryCalls).toBe(2);
    expect(registry.residencyReservations()).toEqual([]);
    await registry.shutdownAsync();
  });

  test("registry rollback closes partial loads and rotation blocks new admission", async () => {
    const fixture = await setup("terminal", "80");
    const registry = new ResidentWorkerRegistry();
    configureAdmission(registry);
    const worker = registry.getOrCreate("rotating", "llama", fixture.worker.modelPath,
      { artifactBytes: 1_000, configuredContext: 1, kv: { bytesPerToken: 1 } });
    const load = worker.load();
    await Bun.sleep(20);
    const rotation = registry.rotateCacheIdentityGeneration();
    await load;
    expect(await rotation).toBe(1);
    expect(registry.residencyReservations()).toEqual([]);
    await expect(worker.load()).rejects.toThrow("retired");
    expect(worker.info().loadState).toBe("not_started");
  });

  test("registry admission awaits eviction before starting its worker", async () => {
    const fixture = await setup();
    const registry = new ResidentWorkerRegistry();
    const lifecycle = new RegistryResidencyLifecycle();
    let release!: () => void;
    lifecycle.evictionGate = new Promise<void>((resolve) => { release = resolve; });
    lifecycle.snapshots = [{
      key: "victim", state: "idle", activeRequests: 0, pinned: false, always: false,
      loadedAtMs: 1, lastUsedAtMs: 1, lifecycleVersion: 1, retainedValueScore: 0,
      memory: { source: "measured", bytes: 600 * 1024 ** 2, basis: "resident_rss" },
    }];
    let samples = 0;
    registry.memorySnapshot = async () => ({ physicalMemoryBytes: 8 * 1024 ** 3,
      availableMemoryBytes: ++samples === 1 ? 100 * 1024 ** 2 : 700 * 1024 ** 2,
      residentBytesByPid: new Map() });
    registry.configureResidency({ lifecycle, osHeadroomBytes: 0, runtimeHeadroomBytes: 0,
      policy: { minimumHeadroomBytes: 0 }, env: {} });
    const worker = registry.getOrCreate("target", "llama", fixture.worker.modelPath,
      { artifactBytes: 1_000, configuredContext: 1, kv: { bytesPerToken: 1 } });
    const load = worker.load();
    await Bun.sleep(20);
    expect(worker.info().loadState).toBe("not_started");
    release();
    await load;
    expect(lifecycle.evicted).toEqual(["victim"]);
    expect(samples).toBe(2);
    await registry.shutdownAsync();
  });
});
