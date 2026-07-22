import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ResidentWorkerRegistry } from "./resident";
import { testResidentChatOptions } from "./test-cache-identity";

const cleanup: Array<() => Promise<void>> = [];

afterEach(async () => {
  while (cleanup.length) await cleanup.pop()!();
});

async function faultWorker(scenario: string) {
  const dir = await mkdtemp(join(tmpdir(), `clap-v1-fault-${scenario}-`));
  const worker = join(dir, "worker");
  const previous = process.env.CLAP_LLAMA_WORKER;
  const previousHome = process.env.CLAP_HOME;
  const source = `#!/usr/bin/env bun
const scenario = ${JSON.stringify(scenario)};
const send = (value) => console.log(typeof value === "string" ? value : JSON.stringify(value));
const scoped = (type, id, sequence, fields = {}) => send({ protocol: 1, type, request_id: id, sequence, ...fields });
if (scenario === "exit_handshake") process.exit(17);
if (scenario === "timeout") await Bun.sleep(10000);
if (scenario === "malformed_ready") { send({ protocol: 1, type: "ready", worker_capabilities: {} }); await Bun.sleep(10000); }
if (scenario === "version_mismatch") { send({ protocol: 2, type: "ready", worker_capabilities: {}, model_capabilities: {} }); await Bun.sleep(10000); }
if (scenario === "non_json") { send("not-json"); await Bun.sleep(10000); }
if (scenario === "unknown_type") { send({ protocol: 1, type: "surprise" }); await Bun.sleep(10000); }
send({ protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {} });
if (scenario === "exit_idle") setTimeout(() => process.exit(23), 100);
const terminal = new Set();
const complete = (id, sequence, result) => { if (terminal.has(id)) return; terminal.add(id); scoped("completed", id, sequence, { result }); };
const decoder = new TextDecoder(); let buffer = "";
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true }); let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
    const command = JSON.parse(raw); const id = command.request_id;
    if (command.type === "shutdown") process.exit(0);
    scoped("accepted", id, 0);
    if (command.type === "load") {
      if (scenario === "exit_load") process.exit(18);
      scoped("started", id, 1); complete(id, 2, { kind: "loaded" }); continue;
    }
    if (command.type === "cancel") {
      if (scenario === "cancel_unknown") scoped("failed", id, 1, { error: { code: "unknown_request", message: "unknown target", retryable: false, fatal: false } });
      else if (scenario === "cancel_terminal") scoped("failed", id, 1, { error: { code: "terminal_request", message: "target already terminal", retryable: false, fatal: false } });
      else {
        complete(command.target_request_id, 3, { kind: "cancelled", finish_reason: "cancel" });
        complete(id, 1, { kind: "cancelled" });
      }
      continue;
    }
    if (command.type !== "generate") {
      if (scenario === "exit_control" && command.type === "set_max_active") process.exit(24);
      scoped("started", id, 1); complete(id, 2, { kind: "unloaded" }); continue;
    }
    if (scenario === "exit_before_start") process.exit(19);
    if (scenario === "duplicate_accepted") { scoped("accepted", id, 1); continue; }
    scoped("started", id, 1);
    if (scenario === "exit_prefill") { scoped("prefill_progress", id, 2, { completed: 1, total: 2 }); process.exit(20); }
    if (scenario === "exit_decode") { scoped("token", id, 2, { text: "valid" }); process.exit(21); }
    if (scenario === "unknown_id") { scoped("token", "other-request", 0, { text: "poison" }); continue; }
    if (scenario === "missing_id") { send({ protocol: 1, type: "token", sequence: 2, text: "poison" }); continue; }
    if (scenario === "missing_sequence") { send({ protocol: 1, type: "token", request_id: id, text: "poison" }); continue; }
    if (scenario === "duplicate_sequence") { scoped("token", id, 1, { text: "poison" }); continue; }
    if (scenario === "gap_sequence") { scoped("token", id, 3, { text: "poison" }); continue; }
    if (scenario === "regressed_sequence") { scoped("token", id, 2, { text: "valid" }); scoped("token", id, 2, { text: "poison" }); continue; }
    if (scenario === "scoped_telemetry") { send({ protocol: 1, type: "telemetry", request_id: id, sequence: 2, telemetry: {} }); continue; }
    if (scenario === "diagnostic_exit") { send({ protocol: 1, type: "diagnostic", level: "error", message: "EXACT diagnostic: model exploded" }); process.exit(22); }
    if (scenario === "duplicate_terminal") { complete(id, 2, { kind: "generated", content: "ok", finish_reason: "stop" }); scoped("completed", id, 3, { result: { kind: "generated", content: "poison" } }); continue; }
    if (scenario === "late_token") { complete(id, 2, { kind: "generated", content: "ok", finish_reason: "stop" }); scoped("token", id, 3, { text: "poison" }); continue; }
    if (scenario === "cancel_race") { scoped("token", id, 2, { text: "race" }); continue; }
    complete(id, 2, { kind: "generated", content: "ok", finish_reason: "stop" });
  }
}`;
  await writeFile(worker, source); await chmod(worker, 0o755);
  process.env.CLAP_LLAMA_WORKER = worker;
  process.env.CLAP_HOME = join(dir, "home");
  cleanup.push(async () => {
    if (previous === undefined) delete process.env.CLAP_LLAMA_WORKER;
    else process.env.CLAP_LLAMA_WORKER = previous;
    if (previousHome === undefined) delete process.env.CLAP_HOME;
    else process.env.CLAP_HOME = previousHome;
    await rm(dir, { recursive: true, force: true });
  });
  const registry = new ResidentWorkerRegistry();
  cleanup.push(async () => registry.shutdownAsync());
  return registry.getOrCreate(scenario, "llama", join(dir, "model.gguf"));
}

const request = { model: "model", messages: [{ role: "user" as const, content: "hello" }], stream: true };

function protocolCode(error: unknown) {
  expect(error).toMatchObject({ code: "worker_protocol_error" });
  return String((error as Error).message).match(/worker protocol ([a-z_]+)/)?.[1];
}

describe.serial("resident v1 fault-injection matrix", () => {
  for (const [scenario, code] of [
    ["malformed_ready", "malformed_event"], ["version_mismatch", "version_mismatch"],
    ["timeout", "handshake_timeout"], ["non_json", "malformed_stdout"],
    ["unknown_type", "unknown_event_type"],
  ] as const) {
    test(`${scenario} fails the handshake with a stable protocol code`, async () => {
      const worker = await faultWorker(scenario);
      try { await worker.load(); throw new Error("load unexpectedly succeeded"); }
      catch (error) { expect(protocolCode(error)).toBe(code); }
      expect(worker.info().state).toBe("not_started");
    });
  }

  for (const [scenario, code] of [
    ["unknown_id", "unknown_request_id"], ["missing_id", "malformed_event"],
    ["missing_sequence", "malformed_event"], ["duplicate_sequence", "sequence_violation"],
    ["gap_sequence", "sequence_violation"], ["regressed_sequence", "sequence_violation"],
    ["duplicate_accepted", "state_violation"], ["scoped_telemetry", "scope_violation"],
  ] as const) {
    test(`${scenario} poisons no output and makes the worker unhealthy`, async () => {
      const worker = await faultWorker(scenario); const tokens: string[] = [];
      try { await worker.chat(request, (token) => tokens.push(token), undefined, undefined, undefined, testResidentChatOptions); throw new Error("chat unexpectedly succeeded"); }
      catch (error) { expect(protocolCode(error)).toBe(code); }
      expect(tokens).not.toContain("poison");
      expect(worker.info().state).toBe("not_started");
    });
  }

  for (const scenario of ["duplicate_terminal", "late_token"] as const) {
    test(`${scenario} cannot mutate resolved content and terminates the worker`, async () => {
      const worker = await faultWorker(scenario);
      const result = await worker.chat(request, undefined, undefined, undefined, undefined, testResidentChatOptions);
      expect(result.content).toBe("ok");
      await Bun.sleep(20);
      expect(worker.info().state).toBe("not_started");
    });
  }

  for (const [scenario, phase] of [
    ["exit_handshake", "protocol handshake"], ["exit_load", "during request"],
    ["exit_before_start", "during request"], ["exit_prefill", "during request"],
    ["exit_decode", "during request"],
  ] as const) {
    test(`${scenario} rejects in-flight work without fabricating completion`, async () => {
      const worker = await faultWorker(scenario); const tokens: string[] = [];
      const operation = scenario === "exit_handshake" || scenario === "exit_load"
        ? worker.load() : worker.chat(request, (token) => tokens.push(token), undefined, undefined, undefined, testResidentChatOptions);
      await expect(operation).rejects.toThrow(phase);
      expect(tokens.every((token) => token === "valid")).toBe(true);
      expect(worker.info().state).toBe("not_started");
      expect(worker.info().stderrLogPath).toContain(worker.info().launchId!);
      const expectedClassification = scenario === "exit_handshake" ? "handshake"
        : scenario === "exit_load" ? "load"
        : scenario === "exit_before_start" ? "prefill"
        : scenario === "exit_prefill" ? "prefill" : "decode";
      expect(worker.info().crashClassification).toBe(expectedClassification);
    });
  }

  test("an idle exit is classified and links its exact stderr", async () => {
    const worker = await faultWorker("exit_idle");
    await worker.load().catch(() => {});
    await Bun.sleep(150);
    expect(worker.info()).toMatchObject({ state: "not_started", crashClassification: "idle" });
    expect(worker.info().stderrLogPath).toContain(worker.info().launchId!);
  });

  test("an exit during a control command is classified", async () => {
    const worker = await faultWorker("exit_control");
    await worker.load();
    await expect(worker.setMaxActive!(2)).rejects.toThrow("during request");
    expect(worker.info().crashClassification).toBe("idle");
  });

  test("an exact diagnostic is retained on the subsequent worker failure", async () => {
    const worker = await faultWorker("diagnostic_exit");
    await expect(worker.chat(request, undefined, undefined, undefined, undefined, testResidentChatOptions)).rejects.toThrow("EXACT diagnostic: model exploded");
  });

  test("a crash invokes restart backoff before the next load attempt", async () => {
    const worker = await faultWorker("exit_load");
    await expect(worker.load()).rejects.toThrow("code 18");
    expect(worker.info().crashes).toBe(1);
    const started = performance.now();
    await expect(worker.load()).rejects.toThrow("code 18");
    expect(performance.now() - started).toBeGreaterThanOrEqual(900);
    expect(worker.info().crashes).toBe(2);
  });

  test("an unknown request ID rejects every pending request without cross-ID tokens", async () => {
    const worker = await faultWorker("unknown_id"); const left: string[] = []; const right: string[] = [];
    const results = await Promise.allSettled([
      worker.chat(request, (token) => left.push(token), undefined, undefined, undefined, testResidentChatOptions),
      worker.chat(request, (token) => right.push(token), undefined, undefined, undefined, testResidentChatOptions),
    ]);
    expect(results.map(({ status }) => status)).toEqual(["rejected", "rejected"]);
    expect([...left, ...right]).toEqual([]);
  });

  for (const scenario of ["cancel_race", "cancel_unknown", "cancel_terminal"] as const) {
    test(`${scenario} gives generation exactly one terminal outcome`, async () => {
      const worker = await faultWorker(scenario); const controller = new AbortController();
      if (scenario !== "cancel_race") controller.abort();
      const result = await worker.chat(request, () => controller.abort(), controller.signal, undefined, undefined, testResidentChatOptions);
      if (scenario === "cancel_race") expect(result.finishReason).toBe("cancel");
      else expect(result).toMatchObject({ content: "ok", finishReason: "stop" });
    });
  }
});
