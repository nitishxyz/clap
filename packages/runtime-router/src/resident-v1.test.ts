import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ResidentWorkerRegistry } from "./resident";
import { testResidentChatOptions as cacheOptions } from "./test-cache-identity";

function restore(name: string, value: string | undefined) {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

async function executable(path: string, source: string) {
  await writeFile(path, source);
  await Bun.write(path, source);
  await import("node:fs/promises").then(({ chmod }) => chmod(path, 0o755));
  return path;
}

async function v1Worker(dir: string) {
  const log = join(dir, "v1-commands.jsonl");
  const path = join(dir, "v1-worker");
  await executable(path, `#!/usr/bin/env bun
import { appendFileSync } from "node:fs";
const log = ${JSON.stringify(log)};
console.log(JSON.stringify({ protocol: 1, type: "ready", worker_capabilities: { streaming: true }, model_capabilities: {} }));
const decoder = new TextDecoder(); let buffer = ""; const next = new Map();
const send = (type, id, fields = {}) => { const sequence = next.get(id) ?? 0; next.set(id, sequence + 1); console.log(JSON.stringify({ protocol: 1, type, request_id: id, sequence, ...fields })); };
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true }); let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
    const command = JSON.parse(raw); appendFileSync(log, raw + "\\n");
    if (command.type === "shutdown") process.exit(0);
    send("accepted", command.request_id);
    if (command.type === "cancel") {
      send("completed", command.target_request_id, { result: { kind: "cancelled", finish_reason: "cancel" } });
      send("completed", command.request_id, { result: { kind: "cancelled" } }); continue;
    }
    send("started", command.request_id);
    if (command.type === "load") send("completed", command.request_id, { result: { kind: "loaded", token_capabilities: { model_context_window: 4096, effective_context_window: 4096, max_input_tokens: 4095, max_output_tokens: null, backend_allocation_cap: 4096, user_configured_override: null, model_context_window_source: "worker" } } });
    if (command.type === "generate") {
      send("prefill_progress", command.request_id, { completed: 2, total: 2 });
      send("token", command.request_id, { text: "v1 " });
      const request = command.request;
      if (request.messages[0].content !== "cancel") {
        send("content", command.request_id, { content: "response" });
        send("completed", command.request_id, { result: { kind: "generated", content: "v1 response", finish_reason: "stop", usage: { prompt_tokens: 2, completion_tokens: 2 }, cache: { hit: true, reused_tokens: 2, reuse_kind: "slot" }, timing: { prefill_ms: 3, first_emit_ms: 4 } } });
      }
    }
    if (command.type === "unload") send("completed", command.request_id, { result: { kind: "unloaded" } });
  }
}`);
  return { path, log };
}

async function noReadyWorker(dir: string) {
  const log = join(dir, "no-ready-commands.jsonl");
  const path = join(dir, "no-ready-worker");
  await executable(path, `#!/usr/bin/env bun
import { appendFileSync } from "node:fs";
const log = ${JSON.stringify(log)}; const decoder = new TextDecoder(); let buffer = "";
for await (const chunk of Bun.stdin.stream()) { buffer += decoder.decode(chunk, { stream: true }); let newline;
while ((newline = buffer.indexOf("\\n")) >= 0) { const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); if (!raw) continue;
const command = JSON.parse(raw); appendFileSync(log, raw + "\\n"); if (command.type === "shutdown") process.exit(0);
if (command.type === "load") console.log(JSON.stringify({ id: command.id, loaded: true, done: true }));
if (command.type === "generate") console.log(JSON.stringify({ id: command.request_id, content: "old-shape", done: true }));
}}`);
  return { path, log };
}

describe.serial("resident worker v1 migration", () => {
  test("cache identity rotation drains and removes every resident", async () => {
    const previous = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-rotation-"));
    try {
      const fake = await v1Worker(dir);
      process.env.CLAP_LLAMA_WORKER = fake.path;
      const model = join(dir, "model.gguf");
      await writeFile(model, "gguf");
      const registry = new ResidentWorkerRegistry();
      await registry.getOrCreate("rotation", "llama", model).load();
      expect(await registry.rotateCacheIdentityGeneration()).toBe(1);
      expect(registry.get("rotation")).toBeUndefined();
      expect(await registry.rotateCacheIdentityGeneration()).toBe(0);
    } finally {
      restore("CLAP_LLAMA_WORKER", previous);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("terminates a worker that violates the v1 handshake", async () => {
    const previous = process.env.CLAP_LLAMA_WORKER; const dir = await mkdtemp(join(tmpdir(), "clap-resident-v1-bad-"));
    try {
      process.env.CLAP_LLAMA_WORKER = await executable(join(dir, "bad-worker"),
        "#!/usr/bin/env bun\nconsole.log(JSON.stringify({ protocol: 2, type: 'ready', worker_capabilities: {}, model_capabilities: {} }));\nawait Bun.sleep(10000);\n");
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("bad", "llama", join(dir, "model.gguf"));
      await expect(worker.load()).rejects.toMatchObject({ code: "worker_protocol_error" });
      expect(worker.info().state).toBe("not_started");
      registry.shutdownAll();
    } finally { restore("CLAP_LLAMA_WORKER", previous); await rm(dir, { recursive: true, force: true }); }
  });

  test("reports exits during mandatory-v1 handshake", async () => {
    const previous = process.env.CLAP_LLAMA_WORKER; const dir = await mkdtemp(join(tmpdir(), "clap-resident-v1-exit-"));
    try {
      process.env.CLAP_LLAMA_WORKER = await executable(join(dir, "exit-worker"), "#!/usr/bin/env bun\nprocess.exit(0);\n");
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("exit", "llama", join(dir, "model.gguf"));
      await expect(worker.load()).rejects.toThrow("exited during protocol handshake with code 0");
      registry.shutdownAll();
    } finally { restore("CLAP_LLAMA_WORKER", previous); await rm(dir, { recursive: true, force: true }); }
  });

  test("negotiates v1, sends envelopes, and maps stream facts", async () => {
    const previous = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-v1-"));
    try {
      const fake = await v1Worker(dir); process.env.CLAP_LLAMA_WORKER = fake.path;
      await writeFile(join(dir, "model.gguf"), "gguf");
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("v1", "llama", join(dir, "model.gguf"));
      const tokens: string[] = []; const progress: Array<[number, number]> = []; let dispatches = 0;
      const result = await worker.chat({ model: "model", messages: [{ role: "user", content: "hello" }], stream: true,
        cache_identity: { secret: "caller-controlled" } } as never,
        (token) => tokens.push(token), undefined, (done, total) => progress.push([done, total]), () => dispatches++, cacheOptions);
      expect(result).toMatchObject({
        content: "v1 response",
        finishReason: "stop",
        usage: { promptTokens: 2, completionTokens: 2 },
        cache: { hit: true, reusedTokens: 2, reuseKind: "slot" },
        timing: { prefillMs: 3, firstEmitMs: 4 },
      });
      expect(tokens).toEqual(["v1 ", "response"]); expect(progress).toEqual([[2, 2]]); expect(dispatches).toBe(1);
      expect(worker.info().tokenCapabilities?.modelContextWindowSource).toBe("worker");
      const commands = (await readFile(fake.log, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
      expect(commands.map(({ type }) => type)).toEqual(["load", "generate"]);
      expect(commands.every(({ protocol, request_id }) => protocol === 1 && request_id.startsWith("req_"))).toBe(true);
      expect(commands[1].request.messages[0].content).toBe("hello");
      expect(commands[1].cache_identity).toEqual(cacheOptions.cacheIdentity);
      expect(commands[1].request.cache_identity).toBeUndefined();
      registry.shutdownAll();
    } finally { restore("CLAP_LLAMA_WORKER", previous); await rm(dir, { recursive: true, force: true }); }
  });

  test("uses a distinct command ID when cancelling v1 generation", async () => {
    const previous = process.env.CLAP_LLAMA_WORKER; const dir = await mkdtemp(join(tmpdir(), "clap-resident-v1-cancel-"));
    try {
      const fake = await v1Worker(dir); process.env.CLAP_LLAMA_WORKER = fake.path; await writeFile(join(dir, "model.gguf"), "gguf");
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("v1", "llama", join(dir, "model.gguf")); const controller = new AbortController();
      const result = await worker.chat({ model: "model", messages: [{ role: "user", content: "cancel" }], stream: true }, () => controller.abort(), controller.signal, undefined, undefined, cacheOptions);
      expect(result.finishReason).toBe("cancel");
      const commands = (await readFile(fake.log, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
      const generate = commands.find(({ type }) => type === "generate"); const cancel = commands.find(({ type }) => type === "cancel");
      expect(cancel.target_request_id).toBe(generate.request_id); expect(cancel.request_id).not.toBe(generate.request_id); expect(cancel.request_id).toStartWith("cmd_");
      registry.shutdownAll();
    } finally { restore("CLAP_LLAMA_WORKER", previous); await rm(dir, { recursive: true, force: true }); }
  });

  test("rejects configured workers that do not send ready/v1", async () => {
    const previous = process.env.CLAP_LLAMA_WORKER; const dir = await mkdtemp(join(tmpdir(), "clap-resident-auto-"));
    try {
      const fake = await noReadyWorker(dir); process.env.CLAP_LLAMA_WORKER = fake.path; await writeFile(join(dir, "model.gguf"), "gguf");
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("no-ready", "llama", join(dir, "model.gguf"));
      await expect(worker.load()).rejects.toMatchObject({ code: "worker_protocol_error" });
      await expect(Bun.file(fake.log).exists()).resolves.toBe(false);
      registry.shutdownAll();
    } finally { restore("CLAP_LLAMA_WORKER", previous); await rm(dir, { recursive: true, force: true }); }
  });
});
