import { describe, expect, test } from "bun:test";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { ResidentWorkerRegistry } from "./resident";

describe("resident worker registry", () => {
  test("loads, reuses one pid for chats, and shuts down", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-registry-test-"));
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeResidentWorker(dir);
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("key", "llama", join(dir, "model.gguf"));
      await writeFile(join(dir, "model.gguf"), "gguf");

      const info = await worker.load();
      expect(info.state).toBe("resident");
      expect(info.pid).toBeNumber();
      const tokens: string[] = [];
      let dispatches = 0;
      const first = await worker.chat(
        { model: join(dir, "model.gguf"), messages: [{ role: "user", content: "one" }], stream: false },
        (token) => tokens.push(token),
        undefined,
        undefined,
        () => { dispatches += 1; },
      );
      const second = await worker.chat({ model: join(dir, "model.gguf"), messages: [{ role: "user", content: "two" }], stream: false });
      expect(first.content).toBe("resident response");
      expect(first.usage).toEqual({ promptTokens: 12, completionTokens: 2 });
      expect(first.finishReason).toBe("stop");
      expect(tokens).toEqual(["resident ", "response"]);
      expect(dispatches).toBe(1);
      expect(second.content).toBe("resident response");
      expect(worker.info().pid).toBe(info.pid);

      registry.shutdownAll();
      expect(worker.info().state).toBe("not_started");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("sends cancel to the worker when the signal aborts", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-cancel-test-"));
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeCancellableWorker(dir);
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("key", "llama", join(dir, "model.gguf"));
      await writeFile(join(dir, "model.gguf"), "gguf");

      const controller = new AbortController();
      const tokens: string[] = [];
      const pending = worker.chat(
        { model: join(dir, "model.gguf"), messages: [{ role: "user", content: "long" }], stream: true },
        (token) => {
          tokens.push(token);
          if (tokens.length === 3) controller.abort();
        },
        controller.signal,
      );
      const result = await pending;
      expect(result.finishReason).toBe("cancel");
      expect(tokens.length).toBeLessThan(50);
      registry.shutdownAll();
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });
});

async function fakeResidentWorker(dir: string): Promise<string> {
  const path = join(dir, "fake-resident-worker");
  await writeFile(path, `#!/usr/bin/env bun
const decoder = new TextDecoder();
let buffer = "";
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const request = JSON.parse(line);
    if (request.type === "shutdown") process.exit(0);
    if (request.type === "load") {
      console.log(JSON.stringify({ id: request.id, loaded: true, done: true }));
      continue;
    }
    if (request.type === "unload") {
      console.log(JSON.stringify({ id: request.id, unloaded: true, done: true }));
      continue;
    }
    console.log(JSON.stringify({ id: request.id, started: true }));
    console.log(JSON.stringify({ id: request.id, token: "resident " }));
    console.log(JSON.stringify({ id: request.id, token: "response" }));
    console.log(JSON.stringify({ id: request.id, done: true, finish_reason: "stop", usage: { prompt_tokens: 12, completion_tokens: 2 } }));
  }
}
`);
  await chmod(path, 0o755);
  return path;
}

async function fakeCancellableWorker(dir: string): Promise<string> {
  const path = join(dir, "fake-cancellable-worker");
  await writeFile(path, `#!/usr/bin/env bun
const decoder = new TextDecoder();
let buffer = "";
let active = null;
let cancelled = false;
async function generate(id) {
  for (let i = 0; i < 50; i += 1) {
    if (cancelled) {
      console.log(JSON.stringify({ id, done: true, cancelled: true, finish_reason: "cancel" }));
      return;
    }
    console.log(JSON.stringify({ id, token: "t" + i + " " }));
    await Bun.sleep(5);
  }
  console.log(JSON.stringify({ id, done: true, finish_reason: "stop" }));
}
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const request = JSON.parse(line);
    if (request.type === "shutdown") process.exit(0);
    if (request.type === "cancel") {
      cancelled = true;
      continue;
    }
    if (request.type === "load") {
      console.log(JSON.stringify({ id: request.id, loaded: true, done: true }));
      continue;
    }
    if (request.type === "chat") {
      cancelled = false;
      active = generate(request.id);
    }
  }
}
`);
  await chmod(path, 0o755);
  return path;
}

function restoreEnv(name: string, value: string | undefined) {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}
