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
      const first = await worker.chat({ model: join(dir, "model.gguf"), messages: [{ role: "user", content: "one" }], stream: false });
      const second = await worker.chat({ model: join(dir, "model.gguf"), messages: [{ role: "user", content: "two" }], stream: false });
      expect(first).toBe("resident response");
      expect(second).toBe("resident response");
      expect(worker.info().pid).toBe(info.pid);

      registry.shutdownAll();
      expect(worker.info().state).toBe("not_started");
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
    console.log(JSON.stringify({ id: request.id, content: "resident response" }));
    console.log(JSON.stringify({ id: request.id, done: true }));
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
