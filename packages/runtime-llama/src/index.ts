import type { Backend, ChatCompletionRequest } from "@clap/api";
import { existsSync } from "node:fs";
import { access, mkdir } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

export type LlamaWorkerStatus = {
  available: boolean;
  command?: string[];
  source: "configured" | "bundled" | "missing";
  logPath: string;
  reason?: string;
};

export type LlamaCompletionOptions = {
  request: ChatCompletionRequest;
  stream?: boolean;
};

export class LlamaWorkerError extends Error {
  constructor(message: string, public readonly code = "llama_worker_error") {
    super(message);
    this.name = "LlamaWorkerError";
  }
}

export function llamaBackendStatus(): Backend {
  const status = getLlamaWorkerStatus();
  return {
    id: "llama",
    name: "llama.cpp",
    formats: ["gguf"],
    status: status.available ? "available" : "not_installed",
    reason: status.available ? `${status.source}: ${status.command?.join(" ")}` : status.reason,
  };
}

export const llamaBackend = llamaBackendStatus();

export function isGgufModel(model: string): boolean {
  return model.toLowerCase().endsWith(".gguf");
}

export async function assertGgufModelPath(model: string): Promise<void> {
  if (!isGgufModel(model)) return;
  try {
    await access(model);
  } catch {
    throw new LlamaWorkerError(`GGUF model not found: ${model}`, "model_not_found");
  }
}

export function getLlamaWorkerStatus(): LlamaWorkerStatus {
  const resolved = resolveLlamaWorkerCommand();
  const logPath = llamaLogPath();
  if (!resolved) {
    return {
      available: false,
      source: "missing",
      logPath,
      reason: "missing: clap-llama worker not found. Set CLAP_LLAMA_WORKER, install bundled libexec/clap-llama, or run bun run bundle:check for packaging diagnostics.",
    };
  }
  return { available: true, command: resolved.command, source: resolved.source, logPath };
}

export async function completeWithLlama(options: LlamaCompletionOptions): Promise<string> {
  const tokens: string[] = [];
  for await (const token of streamWithLlama(options)) {
    tokens.push(token);
  }
  return tokens.join("");
}

export async function* streamWithLlama(options: LlamaCompletionOptions): AsyncGenerator<string> {
  await assertGgufModelPath(options.request.model);
  const status = getLlamaWorkerStatus();
  if (!status.command) {
    throw new LlamaWorkerError(status.reason ?? "clap-llama worker is not available", "worker_not_found");
  }

  await mkdir(dirname(status.logPath), { recursive: true });
  const proc = Bun.spawn(status.command, {
    stdin: "pipe",
    stdout: "pipe",
    stderr: Bun.file(status.logPath),
    env: process.env,
  });

  proc.stdin.write(`${JSON.stringify({ type: "chat", ...options.request, stream: options.stream ?? true })}\n`);
  proc.stdin.end();

  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of proc.stdout) {
    buffer += decoder.decode(chunk, { stream: true });
    let newlineIndex: number;
    while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;
      const token = parseWorkerLine(line);
      if (token) yield token;
    }
  }

  const tail = buffer.trim();
  if (tail) {
    const token = parseWorkerLine(tail);
    if (token) yield token;
  }

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new LlamaWorkerError(`clap-llama exited with code ${exitCode}; see ${status.logPath}`);
  }
}

export function ggufModelDisplayName(model: string): string {
  return basename(model);
}

function parseWorkerLine(line: string): string | null {
  try {
    const message = JSON.parse(line) as { token?: unknown; content?: unknown; done?: unknown; error?: unknown };
    if (message.error) throw new LlamaWorkerError(String(message.error));
    if (message.done) return null;
    if (typeof message.token === "string") return message.token;
    if (typeof message.content === "string") return message.content;
    return null;
  } catch (error) {
    if (error instanceof LlamaWorkerError) throw error;
    return line;
  }
}

function resolveLlamaWorkerCommand(): { command: string[]; source: LlamaWorkerStatus["source"] } | null {
  const configured = process.env.CLAP_LLAMA_WORKER;
  if (configured) {
    const command = splitCommand(configured);
    const source = isEmbeddedWorkerCommand(command[0], "clap-llama") ? "bundled" : "configured";
    return isExecutableFile(command[0]) ? { command, source } : null;
  }
  for (const bundled of bundledWorkerCandidates("clap-llama")) {
    if (isExecutableFile(bundled)) return { command: [bundled], source: "bundled" };
  }
  return null;
}

function isEmbeddedWorkerCommand(command: string | undefined, name: string): boolean {
  const build = process.env.CLAP_EMBED_BUILD;
  if (!command || !build) return false;
  const home = process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap");
  return resolve(command) === resolve(home, "libexec", build, name);
}

function bundledWorkerCandidates(name: string): string[] {
  const packageRoot = resolve(import.meta.dir, "..", "..", "..");
  return [
    join(dirname(process.execPath), "..", "libexec", name),
    join(dirname(process.execPath), "libexec", name),
    join(packageRoot, "libexec", name),
    join(packageRoot, "libexec", "runtimes", name),
  ];
}

function splitCommand(command: string): string[] {
  return command.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)?.map((part) => part.replace(/^['"]|['"]$/g, "")) ?? [command];
}

function isExecutableFile(path: string | undefined): boolean {
  return Boolean(path && existsSync(path) && Bun.file(path).size > 0);
}

function llamaLogPath(): string {
  return join(process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap"), "llama-worker.err.log");
}
