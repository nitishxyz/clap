import type { Backend } from "@clap/api";
import { existsSync } from "node:fs";
import { access } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

export type LlamaWorkerStatus = {
  available: boolean;
  command?: string[];
  source: "configured" | "bundled" | "missing";
  reason?: string;
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
  if (!resolved) {
    return {
      available: false,
      source: "missing",
      reason: "missing: clap-llama worker not found. Set CLAP_LLAMA_WORKER, install bundled libexec/clap-llama, or run bun run bundle:check for packaging diagnostics.",
    };
  }
  return { available: true, command: resolved.command, source: resolved.source };
}

export function ggufModelDisplayName(model: string): string {
  return basename(model);
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
