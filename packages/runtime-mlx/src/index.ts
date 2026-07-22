import type { Backend, ChatCompletionRequest } from "@clap/api";
import { existsSync, readdirSync, statSync } from "node:fs";
import { mkdir, readFile, readdir, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

export type MlxWorkerStatus = {
  available: boolean;
  command?: string[];
  source: "configured" | "bundled" | "missing";
  logPath: string;
  reason?: string;
};

export type MlxCompletionOptions = {
  request: ChatCompletionRequest;
  stream?: boolean;
};

export class MlxWorkerError extends Error {
  constructor(message: string, public readonly code = "mlx_worker_error") {
    super(message);
    this.name = "MlxWorkerError";
  }
}

export function mlxBackendStatus(): Backend {
  const status = getMlxWorkerStatus();
  return {
    id: "mlx",
    name: "Swift MLX",
    formats: ["mlx"],
    status: status.available ? "available" : isMlxPlatformSupported() ? "not_installed" : "unsupported",
    reason: status.available ? `${status.source}: ${status.command?.join(" ")}` : status.reason,
  };
}

export const mlxBackend = mlxBackendStatus();

export function isMlxPlatformSupported(): boolean {
  return process.platform === "darwin" && process.arch === "arm64";
}

export function isMlxModel(model: string): boolean {
  const lower = model.toLowerCase();
  if (lower.endsWith(".gguf")) return false;
  if (lower.endsWith(".mlx") || lower.includes("mlx-community")) return true;
  return mlxModelPaths().includes(model);
}

export async function isMlxModelDirectory(model: string): Promise<boolean> {
  if (isMlxModel(model)) return true;
  try {
    const modelStat = await stat(model);
    if (!modelStat.isDirectory()) return false;
    return hasMlxDirectoryLayout(await readdir(model));
  } catch {
    return false;
  }
}

export function isMlxModelDirectorySync(model: string): boolean {
  if (isMlxModel(model)) return true;
  try {
    const modelStat = statSync(model);
    if (!modelStat.isDirectory()) return false;
    return hasMlxDirectoryLayout(readdirSync(model));
  } catch {
    return false;
  }
}

export async function assertMlxModelPath(model: string): Promise<void> {
  try {
    const modelStat = await stat(model);
    if (!modelStat.isDirectory()) throw new Error("not a directory");
    const entries = await readdir(model);
    if (!hasMlxDirectoryLayout(entries)) {
      throw new MlxWorkerError(`MLX model directory is missing required files (config.json, tokenizer.json/tokenizer_config.json, and .safetensors): ${model}`, "invalid_model_directory");
    }
  } catch (error) {
    if (error instanceof MlxWorkerError) throw error;
    throw new MlxWorkerError(`MLX model directory not found: ${model}`, "model_not_found");
  }
}

export function getMlxWorkerStatus(): MlxWorkerStatus {
  const logPath = mlxLogPath();
  if (!isMlxPlatformSupported()) {
    return {
      available: false,
      source: "missing",
      logPath,
      reason: "MLX backend requires macOS arm64 and a bundled clap-mlx worker.",
    };
  }

  const resolved = resolveMlxWorkerCommand();
  if (!resolved) {
    return {
      available: false,
      source: "missing",
      logPath,
      reason: "missing: clap-mlx worker not found. Set CLAP_MLX_WORKER, install bundled libexec/clap-mlx, or run bun run bundle:check for packaging diagnostics.",
    };
  }
  const metalLibraryReason = missingMetalLibraryReason(resolved.command[0]);
  if (metalLibraryReason) {
    return {
      available: false,
      source: resolved.source,
      logPath,
      reason: metalLibraryReason,
    };
  }
  return { available: true, command: resolved.command, source: resolved.source, logPath };
}

export async function completeWithMlx(options: MlxCompletionOptions): Promise<string> {
  const tokens: string[] = [];
  for await (const token of streamWithMlx(options)) {
    tokens.push(token);
  }
  return tokens.join("");
}

export async function* streamWithMlx(options: MlxCompletionOptions): AsyncGenerator<string> {
  await assertMlxModelPath(options.request.model);
  const status = getMlxWorkerStatus();
  if (!isMlxPlatformSupported()) {
    throw new MlxWorkerError(status.reason ?? "MLX backend is not supported on this platform", "platform_unsupported");
  }
  if (!status.command) {
    throw new MlxWorkerError(status.reason ?? "clap-mlx worker is not available", "worker_not_found");
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
  const stdoutDiagnostics: string[] = [];
  const workerErrors: string[] = [];
  for await (const chunk of proc.stdout) {
    buffer += decoder.decode(chunk, { stream: true });
    let newlineIndex: number;
    while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;
      const parsed = parseWorkerLine(line);
      if (parsed.kind === "token") yield parsed.value;
      if (parsed.kind === "error") workerErrors.push(parsed.value);
      if (parsed.kind === "diagnostic") stdoutDiagnostics.push(parsed.value);
    }
  }

  const tail = buffer.trim();
  if (tail) {
    const parsed = parseWorkerLine(tail);
    if (parsed.kind === "token") yield parsed.value;
    if (parsed.kind === "error") workerErrors.push(parsed.value);
    if (parsed.kind === "diagnostic") stdoutDiagnostics.push(parsed.value);
  }

  const exitCode = await proc.exited;
  const stderr = await readWorkerLog(status.logPath);
  if (workerErrors.length > 0) {
    throw new MlxWorkerError(workerFailureMessage({
      reason: workerErrors.join("; "),
      exitCode,
      stdoutDiagnostics,
      stderr,
      logPath: status.logPath,
    }));
  }
  if (exitCode !== 0) {
    throw new MlxWorkerError(workerFailureMessage({
      reason: `clap-mlx exited with code ${exitCode}`,
      exitCode,
      stdoutDiagnostics,
      stderr,
      logPath: status.logPath,
    }));
  }
}

export function mlxModelDisplayName(model: string): string {
  return basename(model);
}

export function mlxModelPaths(): string[] {
  return (process.env.CLAP_MLX_MODEL_PATHS ?? "")
    .split(",")
    .map((model) => model.trim())
    .filter(Boolean);
}

type ParsedWorkerLine =
  | { kind: "token"; value: string }
  | { kind: "done" }
  | { kind: "error"; value: string }
  | { kind: "diagnostic"; value: string };

function parseWorkerLine(line: string): ParsedWorkerLine {
  try {
    const message = JSON.parse(line) as { token?: unknown; content?: unknown; done?: unknown; error?: unknown };
    if (message.error) return { kind: "error", value: String(message.error) };
    if (message.done) return { kind: "done" };
    if (typeof message.token === "string") return { kind: "token", value: message.token };
    if (typeof message.content === "string") return { kind: "token", value: message.content };
    return { kind: "done" };
  } catch {
    return { kind: "diagnostic", value: line };
  }
}

function hasMlxDirectoryLayout(entries: string[]): boolean {
  const entrySet = new Set(entries);
  return entrySet.has("config.json") && (
    entrySet.has("tokenizer.json") ||
    entrySet.has("tokenizer_config.json")
  ) && entries.some((entry) => entry.endsWith(".safetensors"));
}

function missingMetalLibraryReason(workerPath: string | undefined): string | undefined {
  if (!workerPath) return "missing: clap-mlx worker path is empty";
  const workerDir = dirname(workerPath);
  const candidates = [
    join(workerDir, "mlx.metallib"),
    join(workerDir, "Resources", "mlx.metallib"),
    join(workerDir, "Resources", "default.metallib"),
  ];
  if (candidates.some((candidate) => existsSync(candidate) && Bun.file(candidate).size > 0)) return undefined;
  return `missing: clap-mlx Metal shader library not found next to ${workerPath}. Expected mlx.metallib or Resources/mlx.metallib; rebuild with bun run mlx:build and include the generated metallib in libexec.`;
}

async function readWorkerLog(logPath: string): Promise<string> {
  try {
    return truncateDiagnostic((await readFile(logPath, "utf8")).trim());
  } catch {
    return "";
  }
}

function workerFailureMessage(options: { reason: string; exitCode: number; stdoutDiagnostics: string[]; stderr: string; logPath: string }): string {
  const details = [options.reason, `exitCode=${options.exitCode}`];
  const stdout = truncateDiagnostic(options.stdoutDiagnostics.join("\n").trim());
  if (stdout) details.push(`stdout=${JSON.stringify(stdout)}`);
  if (options.stderr) details.push(`stderr=${JSON.stringify(options.stderr)}`);
  details.push(`stderrLog=${options.logPath}`);
  return details.join("; ");
}

function truncateDiagnostic(value: string): string {
  return value.length > 4_000 ? `${value.slice(0, 4_000)}…` : value;
}

function resolveMlxWorkerCommand(): { command: string[]; source: MlxWorkerStatus["source"] } | null {
  const configured = process.env.CLAP_MLX_WORKER;
  if (configured) {
    const command = splitCommand(configured);
    const source = isEmbeddedWorkerCommand(command[0], "clap-mlx") ? "bundled" : "configured";
    return isExecutableFile(command[0]) ? { command, source } : null;
  }
  for (const bundled of bundledWorkerCandidates("clap-mlx")) {
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
  const packageRoot = join(import.meta.dir, "..", "..", "..");
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

function mlxLogPath(): string {
  return join(process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap"), "mlx-worker.err.log");
}
