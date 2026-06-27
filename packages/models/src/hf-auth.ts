import { existsSync, statSync } from "node:fs";
import { chmod, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

export const hfTokenEnvVars = ["CLAP_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"] as const;

export type HfCredentialSource = "env" | "keychain" | "libsecret" | "file" | "none";

export type HfCredentialStatus = {
  authenticated: boolean;
  source: HfCredentialSource;
  detail?: string;
  tokenPreview?: string;
  envVar?: string;
};

const serviceName = "dev.clap.huggingface";
const accountName = "huggingface-token";

export async function resolveHfToken(): Promise<{ token?: string; source: HfCredentialSource; envVar?: string }> {
  const env = envToken();
  if (env) return { token: env.token, source: "env", envVar: env.name };

  const stored = await readStoredHfToken();
  if (stored.token) return stored;
  return { source: "none" };
}

export async function hfAuthStatus(): Promise<HfCredentialStatus> {
  const resolved = await resolveHfToken();
  if (!resolved.token) {
    return { authenticated: false, source: "none", detail: storageDetail() };
  }
  return {
    authenticated: true,
    source: resolved.source,
    detail: resolved.envVar ?? storageDetail(resolved.source),
    tokenPreview: redactToken(resolved.token),
    envVar: resolved.envVar,
  };
}

export async function storeHfToken(token: string): Promise<HfCredentialStatus> {
  const trimmed = token.trim();
  if (!trimmed) throw new Error("Hugging Face token cannot be empty");

  const backend = preferredStorageBackend();
  if (backend === "keychain" && await hasCommand("security")) {
    const result = await runCommand(["security", "add-generic-password", "-a", accountName, "-s", serviceName, "-w", trimmed, "-U"]);
    if (result.ok) return statusForStoredToken(trimmed, "keychain");
  }
  if (backend === "libsecret" && await hasCommand("secret-tool")) {
    const result = await runCommand(["secret-tool", "store", "--label=Clap Hugging Face Token", "service", serviceName, "account", accountName], trimmed);
    if (result.ok) return statusForStoredToken(trimmed, "libsecret");
  }

  await writeFileToken(trimmed);
  return statusForStoredToken(trimmed, "file");
}

export async function deleteStoredHfToken(): Promise<HfCredentialStatus> {
  let deleted = false;
  if (await hasCommand("security")) {
    const result = await runCommand(["security", "delete-generic-password", "-a", accountName, "-s", serviceName]);
    deleted ||= result.ok;
  }
  if (await hasCommand("secret-tool")) {
    const result = await runCommand(["secret-tool", "clear", "service", serviceName, "account", accountName]);
    deleted ||= result.ok;
  }
  if (existsSync(tokenFilePath())) {
    await rm(tokenFilePath(), { force: true });
    deleted = true;
  }
  return { authenticated: false, source: "none", detail: deleted ? "stored credential removed" : "no stored credential found" };
}

export function isHfAuthError(error: unknown): boolean {
  return error instanceof Error && error.message.includes("Hugging Face authentication failed");
}

export function hfAuthGuidance(): string {
  return `Run clap auth login or set ${hfTokenEnvVars.join(", ")}.`;
}

function envToken(): { token: string; name: string } | undefined {
  for (const name of hfTokenEnvVars) {
    const token = process.env[name]?.trim();
    if (token) return { token, name };
  }
  return undefined;
}

async function readStoredHfToken(): Promise<{ token?: string; source: HfCredentialSource }> {
  const backend = preferredStorageBackend();
  if (backend === "keychain" && await hasCommand("security")) {
    const result = await runCommand(["security", "find-generic-password", "-a", accountName, "-s", serviceName, "-w"]);
    if (result.ok && result.stdout.trim()) return { token: result.stdout.trim(), source: "keychain" };
  }
  if (backend === "libsecret" && await hasCommand("secret-tool")) {
    const result = await runCommand(["secret-tool", "lookup", "service", serviceName, "account", accountName]);
    if (result.ok && result.stdout.trim()) return { token: result.stdout.trim(), source: "libsecret" };
  }
  if (existsSync(tokenFilePath())) {
    const token = (await readFile(tokenFilePath(), "utf8")).trim();
    if (token) return { token, source: "file" };
  }
  return { source: "none" };
}

function preferredStorageBackend(): HfCredentialSource {
  const forced = process.env.CLAP_HF_AUTH_BACKEND;
  if (forced === "file" || forced === "keychain" || forced === "libsecret") return forced;
  if (process.platform === "darwin") return "keychain";
  if (process.platform === "linux") return "libsecret";
  return "file";
}

function storageDetail(source: HfCredentialSource = preferredStorageBackend()): string {
  if (source === "keychain") return "macOS Keychain service dev.clap.huggingface";
  if (source === "libsecret") return "libsecret service dev.clap.huggingface";
  if (source === "file") return tokenFilePath();
  return "no credential source configured";
}

function statusForStoredToken(token: string, source: HfCredentialSource): HfCredentialStatus {
  return {
    authenticated: true,
    source,
    detail: storageDetail(source),
    tokenPreview: redactToken(token),
  };
}

async function writeFileToken(token: string): Promise<void> {
  const file = tokenFilePath();
  await mkdir(dirname(file), { recursive: true, mode: 0o700 });
  await chmod(dirname(file), 0o700);
  await writeFile(file, `${token}\n`, { mode: 0o600 });
  await chmod(file, 0o600);
}

function tokenFilePath(): string {
  return join(clapHome(), "auth", "huggingface-token");
}

function clapHome(): string {
  return process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap");
}

function redactToken(token: string): string {
  if (token.length <= 8) return `${token.slice(0, 2)}...${token.slice(-2)}`;
  return `${token.slice(0, 6)}...${token.slice(-4)}`;
}

async function hasCommand(command: string): Promise<boolean> {
  if (process.env.CLAP_HF_AUTH_BACKEND === "file") return false;
  const result = await runCommand(["/bin/sh", "-lc", `command -v ${command} >/dev/null 2>&1`]);
  return result.ok;
}

async function runCommand(command: string[], stdin?: string): Promise<{ ok: boolean; stdout: string; stderr: string }> {
  try {
    const proc = Bun.spawn(command, {
      stdin: stdin === undefined ? "ignore" : "pipe",
      stdout: "pipe",
      stderr: "pipe",
      env: process.env,
    });
    if (stdin !== undefined) {
      proc.stdin!.write(stdin);
      proc.stdin!.end();
    }
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);
    return { ok: exitCode === 0, stdout, stderr };
  } catch (error) {
    return { ok: false, stdout: "", stderr: error instanceof Error ? error.message : String(error) };
  }
}

export function assertFileCredentialPermissions(): { fileMode?: number; dirMode?: number } {
  const file = tokenFilePath();
  const dir = dirname(file);
  return {
    fileMode: existsSync(file) ? statSync(file).mode & 0o777 : undefined,
    dirMode: existsSync(dir) ? statSync(dir).mode & 0o777 : undefined,
  };
}
