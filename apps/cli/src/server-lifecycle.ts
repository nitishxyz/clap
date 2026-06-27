import { createClapClient, defaultBaseURL, type HealthResponse } from "@clap/api";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

export type ServerMetadata = {
  pid: number;
  port: number;
  baseURL: string;
  startedAt: string;
};

export type ServerPaths = {
  home: string;
  stateFile: string;
  lockDir: string;
  stdoutLog: string;
  stderrLog: string;
};

export function clapHome(): string {
  return process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap");
}

export function serverPaths(home = clapHome()): ServerPaths {
  return {
    home,
    stateFile: join(home, "server.json"),
    lockDir: join(home, "server.lock"),
    stdoutLog: join(home, "server.log"),
    stderrLog: join(home, "server.err.log"),
  };
}

export function baseURLFromEnv(): string {
  return (process.env.CLAP_BASE_URL ?? defaultBaseURL).replace(/\/$/, "");
}

export function portFromBaseURL(baseURL: string): number {
  return Number(new URL(baseURL).port || "80");
}

export async function readServerMetadata(paths = serverPaths()): Promise<ServerMetadata | null> {
  try {
    return JSON.parse(await readFile(paths.stateFile, "utf8")) as ServerMetadata;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

export async function writeServerMetadata(metadata: ServerMetadata, paths = serverPaths()): Promise<void> {
  await mkdir(paths.home, { recursive: true });
  await writeFile(paths.stateFile, `${JSON.stringify(metadata, null, 2)}\n`);
}

export async function removeServerMetadata(paths = serverPaths()): Promise<void> {
  await rm(paths.stateFile, { force: true });
}

export async function healthCheck(baseURL = baseURLFromEnv()): Promise<HealthResponse | null> {
  try {
    return await createClapClient({ baseURL }).health();
  } catch {
    return null;
  }
}

export async function withServerStartLock<T>(fn: () => Promise<T>, paths = serverPaths()): Promise<T> {
  await mkdir(paths.home, { recursive: true });
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      await mkdir(paths.lockDir);
      try {
        return await fn();
      } finally {
        await rm(paths.lockDir, { recursive: true, force: true });
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      await Bun.sleep(100);
    }
  }
  throw new Error(`timed out waiting for server start lock at ${paths.lockDir}`);
}

export async function waitForHealthy(baseURL: string, timeoutMs = 10_000): Promise<HealthResponse | null> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const health = await healthCheck(baseURL);
    if (health?.status === "ok") return health;
    await Bun.sleep(100);
  }
  return null;
}

export async function tailFile(path: string, lineCount: number): Promise<string> {
  try {
    const text = await readFile(path, "utf8");
    return text.split("\n").slice(-lineCount).join("\n");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return "";
    throw error;
  }
}

export function currentCliCommand(): string[] {
  return [process.execPath, import.meta.resolve("./index.ts").replace("file://", "")];
}

export function launchdPlist(command: string[], paths = serverPaths()): string {
  const args = [...command, "serve"].map((arg) => `    <string>${escapeXml(arg)}</string>`).join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.clap.server</string>
  <key>ProgramArguments</key>
  <array>
${args}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${escapeXml(paths.stdoutLog)}</string>
  <key>StandardErrorPath</key>
  <string>${escapeXml(paths.stderrLog)}</string>
</dict>
</plist>
`;
}

export function systemdService(command: string[], paths = serverPaths()): string {
  return `[Unit]
Description=Clap local model server

[Service]
ExecStart=${command.map(quoteSystemdArg).join(" ")} serve
Restart=on-failure
StandardOutput=append:${paths.stdoutLog}
StandardError=append:${paths.stderrLog}

[Install]
WantedBy=default.target
`;
}

function escapeXml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function quoteSystemdArg(value: string): string {
  if (/^[A-Za-z0-9_./:@%+=,-]+$/.test(value)) return value;
  return `'${value.replaceAll("'", "'\\''")}'`;
}
