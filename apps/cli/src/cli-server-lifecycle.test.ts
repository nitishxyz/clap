import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readServerMetadata, serverPaths, writeServerMetadata } from "./server-lifecycle";

const cli = [process.execPath, "apps/cli/src/index.ts"];
const managedHomes: string[] = [];
const dummyServers: Bun.Subprocess[] = [];

afterEach(async () => {
  for (const proc of dummyServers.splice(0)) {
    try {
      proc.kill("SIGTERM");
    } catch {
      // already exited
    }
    await Promise.race([proc.exited, Bun.sleep(500)]);
  }
  for (const home of managedHomes.splice(0)) {
    const metadata = await readServerMetadata(serverPaths(home));
    if (metadata?.baseURL) await runCli(["server", "stop", "--force"], { home, baseURL: metadata.baseURL, allowFailure: true });
    await rm(home, { recursive: true, force: true });
  }
});

describe("CLI server lifecycle", () => {
  test("removes stale pid metadata when server is unhealthy", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-cli-test-"));
    managedHomes.push(home);
    const baseURL = `http://127.0.0.1:${await freePort()}`;
    const paths = serverPaths(home);
    await writeServerMetadata({
      pid: 999_999,
      port: new URL(baseURL).port ? Number(new URL(baseURL).port) : 80,
      baseURL,
      startedAt: "2026-06-27T00:00:00.000Z",
      managed: true,
    }, paths);

    const result = await runCli(["server", "stop"], { home, baseURL });

    expect(result.stdout).toContain("removed stale metadata");
    await expect(readServerMetadata(paths)).resolves.toBeNull();
  });

  test("start does not write pid 0 metadata for an unmanaged healthy server", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-cli-test-"));
    managedHomes.push(home);
    const dummy = await startDummyHealthyServer();

    const result = await runCli(["server", "start"], { home, baseURL: dummy.baseURL });

    expect(result.stdout).toContain("not managed by this CLI");
    await expect(readServerMetadata(serverPaths(home))).resolves.toBeNull();
  });

  test("restart replaces a managed server", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-cli-test-"));
    managedHomes.push(home);
    const baseURL = `http://127.0.0.1:${await freePort()}`;

    await runCli(["server", "start"], { home, baseURL });
    const first = await readServerMetadata(serverPaths(home));
    expect(first?.pid).toBeGreaterThan(0);

    await runCli(["server", "restart"], { home, baseURL });
    const second = await readServerMetadata(serverPaths(home));

    expect(second?.pid).toBeGreaterThan(0);
    expect(second?.pid).not.toBe(first?.pid);
  });

  test("restart fails clearly for an unmanaged healthy server without force", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-cli-test-"));
    managedHomes.push(home);
    const dummy = await startDummyHealthyServer();

    const result = await runCli(["server", "restart"], { home, baseURL: dummy.baseURL, allowFailure: true });

    expect(result.exitCode).not.toBe(0);
    expect(result.stderr).toContain("no managed server metadata exists");
    expect(result.stderr).toContain("restart --force");
  });

  test("stop --force kills an unmanaged healthy server", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-cli-test-"));
    managedHomes.push(home);
    const dummy = await startDummyHealthyServer();

    const result = await runCli(["server", "stop", "--force"], { home, baseURL: dummy.baseURL });

    expect(result.stdout).toContain("stopped with --force");
    expect(await isHealthy(dummy.baseURL)).toBe(false);
  });

  test("restart --force replaces an unmanaged healthy server with a managed server", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-cli-test-"));
    managedHomes.push(home);
    const dummy = await startDummyHealthyServer();

    const result = await runCli(["server", "restart", "--force"], { home, baseURL: dummy.baseURL });
    const metadata = await readServerMetadata(serverPaths(home));

    expect(result.stdout).toContain("clap server started");
    expect(metadata?.pid).toBeGreaterThan(0);
    expect(metadata?.managed).toBe(true);
  });
});

async function runCli(argv: string[], options: { home: string; baseURL: string; allowFailure?: boolean }) {
  const proc = Bun.spawn([...cli, ...argv], {
    env: { ...process.env, CLAP_HOME: options.home, CLAP_BASE_URL: options.baseURL },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  if (!options.allowFailure && exitCode !== 0) throw new Error(`clap ${argv.join(" ")} failed (${exitCode})\nstdout: ${stdout}\nstderr: ${stderr}`);
  return { stdout, stderr, exitCode };
}

async function startDummyHealthyServer() {
  const port = await freePort();
  const script = `
    Bun.serve({
      hostname: "127.0.0.1",
      port: ${port},
      fetch(request) {
        const url = new URL(request.url);
        if (url.pathname === "/clap/v1/health") {
          return Response.json({ status: "ok", version: "test-stale", uptimeMs: 1 });
        }
        return new Response("not found", { status: 404 });
      },
    });
    await new Promise(() => undefined);
  `;
  const proc = Bun.spawn([process.execPath, "-e", script], { stdout: "ignore", stderr: "pipe" });
  dummyServers.push(proc);
  const baseURL = `http://127.0.0.1:${port}`;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (await isHealthy(baseURL)) return { proc, baseURL };
    await Bun.sleep(100);
  }
  const stderr = await new Response(proc.stderr).text();
  throw new Error(`dummy server did not start: ${stderr}`);
}

async function isHealthy(baseURL: string) {
  try {
    const response = await fetch(`${baseURL}/clap/v1/health`);
    return response.ok;
  } catch {
    return false;
  }
}

async function freePort() {
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: () => new Response("ok"),
  });
  const port = server.port;
  await server.stop(true);
  return port;
}
