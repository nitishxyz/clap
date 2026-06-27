import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  launchdPlist,
  readServerMetadata,
  serverPaths,
  systemdService,
  tailFile,
  writeServerMetadata,
} from "./server-lifecycle";

describe("server lifecycle helpers", () => {
  test("stores server metadata under the Clap state path", async () => {
    const home = await mkdtemp(join(tmpdir(), "clap-test-"));
    try {
      const paths = serverPaths(home);
      await writeServerMetadata({
        pid: 123,
        port: 11435,
        baseURL: "http://127.0.0.1:11435",
        startedAt: "2026-06-27T00:00:00.000Z",
      }, paths);

      await expect(readServerMetadata(paths)).resolves.toEqual({
        pid: 123,
        port: 11435,
        baseURL: "http://127.0.0.1:11435",
        startedAt: "2026-06-27T00:00:00.000Z",
      });
    } finally {
      await rm(home, { recursive: true, force: true });
    }
  });

  test("renders launchd and systemd service templates", () => {
    const paths = serverPaths("/tmp/clap-state");
    const command = ["/usr/local/bin/bun", "/opt/clap/index.ts"];

    expect(launchdPlist(command, paths)).toContain("dev.clap.server");
    expect(launchdPlist(command, paths)).toContain("/tmp/clap-state/server.log");
    expect(systemdService(command, paths)).toContain("ExecStart=/usr/local/bin/bun /opt/clap/index.ts serve");
    expect(systemdService(command, paths)).toContain("StandardError=append:/tmp/clap-state/server.err.log");
  });

  test("tails missing log files as empty", async () => {
    await expect(tailFile("/tmp/clap-missing-test.log", 10)).resolves.toBe("");
  });
});
