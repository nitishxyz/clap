import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  installationSecretPath,
  loadOrCreateInstallationSecret,
  rotateInstallationSecret,
} from "./installation-secret";

const previousClapHome = process.env.CLAP_HOME;
let root = "";

describe("cache identity installation secret", () => {
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "clap-cache-identity-"));
    process.env.CLAP_HOME = join(root, "home");
  });

  afterEach(async () => {
    if (previousClapHome === undefined) delete process.env.CLAP_HOME;
    else process.env.CLAP_HOME = previousClapHome;
    await rm(root, { recursive: true, force: true });
  });

  test("creates a versioned 32-byte secret with owner-only modes", async () => {
    const loaded = await loadOrCreateInstallationSecret();
    const path = installationSecretPath();
    const document = JSON.parse(await readFile(path, "utf8"));

    expect(loaded.generation).toMatch(/^sec_/);
    expect(loaded.key).toHaveLength(32);
    expect(document).toEqual({
      version: 1,
      generation: loaded.generation,
      createdAt: expect.any(String),
      secret: Buffer.from(loaded.key).toString("base64url"),
    });
    expect((await lstat(dirname(path))).mode & 0o777).toBe(0o700);
    expect((await lstat(path)).mode & 0o777).toBe(0o600);
  });

  test("repairs permissive existing directory and file modes", async () => {
    const first = await loadOrCreateInstallationSecret();
    const path = installationSecretPath();
    await chmod(dirname(path), 0o755);
    await chmod(path, 0o644);

    const loaded = await loadOrCreateInstallationSecret();

    expect(loaded.generation).toBe(first.generation);
    expect(loaded.key).toEqual(first.key);
    expect((await lstat(dirname(path))).mode & 0o777).toBe(0o700);
    expect((await lstat(path)).mode & 0o777).toBe(0o600);
  });

  test("concurrent creators all load the exclusive winner", async () => {
    const loaded = await Promise.all(Array.from({ length: 40 }, () => loadOrCreateInstallationSecret()));

    expect(new Set(loaded.map((entry) => entry.generation)).size).toBe(1);
    expect(new Set(loaded.map((entry) => Buffer.from(entry.key).toString("hex"))).size).toBe(1);
  });

  test("loads the same key after restart", async () => {
    const first = await loadOrCreateInstallationSecret();
    const second = await loadOrCreateInstallationSecret();

    expect(second.generation).toBe(first.generation);
    expect(second.key).toEqual(first.key);
    expect(second.key).not.toBe(first.key);
  });

  test.each([
    ["invalid JSON", "not-json"],
    ["unsupported version", JSON.stringify(validDocument({ version: 2 }))],
    ["unexpected field", JSON.stringify(validDocument({ leakedMetadata: true }))],
    ["invalid generation", JSON.stringify(validDocument({ generation: "sec_not-a-uuid" }))],
    ["invalid timestamp", JSON.stringify(validDocument({ createdAt: "yesterday" }))],
    ["short key", JSON.stringify(validDocument({ secret: Buffer.alloc(31).toString("base64url") }))],
    ["long key", JSON.stringify(validDocument({ secret: Buffer.alloc(33).toString("base64url") }))],
    ["padded key", JSON.stringify(validDocument({ secret: `${Buffer.alloc(32).toString("base64url")}=` }))],
  ])("rejects a malformed document: %s", async (_label, contents) => {
    const path = installationSecretPath();
    await mkdir(dirname(path), { recursive: true, mode: 0o700 });
    await writeFile(path, contents, { mode: 0o600 });

    await expect(loadOrCreateInstallationSecret()).rejects.toThrow(
      "Unable to load or create the cache identity installation secret securely",
    );
  });

  test("rejects a symlink without reading its target", async () => {
    const target = join(root, "target.json");
    const marker = "do-not-disclose-this-target-value";
    await writeFile(target, marker);
    await mkdir(dirname(installationSecretPath()), { recursive: true, mode: 0o700 });
    await symlink(target, installationSecretPath());

    let message = "";
    try {
      await loadOrCreateInstallationSecret();
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    expect(message).toContain("Unable to load or create");
    expect(message).not.toContain(marker);
    expect(await readFile(target, "utf8")).toBe(marker);
  });

  test("rejects a symlinked auth directory", async () => {
    const targetDirectory = join(root, "target-auth");
    await mkdir(targetDirectory);
    await mkdir(process.env.CLAP_HOME!);
    await symlink(targetDirectory, join(process.env.CLAP_HOME!, "auth"));

    await expect(loadOrCreateInstallationSecret()).rejects.toThrow("securely");
    expect(await readdir(targetDirectory)).toEqual([]);
  });

  test("rotation atomically installs a new generation and key", async () => {
    const first = await loadOrCreateInstallationSecret();
    const rotated = await rotateInstallationSecret();
    const loaded = await loadOrCreateInstallationSecret();

    expect(rotated.generation).not.toBe(first.generation);
    expect(rotated.key).not.toEqual(first.key);
    expect(loaded.generation).toBe(rotated.generation);
    expect(loaded.key).toEqual(rotated.key);
    expect((await lstat(installationSecretPath())).mode & 0o777).toBe(0o600);
  });

  test("a failed rotation rename preserves the old secret and removes its temp file", async () => {
    const first = await loadOrCreateInstallationSecret();
    let replacementPath = "";

    await expect(rotateInstallationSecret({
      rename: async (from) => {
        replacementPath = from;
        throw new Error("injected rename failure");
      },
    })).rejects.toThrow("Unable to rotate the cache identity installation secret securely");

    const loaded = await loadOrCreateInstallationSecret();
    expect(loaded.generation).toBe(first.generation);
    expect(loaded.key).toEqual(first.key);
    expect(replacementPath).not.toBe("");
    expect(await readdir(dirname(installationSecretPath()))).toEqual(["cache-identity-secret.json"]);
  });

  test("errors redact stored and newly generated secret material", async () => {
    const stored = await loadOrCreateInstallationSecret();
    const storedText = Buffer.from(stored.key).toString("base64url");
    let replacementText = "";
    let message = "";

    try {
      await rotateInstallationSecret({
        rename: async (from) => {
          replacementText = JSON.parse(await readFile(from, "utf8")).secret;
          throw new Error(`rename failed for ${storedText} and ${replacementText}`);
        },
      });
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }

    expect(replacementText).toHaveLength(43);
    expect(message).not.toContain(storedText);
    expect(message).not.toContain(replacementText);
    expect(message).toBe("Unable to rotate the cache identity installation secret securely");
  });
});

function validDocument(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    version: 1,
    generation: "sec_123e4567-e89b-42d3-a456-426614174000",
    createdAt: "2026-07-23T12:00:00.000Z",
    secret: Buffer.alloc(32, 7).toString("base64url"),
    ...overrides,
  };
}
