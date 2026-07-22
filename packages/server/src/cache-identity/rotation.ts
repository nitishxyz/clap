import { constants } from "node:fs";
import { lstat, open, readFile, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import {
  installationSecretPath,
  loadOrCreateInstallationSecret,
  rotateInstallationSecret,
} from "./installation-secret";
import type { InstallationSecret } from "./types";

const LOCK_FILE_NAME = "cache-identity-rotation.lock";
const LOCK_RETRY_MS = 10;
const LOCK_TIMEOUT_MS = 30_000;
const STALE_LOCK_MS = 5 * 60_000;

type RotateSecret = typeof rotateInstallationSecret;

export type InstallationSecretProviderOptions = {
  load?: typeof loadOrCreateInstallationSecret;
  rotate?: RotateSecret;
  now?: () => number;
  lockTimeoutMs?: number;
};

export type InstallationSecretLease = {
  secret: InstallationSecret;
  release(): void;
};

export type InstallationSecretRotation = {
  previousGeneration: string;
  newGeneration: string;
  rotatedAt: string;
};

/** Writer-priority barrier keeps new derivations blocked through resident draining. */
export class InstallationSecretProvider {
  private current?: InstallationSecret;
  private activeReaders = 0;
  private pendingWriters = 0;
  private writerActive = false;
  private readonly waiters = new Set<() => void>();
  private readonly loadSecret: typeof loadOrCreateInstallationSecret;
  private readonly rotateSecret: RotateSecret;
  private readonly now: () => number;
  private readonly lockTimeoutMs: number;

  constructor(options: InstallationSecretProviderOptions = {}) {
    this.loadSecret = options.load ?? loadOrCreateInstallationSecret;
    this.rotateSecret = options.rotate ?? rotateInstallationSecret;
    this.now = options.now ?? Date.now;
    this.lockTimeoutMs = options.lockTimeoutMs ?? LOCK_TIMEOUT_MS;
  }

  async withSecret<T>(use: (secret: InstallationSecret) => T | Promise<T>): Promise<T> {
    const lease = await this.acquireSecret();
    try {
      return await use(lease.secret);
    } finally {
      lease.release();
    }
  }

  async acquireSecret(): Promise<InstallationSecretLease> {
    await this.acquireReader();
    let released = false;
    try {
      const secret = await this.secret();
      return {
        secret,
        release: () => {
          if (released) return;
          released = true;
          this.activeReaders -= 1;
          this.notify();
        },
      };
    } catch (error) {
      this.activeReaders -= 1;
      this.notify();
      throw error;
    }
  }

  async rotate<T>(afterPersist: (rotation: InstallationSecretRotation) => T | Promise<T>): Promise<T> {
    await this.acquireWriter();
    let releaseLock: (() => Promise<void>) | undefined;
    try {
      await this.loadSecret(); // Ensures the secure auth directory exists before locking.
      releaseLock = await acquireRotationFileLock(this.now, this.lockTimeoutMs);
      // Refresh under the cross-process lock so previousGeneration reflects the winner.
      const previous = await this.loadSecret();
      const next = await this.rotateSecret();
      this.current = next;
      const rotation = {
        previousGeneration: previous.generation,
        newGeneration: next.generation,
        rotatedAt: new Date(this.now()).toISOString(),
      };
      return await afterPersist(rotation);
    } finally {
      await releaseLock?.();
      this.writerActive = false;
      this.notify();
    }
  }

  private async secret(): Promise<InstallationSecret> {
    return this.current ??= await this.loadSecret();
  }

  private async acquireReader(): Promise<void> {
    while (this.writerActive || this.pendingWriters > 0) await this.wait();
    this.activeReaders += 1;
  }

  private async acquireWriter(): Promise<void> {
    this.pendingWriters += 1;
    try {
      while (this.writerActive || this.activeReaders > 0) await this.wait();
      this.writerActive = true;
    } finally {
      this.pendingWriters -= 1;
    }
  }

  private wait(): Promise<void> {
    return new Promise((resolve) => this.waiters.add(resolve));
  }

  private notify(): void {
    for (const resolve of this.waiters) resolve();
    this.waiters.clear();
  }
}

export function installationSecretRotationLockPath(): string {
  return join(dirname(installationSecretPath()), LOCK_FILE_NAME);
}

async function acquireRotationFileLock(now: () => number, timeoutMs: number): Promise<() => Promise<void>> {
  const path = installationSecretRotationLockPath();
  const started = now();
  while (true) {
    try {
      const handle = await open(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, 0o600);
      const opened = await handle.stat();
      try {
        await handle.writeFile(`${JSON.stringify({ version: 1, pid: process.pid, createdAt: new Date(now()).toISOString() })}\n`);
        await handle.chmod(0o600);
        await handle.sync();
      } finally {
        await handle.close();
      }
      return async () => {
        try {
          const metadata = await lstat(path);
          if (metadata.isFile() && metadata.dev === opened.dev && metadata.ino === opened.ino) await rm(path);
        } catch {
          // Lock already removed; never unlink a replacement.
        }
      };
    } catch (error) {
      if (!hasCode(error, "EEXIST")) throw new Error("Unable to lock cache identity rotation securely");
      await removeStaleLock(path, now());
      if (now() - started >= timeoutMs) throw new Error("Timed out waiting for cache identity rotation lock");
      await new Promise((resolve) => setTimeout(resolve, LOCK_RETRY_MS));
    }
  }
}

async function removeStaleLock(path: string, now: number): Promise<void> {
  try {
    const metadata = await lstat(path);
    const uid = typeof process.getuid === "function" ? process.getuid() : undefined;
    if (!metadata.isFile() || metadata.isSymbolicLink() || (uid !== undefined && metadata.uid !== uid)) {
      throw new Error("Unsafe cache identity rotation lock");
    }
    const parsed = JSON.parse(await readFile(path, "utf8")) as { version?: unknown; pid?: unknown; createdAt?: unknown };
    if (parsed.version !== 1 || typeof parsed.pid !== "number" || typeof parsed.createdAt !== "string") {
      throw new Error("Malformed cache identity rotation lock");
    }
    const age = now - Date.parse(parsed.createdAt);
    if (age < STALE_LOCK_MS || processIsAlive(parsed.pid)) return;
    const before = await lstat(path);
    if (before.dev === metadata.dev && before.ino === metadata.ino) await rm(path);
  } catch (error) {
    if (hasCode(error, "ENOENT")) return;
    throw error;
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return hasCode(error, "EPERM");
  }
}

function hasCode(error: unknown, code: string): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === code;
}
