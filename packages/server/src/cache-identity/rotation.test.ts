import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { rotateInstallationSecret } from "./installation-secret";
import { InstallationSecretProvider, installationSecretRotationLockPath } from "./rotation";

const previousHome = process.env.CLAP_HOME;
let root = "";

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "clap-secret-provider-"));
  process.env.CLAP_HOME = join(root, "home");
});

afterEach(async () => {
  if (previousHome === undefined) delete process.env.CLAP_HOME;
  else process.env.CLAP_HOME = previousHome;
  await rm(root, { recursive: true, force: true });
});

describe("installation secret rotation provider", () => {
  test("serializes concurrent rotations and exposes each winner to the next", async () => {
    const provider = new InstallationSecretProvider();
    const order: string[] = [];
    const rotations = await Promise.all([1, 2, 3].map((id) => provider.rotate(async (rotation) => {
      order.push(`start-${id}`);
      await Bun.sleep(5);
      order.push(`end-${id}`);
      return rotation;
    })));

    expect(order).toEqual(["start-1", "end-1", "start-2", "end-2", "start-3", "end-3"]);
    expect(rotations[1]?.previousGeneration).toBe(rotations[0]?.newGeneration);
    expect(rotations[2]?.previousGeneration).toBe(rotations[1]?.newGeneration);
    expect(new Set(rotations.map((rotation) => rotation.newGeneration)).size).toBe(3);
    await expect(readFile(installationSecretRotationLockPath())).rejects.toThrow();
  });

  test("exclusive lock serializes independent provider instances", async () => {
    const providers = [new InstallationSecretProvider(), new InstallationSecretProvider()];
    let active = 0;
    let maximumActive = 0;
    const rotations = await Promise.all(providers.map((provider) => provider.rotate(async (rotation) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      await Bun.sleep(15);
      active -= 1;
      return rotation;
    })));
    expect(maximumActive).toBe(1);
    expect(rotations.some((left) => rotations.some((right) => left.newGeneration === right.previousGeneration))).toBe(true);
  });

  test("blocks new derivations until post-persist resident draining completes", async () => {
    const provider = new InstallationSecretProvider();
    const initial = await provider.withSecret((secret) => secret.generation);
    let releaseDrain!: () => void;
    const drain = new Promise<void>((resolve) => { releaseDrain = resolve; });
    let persisted = false;
    const rotating = provider.rotate(async () => {
      persisted = true;
      await drain;
    });
    while (!persisted) await Bun.sleep(1);

    let derived = false;
    const waiting = provider.withSecret((secret) => {
      derived = true;
      return secret.generation;
    });
    await Bun.sleep(10);
    expect(derived).toBe(false);
    releaseDrain();
    await rotating;
    const next = await waiting;
    expect(next).not.toBe(initial);
  });

  test("failed persistence leaves post-persist cleanup untouched", async () => {
    let cleanupCalls = 0;
    const provider = new InstallationSecretProvider({
      rotate: async () => {
        throw new Error("injected persistence failure with secret material");
      },
    });
    const initial = await provider.withSecret((secret) => secret.generation);

    await expect(provider.rotate(async () => { cleanupCalls += 1; })).rejects.toThrow("injected persistence failure");
    expect(cleanupCalls).toBe(0);
    expect(await provider.withSecret((secret) => secret.generation)).toBe(initial);
  });

  test("post-persist failure still publishes the durable new generation", async () => {
    const provider = new InstallationSecretProvider({ rotate: rotateInstallationSecret });
    const initial = await provider.withSecret((secret) => secret.generation);
    await expect(provider.rotate(async () => { throw new Error("resident crashed while draining"); }))
      .rejects.toThrow("resident crashed while draining");
    expect(await provider.withSecret((secret) => secret.generation)).not.toBe(initial);
  });
});
