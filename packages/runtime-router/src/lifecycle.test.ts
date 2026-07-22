import { describe, expect, test } from "bun:test";
import { ModelLifecycleManager, parseKeepAliveMs } from "./lifecycle";
import { measuredMemory } from "./residency/types";
import type { ResolvedModel } from "@clap/models";

const resolved: ResolvedModel = {
  id: "acme/tiny",
  input: "acme/tiny",
  backend: "llama",
  format: "gguf",
  modelPath: "/models/tiny.gguf",
  status: "available",
};

describe("runtime model lifecycle", () => {
  test("parses keep-alive durations and always", () => {
    expect(parseKeepAliveMs("30s")).toBe(30_000);
    expect(parseKeepAliveMs("15m")).toBe(900_000);
    expect(parseKeepAliveMs("1h")).toBe(3_600_000);
    expect(parseKeepAliveMs("always")).toBeNull();
    expect(() => parseKeepAliveMs("soon")).toThrow("invalid keep-alive duration");
  });

  test("extends expiry on usage and expires idle models", () => {
    let now = Date.parse("2026-01-01T00:00:00.000Z");
    const manager = new ModelLifecycleManager(() => now);
    const loaded = manager.load(resolved, { keepAlive: "30s" });
    expect(loaded.expiresAt).toBe("2026-01-01T00:00:30.000Z");

    now += 10_000;
    const active = manager.beginUsage(resolved);
    manager.finishUsage(active);
    expect(active.expiresAt).toBe("2026-01-01T00:00:40.000Z");

    now += 31_000;
    expect(manager.list()).toEqual([]);
  });

  test("keeps always-loaded models until manual cleanup", () => {
    let now = Date.parse("2026-01-01T00:00:00.000Z");
    const manager = new ModelLifecycleManager(() => now);
    const loaded = manager.load(resolved, { keepAlive: "always" });
    expect(loaded.expiresAt).toBeNull();
    expect(loaded.always).toBe(true);

    now += 86_400_000;
    expect(manager.list()).toHaveLength(1);
    expect(manager.unload(resolved).unloaded).toBe(true);
    expect(manager.list()).toEqual([]);
  });

  test("exposes honest residency state, memory, and loading transitions", () => {
    const manager = new ModelLifecycleManager(() => Date.parse("2026-01-01T00:00:00.000Z"));
    const loaded = manager.load(resolved);
    const memory = measuredMemory(4_096, "resident_rss");
    expect(manager.snapshotForResidency({
      memoryByKey: new Map([[loaded.key, memory]]),
      retainedValueByKey: new Map([[loaded.key, 7]]),
    })[0]).toMatchObject({ state: "idle", activeRequests: 0, memory, retainedValueScore: 7 });

    manager.setResidencyTransition(loaded.key, "loading");
    expect(manager.snapshotForResidency()[0]?.state).toBe("loading");
    manager.clearResidencyTransition(loaded.key);
    manager.beginUsage(resolved);
    expect(manager.snapshotForResidency()[0]?.state).toBe("active");

    manager.setResidencyTransition("not-yet-loaded", "starting");
    expect(manager.snapshotForResidency().find((entry) => entry.key === "not-yet-loaded")?.state).toBe("starting");
  });

  test("atomically rejects stale snapshots after touch, activity, pinning, or transition", async () => {
    let now = Date.parse("2026-01-01T00:00:00.000Z");
    const cases: Array<(manager: ModelLifecycleManager, entry: ReturnType<ModelLifecycleManager["load"]>) => void> = [
      (manager, entry) => { now += 1; manager.touch(entry); },
      (manager) => { manager.beginUsage(resolved); },
      (manager) => { const active = manager.beginUsage(resolved); manager.finishUsage(active); },
      (_manager, entry) => { entry.pinned = true; },
      (_manager, entry) => { entry.always = true; },
      (manager, entry) => { manager.setResidencyTransition(entry.key, "starting"); },
    ];
    for (const mutate of cases) {
      const manager = new ModelLifecycleManager(() => now);
      const entry = manager.load(resolved);
      const snapshot = manager.snapshotForResidency()[0]!;
      mutate(manager, entry);
      expect(await manager.tryEvictIdle(snapshot)).toBe("changed");
      manager.cleanup();
    }
  });

  test("awaits the removal hook before idle eviction completes", async () => {
    let release!: () => void;
    const removed = new Promise<void>((resolve) => { release = resolve; });
    let hookFinished = false;
    const manager = new ModelLifecycleManager(() => 1, async () => {
      await removed;
      hookFinished = true;
    });
    manager.load(resolved);
    const eviction = manager.tryEvictIdle(manager.snapshotForResidency()[0]!);

    await Promise.resolve();
    expect(hookFinished).toBe(false);
    expect(manager.list()).toEqual([]);
    release();
    expect(await eviction).toBe("evicted");
    expect(hookFinished).toBe(true);
  });
});
