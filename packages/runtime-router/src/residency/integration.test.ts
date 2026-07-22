import { describe, expect, test } from "bun:test";
import type { LifecycleResidencySnapshot } from "../lifecycle";
import {
  InsufficientModelMemoryError,
  ResidencyCoordinator,
  measuredMemory,
  unavailableMemory,
  type ResidencyCoordinatorEvent,
  type ResidencyLifecycleAdapter,
  type ResidencyMemorySnapshot,
  type ResidencyModelDescriptor,
} from "./index";

const MIB = 1024 ** 2;
const PHYSICAL = 8 * 1024 ** 3;
const descriptor = (modelKey: string): ResidencyModelDescriptor => ({
  modelKey,
  modelId: `fixture/${modelKey}`,
  backend: "llama",
  artifactBytes: 1_000,
  configuredContext: 1,
  kv: { bytesPerToken: 1 },
});

function memory(availableBytes: number): ResidencyMemorySnapshot {
  return { available: measuredMemory(availableBytes, "os_available"), physicalMemoryBytes: PHYSICAL };
}

function resident(key: string, overrides: Partial<LifecycleResidencySnapshot> = {}): LifecycleResidencySnapshot {
  return {
    key,
    state: "idle",
    activeRequests: 0,
    pinned: false,
    always: false,
    loadedAtMs: 1,
    lastUsedAtMs: 1,
    lifecycleVersion: 1,
    retainedValueScore: 0,
    memory: measuredMemory(600 * MIB, "resident_rss"),
    ...overrides,
  };
}

class IntegrationLifecycle implements ResidencyLifecycleAdapter {
  snapshots: LifecycleResidencySnapshot[] = [];
  transitions = new Map<string, string>();
  evictionAttempts: string[] = [];
  shutdowns: string[] = [];
  staleOnce = false;
  failEviction?: Error;
  evictionGate?: Promise<void>;

  snapshotForResidency() { return this.snapshots; }
  async tryEvictIdle(snapshot: LifecycleResidencySnapshot) {
    this.evictionAttempts.push(snapshot.key);
    if (this.failEviction) throw this.failEviction;
    if (this.staleOnce) {
      this.staleOnce = false;
      this.snapshots = this.snapshots.map((entry) => entry.key === snapshot.key
        ? { ...entry, lastUsedAtMs: entry.lastUsedAtMs + 10, lifecycleVersion: entry.lifecycleVersion + 1 }
        : entry);
      return "changed" as const;
    }
    await this.evictionGate;
    this.shutdowns.push(snapshot.key);
    this.snapshots = this.snapshots.filter((entry) => entry.key !== snapshot.key);
    return "evicted" as const;
  }
  setResidencyTransition(key: string, state: "starting" | "loading" | "closing") {
    this.transitions.set(key, state);
  }
  clearResidencyTransition(key: string) { this.transitions.delete(key); }
}

function harness(
  memorySnapshot: () => Promise<ResidencyMemorySnapshot>,
  lifecycle = new IntegrationLifecycle(),
) {
  const events: ResidencyCoordinatorEvent[] = [];
  const coordinator = new ResidencyCoordinator({
    lifecycle,
    memorySnapshot,
    osHeadroomBytes: 0,
    runtimeHeadroomBytes: 0,
    policy: { minimumHeadroomBytes: 0 },
    env: {},
    onEvent: (event) => events.push(event),
  });
  return { coordinator, lifecycle, events };
}

describe("residency admission integration matrix", () => {
  test("serializes simultaneous loads, exposes one reservation, and rejects stale overcommit", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let samples = 0;
    let physicalLoads = 0;
    const setup = harness(async () => memory(++samples === 1 ? 700 * MIB : 100 * MIB));

    const first = setup.coordinator.load(descriptor("large-a"), {
      performLoad: async () => { physicalLoads += 1; await gate; return "a"; },
    });
    const second = setup.coordinator.load(descriptor("large-b"), {
      performLoad: async () => { physicalLoads += 1; return "b"; },
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(setup.coordinator.reservations()).toHaveLength(1);
    expect(physicalLoads).toBe(1);
    release();
    await first;
    await expect(second).rejects.toMatchObject({
      code: "insufficient_model_memory",
      details: { availableBytes: 100 * MIB, reservedBytes: 0 },
    });
    expect(samples).toBe(2);
    expect(physicalLoads).toBe(1);
    expect(setup.coordinator.reservations()).toEqual([]);
  });

  test("protects active, pinned, loading, target, and reserved residents", async () => {
    const lifecycle = new IntegrationLifecycle();
    lifecycle.snapshots = [
      resident("active", { state: "active", activeRequests: 1 }),
      resident("pinned", { pinned: true }),
      resident("loading", { state: "loading" }),
      resident("target"),
    ];
    const setup = harness(async () => memory(100 * MIB), lifecycle);
    await expect(setup.coordinator.load(descriptor("target"), { performLoad: async () => "never" }))
      .rejects.toBeInstanceOf(InsufficientModelMemoryError);
    expect(lifecycle.evictionAttempts).toEqual([]);
    expect(setup.events.at(-1)).toMatchObject({ type: "model_load_rejected", reason: "no_evictable_models" });
  });

  test("replans a stale last-used snapshot, awaits shutdown, and resamples afterward", async () => {
    const lifecycle = new IntegrationLifecycle();
    lifecycle.snapshots = [resident("old", { lastUsedAtMs: 1 }), resident("new", { lastUsedAtMs: 2 })];
    lifecycle.staleOnce = true;
    let release!: () => void;
    lifecycle.evictionGate = new Promise<void>((resolve) => { release = resolve; });
    let samples = 0;
    let loadStarted = false;
    const setup = harness(async () => memory(++samples === 1 ? 100 * MIB : 700 * MIB), lifecycle);
    const load = setup.coordinator.load(descriptor("target"), {
      performLoad: async () => { loadStarted = true; return "loaded"; },
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(loadStarted).toBe(false);
    expect(samples).toBe(1);
    release();
    await load;
    expect(lifecycle.evictionAttempts).toEqual(["old", "new"]);
    expect(lifecycle.shutdowns).toEqual(["new"]);
    expect(samples).toBe(2);
    expect(setup.events.map((event) => event.type)).toContain("model_evicted_for_load");
  });

  test("rolls back reservation and transition after eviction, spawn, or crash failure", async () => {
    const evictionFailure = new Error("shutdown failed");
    const evictionLifecycle = new IntegrationLifecycle();
    evictionLifecycle.snapshots = [resident("victim")];
    evictionLifecycle.failEviction = evictionFailure;
    const eviction = harness(async () => memory(100 * MIB), evictionLifecycle);
    await expect(eviction.coordinator.load(descriptor("target"), { performLoad: async () => "never" }))
      .rejects.toBe(evictionFailure);
    expect(eviction.coordinator.reservations()).toEqual([]);

    for (const failure of [new Error("spawn failure"), new Error("load crash")]) {
      const setup = harness(async () => memory(700 * MIB));
      let cleanupCause: unknown;
      await expect(setup.coordinator.load(descriptor("target"), {
        performLoad: async () => { throw failure; },
        shutdownPartial: async (_loaded, cause) => { cleanupCause = cause; },
      })).rejects.toBe(failure);
      expect(cleanupCause).toBe(failure);
      expect(setup.coordinator.reservations()).toEqual([]);
      expect(setup.lifecycle.transitions.size).toBe(0);
      expect(setup.events.map((event) => event.type)).toContain("model_load_rolled_back");
    }
  });

  test("commits without fabricating RSS and fails closed on unavailable OS memory", async () => {
    const success = harness(async () => memory(700 * MIB));
    await success.coordinator.load(descriptor("no-rss"), {
      performLoad: async () => "loaded",
      observeRss: async () => undefined,
    });
    expect(success.events.at(-1)).toMatchObject({ type: "model_load_committed" });
    expect(success.events.at(-1)?.observedRssBytes).toBeUndefined();

    const unavailable = harness(async () => ({
      available: unavailableMemory("not_supported"),
      physicalMemoryBytes: PHYSICAL,
    }));
    await expect(unavailable.coordinator.load(descriptor("unknown-memory"), {
      performLoad: async () => "never",
    })).rejects.toMatchObject({
      code: "insufficient_model_memory",
      details: { reason: "memory_state_unavailable", availableBytes: null },
    });
    expect(unavailable.coordinator.reservations()).toEqual([]);
  });
});
