import { describe, expect, test } from "bun:test";
import type { LifecycleResidencySnapshot } from "../lifecycle";
import {
  InsufficientModelMemoryError,
  ModelLoadEstimateHistory,
  ResidencyCoordinator,
  measuredMemory,
  estimatedMemory,
  unavailableMemory,
  type ResidencyLifecycleAdapter,
  type ResidencyModelDescriptor,
  type ResidencyMemorySnapshot,
} from "./index";

const MIB = 1024 ** 2;
const model: ResidencyModelDescriptor = {
  modelKey: "model-a",
  modelId: "owner/model-a",
  backend: "llama",
  artifactBytes: 1_000,
  configuredContext: 1,
  kv: { bytesPerToken: 1 },
};

function memory(availableBytes: number): ResidencyMemorySnapshot {
  return { available: measuredMemory(availableBytes, "os_available"), physicalMemoryBytes: 8 * 1024 ** 3 };
}

function idle(key: string, overrides: Partial<LifecycleResidencySnapshot> = {}): LifecycleResidencySnapshot {
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
    memory: measuredMemory(100, "resident_rss"),
    ...overrides,
  };
}

class FakeLifecycle implements ResidencyLifecycleAdapter {
  snapshots: LifecycleResidencySnapshot[] = [];
  transitions = new Map<string, string>();
  evictionResults: Array<"evicted" | "changed"> = [];
  evicted: string[] = [];

  snapshotForResidency() { return this.snapshots; }
  async tryEvictIdle(snapshot: LifecycleResidencySnapshot) {
    const result = this.evictionResults.shift() ?? "evicted";
    if (result === "evicted") {
      this.evicted.push(snapshot.key);
      this.snapshots = this.snapshots.filter((entry) => entry.key !== snapshot.key);
    }
    return result;
  }
  setResidencyTransition(key: string, state: "starting" | "loading" | "closing") {
    this.transitions.set(key, state);
  }
  clearResidencyTransition(key: string) { this.transitions.delete(key); }
}

function coordinator(
  snapshots: ResidencyMemorySnapshot[] | (() => Promise<ResidencyMemorySnapshot>),
  lifecycle = new FakeLifecycle(),
  history = new ModelLoadEstimateHistory(),
) {
  const values = Array.isArray(snapshots) ? [...snapshots] : undefined;
  return {
    lifecycle,
    history,
    coordinator: new ResidencyCoordinator({
      lifecycle,
      history,
      memorySnapshot: Array.isArray(snapshots)
        ? async () => values!.shift() ?? values!.at(-1) ?? memory(2_000 * MIB)
        : snapshots,
      osHeadroomBytes: 0,
      runtimeHeadroomBytes: 0,
      policy: { minimumHeadroomBytes: 0 },
      reservationId: (() => { let id = 0; return () => `reservation-${++id}`; })(),
      now: (() => { let now = 10; return () => ++now; })(),
      env: {},
    }),
  };
}

describe("ResidencyCoordinator", () => {
  test("serializes whole loads and resamples before a second admission", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let memoryCalls = 0;
    let activeLoads = 0;
    let maxActiveLoads = 0;
    const setup = coordinator(async () => {
      memoryCalls += 1;
      return memory(memoryCalls === 1 ? 700 * MIB : 100 * MIB);
    });
    const first = setup.coordinator.load(model, { performLoad: async () => {
      activeLoads += 1;
      maxActiveLoads = Math.max(maxActiveLoads, activeLoads);
      await gate;
      activeLoads -= 1;
      return "first";
    } });
    const second = setup.coordinator.load({ ...model, modelKey: "model-b" }, {
      performLoad: async () => { activeLoads += 1; maxActiveLoads = Math.max(maxActiveLoads, activeLoads); return "second"; },
    });

    await Promise.resolve();
    expect(memoryCalls).toBe(1);
    release();
    expect((await first).value).toBe("first");
    await expect(second).rejects.toBeInstanceOf(InsufficientModelMemoryError);
    expect(memoryCalls).toBe(2);
    expect(maxActiveLoads).toBe(1);
  });

  test("joins same-key in-flight work and exposes one held reservation", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let calls = 0;
    const setup = coordinator([memory(700 * MIB)]);
    const operation = { performLoad: async () => { calls += 1; await gate; return { ready: true }; } };
    const first = setup.coordinator.load(model, operation);
    const second = setup.coordinator.load(model, operation);
    await Promise.resolve();
    await Promise.resolve();

    expect(setup.coordinator.reservations()).toHaveLength(1);
    expect(setup.coordinator.reservations()[0]).toMatchObject({ state: "held", model });
    release();
    expect(await first).toEqual(await second);
    expect(calls).toBe(1);
    expect(setup.coordinator.reservations()).toEqual([]);
  });

  test("commits observed RSS high-water after stabilization", async () => {
    const setup = coordinator([memory(900 * MIB)]);
    const order: string[] = [];
    const result = await setup.coordinator.load(model, {
      performLoad: async () => { order.push("load"); return "worker"; },
      stabilize: async () => { order.push("stabilize"); },
      observeRss: async () => { order.push("observe"); return 800 * MIB; },
    });

    expect(order).toEqual(["load", "stabilize", "observe"]);
    expect(result.reservation.state).toBe("committed");
    expect(result.decision.reason).toBe("within_budget");
    expect(setup.history.get(model)).toBe(800 * MIB);
  });

  test("rolls back reservation, loading state, and partial worker on load failures", async () => {
    for (const failure of [new Error("spawn"), new Error("crash")]) {
      const setup = coordinator([memory(700 * MIB)]);
      const partialCalls: Array<unknown> = [];
      await expect(setup.coordinator.load(model, {
        performLoad: async () => { throw failure; },
        shutdownPartial: async (loaded, cause) => { partialCalls.push(loaded, cause); },
      })).rejects.toBe(failure);
      expect(partialCalls).toEqual([undefined, failure]);
      expect(setup.coordinator.reservations()).toEqual([]);
      expect(setup.lifecycle.transitions.size).toBe(0);
    }
  });

  test("shuts down a partially loaded worker when stabilization or observation fails", async () => {
    const failure = new Error("post-load crash");
    const setup = coordinator([memory(700 * MIB)]);
    const partialCalls: Array<unknown> = [];
    await expect(setup.coordinator.load(model, {
      performLoad: async () => "partial-worker",
      observeRss: async () => { throw failure; },
      shutdownPartial: async (loaded, cause) => { partialCalls.push(loaded, cause); },
    })).rejects.toBe(failure);
    expect(partialCalls).toEqual(["partial-worker", failure]);
    expect(setup.coordinator.reservations()).toEqual([]);
  });

  test("evicts deterministically, replans changed snapshots, awaits, and resamples", async () => {
    const lifecycle = new FakeLifecycle();
    lifecycle.snapshots = [idle("older", { lastUsedAtMs: 1 }), idle("newer", { lastUsedAtMs: 2 })];
    lifecycle.evictionResults = ["changed", "evicted"];
    let memoryCalls = 0;
    const setup = coordinator(async () => {
      memoryCalls += 1;
      return memory(memoryCalls === 1 ? 100 * MIB : 700 * MIB);
    }, lifecycle);

    const result = await setup.coordinator.load(model, { performLoad: async () => "loaded" });
    expect(lifecycle.evicted).toEqual(["older"]);
    expect(memoryCalls).toBe(2);
    expect(result.decision.reason).toBe("within_budget_after_eviction");
    expect(result.decision.evictedModelKeys).toEqual(["older"]);
  });

  test("blocks active and pinned residents with structured capacity details", async () => {
    const lifecycle = new FakeLifecycle();
    lifecycle.snapshots = [
      idle("active", { state: "active", activeRequests: 1 }),
      idle("pinned", { pinned: true }),
    ];
    const setup = coordinator([memory(100 * MIB)], lifecycle);
    try {
      await setup.coordinator.load(model, { performLoad: async () => "never" });
      throw new Error("expected admission failure");
    } catch (error) {
      expect(error).toBeInstanceOf(InsufficientModelMemoryError);
      expect((error as InsufficientModelMemoryError).details).toMatchObject({
        reason: "no_evictable_models",
        requestedBytes: 512 * MIB + 1_201,
        availableBytes: 100 * MIB,
        reservedBytes: 0,
        evictableModelCount: 0,
      });
    }
    expect(setup.coordinator.reservations()).toEqual([]);
  });

  test("fails closed when available memory is unavailable", async () => {
    const setup = coordinator([{
      available: unavailableMemory("not_supported"),
      physicalMemoryBytes: 8 * 1024 ** 3,
    }]);
    await expect(setup.coordinator.load(model, { performLoad: async () => "never" }))
      .rejects.toMatchObject({
        code: "insufficient_model_memory",
        details: { reason: "memory_state_unavailable", availableBytes: null },
      });
    expect(setup.coordinator.reservations()).toEqual([]);
  });

  test("fails closed on estimated availability and accounts explicit environment headroom", async () => {
    const lifecycle = new FakeLifecycle();
    const estimated = new ResidencyCoordinator({
      lifecycle,
      memorySnapshot: async () => ({
        available: estimatedMemory(700 * MIB, "conservative_fallback"),
        physicalMemoryBytes: 8 * 1024 ** 3,
      }),
      policy: { minimumHeadroomBytes: 0 },
      osHeadroomBytes: 0,
      runtimeHeadroomBytes: 0,
      env: {},
    });
    await expect(estimated.load(model, { performLoad: async () => "never" }))
      .rejects.toMatchObject({ details: { reason: "memory_state_unavailable" } });

    const withHeadroom = new ResidencyCoordinator({
      lifecycle: new FakeLifecycle(),
      memorySnapshot: async () => memory(700 * MIB),
      policy: { minimumHeadroomBytes: 0 },
      env: {
        CLAP_MODEL_OS_HEADROOM_BYTES: String(10 * MIB),
        CLAP_MODEL_RUNTIME_HEADROOM_BYTES: String(20 * MIB),
      },
    });
    const result = await withHeadroom.load(model, { performLoad: async () => "loaded" });
    expect(result.decision.headroomBytes).toBe(30 * MIB);
    expect(withHeadroom.policy.maximumConcurrentLoads).toBe(1);
  });
});
