import { describe, expect, test } from "bun:test";
import { deriveTokenCapabilities, evictOneIdleForCriticalPressure, parseWorkerRetention, parseWorkerTokenCapabilities, shouldEvictForCriticalPressure, validateTokenBudget } from "./resident";
import { unavailableMemory } from "./residency";
import type { LifecycleResidencySnapshot } from "./lifecycle";

function pressureSnapshot(key: string, overrides: Partial<LifecycleResidencySnapshot> = {}): LifecycleResidencySnapshot {
  return { key, state: "idle", activeRequests: 0, pinned: false, always: false,
    loadedAtMs: 1, lastUsedAtMs: 1, lifecycleVersion: 1, retainedValueScore: 0,
    memory: unavailableMemory("not_observed"), ...overrides };
}

describe("critical pressure eviction", () => {
  test("triggers only when critical pressure cannot be relieved above minimum concurrency", () => {
    const retention = parseWorkerRetention({
      max_active: 1, active_policy: { mode: "auto", selected_max: 1, backend_ceiling: 8,
        hardware_ceiling: 8, model_ceiling: 8, memory_ceiling: 1, reason: "memory_ceiling", inputs: {} },
      active: 0, retained_total: 0, retained_sessions: 0, retained_anchors: 0,
      retained_bytes: 0, session_bytes: 0, anchor_bytes: 0, budget_bytes: 1,
      retained_bytes_source: "estimated", retained_bytes_basis: "cache_components",
      session_bytes_source: "estimated", session_bytes_basis: "cache_components",
      anchor_bytes_source: "estimated", anchor_bytes_basis: "cache_components",
      high_watermark_bytes: 1, low_watermark_bytes: 1, under_pressure: false,
      hard_ceiling: 1, eviction_count: 0,
    })!;
    expect(shouldEvictForCriticalPressure("critical", [retention])).toBe(true);
    expect(shouldEvictForCriticalPressure("warning", [retention])).toBe(false);
    expect(shouldEvictForCriticalPressure("critical", [{ ...retention, maxActive: 2,
      activePolicy: { ...retention.activePolicy, memoryCeiling: 2 } }])).toBe(false);
  });

  test("protects unsafe residents and selects the oldest eligible candidate", async () => {
    const snapshots = [
      pressureSnapshot("active", { state: "active", activeRequests: 1, lastUsedAtMs: 0 }),
      pressureSnapshot("pinned", { pinned: true, lastUsedAtMs: 0 }),
      pressureSnapshot("loading", { state: "loading", lastUsedAtMs: 0 }),
      pressureSnapshot("always", { always: true, lastUsedAtMs: 0 }),
      pressureSnapshot("reserved", { lastUsedAtMs: 0 }),
      pressureSnapshot("oldest", { lastUsedAtMs: 10 }),
      pressureSnapshot("newest", { lastUsedAtMs: 20 }),
    ];
    const evicted: string[] = [];
    const victim = await evictOneIdleForCriticalPressure({
      lifecycle: {
        snapshotForResidency: () => snapshots,
        tryEvictIdle: async (snapshot) => { evicted.push(snapshot.key); return "evicted"; },
        setResidencyTransition: () => {}, clearResidencyTransition: () => {},
      },
      reservedKeys: new Set(["reserved"]),
    });
    expect(victim?.key).toBe("oldest");
    expect(evicted).toEqual(["oldest"]);
  });

  test("replans stale snapshots, awaits shutdown, resamples, and evicts once", async () => {
    let snapshots = [pressureSnapshot("stale", { lastUsedAtMs: 1 }), pressureSnapshot("next", { lastUsedAtMs: 2 })];
    const order: string[] = [];
    const victim = await evictOneIdleForCriticalPressure({
      lifecycle: {
        snapshotForResidency: () => snapshots,
        tryEvictIdle: async (snapshot) => {
          order.push(`try:${snapshot.key}`);
          if (snapshot.key === "stale") { snapshots = snapshots.slice(1); return "changed"; }
          await Promise.resolve();
          order.push(`shutdown:${snapshot.key}`);
          snapshots = [];
          return "evicted";
        },
        setResidencyTransition: () => {}, clearResidencyTransition: () => {},
      },
      onEvicted: (snapshot) => { order.push(`event:${snapshot.key}`); },
      resample: () => { order.push("resample"); },
    });
    expect(victim?.key).toBe("next");
    expect(order).toEqual(["try:stale", "try:next", "shutdown:next", "event:next", "resample"]);
  });

  test("falls back safely when no eligible candidate exists", async () => {
    let calls = 0;
    expect(await evictOneIdleForCriticalPressure({
      lifecycle: {
        snapshotForResidency: () => [pressureSnapshot("active", { state: "active", activeRequests: 1 })],
        tryEvictIdle: async () => { calls += 1; return "evicted"; },
        setResidencyTransition: () => {}, clearResidencyTransition: () => {},
      },
    })).toBeUndefined();
    expect(calls).toBe(0);
  });

  test("propagates shutdown and resample failures without trying another victim", async () => {
    let attempts = 0;
    await expect(evictOneIdleForCriticalPressure({
      lifecycle: {
        snapshotForResidency: () => [pressureSnapshot("victim"), pressureSnapshot("other")],
        tryEvictIdle: async () => { attempts += 1; throw new Error("shutdown failed"); },
        setResidencyTransition: () => {}, clearResidencyTransition: () => {},
      },
    })).rejects.toThrow("shutdown failed");
    expect(attempts).toBe(1);

    const order: string[] = [];
    await expect(evictOneIdleForCriticalPressure({
      lifecycle: {
        snapshotForResidency: () => [pressureSnapshot("victim")],
        tryEvictIdle: async () => { order.push("shutdown"); return "evicted"; },
        setResidencyTransition: () => {}, clearResidencyTransition: () => {},
      },
      onEvicted: () => { order.push("event"); },
      resample: () => { order.push("resample"); throw new Error("resample failed"); },
    })).rejects.toThrow("resample failed");
    expect(order).toEqual(["shutdown", "event", "resample"]);
  });
});

describe("worker retention telemetry", () => {
  test("parses a complete MLX retention snapshot", () => {
    expect(parseWorkerRetention({
      max_active: 4, queued: 3, previous_max_active: 2,
      last_adjustment_reason: "global_headroom_available", last_adjustment_at: "2026-07-20T11:29:00.000Z",
      retained_growth_reserve_bytes: 1_000, global_resident_memory_bytes: 20_000,
      pressure_state: "normal", active_policy: { mode: "auto", selected_max: 4, backend_ceiling: 16,
        hardware_ceiling: 8, model_ceiling: 16, memory_ceiling: 4, reason: "memory_ceiling",
        inputs: { startup_available_bytes: 12_000, hybrid_or_recurrent: false } },
      active: 2, retained_total: 103, retained_sessions: 100, retained_anchors: 3,
      retained_bytes: 10_000, session_bytes: 8_000, anchor_bytes: 2_000, budget_bytes: 100_000,
      retained_bytes_source: "estimated", retained_bytes_basis: "cache_components",
      session_bytes_source: "estimated", session_bytes_basis: "cache_components",
      anchor_bytes_source: "estimated", anchor_bytes_basis: "cache_components",
      high_watermark_bytes: 90_000, low_watermark_bytes: 75_000, under_pressure: false,
      hard_ceiling: 256, eviction_reason: "byte_pressure", eviction_count: 7,
    })).toEqual({
      maxActive: 4, queued: 3, previousMaxActive: 2,
      lastAdjustmentReason: "global_headroom_available", lastAdjustmentAt: "2026-07-20T11:29:00.000Z",
      retainedGrowthReserveBytes: 1_000, globalResidentMemoryBytes: 20_000,
      pressureState: "normal", activePolicy: { mode: "auto", selectedMax: 4, backendCeiling: 16,
        hardwareCeiling: 8, modelCeiling: 16, memoryCeiling: 4, reason: "memory_ceiling",
        inputs: { startup_available_bytes: 12_000, hybrid_or_recurrent: false } },
      active: 2, retainedTotal: 103, retainedSessions: 100, retainedAnchors: 3,
      retainedBytes: 10_000, sessionBytes: 8_000, anchorBytes: 2_000, budgetBytes: 100_000,
      retainedBytesSource: "estimated", retainedBytesBasis: "cache_components",
      sessionBytesSource: "estimated", sessionBytesBasis: "cache_components",
      anchorBytesSource: "estimated", anchorBytesBasis: "cache_components",
      highWatermarkBytes: 90_000, lowWatermarkBytes: 75_000, underPressure: false,
      hardCeiling: 256, evictionReason: "byte_pressure", evictionCount: 7,
    });
  });

  test("preserves absent telemetry for older and non-MLX workers", () => {
    expect(parseWorkerRetention(undefined)).toBeUndefined();
    expect(parseWorkerRetention({ retained_total: 1 })).toBeUndefined();
    expect(parseWorkerRetention({ max_active: 4, active_policy: { mode: "auto" } })).toBeUndefined();
  });

  test("maps validated retention source and basis companions", () => {
    const legacy = {
      max_active: 1, active_policy: { mode: "fixed", selected_max: 1, backend_ceiling: 1,
        hardware_ceiling: 1, model_ceiling: 1, memory_ceiling: 1, reason: "configured",
        inputs: {} },
      active: 0, retained_total: 1, retained_sessions: 1, retained_anchors: 0,
      retained_bytes: 4096, session_bytes: 4096, anchor_bytes: 0, budget_bytes: 8192,
      high_watermark_bytes: 7000, low_watermark_bytes: 6000, under_pressure: false,
      hard_ceiling: 4, eviction_count: 0,
    };
    expect(parseWorkerRetention({
      ...legacy,
      retained_bytes_source: "measured", retained_bytes_basis: "runtime_allocator",
      session_bytes_source: "estimated", session_bytes_basis: "configured_cache",
      anchor_bytes_source: "estimated", anchor_bytes_basis: "configured_cache",
    })).toMatchObject({
      retainedBytes: 4096, retainedBytesSource: "measured", retainedBytesBasis: "runtime_allocator",
      sessionBytes: 4096, sessionBytesSource: "estimated", sessionBytesBasis: "configured_cache",
    });
    expect(parseWorkerRetention({
      ...legacy, retained_bytes_source: "measured", retained_bytes_basis: "runtime_allocator",
      retained_bytes: 0,
    })).toBeUndefined();
    expect(parseWorkerRetention({
      ...legacy, retained_bytes_source: "unavailable", retained_bytes_basis: "not_reported",
    })).toBeUndefined();
    expect(parseWorkerRetention({
      ...legacy,
      retained_bytes_source: "estimated", retained_bytes_basis: "cache_components",
      session_bytes_source: "estimated", session_bytes_basis: "cache_components",
      anchor_bytes_source: "estimated", anchor_bytes_basis: "cache_components",
      retained_bytes: 0, session_bytes: 0, anchor_bytes: 0,
    })).toMatchObject({
      retainedBytes: 0, retainedBytesSource: "estimated", retainedBytesBasis: "cache_components",
    });
    expect(parseWorkerRetention({
      ...legacy,
      retained_bytes: null, retained_bytes_source: "unavailable", retained_bytes_basis: "not_observed",
      session_bytes: null, session_bytes_source: "unavailable", session_bytes_basis: "not_observed",
      anchor_bytes: null, anchor_bytes_source: "unavailable", anchor_bytes_basis: "not_observed",
      evicted_bytes: null, evicted_bytes_source: "unavailable", evicted_bytes_basis: "not_observed",
      estimated_retained_bytes: 4096, estimated_retained_bytes_source: "estimated",
      estimated_retained_bytes_basis: "context_configuration",
    })).toMatchObject({
      retainedBytes: null, retainedBytesSource: "unavailable",
      estimatedRetainedBytes: 4096, estimatedRetainedBytesSource: "estimated",
    });
  });
});

describe("worker token capabilities", () => {
  test.each([
    ["small llama", 4096, 1024],
    ["long mlx", 131072, 8192],
  ])("parses the %s model profile", (_name, context, output) => {
    expect(parseWorkerTokenCapabilities({
      model_context_window: context,
      effective_context_window: context,
      max_input_tokens: context - 1,
      max_output_tokens: output,
      backend_allocation_cap: context,
      user_configured_override: null,
    })).toEqual({
      modelContextWindow: context,
      effectiveContextWindow: context,
      maxInputTokens: context - 1,
      maxOutputTokens: output,
      backendAllocationCap: context,
      userConfiguredOverride: null,
    });
  });

  test("preserves unknown metadata instead of fabricating limits", () => {
    expect(parseWorkerTokenCapabilities({
      model_context_window: null,
      effective_context_window: null,
      max_input_tokens: null,
      max_output_tokens: null,
      backend_allocation_cap: null,
      user_configured_override: null,
    })).toEqual({
      modelContextWindow: null,
      effectiveContextWindow: null,
      maxInputTokens: null,
      maxOutputTokens: null,
      backendAllocationCap: null,
      userConfiguredOverride: null,
    });
  });

  test("preserves authoritative worker metadata sources", () => {
    expect(parseWorkerTokenCapabilities({
      model_context_window: 131072,
      effective_context_window: 131072,
      max_input_tokens: 131071,
      max_output_tokens: null,
      model_context_window_source: "config.json:text_config.max_position_embeddings",
      max_output_tokens_source: null,
      backend_allocation_cap: 131072,
      user_configured_override: null,
    })).toMatchObject({
      modelContextWindowSource: "config.json:text_config.max_position_embeddings",
      maxOutputTokensSource: null,
    });
  });

  test("applies the minimum model, allocation, and admin override", () => {
    expect(deriveTokenCapabilities({
      modelContextWindow: 131072,
      backendAllocationCap: 65536,
      userConfiguredOverride: 32768,
      maxOutputTokens: 8192,
    })).toEqual({
      modelContextWindow: 131072,
      effectiveContextWindow: 32768,
      maxInputTokens: 32767,
      maxOutputTokens: 8192,
      backendAllocationCap: 65536,
      userConfiguredOverride: 32768,
    });
  });

  test("rejects prompt, output, and combined context overflows", () => {
    const capabilities = deriveTokenCapabilities({ modelContextWindow: 4096, backendAllocationCap: 4096, maxOutputTokens: 1024 });
    expect(validateTokenBudget(capabilities, 4096, 1)).toMatchObject({ code: "context_length_exceeded" });
    expect(validateTokenBudget(capabilities, 100, 1025)).toMatchObject({ code: "max_output_tokens_exceeded" });
    expect(validateTokenBudget(capabilities, 3500, 700)).toMatchObject({ code: "context_length_exceeded" });
    expect(validateTokenBudget(capabilities, 3500)).toEqual({ maxTokens: 596 });
  });

  test("requires an explicit output budget when every capability is unknown", () => {
    const capabilities = deriveTokenCapabilities({});
    expect(validateTokenBudget(capabilities, 100)).toMatchObject({ code: "token_capability_unknown" });
    expect(validateTokenBudget(capabilities, 100, 32)).toEqual({ maxTokens: 32 });
  });
});
