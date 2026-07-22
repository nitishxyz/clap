import { describe, expect, test } from "bun:test";
import { deriveTokenCapabilities, parseWorkerRetention, parseWorkerTokenCapabilities, validateTokenBudget } from "./resident";

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
      highWatermarkBytes: 90_000, lowWatermarkBytes: 75_000, underPressure: false,
      hardCeiling: 256, evictionReason: "byte_pressure", evictionCount: 7,
    });
  });

  test("preserves absent telemetry for older and non-MLX workers", () => {
    expect(parseWorkerRetention(undefined)).toBeUndefined();
    expect(parseWorkerRetention({ retained_total: 1 })).toBeUndefined();
    expect(parseWorkerRetention({ max_active: 4, active_policy: { mode: "auto" } })).toBeUndefined();
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
