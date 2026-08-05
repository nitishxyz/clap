import { describe, expect, test } from "bun:test";
import type { DashboardTotals } from "@/lib/api";
import { cacheSummary } from "./Tiles";

function totals(overrides: Partial<DashboardTotals> = {}): DashboardTotals {
  return {
    requests: 0,
    ok: 0,
    errors: 0,
    cancelled: 0,
    promptTokens: 0,
    completionTokens: 0,
    cacheHits: 0,
    cacheMisses: 0,
    cacheEligible: 0,
    cacheNotEligible: 0,
    cacheIsolatedMisses: 0,
    cacheFreshMisses: 0,
    reusedTokens: 0,
    ...overrides,
  };
}

describe("dashboard cache tiles", () => {
  test("reports reuse for clients that never send cache intent", () => {
    // Shape observed from a stock OpenAI client: no intent, so the strict KPI
    // counters stay at zero while physical reuse is substantial.
    const summary = cacheSummary(totals({
      requests: 24,
      cacheEligible: 0,
      cacheNotEligible: 24,
      cacheHits: 0,
      cacheMisses: 0,
      reusedTokens: 0,
      physicalCacheHits: 23,
      physicalCacheMisses: 1,
      physicalReusedTokens: 100_000,
      physicalPromptTokens: 125_000,
    }));

    expect(summary.hitRate).toBe("96%");
    expect(summary.detail).toBe("24 admitted · 23 hit · 1 miss");
    expect(summary.reusedTokens).toBe(100_000);
    expect(summary.reuseSub).toBe("80% of prompt tokens skipped");
  });

  test("falls back to the intent-gated KPI when a server omits physical totals", () => {
    const summary = cacheSummary(totals({
      cacheEligible: 5, cacheHits: 4, cacheMisses: 1, reusedTokens: 512,
    }));

    expect(summary.hitRate).toBe("80%");
    expect(summary.detail).toBe("5 eligible · 4 hit · 1 miss");
    expect(summary.reusedTokens).toBe(512);
    expect(summary.reuseSub).toBe("prompt tokens skipped");
  });

  test("shows a dash instead of a fabricated rate when nothing was admitted", () => {
    const summary = cacheSummary(totals({
      physicalCacheHits: 0, physicalCacheMisses: 0,
      physicalReusedTokens: 0, physicalPromptTokens: 0,
    }));

    expect(summary.hitRate).toBe("-");
    expect(summary.reuseSub).toBe("prompt tokens skipped");
  });

  test("clamps the reuse ratio so reported reuse never exceeds the prompt", () => {
    const summary = cacheSummary(totals({
      physicalCacheHits: 1, physicalCacheMisses: 0,
      physicalReusedTokens: 140, physicalPromptTokens: 100,
    }));

    expect(summary.reuseSub).toBe("100% of prompt tokens skipped");
  });
});
