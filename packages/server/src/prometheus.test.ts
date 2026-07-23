import { describe, expect, test } from "bun:test";
import { Histogram, renderPrometheus, type PromSnapshot } from "./prometheus";

function snapshot(): PromSnapshot {
  const histogram = () => new Histogram([1]);
  return {
    totals: { requests: 0, ok: 0, errors: 0, cancelled: 0, promptTokens: 0,
      completionTokens: 0, cacheHits: 0, cacheMisses: 0, reusedTokens: 0 },
    activeRequests: 0,
    queue: { inflight: 0, queued: 0, maxInflight: 1, queueDepth: 1 },
    loadedModels: [{
      id: "private/model-id", backend: "mlx", state: "resident",
      retention: {
        maxActive: 1, active: 0, retainedTotal: 0, retainedSessions: 0, retainedAnchors: 0,
        retainedBytes: null, retainedBytesSource: "unavailable", retainedBytesBasis: "not_observed",
        sessionBytes: 0, sessionBytesSource: "estimated", sessionBytesBasis: "cache_components",
        anchorBytes: 0, anchorBytesSource: "estimated", anchorBytesBasis: "cache_components",
        evictedBytes: null, evictedBytesSource: "unavailable", evictedBytesBasis: "not_observed",
        estimatedRetainedBytes: 4096, estimatedRetainedBytesSource: "estimated",
        estimatedRetainedBytesBasis: "cache_components", budgetBytes: 8192,
        highWatermarkBytes: 7000, lowWatermarkBytes: 6000, underPressure: false,
        hardCeiling: 4, evictionCount: 0,
      },
    }],
    uptimeMs: 0,
    histograms: { ttftMs: histogram(), durationMs: histogram(), queuedMs: histogram(), completionTokens: histogram() },
    structuredOutputOutcomes: new Map(),
    residency: { reservedBytes: 0, activeReservations: 0, outcomes: new Map(), evictions: new Map(),
      estimateObservedRatioSum: 0, estimateObservedRatioCount: 0 },
  };
}

describe("honest Prometheus memory telemetry", () => {
  test("omits unavailable values, labels estimates, and never exposes model IDs", () => {
    const output = renderPrometheus(snapshot());
    expect(output).not.toContain("private/model-id");
    expect(output).not.toContain("clap_retention_bytes{");
    expect(output).not.toContain("clap_retention_evicted_bytes{");
    expect(output).toContain('clap_retention_session_bytes{backend="mlx",source="estimated",basis="cache_components"} 0');
    expect(output).toContain('clap_retention_estimated_bytes{backend="mlx",source="estimated",basis="cache_components"} 4096');
    expect(output).not.toContain('source="measured"');
  });
});
