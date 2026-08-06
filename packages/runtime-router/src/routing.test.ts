import { describe, expect, test } from "bun:test";
import { CacheAwareRoutingDirectory, stableRoutingWorkerId, type RoutingWorkerHeartbeat } from "./routing";

function heartbeat(overrides: Partial<RoutingWorkerHeartbeat> & Pick<RoutingWorkerHeartbeat, "workerId">): RoutingWorkerHeartbeat {
  return {
    workerId: overrides.workerId,
    generation: overrides.generation ?? "generation-1",
    modelDomain: overrides.modelDomain ?? "model-a",
    state: overrides.state ?? "ready",
    observedAt: overrides.observedAt ?? 1_000,
    activeRequests: overrides.activeRequests ?? 0,
    queueDepth: overrides.queueDepth ?? 0,
    estimatedQueueWaitMs: overrides.estimatedQueueWaitMs ?? 0,
    pressure: overrides.pressure ?? 0,
    loaded: overrides.loaded ?? true,
    estimatedColdLoadMs: overrides.estimatedColdLoadMs ?? 0,
    prefillTokensPerSecond: overrides.prefillTokensPerSecond ?? 1_000,
  };
}

describe("cache-aware routing directory", () => {
  test("prefers the worker with an exact session location when saved prefill beats queueing", () => {
    const directory = new CacheAwareRoutingDirectory({ now: () => 1_000 });
    directory.heartbeat(heartbeat({ workerId: "worker-a", estimatedQueueWaitMs: 20 }));
    directory.heartbeat(heartbeat({ workerId: "worker-b" }));
    directory.publishLocation({
      modelDomain: "model-a", shareDomain: "tenant-a", kind: "session",
      fingerprint: "session-a", workerId: "worker-a", workerGeneration: "generation-1",
      tokenCount: 8_000, lastSeenAt: 1_000,
    });

    const decision = directory.select({
      modelDomain: "model-a", shareDomain: "tenant-a", sessionFingerprint: "session-a",
      promptTokens: 8_500, affinityFingerprint: "session-a",
    });

    expect(decision).toMatchObject({
      workerId: "worker-a",
      reason: "session_locality",
      directoryHit: true,
      matchedTokens: 8_000,
      candidateCount: 2,
    });
  });

  test("chooses idle recomputation when the cached worker queue is more expensive", () => {
    const directory = new CacheAwareRoutingDirectory({ now: () => 1_000, pressurePenaltyMs: 1_000 });
    directory.heartbeat(heartbeat({ workerId: "cached", estimatedQueueWaitMs: 5_000 }));
    directory.heartbeat(heartbeat({ workerId: "idle" }));
    directory.publishLocation({
      modelDomain: "model-a", shareDomain: "tenant-a", kind: "session",
      fingerprint: "session-a", workerId: "cached", workerGeneration: "generation-1",
      tokenCount: 1_000, lastSeenAt: 1_000,
    });

    const decision = directory.select({
      modelDomain: "model-a", shareDomain: "tenant-a", sessionFingerprint: "session-a",
      promptTokens: 2_000, affinityFingerprint: "session-a",
    });

    expect(decision?.workerId).toBe("idle");
    expect(decision?.reason).toBe("lowest_cost");
    expect(decision?.directoryHit).toBe(false);
    expect(decision?.candidates.find((candidate) => candidate.workerId === "cached")?.totalCostMs).toBe(6_000);
    expect(decision?.candidates.find((candidate) => candidate.workerId === "idle")?.totalCostMs).toBe(2_000);
  });

  test("fails closed across tenant and model compatibility domains", () => {
    const directory = new CacheAwareRoutingDirectory({ now: () => 1_000 });
    directory.heartbeat(heartbeat({ workerId: "model-a-worker" }));
    directory.heartbeat(heartbeat({ workerId: "model-b-worker", modelDomain: "model-b" }));
    directory.publishLocation({
      modelDomain: "model-a", shareDomain: "tenant-a", kind: "session",
      fingerprint: "same-label", workerId: "model-a-worker", workerGeneration: "generation-1",
      tokenCount: 4_000, lastSeenAt: 1_000,
    });

    const tenantMismatch = directory.select({
      modelDomain: "model-a", shareDomain: "tenant-b", sessionFingerprint: "same-label",
      promptTokens: 4_000, affinityFingerprint: "same-label",
    });
    const modelMismatch = directory.select({
      modelDomain: "model-b", shareDomain: "tenant-a", sessionFingerprint: "same-label",
      promptTokens: 4_000, affinityFingerprint: "same-label",
    });

    expect(tenantMismatch?.directoryHit).toBe(false);
    expect(modelMismatch).toMatchObject({ workerId: "model-b-worker", directoryHit: false });
  });

  test("ignores stale generations and expires bounded soft state", () => {
    let now = 1_000;
    const directory = new CacheAwareRoutingDirectory({
      now: () => now,
      workerTtlMs: 100,
      locationTtlMs: 200,
      maxLocations: 2,
    });
    directory.heartbeat(heartbeat({ workerId: "worker-a", observedAt: now }));
    directory.publishLocation({
      modelDomain: "model-a", shareDomain: "tenant-a", kind: "session",
      fingerprint: "old", workerId: "worker-a", workerGeneration: "generation-1",
      tokenCount: 100, lastSeenAt: now,
    });
    directory.heartbeat(heartbeat({ workerId: "worker-a", generation: "generation-2", observedAt: now }));

    const decision = directory.select({
      modelDomain: "model-a", shareDomain: "tenant-a", sessionFingerprint: "old",
      promptTokens: 100, affinityFingerprint: "old",
    });
    expect(decision).toMatchObject({ directoryHit: false, staleLocationsIgnored: 1 });

    directory.publishLocation({
      modelDomain: "model-a", shareDomain: "tenant-a", kind: "session",
      fingerprint: "new-1", workerId: "worker-a", workerGeneration: "generation-2",
      tokenCount: 100, lastSeenAt: now,
    });
    directory.publishLocation({
      modelDomain: "model-a", shareDomain: "tenant-a", kind: "session",
      fingerprint: "new-2", workerId: "worker-a", workerGeneration: "generation-2",
      tokenCount: 100, lastSeenAt: now,
    });
    expect(directory.snapshot().locations.map((location) => location.fingerprint)).toEqual(["new-1", "new-2"]);

    now = 1_201;
    expect(directory.cleanup()).toEqual({ workers: 1, locations: 2 });
    expect(directory.snapshot()).toMatchObject({ workers: [], locations: [] });
  });

  test("uses deterministic sticky routing when workers have equal cost", () => {
    const directory = new CacheAwareRoutingDirectory({ now: () => 1_000 });
    directory.heartbeat(heartbeat({ workerId: "worker-a" }));
    directory.heartbeat(heartbeat({ workerId: "worker-b" }));
    const query = {
      modelDomain: "model-a", shareDomain: "tenant-a", promptTokens: 10,
      affinityFingerprint: "affinity-a",
    };
    const first = directory.select(query);
    const second = directory.select(query);
    expect(first?.reason).toBe("sticky_fallback");
    expect(second?.workerId).toBe(first?.workerId);
  });

  test("derives opaque stable worker IDs without leaking node or model labels", () => {
    const first = stableRoutingWorkerId("pod-a", "model-a", 0);
    expect(first).toBe(stableRoutingWorkerId("pod-a", "model-a", 0));
    expect(first).not.toBe(stableRoutingWorkerId("pod-a", "model-a", 1));
    expect(first).not.toContain("pod-a");
    expect(first).not.toContain("model-a");
  });
});
