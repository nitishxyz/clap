import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CacheEventStore } from "./cache-event-store";
import { MetricsCollector } from "./metrics";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const tempRoots: string[] = [];
afterEach(() => {
  for (const path of tempRoots.splice(0)) rmSync(path, { recursive: true, force: true });
});
function tempDir(): string {
  const path = mkdtempSync(join(tmpdir(), "clap-metrics-"));
  tempRoots.push(path);
  return path;
}

describe("metrics queue accounting", () => {
  test("structured telemetry fingerprints schemas and uses bounded metric labels", () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("secret-model-id", "/v1/chat/completions", false);
    handle.capture({ messages: [{ role: "user", content: "x" }], response_format: {
      type: "json_schema", constraint: "required",
      json_schema: { schema: { type: "object", properties: { secretField: { const: "secret-value" } } } },
    } });
    handle.finish({ status: "ok", structuredOutput: {
      backendMode: "native", outcome: "native_validated", repairApplied: false, validationMs: 1.25,
    } });
    const record = metrics.recent(1)[0]!;
    expect(record.structuredOutput).toMatchObject({
      kind: "json_schema", requestedStrength: "required", backendMode: "native",
      outcome: "native_validated", schemaSize: expect.any(Number), schemaFingerprint: expect.any(String),
    });
    expect(JSON.stringify(record.structuredOutput)).not.toContain("secretField");
    expect(JSON.stringify(record.structuredOutput)).not.toContain("secret-value");
    expect([...metrics.structuredOutputOutcomes.keys()].join(" ")).not.toContain("secret-model-id");
    expect([...metrics.structuredOutputOutcomes.keys()].join(" ")).not.toContain("secretField");
  });
  test("time waiting in queue is tracked separately from ttft", async () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", false);
    handle.phase("queued");
    await sleep(120);
    // Worker picks the request up and starts prefill.
    handle.prefill(10, 100);
    await sleep(60);
    handle.firstToken();
    await sleep(30);
    handle.finish({ status: "ok", promptTokens: 100, completionTokens: 20 });

    const record = handle.record;
    expect(record.queuedMs).toBeGreaterThanOrEqual(100);
    expect(record.queuedMs!).toBeLessThan(400);
    // TTFT is measured from dispatch, not from arrival: it must not include
    // the 120ms spent queued.
    expect(record.ttftMs).toBeGreaterThanOrEqual(40);
    expect(record.ttftMs!).toBeLessThan(115);
    expect(record.durationMs!).toBeGreaterThanOrEqual(200);
  });

  test("requests cancelled while queued record queue time", async () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", true);
    handle.phase("queued");
    await sleep(80);
    handle.finish({ status: "cancelled", finishReason: "cancel" });
    expect(handle.record.queuedMs).toBeGreaterThanOrEqual(60);
    expect(handle.record.ttftMs).toBeUndefined();
  });

  test("fast unqueued requests do not report noise queue time", () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", false);
    handle.prefill(1, 10);
    handle.firstToken();
    handle.finish({ status: "ok", promptTokens: 10, completionTokens: 5 });
    expect(handle.record.queuedMs).toBeUndefined();
  });

  test("preserves normalized cache coordinator telemetry", () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", false);
    handle.finish({
      status: "ok",
      cacheHit: true,
      reusedTokens: 42,
      reuseKind: "branch",
      reuseScope: "project",
      sideRequest: true,
      slot: 3,
      cacheNamespace: "tenant-a",
      donorSlot: 1,
      targetSlot: 3,
      evictedSlots: [0, 2],
      cacheDecisionUs: 17,
      plannedReuseTokens: 44,
      realizedReuseTokens: 42,
      cacheFallback: "copy_retry",
    });
    expect(handle.record).toMatchObject({
      cacheNamespace: "tenant-a",
      donorSlot: 1,
      targetSlot: 3,
      evictedSlots: [0, 2],
      cacheDecisionUs: 17,
      plannedReuseTokens: 44,
      realizedReuseTokens: 42,
      cacheFallback: "copy_retry",
    });
  });

  test("classifies cold only for the first decision of a model+worker domain", () => {
    const metrics = new MetricsCollector(undefined, { checkpointMinimumTokens: () => 2048 });
    const first = metrics.start("model-a", "/v1/chat/completions", false);
    first.finish({
      status: "ok",
      cacheHit: false,
      reusedTokens: 0,
      workerLaunchId: "worker-1",
      cacheCandidates: [],
      promptTokenCount: 4096,
    });
    expect(first.record.cacheOutcome?.category).toBe("cold");
    expect(first.record.cacheOutcome?.evidence).toContain("isFirstDecisionForWorkerModelDomain=true");

    const second = metrics.start("model-a", "/v1/chat/completions", false);
    second.finish({
      status: "ok",
      cacheHit: false,
      reusedTokens: 0,
      workerLaunchId: "worker-1",
      cacheCandidates: [],
      promptTokenCount: 4096,
    });
    expect(second.record.cacheOutcome?.category).toBe("miss_reason_unavailable");

    const otherDomain = metrics.start("model-b", "/v1/chat/completions", false);
    otherDomain.finish({
      status: "ok",
      cacheHit: false,
      reusedTokens: 0,
      workerLaunchId: "worker-1",
      cacheCandidates: [],
      promptTokenCount: 4096,
    });
    expect(otherDomain.record.cacheOutcome?.category).toBe("cold");
  });

  test("does not guess cold without a worker launch id", () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("model-a", "/v1/chat/completions", false);
    handle.finish({
      status: "ok",
      cacheHit: false,
      reusedTokens: 0,
      cacheCandidates: [],
      promptTokenCount: 4096,
    });
    expect(handle.record.cacheOutcome?.category).toBe("miss_reason_unavailable");
  });

  test("exposes session display identity from cache.session fingerprint", () => {
    const store = new CacheEventStore({ directory: tempDir() });
    const metrics = new MetricsCollector(store);
    const sessions = ["S1-session", "S2-session", "S3-session", "S4-session"];
    const displays: string[] = [];
    let sharedPrefix: string | undefined;
    for (const session of sessions) {
      const handle = metrics.start("model-a", "/v1/chat/completions", false);
      handle.capture({
        messages: [{ role: "system", content: "same system" }, { role: "user", content: "same first user" }],
        cache: { session, namespace: "tenant", priority: "normal" },
      });
      handle.finish({ status: "ok", cacheHit: false, reusedTokens: 0, workerLaunchId: "w", cacheCandidates: [] });
      expect(handle.record.sessionIdentityKind).toBe("cache_session");
      expect(handle.record.sessionDisplayId).toBeDefined();
      expect(handle.record.sessionFingerprint).toBeDefined();
      expect(handle.record.sessionDisplayId).toBe(handle.record.sessionFingerprint!.slice(0, 8));
      displays.push(handle.record.sessionDisplayId!);
      // Same prompt prefix for all four.
      expect(handle.record.conversation).toBeDefined();
      if (!sharedPrefix) sharedPrefix = handle.record.conversation;
      else expect(handle.record.conversation).toBe(sharedPrefix);
    }
    expect(new Set(displays).size).toBe(4);
    // Shared prompt prefix must not be used as the session display id.
    expect(sharedPrefix).toBeDefined();
    for (const display of displays) expect(display).not.toBe(sharedPrefix);
  });

  test("no-session requests label identity as prompt_prefix", () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("model-a", "/v1/chat/completions", false);
    handle.capture({ messages: [{ role: "user", content: "hello" }] });
    handle.finish({ status: "ok" });
    expect(handle.record.sessionIdentityKind).toBe("prompt_prefix");
    expect(handle.record.sessionDisplayId).toBe(handle.record.conversation);
    expect(handle.record.sessionFingerprint).toBeUndefined();
  });

  test("records critical pressure evictions with a safe reason", () => {
    const metrics = new MetricsCollector();
    metrics.residencyEvent({
      type: "model_evicted_for_pressure", backend: "mlx", reason: "critical_memory_pressure",
      reservationBytes: 0, activeReservations: 0,
    });
    expect(metrics.residency.evictions.get("mlx\0critical_memory_pressure")).toBe(1);
    expect(metrics.events(1)[0]).toMatchObject({
      type: "model_evicted_for_pressure", backend: "mlx", reason: "critical_memory_pressure",
    });
  });
});
