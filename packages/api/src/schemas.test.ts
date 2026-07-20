import { describe, expect, test } from "bun:test";
import { ChatCompletionRequestSchema, LoadedModelSchema } from "./schemas";

describe("loaded model retention schema", () => {
  const model = {
    key: "mlx:model", id: "mlx/model", backend: "mlx", format: "mlx", localPath: "/models/mlx",
    state: "warm", activeRequests: 0, loadedAt: "now", lastUsedAt: "now", keepAlive: "5m",
    expiresAt: null, pinned: false, always: false, worker: { state: "resident" },
  };

  test("accepts MLX retention telemetry and keeps it optional", () => {
    expect(LoadedModelSchema.parse(model).worker.retention).toBeUndefined();
    const retention = {
      maxActive: 4, queued: 2, previousMaxActive: 2,
      lastAdjustmentReason: "global_headroom_available", lastAdjustmentAt: "2026-07-20T11:29:00.000Z",
      retainedGrowthReserveBytes: 65_536, globalResidentMemoryBytes: 8192,
      pressureState: "normal" as const,
      activePolicy: { mode: "auto" as const, selectedMax: 4, backendCeiling: 16,
        hardwareCeiling: 8, modelCeiling: 16, memoryCeiling: 4, reason: "memory_ceiling",
        inputs: { model_active_bytes: 4096, hybrid_or_recurrent: false } },
      active: 0, retainedTotal: 101, retainedSessions: 100, retainedAnchors: 1,
      retainedBytes: 4096, sessionBytes: 3072, anchorBytes: 1024, budgetBytes: 1_000_000,
      highWatermarkBytes: 900_000, lowWatermarkBytes: 750_000, underPressure: false,
      hardCeiling: 256, evictionCount: 0,
    };
    expect(LoadedModelSchema.parse({ ...model, worker: { ...model.worker, retention } }).worker.retention).toEqual(retention);
  });
});

describe("cache intent schema", () => {
  test("accepts and preserves organizational cache intent", () => {
    const parsed = ChatCompletionRequestSchema.parse({
      model: "org/model",
      messages: [{ role: "user", content: "hello" }],
      cache: {
        tenant: "acme",
        namespace: "acme-private",
        project: "payments",
        harness: "coding-v4",
        agent: "reviewer",
        session: "conversation-123",
        priority: "background",
        side_request: true,
      },
    });
    expect(parsed.cache).toEqual({
      tenant: "acme",
      namespace: "acme-private",
      project: "payments",
      harness: "coding-v4",
      agent: "reviewer",
      session: "conversation-123",
      priority: "background",
      side_request: true,
    });
  });

  test("requires an isolation namespace", () => {
    expect(() => ChatCompletionRequestSchema.parse({
      model: "org/model",
      messages: [{ role: "user", content: "hello" }],
      cache: { project: "payments" },
    })).toThrow();
  });
});
