import { describe, expect, test } from "bun:test";
import { ChatCompletionRequestSchema, LoadedModelSchema, ResponseRequestSchema } from "./schemas";

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

  test("accepts ordered generic slices with zero-based inclusive message indexes", () => {
    const request = ChatCompletionRequestSchema.parse({
      model: "org/model",
      messages: [
        { role: "system", content: "stable section zero" },
        { role: "user", content: "stable section one" },
        { role: "user", content: "changing suffix" },
      ],
      tools: [{ type: "function", function: { name: "lookup" } }],
      cache: {
        tenant: "acme",
        boundaries: [
          { kind: "tools", label: "slice:tools-v2" },
          { kind: "messages", through_message: 0, label: "prefix-0" },
          { kind: "messages", through_message: 1, label: "checkpoint.alpha" },
        ],
      },
    });
    expect(request.cache?.boundaries).toEqual([
      { kind: "tools", label: "slice:tools-v2" },
      { kind: "messages", through_message: 0, label: "prefix-0" },
      { kind: "messages", through_message: 1, label: "checkpoint.alpha" },
    ]);
  });

  test("labels are optional domain-neutral telemetry and cannot change structural resolution", () => {
    const base = {
      model: "org/model",
      messages: [{ role: "user" as const, content: "stable slice" }],
      cache: { tenant: "acme" },
    };
    const unlabeled = ChatCompletionRequestSchema.parse({
      ...base, cache: { ...base.cache, boundaries: [{ kind: "messages", through_message: 0 }] },
    });
    const first = ChatCompletionRequestSchema.parse({
      ...base, cache: { ...base.cache, boundaries: [{ kind: "messages", through_message: 0, label: "prefix-0" }] },
    });
    const second = ChatCompletionRequestSchema.parse({
      ...base, cache: { ...base.cache, boundaries: [{ kind: "messages", through_message: 0, label: "checkpoint.alpha" }] },
    });
    expect(unlabeled.cache?.boundaries).toEqual([{ kind: "messages", through_message: 0 }]);
    expect(first.cache?.boundaries?.map(({ label: _label, ...boundary }) => boundary)).toEqual(
      second.cache?.boundaries?.map(({ label: _label, ...boundary }) => boundary),
    );
  });

  test("rejects empty, oversized, or unsafe boundary labels", () => {
    const base = { model: "org/model", messages: [{ role: "user", content: "hello" }] };
    for (const label of ["", "contains space", "raw/path", `x${"a".repeat(64)}`]) {
      expect(ChatCompletionRequestSchema.safeParse({
        ...base,
        cache: { tenant: "acme", boundaries: [{ kind: "messages", through_message: 0, label }] },
      }).success).toBe(false);
    }
  });

  test("rejects out-of-range, duplicate, unordered, unsafe tools, and more than eight boundaries", () => {
    const base = { model: "org/model", messages: [{ role: "user", content: "hello" }] };
    const invalid = [
      [{ kind: "messages", through_message: 1 }],
      [{ kind: "messages", through_message: 0 }, { kind: "messages", through_message: 0 }],
      [{ kind: "messages", through_message: 0 }, { kind: "tools" }],
      [{ kind: "tools" }],
      Array.from({ length: 9 }, (_, through_message) => ({ kind: "messages", through_message })),
    ];
    for (const boundaries of invalid) {
      expect(ChatCompletionRequestSchema.safeParse({ ...base, cache: { tenant: "acme", boundaries } }).success).toBe(false);
    }
  });

  test("supports the same current-request contract on Responses cache intent", () => {
    const parsed = ResponseRequestSchema.parse({
      model: "org/model",
      input: [{ role: "user", content: "hello" }],
      cache: { tenant: "acme", boundaries: [{ kind: "messages", through_message: 0, label: "checkpoint.alpha" }] },
    });
    expect(parsed.cache?.boundaries?.[0]).toEqual({ kind: "messages", through_message: 0, label: "checkpoint.alpha" });
  });
});
