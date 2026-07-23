import { describe, expect, test } from "bun:test";
import { ChatCompletionRequestSchema, ErrorResponseSchema, LoadedModelSchema, ResponseFormatSchema, ResponseRequestSchema } from "./schemas";

describe("structured response format schema", () => {
  test("accepts optional constraints and maps legacy strict schemas to required", () => {
    expect(ResponseFormatSchema.parse({ type: "json_object" })).toEqual({ type: "json_object" });
    expect(ResponseFormatSchema.parse({ type: "json_object", constraint: "best_effort" })).toEqual({
      type: "json_object", constraint: "best_effort",
    });
    expect(ResponseFormatSchema.parse({
      type: "json_schema",
      json_schema: { name: "Result", strict: true, schema: { type: "object" } },
    })).toMatchObject({ type: "json_schema", constraint: "required" });
  });

  test("requires schemas and accepts local references", () => {
    expect(ResponseFormatSchema.safeParse({
      type: "json_schema", json_schema: { name: "Missing" },
    }).success).toBe(false);
    expect(ResponseFormatSchema.safeParse({
      type: "json_schema",
      json_schema: {
        name: "LocalRef",
        schema: { $defs: { item: { type: "string" } }, type: "object", properties: { item: { $ref: "#/$defs/item" } } },
      },
    }).success).toBe(true);
  });

  test("rejects remote references and bounded schema violations", () => {
    const nested: Record<string, unknown> = { type: "object" };
    let cursor = nested;
    for (let index = 0; index < 33; index += 1) {
      cursor.properties = { child: { type: "object" } };
      cursor = (cursor.properties as Record<string, Record<string, unknown>>).child!;
    }
    const formats = [
      { type: "json_schema", json_schema: { name: "Remote", schema: { $ref: "https://example.com/schema.json" } } },
      { type: "json_schema", json_schema: { name: "Large", schema: { description: "x".repeat(65_536) } } },
      { type: "json_schema", json_schema: { name: "Deep", schema: nested } },
      { type: "json_schema", json_schema: { name: "Wide", schema: {
        type: "object", properties: Object.fromEntries(Array.from({ length: 1025 }, (_, index) => [`p${index}`, { type: "string" }])),
      } } },
    ];
    for (const format of formats) expect(ResponseFormatSchema.safeParse(format).success).toBe(false);
  });
});

describe("loaded model retention schema", () => {
  const model = {
    key: "mlx:model", id: "mlx/model", backend: "mlx", format: "mlx", localPath: "/models/mlx",
    state: "warm", activeRequests: 0, loadedAt: "now", lastUsedAt: "now", keepAlive: "5m",
    expiresAt: null, pinned: false, always: false, worker: { state: "resident" },
  };

  test("preserves nullable honest memory provenance", () => {
    const memory = { activeBytes: 1024, activeBytesSource: "measured" as const, activeBytesBasis: "worker_allocator",
      cacheBytes: null, cacheBytesSource: "unavailable" as const, cacheBytesBasis: "not_observed",
      peakActiveBytes: 2048, peakActiveBytesSource: "measured" as const, peakActiveBytesBasis: "worker_allocator" };
    expect(LoadedModelSchema.parse({ ...model, worker: { ...model.worker, memory } }).worker.memory).toEqual(memory);
  });

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

  test("preserves exact worker launch diagnostics", () => {
    const launch = {
      launchId: "launch-123",
      stderrLogPath: "/clap/logs/workers/mlx/hash/launch-123.stderr.log",
      launchMetadataPath: "/clap/logs/workers/mlx/hash/launch-123.json",
      crashClassification: "decode",
    };
    expect(LoadedModelSchema.parse({ ...model, worker: { ...model.worker, ...launch } }).worker)
      .toMatchObject(launch);
  });

  test("preserves residency state without treating estimates as observed RSS", () => {
    const residency = {
      estimateBytes: 1_000,
      estimateSource: "model_artifacts" as const,
      observedRssBytes: null,
      observedRssSource: null,
      reservationBytes: 1_000,
      lastAdmissionReason: "within_budget" as const,
      lastEvictionReason: null,
    };
    const parsed = LoadedModelSchema.parse({
      ...model,
      worker: { ...model.worker, loadState: "resident", residency },
    });
    expect(parsed.worker).toMatchObject({ loadState: "resident", residency });
    expect(() => LoadedModelSchema.parse({
      ...model,
      worker: { ...model.worker, residency: { ...residency, observedRssBytes: 1_000, observedRssSource: null } },
    })).toThrow();
  });

  test("exposes strict effective capability groups on loaded workers", () => {
    const effectiveCapabilities = {
      cache: { partialSuffixTrim: true, partialPrefixBranch: false, wholeStateCopy: true,
        promptBoundarySnapshots: true, quantizedKv: false },
      generation: { structuredOutput: { json_object: "native" as const, json_schema: "post_validate" as const,
        post_validation: true, max_schema_bytes: 65_536 }, toolTemplateSupport: true },
      modalities: { input: ["text"] as ["text"], output: ["text"] as ["text"] },
    };
    expect(LoadedModelSchema.parse({ ...model, worker: { ...model.worker, effectiveCapabilities } })
      .worker.effectiveCapabilities).toEqual(effectiveCapabilities);
    expect(() => LoadedModelSchema.parse({ ...model, worker: { ...model.worker,
      effectiveCapabilities: { ...effectiveCapabilities, generation: { toolTemplateSupport: true } },
    } })).toThrow();
  });
});

describe("structured model memory errors", () => {
  test("accepts safe scalar admission details", () => {
    expect(ErrorResponseSchema.parse({
      error: {
        message: "Insufficient memory to load the requested model safely",
        type: "model_error",
        code: "insufficient_model_memory",
        details: { requestedBytes: 10, availableBytes: null, retryable: true },
      },
    }).error.details).toEqual({ requestedBytes: 10, availableBytes: null, retryable: true });
  });
});

describe("cache intent schema", () => {
  test("accepts and preserves organizational cache intent", () => {
    const parsed = ChatCompletionRequestSchema.parse({
      model: "org/model",
      messages: [{ role: "user", content: "hello" }],
      cache: {
        tenant: "acme-private",
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
      tenant: "acme-private",
      namespace: "acme-private",
      project: "payments",
      harness: "coding-v4",
      agent: "reviewer",
      session: "conversation-123",
      priority: "background",
      side_request: true,
    });
  });

  test("allows cache intent without identity labels", () => {
    const parsed = ChatCompletionRequestSchema.parse({
      model: "org/model",
      messages: [{ role: "user", content: "hello" }],
      cache: { project: "payments", priority: "interactive", side_request: false },
    });
    expect(parsed.cache).toEqual({ project: "payments", priority: "interactive", side_request: false });
  });

  test("bounds all public labels and rejects conflicting deprecated tenant alias", () => {
    const base = { model: "org/model", messages: [{ role: "user", content: "hello" }] };
    for (const label of ["namespace", "tenant", "project", "harness", "agent", "session"] as const) {
      expect(() => ChatCompletionRequestSchema.parse({
        ...base,
        cache: { [label]: "x".repeat(129) },
      })).toThrow();
      expect(ChatCompletionRequestSchema.parse({
        ...base,
        cache: { [label]: `  ${label}  ` },
      }).cache?.[label]).toBe(label);
    }
    expect(() => ChatCompletionRequestSchema.parse({
      ...base,
      cache: { namespace: "preferred", tenant: "different" },
    })).toThrow("deprecated tenant alias must match");
    expect(ChatCompletionRequestSchema.parse({
      ...base,
      cache: { namespace: "same", tenant: "same" },
    }).cache).toMatchObject({ namespace: "same", tenant: "same" });
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
