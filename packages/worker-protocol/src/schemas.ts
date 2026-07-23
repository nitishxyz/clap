import { z } from "zod";
import { WORKER_PROTOCOL_VERSION } from "./types";

const extensibleObject = <T extends z.ZodRawShape>(shape: T) => z.object(shape).passthrough();
const protocol = z.literal(WORKER_PROTOCOL_VERSION);
const requestId = z.string().min(1, "request_id must not be empty");
const sequence = z.number().int().nonnegative();
const content = z.union([
  z.string(), z.number(), z.boolean(), z.null(), z.record(z.unknown()), z.array(z.unknown()),
]);

const requestBase = { protocol, request_id: requestId };
const fingerprint = z.string().regex(/^[0-9a-f]{64}$/);
const displayLabel = z.string().min(1).max(128);
const memoryBytes = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const measuredMemoryBasis = z.enum(["resident_rss", "runtime_allocator", "worker_allocator", "os_available"]);
const estimatedMemoryBasis = z.enum([
  "prior_observation", "model_artifacts", "architecture_metadata", "configured_cache", "cache_components", "context_configuration",
  "conservative_fallback",
]);
const unavailableMemoryBasis = z.enum(["not_observed", "not_supported", "not_reported"]);
export const MemoryValueSchema = z.discriminatedUnion("source", [
  z.object({ bytes: memoryBytes.positive(), source: z.literal("measured"), basis: measuredMemoryBasis }).strict(),
  z.object({ bytes: memoryBytes, source: z.literal("estimated"), basis: estimatedMemoryBasis }).strict(),
  z.object({ bytes: z.null(), source: z.literal("unavailable"), basis: unavailableMemoryBasis }).strict(),
]);

const memorySource = z.enum(["measured", "estimated", "unavailable"]);
const memoryBasis = z.union([measuredMemoryBasis, estimatedMemoryBasis, unavailableMemoryBasis]);
const addMemoryCompanionValidation = <T extends z.ZodRawShape>(shape: T, fields: readonly string[]) =>
  extensibleObject(shape).superRefine((value, context) => {
    const record = value as Record<string, unknown>;
    for (const field of fields) {
      const sourceKey = `${field}_source`;
      const basisKey = `${field}_basis`;
      const hasSource = record[sourceKey] !== undefined;
      const hasBasis = record[basisKey] !== undefined;
      if (!hasSource && !hasBasis) {
        if (record[field] === null) context.addIssue({ code: "custom", path: [field], message: "null bytes require source and basis" });
        continue;
      }
      if (!hasSource || !hasBasis) {
        context.addIssue({ code: "custom", path: [!hasSource ? sourceKey : basisKey], message: "memory source and basis must be provided together" });
        continue;
      }
      const parsed = MemoryValueSchema.safeParse({
        bytes: record[field], source: record[sourceKey], basis: record[basisKey],
      });
      if (!parsed.success) context.addIssue({ code: "custom", path: [field], message: "invalid memory value" });
    }
  });

export const WorkerMemoryTelemetrySchema = addMemoryCompanionValidation({
  active_bytes: memoryBytes.nullable(), active_bytes_source: memorySource.optional(), active_bytes_basis: memoryBasis.optional(),
  cache_bytes: memoryBytes.nullable(), cache_bytes_source: memorySource.optional(), cache_bytes_basis: memoryBasis.optional(),
  peak_active_bytes: memoryBytes.nullable(), peak_active_bytes_source: memorySource.optional(), peak_active_bytes_basis: memoryBasis.optional(),
}, ["active_bytes", "cache_bytes", "peak_active_bytes"]);

export const WorkerRetentionTelemetrySchema = addMemoryCompanionValidation({
  retained_bytes: memoryBytes.nullable().optional(), retained_bytes_source: memorySource.optional(), retained_bytes_basis: memoryBasis.optional(),
  session_bytes: memoryBytes.nullable().optional(), session_bytes_source: memorySource.optional(), session_bytes_basis: memoryBasis.optional(),
  anchor_bytes: memoryBytes.nullable().optional(), anchor_bytes_source: memorySource.optional(), anchor_bytes_basis: memoryBasis.optional(),
  evicted_bytes: memoryBytes.nullable().optional(), evicted_bytes_source: memorySource.optional(), evicted_bytes_basis: memoryBasis.optional(),
  estimated_retained_bytes: memoryBytes.nullable().optional(), estimated_retained_bytes_source: memorySource.optional(), estimated_retained_bytes_basis: memoryBasis.optional(),
}, ["retained_bytes", "session_bytes", "anchor_bytes", "evicted_bytes", "estimated_retained_bytes"]);
export const CacheIdentitySchema = z.object({
  version: z.literal(1),
  generation: z.string().min(1).max(64),
  tenant_root: fingerprint,
  project_fingerprint: fingerprint.optional(),
  harness_fingerprint: fingerprint.optional(),
  agent_fingerprint: fingerprint.optional(),
  session_fingerprint: fingerprint.optional(),
  scope: z.enum(["tenant", "project", "harness", "agent", "session"]),
  scope_fingerprint: fingerprint,
  namespace_fingerprint: fingerprint,
  namespace_id: z.string().regex(/^[1-9][0-9]{0,19}$/)
    .refine((value) => BigInt(value) <= 0xffff_ffff_ffff_ffffn, "namespace_id must fit unsigned 64-bit"),
  priority: z.enum(["interactive", "normal", "background"]).default("normal"),
  side_request: z.boolean(),
  display: z.object({
    namespace: displayLabel.optional(),
    project: displayLabel.optional(),
    harness: displayLabel.optional(),
    agent: displayLabel.optional(),
    session: displayLabel.optional(),
  }).strict(),
  physical: z.object({
    fingerprint,
    backend: z.enum(["llama", "mlx"]),
    resolved_revision: z.string().min(1).max(256),
    model_artifact_fingerprint: fingerprint,
    tokenizer_fingerprint: fingerprint,
    context_allocation: z.number().int().nonnegative(),
    kv_format: z.string().min(1).max(64),
    unified_kv: z.boolean(),
    layout_version: z.number().int().positive(),
  }).strict(),
}).strict();
export const StructuredOutputContractSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("json_object"),
    strength: z.enum(["best_effort", "required"]),
    schema: z.never().optional(),
  }).strict(),
  z.object({
    kind: z.literal("json_schema"),
    strength: z.enum(["best_effort", "required"]),
    schema: z.record(z.unknown()),
  }).strict(),
]);
export const LoadRequestSchema = extensibleObject({ ...requestBase, type: z.literal("load"), model: z.string().min(1) });
export const GenerateRequestSchema = extensibleObject({
  ...requestBase, type: z.literal("generate"), prompt: z.string(), cache_identity: CacheIdentitySchema,
  structured_output: StructuredOutputContractSchema.optional(),
});
export const CancelRequestSchema = extensibleObject({ ...requestBase, type: z.literal("cancel"), target_request_id: requestId });
export const SetMaxActiveRequestSchema = extensibleObject({ ...requestBase, type: z.literal("set_max_active"), max_active: z.number().int().positive() });
export const UnloadRequestSchema = extensibleObject({ ...requestBase, type: z.literal("unload") });
export const ShutdownRequestSchema = extensibleObject({ ...requestBase, type: z.literal("shutdown") });

export const WorkerRequestSchema = z.discriminatedUnion("type", [
  LoadRequestSchema, GenerateRequestSchema, CancelRequestSchema, SetMaxActiveRequestSchema,
  UnloadRequestSchema, ShutdownRequestSchema,
]);

const structuredOutputMode = z.enum(["native", "post_validate", "unsupported"]);
export const StructuredOutputCapabilitiesSchema = z.object({
  json_object: structuredOutputMode,
  json_schema: structuredOutputMode,
  post_validation: z.boolean(),
  max_schema_bytes: z.number().int().nonnegative(),
}).strict();
export const WorkerCapabilitiesSchema = z.object({
  backend: z.enum(["llama", "mlx"]),
  streaming: z.boolean(),
  scheduling: z.object({
    fused_multi_sequence_batching: z.boolean(),
    interleaved: z.boolean(),
    priority_aware: z.boolean(),
  }).strict(),
}).strict();
export const CacheCapabilitiesSchema = z.object({
  partial_suffix_trim: z.boolean(),
  partial_prefix_branch: z.boolean(),
  whole_state_copy: z.boolean(),
  prompt_boundary_snapshots: z.boolean(),
  quantized_kv: z.boolean(),
}).strict();
export const GenerationCapabilitiesSchema = z.object({
  structured_output: StructuredOutputCapabilitiesSchema,
  tool_templates: z.boolean(),
}).strict();
export const ModalCapabilitiesSchema = z.object({
  input: z.tuple([z.literal("text")]),
  output: z.tuple([z.literal("text")]),
}).strict();
export const EffectiveModelCapabilitiesSchema = z.object({
  cache: CacheCapabilitiesSchema,
  generation: GenerationCapabilitiesSchema,
  modalities: ModalCapabilitiesSchema,
}).strict();
export const WorkerTokenCapabilitiesSchema = z.object({
  model_context_window: z.number().int().positive().nullable(),
  effective_context_window: z.number().int().positive(),
  max_input_tokens: z.number().int().nonnegative(),
  max_output_tokens: z.number().int().positive().nullable(),
  backend_allocation_cap: z.number().int().positive(),
  user_configured_override: z.number().int().positive().nullable(),
  model_context_window_source: z.string().nullable().optional(),
  max_output_tokens_source: z.string().nullable().optional(),
}).strict();

const resultSchemas = [
  extensibleObject({ kind: z.literal("loaded"),
    effective_model_capabilities: EffectiveModelCapabilitiesSchema,
    token_capabilities: WorkerTokenCapabilitiesSchema }),
  extensibleObject({ kind: z.literal("generated"), content: z.string() }),
  extensibleObject({ kind: z.literal("cancelled") }),
  extensibleObject({ kind: z.literal("max_active_updated"), max_active: z.number().int().positive() }),
  extensibleObject({ kind: z.literal("unloaded") }),
  extensibleObject({ kind: z.literal("shutdown") }),
] as const;
export const CompletedResultSchema = z.discriminatedUnion("kind", resultSchemas);

export const ProtocolErrorSchema = extensibleObject({
  code: z.string().min(1), message: z.string().min(1), retryable: z.boolean(), fatal: z.boolean(),
  details: z.unknown().optional(),
});

const scopedBase = { protocol, request_id: requestId, sequence };
const unsolicitedBase = { protocol, request_id: z.never().optional(), sequence: z.never().optional() };

export const ReadyEventSchema = extensibleObject({
  ...unsolicitedBase,
  type: z.literal("ready"),
  worker_capabilities: WorkerCapabilitiesSchema,
  model_capabilities: z.null(),
});
export const AcceptedEventSchema = extensibleObject({ ...scopedBase, type: z.literal("accepted") });
export const StartedEventSchema = extensibleObject({ ...scopedBase, type: z.literal("started") });
export const TokenEventSchema = extensibleObject({ ...scopedBase, type: z.literal("token"), text: z.string() });
export const ContentEventSchema = extensibleObject({ ...scopedBase, type: z.literal("content"), content });
export const PrefillProgressEventSchema = extensibleObject({
  ...scopedBase, type: z.literal("prefill_progress"), completed: z.number().int().nonnegative(), total: z.number().int().nonnegative(),
});
export const CompletedEventSchema = extensibleObject({ ...scopedBase, type: z.literal("completed"), result: CompletedResultSchema });
export const FailedEventSchema = extensibleObject({ ...scopedBase, type: z.literal("failed"), error: ProtocolErrorSchema });
export const WorkerTelemetrySchema = extensibleObject({
  memory: WorkerMemoryTelemetrySchema.optional(),
  retention: WorkerRetentionTelemetrySchema.optional(),
});
export const TelemetryEventSchema = extensibleObject({
  ...unsolicitedBase, type: z.literal("telemetry"), telemetry: WorkerTelemetrySchema,
});
export const DiagnosticEventSchema = extensibleObject({
  ...unsolicitedBase, type: z.literal("diagnostic"), level: z.enum(["debug", "info", "warning", "error"]), message: z.string().min(1),
});

export const WorkerEventSchema = z.discriminatedUnion("type", [
  ReadyEventSchema, AcceptedEventSchema, StartedEventSchema, TokenEventSchema, ContentEventSchema,
  PrefillProgressEventSchema, CompletedEventSchema, FailedEventSchema, TelemetryEventSchema, DiagnosticEventSchema,
]);
