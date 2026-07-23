export const WORKER_PROTOCOL_VERSION = 1 as const;

export const WORKER_REQUEST_TYPES = [
  "load", "generate", "cancel", "set_max_active", "unload", "shutdown",
] as const;

export const WORKER_EVENT_TYPES = [
  "ready", "accepted", "started", "token", "content", "prefill_progress",
  "completed", "failed", "telemetry", "diagnostic",
] as const;

export const COMPLETED_RESULT_KINDS = [
  "loaded", "generated", "cancelled", "max_active_updated", "unloaded", "shutdown",
] as const;

export const STRUCTURED_OUTPUT_KINDS = ["json_object", "json_schema"] as const;
export const STRUCTURED_OUTPUT_STRENGTHS = ["best_effort", "required"] as const;
export const STRUCTURED_OUTPUT_MODES = ["native", "post_validate", "unsupported"] as const;

export type WorkerProtocolVersion = typeof WORKER_PROTOCOL_VERSION;
export type WorkerRequestType = typeof WORKER_REQUEST_TYPES[number];
export type WorkerEventType = typeof WORKER_EVENT_TYPES[number];
export type CompletedResultKind = typeof COMPLETED_RESULT_KINDS[number];
export type StructuredOutputKind = typeof STRUCTURED_OUTPUT_KINDS[number];
export type StructuredOutputStrength = typeof STRUCTURED_OUTPUT_STRENGTHS[number];
export type StructuredOutputMode = typeof STRUCTURED_OUTPUT_MODES[number];

export type MemorySource = "measured" | "estimated" | "unavailable";
export type MeasuredMemoryBasis = "resident_rss" | "runtime_allocator" | "worker_allocator" | "os_available";
export type EstimatedMemoryBasis =
  | "prior_observation"
  | "model_artifacts"
  | "architecture_metadata"
  | "configured_cache"
  | "cache_components"
  | "context_configuration"
  | "conservative_fallback";
export type UnavailableMemoryBasis = "not_observed" | "not_supported" | "not_reported";
export type MemoryBasis = MeasuredMemoryBasis | EstimatedMemoryBasis | UnavailableMemoryBasis;
export type MemoryValue =
  | Readonly<{ bytes: number; source: "measured"; basis: MeasuredMemoryBasis }>
  | Readonly<{ bytes: number; source: "estimated"; basis: EstimatedMemoryBasis }>
  | Readonly<{ bytes: null; source: "unavailable"; basis: UnavailableMemoryBasis }>;

export type WorkerMemoryTelemetry = {
  active_bytes: number | null;
  active_bytes_source?: MemorySource;
  active_bytes_basis?: MemoryBasis;
  cache_bytes: number | null;
  cache_bytes_source?: MemorySource;
  cache_bytes_basis?: MemoryBasis;
  peak_active_bytes: number | null;
  peak_active_bytes_source?: MemorySource;
  peak_active_bytes_basis?: MemoryBasis;
  [key: string]: unknown;
};

export type WorkerRetentionTelemetry = {
  retained_bytes?: number | null;
  retained_bytes_source?: MemorySource;
  retained_bytes_basis?: MemoryBasis;
  session_bytes?: number | null;
  session_bytes_source?: MemorySource;
  session_bytes_basis?: MemoryBasis;
  anchor_bytes?: number | null;
  anchor_bytes_source?: MemorySource;
  anchor_bytes_basis?: MemoryBasis;
  evicted_bytes?: number | null;
  evicted_bytes_source?: MemorySource;
  evicted_bytes_basis?: MemoryBasis;
  estimated_retained_bytes?: number | null;
  estimated_retained_bytes_source?: MemorySource;
  estimated_retained_bytes_basis?: MemoryBasis;
  [key: string]: unknown;
};

export type ProtocolError = {
  code: string;
  message: string;
  retryable: boolean;
  fatal: boolean;
  details?: unknown;
  [key: string]: unknown;
};

export type LoadRequest = { protocol: 1; type: "load"; request_id: string; model: string; [key: string]: unknown };
export type CacheIdentity = {
  version: 1;
  generation: string;
  tenant_root: string;
  project_fingerprint?: string;
  harness_fingerprint?: string;
  agent_fingerprint?: string;
  session_fingerprint?: string;
  scope: "tenant" | "project" | "harness" | "agent" | "session";
  scope_fingerprint: string;
  namespace_fingerprint: string;
  namespace_id: string;
  priority: "interactive" | "normal" | "background";
  side_request: boolean;
  display: { namespace?: string; project?: string; harness?: string; agent?: string; session?: string };
  physical: {
    fingerprint: string;
    backend: "llama" | "mlx";
    resolved_revision: string;
    model_artifact_fingerprint: string;
    tokenizer_fingerprint: string;
    context_allocation: number;
    kv_format: string;
    unified_kv: boolean;
    layout_version: number;
  };
};
export type StructuredOutputContract =
  | { kind: "json_object"; strength: StructuredOutputStrength; schema?: never }
  | { kind: "json_schema"; strength: StructuredOutputStrength; schema: Record<string, unknown> };
export type GenerateRequest = { protocol: 1; type: "generate"; request_id: string; prompt: string; cache_identity: CacheIdentity; structured_output?: StructuredOutputContract; [key: string]: unknown };
export type CancelRequest = { protocol: 1; type: "cancel"; request_id: string; target_request_id: string; [key: string]: unknown };
export type SetMaxActiveRequest = { protocol: 1; type: "set_max_active"; request_id: string; max_active: number; [key: string]: unknown };
export type UnloadRequest = { protocol: 1; type: "unload"; request_id: string; [key: string]: unknown };
export type ShutdownRequest = { protocol: 1; type: "shutdown"; request_id: string; [key: string]: unknown };
export type WorkerRequest = LoadRequest | GenerateRequest | CancelRequest | SetMaxActiveRequest | UnloadRequest | ShutdownRequest;

export type GeneratedResult = { kind: "generated"; content: string; [key: string]: unknown };
export type CancelledResult = { kind: "cancelled"; [key: string]: unknown };
export type MaxActiveUpdatedResult = { kind: "max_active_updated"; max_active: number; [key: string]: unknown };
export type UnloadedResult = { kind: "unloaded"; [key: string]: unknown };
export type ShutdownResult = { kind: "shutdown"; [key: string]: unknown };
export type CompletedResult = LoadedResult | GeneratedResult | CancelledResult | MaxActiveUpdatedResult | UnloadedResult | ShutdownResult;

export type StructuredOutputCapabilities = {
  json_object: StructuredOutputMode;
  json_schema: StructuredOutputMode;
  post_validation: boolean;
  max_schema_bytes: number;
};
export type WorkerCapabilities = {
  backend: "llama" | "mlx";
  streaming: boolean;
  scheduling: {
    fused_multi_sequence_batching: boolean;
    interleaved: boolean;
    priority_aware: boolean;
  };
};
export type CacheCapabilities = {
  partial_suffix_trim: boolean;
  partial_prefix_branch: boolean;
  whole_state_copy: boolean;
  prompt_boundary_snapshots: boolean;
  quantized_kv: boolean;
};
export type GenerationCapabilities = {
  structured_output: StructuredOutputCapabilities;
  tool_templates: boolean;
};
export type ModalCapabilities = {
  input: ["text"];
  output: ["text"];
};
export type EffectiveModelCapabilities = {
  cache: CacheCapabilities;
  generation: GenerationCapabilities;
  modalities: ModalCapabilities;
};
export type WorkerTokenCapabilities = {
  model_context_window: number | null;
  effective_context_window: number;
  max_input_tokens: number;
  max_output_tokens: number | null;
  backend_allocation_cap: number;
  user_configured_override: number | null;
  model_context_window_source?: string | null;
  max_output_tokens_source?: string | null;
};
export type LoadedResult = {
  kind: "loaded";
  effective_model_capabilities: EffectiveModelCapabilities;
  token_capabilities: WorkerTokenCapabilities;
  [key: string]: unknown;
};
export type ReadyEvent = { protocol: 1; type: "ready"; worker_capabilities: WorkerCapabilities; model_capabilities: null; [key: string]: unknown };
export type ScopedEventBase = { protocol: 1; request_id: string; sequence: number; [key: string]: unknown };
export type AcceptedEvent = ScopedEventBase & { type: "accepted" };
export type StartedEvent = ScopedEventBase & { type: "started" };
export type TokenEvent = ScopedEventBase & { type: "token"; text: string };
export type ContentEvent = ScopedEventBase & { type: "content"; content: unknown };
export type PrefillProgressEvent = ScopedEventBase & { type: "prefill_progress"; completed: number; total: number };
export type CompletedEvent = ScopedEventBase & { type: "completed"; result: CompletedResult };
export type FailedEvent = ScopedEventBase & { type: "failed"; error: ProtocolError };
export type WorkerTelemetry = {
  memory?: WorkerMemoryTelemetry;
  retention?: WorkerRetentionTelemetry;
  [key: string]: unknown;
};
export type TelemetryEvent = { protocol: 1; type: "telemetry"; telemetry: WorkerTelemetry; [key: string]: unknown };
export type DiagnosticEvent = { protocol: 1; type: "diagnostic"; level: "debug" | "info" | "warning" | "error"; message: string; [key: string]: unknown };
export type WorkerEvent = ReadyEvent | AcceptedEvent | StartedEvent | TokenEvent | ContentEvent | PrefillProgressEvent | CompletedEvent | FailedEvent | TelemetryEvent | DiagnosticEvent;
