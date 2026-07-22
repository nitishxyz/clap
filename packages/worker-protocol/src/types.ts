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

export type WorkerProtocolVersion = typeof WORKER_PROTOCOL_VERSION;
export type WorkerRequestType = typeof WORKER_REQUEST_TYPES[number];
export type WorkerEventType = typeof WORKER_EVENT_TYPES[number];
export type CompletedResultKind = typeof COMPLETED_RESULT_KINDS[number];

export type ProtocolError = {
  code: string;
  message: string;
  retryable: boolean;
  fatal: boolean;
  details?: unknown;
  [key: string]: unknown;
};

export type LoadRequest = { protocol: 1; type: "load"; request_id: string; model: string; [key: string]: unknown };
export type GenerateRequest = { protocol: 1; type: "generate"; request_id: string; prompt: string; [key: string]: unknown };
export type CancelRequest = { protocol: 1; type: "cancel"; request_id: string; target_request_id: string; [key: string]: unknown };
export type SetMaxActiveRequest = { protocol: 1; type: "set_max_active"; request_id: string; max_active: number; [key: string]: unknown };
export type UnloadRequest = { protocol: 1; type: "unload"; request_id: string; [key: string]: unknown };
export type ShutdownRequest = { protocol: 1; type: "shutdown"; request_id: string; [key: string]: unknown };
export type WorkerRequest = LoadRequest | GenerateRequest | CancelRequest | SetMaxActiveRequest | UnloadRequest | ShutdownRequest;

export type LoadedResult = { kind: "loaded"; [key: string]: unknown };
export type GeneratedResult = { kind: "generated"; content: string; [key: string]: unknown };
export type CancelledResult = { kind: "cancelled"; [key: string]: unknown };
export type MaxActiveUpdatedResult = { kind: "max_active_updated"; max_active: number; [key: string]: unknown };
export type UnloadedResult = { kind: "unloaded"; [key: string]: unknown };
export type ShutdownResult = { kind: "shutdown"; [key: string]: unknown };
export type CompletedResult = LoadedResult | GeneratedResult | CancelledResult | MaxActiveUpdatedResult | UnloadedResult | ShutdownResult;

export type ReadyEvent = { protocol: 1; type: "ready"; worker_capabilities: Record<string, unknown>; model_capabilities: Record<string, unknown>; [key: string]: unknown };
export type ScopedEventBase = { protocol: 1; request_id: string; sequence: number; [key: string]: unknown };
export type AcceptedEvent = ScopedEventBase & { type: "accepted" };
export type StartedEvent = ScopedEventBase & { type: "started" };
export type TokenEvent = ScopedEventBase & { type: "token"; text: string };
export type ContentEvent = ScopedEventBase & { type: "content"; content: unknown };
export type PrefillProgressEvent = ScopedEventBase & { type: "prefill_progress"; completed: number; total: number };
export type CompletedEvent = ScopedEventBase & { type: "completed"; result: CompletedResult };
export type FailedEvent = ScopedEventBase & { type: "failed"; error: ProtocolError };
export type TelemetryEvent = { protocol: 1; type: "telemetry"; telemetry: Record<string, unknown>; [key: string]: unknown };
export type DiagnosticEvent = { protocol: 1; type: "diagnostic"; level: "debug" | "info" | "warning" | "error"; message: string; [key: string]: unknown };
export type WorkerEvent = ReadyEvent | AcceptedEvent | StartedEvent | TokenEvent | ContentEvent | PrefillProgressEvent | CompletedEvent | FailedEvent | TelemetryEvent | DiagnosticEvent;
