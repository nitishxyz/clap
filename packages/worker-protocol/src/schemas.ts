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
export const LoadRequestSchema = extensibleObject({ ...requestBase, type: z.literal("load"), model: z.string().min(1) });
export const GenerateRequestSchema = extensibleObject({ ...requestBase, type: z.literal("generate"), prompt: z.string() });
export const CancelRequestSchema = extensibleObject({ ...requestBase, type: z.literal("cancel"), target_request_id: requestId });
export const SetMaxActiveRequestSchema = extensibleObject({ ...requestBase, type: z.literal("set_max_active"), max_active: z.number().int().positive() });
export const UnloadRequestSchema = extensibleObject({ ...requestBase, type: z.literal("unload") });
export const ShutdownRequestSchema = extensibleObject({ ...requestBase, type: z.literal("shutdown") });

export const WorkerRequestSchema = z.discriminatedUnion("type", [
  LoadRequestSchema, GenerateRequestSchema, CancelRequestSchema, SetMaxActiveRequestSchema,
  UnloadRequestSchema, ShutdownRequestSchema,
]);

const resultSchemas = [
  extensibleObject({ kind: z.literal("loaded") }),
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
  worker_capabilities: z.record(z.unknown()),
  model_capabilities: z.record(z.unknown()),
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
export const TelemetryEventSchema = extensibleObject({ ...unsolicitedBase, type: z.literal("telemetry"), telemetry: z.record(z.unknown()) });
export const DiagnosticEventSchema = extensibleObject({
  ...unsolicitedBase, type: z.literal("diagnostic"), level: z.enum(["debug", "info", "warning", "error"]), message: z.string().min(1),
});

export const WorkerEventSchema = z.discriminatedUnion("type", [
  ReadyEventSchema, AcceptedEventSchema, StartedEventSchema, TokenEventSchema, ContentEventSchema,
  PrefillProgressEventSchema, CompletedEventSchema, FailedEventSchema, TelemetryEventSchema, DiagnosticEventSchema,
]);
