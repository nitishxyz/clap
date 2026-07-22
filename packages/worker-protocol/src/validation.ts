import { z, type ZodIssue, type ZodType } from "zod";
import { WorkerEventSchema, WorkerRequestSchema } from "./schemas";
import type { WorkerEvent, WorkerRequest } from "./types";

export type ProtocolMessageKind = "request" | "event";
export type ProtocolValidationErrorCode = "invalid_json" | "invalid_request" | "invalid_event";

export class ProtocolValidationError extends Error {
  readonly name = "ProtocolValidationError";
  constructor(
    readonly code: ProtocolValidationErrorCode,
    readonly kind: ProtocolMessageKind,
    readonly issues: readonly ZodIssue[],
    readonly input: unknown,
  ) {
    super(code === "invalid_json" ? `Invalid worker protocol ${kind} JSON` : `Invalid worker protocol ${kind}`);
  }

  toJSON() {
    return { name: this.name, code: this.code, kind: this.kind, message: this.message, issues: this.issues };
  }
}

function parseJson(input: string, kind: ProtocolMessageKind): unknown {
  try {
    return JSON.parse(input);
  } catch {
    throw new ProtocolValidationError("invalid_json", kind, [{ code: "custom", path: [], message: "Input is not valid JSON" }], input);
  }
}

function decode<T>(input: string | unknown, kind: ProtocolMessageKind, schema: ZodType<T>): T {
  const value = typeof input === "string" ? parseJson(input, kind) : input;
  const result = schema.safeParse(value);
  if (!result.success) throw new ProtocolValidationError(`invalid_${kind}`, kind, result.error.issues, value);
  return result.data;
}

export function decodeWorkerRequest(input: string | unknown): WorkerRequest {
  return decode(input, "request", WorkerRequestSchema) as WorkerRequest;
}

export function decodeWorkerEvent(input: string | unknown): WorkerEvent {
  return decode(input, "event", WorkerEventSchema) as WorkerEvent;
}

export function encodeWorkerRequest(input: WorkerRequest): string {
  return JSON.stringify(decodeWorkerRequest(input));
}

export function encodeWorkerEvent(input: WorkerEvent): string {
  return JSON.stringify(decodeWorkerEvent(input));
}

export const decodeRequest = decodeWorkerRequest;
export const decodeEvent = decodeWorkerEvent;
export const encodeRequest = encodeWorkerRequest;
export const encodeEvent = encodeWorkerEvent;
