import {
  ProtocolValidationError,
  WORKER_EVENT_TYPES,
  WORKER_PROTOCOL_VERSION,
  decodeWorkerEvent,
  type WorkerEvent,
} from "@clap/worker-protocol";
import { WorkerProtocolFault, protocolFault } from "./errors";

export const MAX_WORKER_PROTOCOL_LINE_BYTES = 8 * 1024 * 1024;
const eventTypes = new Set<string>(WORKER_EVENT_TYPES);

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

export class V1WorkerProtocolDecoder {
  private ready = false;

  constructor(readonly maxLineBytes = MAX_WORKER_PROTOCOL_LINE_BYTES) {
    if (!Number.isSafeInteger(maxLineBytes) || maxLineBytes <= 0) throw new RangeError("maxLineBytes must be a positive integer");
  }

  get negotiated(): boolean {
    return this.ready;
  }

  decode(line: string): WorkerEvent {
    const bytes = Buffer.byteLength(line, "utf8");
    if (bytes > this.maxLineBytes) {
      throw protocolFault("line_too_large", `Worker protocol line is ${bytes} bytes; limit is ${this.maxLineBytes}`);
    }

    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (cause) {
      throw protocolFault("malformed_stdout", "Worker stdout line is not valid JSON", undefined, cause);
    }

    const object = record(value);
    if (!object) throw protocolFault("malformed_stdout", "Worker stdout JSON must be an object");
    const requestId = typeof object.request_id === "string" && object.request_id.length > 0
      ? object.request_id
      : undefined;
    if (object.protocol !== WORKER_PROTOCOL_VERSION) {
      throw protocolFault(
        "version_mismatch",
        `Unsupported worker protocol version ${String(object.protocol)}; expected ${WORKER_PROTOCOL_VERSION}`,
      );
    }
    if (typeof object.type !== "string" || !eventTypes.has(object.type)) {
      throw protocolFault("unknown_event_type", `Unknown worker event type ${String(object.type)}`, requestId);
    }
    if (!this.ready && object.type !== "ready") {
      throw protocolFault("ready_required", "Worker must send ready before other protocol events", requestId);
    }
    if (this.ready && object.type === "ready") {
      throw protocolFault("duplicate_ready", "Worker sent more than one ready event");
    }

    let event: WorkerEvent;
    try {
      event = decodeWorkerEvent(value);
    } catch (cause) {
      if (!(cause instanceof ProtocolValidationError)) throw cause;
      const unsolicited = object.type === "ready" || object.type === "telemetry" || object.type === "diagnostic";
      const hasScope = "request_id" in object || "sequence" in object;
      const code = unsolicited && hasScope ? "scope_violation" : "malformed_event";
      throw protocolFault(code, `Malformed ${object.type} worker event`, unsolicited ? undefined : requestId, cause);
    }

    if (event.type === "ready") this.ready = true;
    return event;
  }
}
