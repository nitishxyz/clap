import type { StructuredOutputCapabilities, WorkerEvent } from "@clap/worker-protocol";
import { V1WorkerProtocolDecoder } from "./v1-decoder";
import { WorkerProtocolFault, protocolFault } from "./errors";

export type TrackedRequestState = "registered" | "accepted" | "started";

export type ResidentProtocolFact =
  | { kind: "ready"; workerCapabilities: Record<string, unknown>; modelCapabilities: Record<string, unknown>;
      structuredOutputCapabilities?: StructuredOutputCapabilities }
  | { kind: "accepted"; requestId: string }
  | { kind: "started"; requestId: string }
  | { kind: "token"; requestId: string; text: string }
  | { kind: "content"; requestId: string; content: unknown }
  | { kind: "prefill_progress"; requestId: string; done: number; total: number }
  | { kind: "completed"; requestId: string; result: Extract<WorkerEvent, { type: "completed" }>["result"] }
  | { kind: "failed"; requestId: string; error: Extract<WorkerEvent, { type: "failed" }>["error"] }
  | { kind: "telemetry"; telemetry: Record<string, unknown> }
  | { kind: "diagnostic"; level: "debug" | "info" | "warning" | "error"; message: string };

type ActiveRequest = { state: TrackedRequestState; nextSequence: number };

export class V1RequestTracker {
  private readonly active = new Map<string, ActiveRequest>();
  private readonly tombstones = new Map<string, true>();

  constructor(
    readonly decoder = new V1WorkerProtocolDecoder(),
    readonly maxTombstones = 1024,
  ) {
    if (!Number.isSafeInteger(maxTombstones) || maxTombstones < 0) throw new RangeError("maxTombstones must be a nonnegative integer");
  }

  register(requestId: string): void {
    if (!requestId) throw new TypeError("requestId must not be empty");
    if (this.active.has(requestId) || this.tombstones.has(requestId)) {
      throw protocolFault("state_violation", `Request ID ${requestId} is already tracked`, requestId);
    }
    this.active.set(requestId, { state: "registered", nextSequence: 0 });
  }

  consumeLine(line: string): ResidentProtocolFact {
    try {
      return this.consume(this.decoder.decode(line));
    } catch (error) {
      if (!(error instanceof WorkerProtocolFault) || !error.requestId) throw error;
      if (this.active.has(error.requestId) || this.tombstones.has(error.requestId)) throw error;
      throw new WorkerProtocolFault("unknown_request_id", `Unknown request ID ${error.requestId}`, "worker", undefined, error);
    }
  }

  consume(event: WorkerEvent): ResidentProtocolFact {
    if (event.type === "ready") {
      return {
        kind: "ready",
        workerCapabilities: event.worker_capabilities,
        modelCapabilities: event.model_capabilities,
        ...(event.structured_output ? { structuredOutputCapabilities: event.structured_output } : {}),
      };
    }
    if (event.type === "telemetry") return { kind: "telemetry", telemetry: event.telemetry };
    if (event.type === "diagnostic") return { kind: "diagnostic", level: event.level, message: event.message };

    const requestId = event.request_id;
    const tracked = this.active.get(requestId);
    if (!tracked) {
      const detail = this.tombstones.has(requestId) ? "terminal request ID" : "unknown request ID";
      throw protocolFault(this.tombstones.has(requestId) ? "state_violation" : "unknown_request_id", `${event.type} references ${detail} ${requestId}`, this.tombstones.has(requestId) ? requestId : undefined);
    }
    if (event.sequence !== tracked.nextSequence) {
      throw protocolFault("sequence_violation", `Expected sequence ${tracked.nextSequence} for ${requestId}, received ${event.sequence}`, requestId);
    }

    if (event.type === "accepted") {
      if (tracked.state !== "registered") throw protocolFault("state_violation", `Duplicate or late accepted event for ${requestId}`, requestId);
      tracked.state = "accepted";
    } else if (event.type === "started") {
      if (tracked.state !== "accepted") throw protocolFault("state_violation", `Started event before accepted for ${requestId}`, requestId);
      tracked.state = "started";
    } else if (event.type === "completed" || event.type === "failed") {
      if (tracked.state !== "accepted" && tracked.state !== "started") {
        throw protocolFault("state_violation", `Terminal event before accepted for ${requestId}`, requestId);
      }
      this.finish(requestId);
    } else if (tracked.state !== "started") {
      throw protocolFault("state_violation", `${event.type} event before started for ${requestId}`, requestId);
    }
    if (this.active.has(requestId)) tracked.nextSequence += 1;

    switch (event.type) {
      case "accepted": return { kind: "accepted", requestId };
      case "started": return { kind: "started", requestId };
      case "token": return { kind: "token", requestId, text: event.text };
      case "content": return { kind: "content", requestId, content: event.content };
      case "prefill_progress": return { kind: "prefill_progress", requestId, done: event.completed, total: event.total };
      case "completed": return { kind: "completed", requestId, result: event.result };
      case "failed": return { kind: "failed", requestId, error: event.error };
    }
  }

  state(requestId: string): TrackedRequestState | "terminal" | undefined {
    return this.active.get(requestId)?.state ?? (this.tombstones.has(requestId) ? "terminal" : undefined);
  }

  private finish(requestId: string): void {
    this.active.delete(requestId);
    if (this.maxTombstones === 0) return;
    this.tombstones.set(requestId, true);
    while (this.tombstones.size > this.maxTombstones) {
      const oldest = this.tombstones.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.tombstones.delete(oldest);
    }
  }
}
