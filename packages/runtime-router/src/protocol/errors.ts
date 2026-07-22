export type ProtocolFaultCode =
  | "line_too_large"
  | "malformed_stdout"
  | "version_mismatch"
  | "handshake_timeout"
  | "ready_required"
  | "duplicate_ready"
  | "unknown_event_type"
  | "malformed_event"
  | "scope_violation"
  | "unknown_request_id"
  | "sequence_violation"
  | "state_violation";

export type ProtocolFaultScope = "request" | "worker";

export class WorkerProtocolFault extends Error {
  readonly name = "WorkerProtocolFault";

  constructor(
    readonly code: ProtocolFaultCode,
    message: string,
    readonly scope: ProtocolFaultScope,
    readonly requestId?: string,
    readonly cause?: unknown,
  ) {
    super(message);
  }

  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      scope: this.scope,
      request_id: this.requestId,
    };
  }
}

export function protocolFault(
  code: ProtocolFaultCode,
  message: string,
  requestId?: string,
  cause?: unknown,
): WorkerProtocolFault {
  return new WorkerProtocolFault(code, message, requestId ? "request" : "worker", requestId, cause);
}
