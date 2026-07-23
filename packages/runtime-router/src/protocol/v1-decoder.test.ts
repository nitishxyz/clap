import { describe, expect, test } from "bun:test";
import { WorkerProtocolFault } from "./errors";
import { MAX_WORKER_PROTOCOL_LINE_BYTES, V1WorkerProtocolDecoder } from "./v1-decoder";

const line = (value: unknown) => JSON.stringify(value);
const ready = {
  protocol: 1,
  type: "ready",
  worker_capabilities: { backend: "llama", streaming: true,
    scheduling: { fused_multi_sequence_batching: true, interleaved: true } },
  model_capabilities: null,
};

function fault(run: () => unknown): WorkerProtocolFault {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(WorkerProtocolFault);
    return error as WorkerProtocolFault;
  }
  throw new Error("expected WorkerProtocolFault");
}

describe("V1WorkerProtocolDecoder", () => {
  test("negotiates ready exactly once", () => {
    const decoder = new V1WorkerProtocolDecoder();
    expect(decoder.negotiated).toBe(false);
    expect(decoder.decode(line(ready))).toMatchObject({ type: "ready" });
    expect(decoder.negotiated).toBe(true);
    expect(fault(() => decoder.decode(line(ready)))).toMatchObject({ code: "duplicate_ready", scope: "worker" });
  });

  test("requires ready before request and unsolicited events", () => {
    const decoder = new V1WorkerProtocolDecoder();
    expect(fault(() => decoder.decode(line({ protocol: 1, type: "accepted", request_id: "a", sequence: 0 }))))
      .toMatchObject({ code: "ready_required", scope: "request", requestId: "a" });
    expect(fault(() => decoder.decode(line({ protocol: 1, type: "telemetry", telemetry: {} }))))
      .toMatchObject({ code: "ready_required", scope: "worker" });
  });

  test("rejects missing and mismatched protocol versions", () => {
    for (const protocol of [undefined, 0, 2, "1"]) {
      const event = { ...ready, protocol };
      expect(fault(() => new V1WorkerProtocolDecoder().decode(line(event))))
        .toMatchObject({ code: "version_mismatch", scope: "worker" });
    }
  });

  test("bounds UTF-8 encoded lines to 8 MiB by default", () => {
    expect(MAX_WORKER_PROTOCOL_LINE_BYTES).toBe(8 * 1024 * 1024);
    const decoder = new V1WorkerProtocolDecoder(16);
    const error = fault(() => decoder.decode("😀😀😀😀😀"));
    expect(error).toMatchObject({ code: "line_too_large", scope: "worker" });
  });

  test("classifies malformed stdout and unknown event types", () => {
    expect(fault(() => new V1WorkerProtocolDecoder().decode("native log output")))
      .toMatchObject({ code: "malformed_stdout", scope: "worker" });
    expect(fault(() => new V1WorkerProtocolDecoder().decode("[]")))
      .toMatchObject({ code: "malformed_stdout", scope: "worker" });
    const decoder = new V1WorkerProtocolDecoder();
    decoder.decode(line(ready));
    expect(fault(() => decoder.decode(line({ protocol: 1, type: "future" }))))
      .toMatchObject({ code: "unknown_event_type", scope: "worker" });
    expect(fault(() => decoder.decode(line({ protocol: 1, type: "future", request_id: "a" }))))
      .toMatchObject({ code: "unknown_event_type", scope: "request", requestId: "a" });
  });

  test("classifies unsolicited scope as a worker fault", () => {
    for (const event of [
      { ...ready, request_id: "a" },
      { protocol: 1, type: "telemetry", telemetry: {}, sequence: 0 },
      { protocol: 1, type: "diagnostic", level: "info", message: "x", request_id: "a" },
    ]) {
      const decoder = new V1WorkerProtocolDecoder();
      if (event.type !== "ready") decoder.decode(line(ready));
      expect(fault(() => decoder.decode(line(event)))).toMatchObject({ code: "scope_violation", scope: "worker" });
    }
  });
});
