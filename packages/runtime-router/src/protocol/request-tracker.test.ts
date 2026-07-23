import { describe, expect, test } from "bun:test";
import { WorkerProtocolFault } from "./errors";
import { V1RequestTracker } from "./request-tracker";

const json = (value: unknown) => JSON.stringify(value);
const ready = {
  protocol: 1, type: "ready", worker_capabilities: { backend: "llama", streaming: true,
    scheduling: { fused_multi_sequence_batching: true, interleaved: true, priority_aware: true } }, model_capabilities: null,
} as const;
const scoped = (type: string, requestId: string, sequence: number, fields: Record<string, unknown> = {}) =>
  json({ protocol: 1, type, request_id: requestId, sequence, ...fields });

function tracker(maxTombstones = 1024) {
  const result = new V1RequestTracker(undefined, maxTombstones);
  expect(result.consumeLine(json(ready))).toEqual({
    kind: "ready", workerCapabilities: ready.worker_capabilities, modelCapabilities: null,
  });
  return result;
}

function fault(run: () => unknown): WorkerProtocolFault {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(WorkerProtocolFault);
    return error as WorkerProtocolFault;
  }
  throw new Error("expected WorkerProtocolFault");
}

function acceptAndStart(subject: V1RequestTracker, requestId: string) {
  subject.register(requestId);
  subject.consumeLine(scoped("accepted", requestId, 0));
  subject.consumeLine(scoped("started", requestId, 1));
}

describe("V1RequestTracker", () => {
  test("tracks concurrent IDs and maps resident-compatible facts", () => {
    const subject = tracker();
    subject.register("a");
    subject.register("b");
    expect(subject.consumeLine(scoped("accepted", "a", 0))).toEqual({ kind: "accepted", requestId: "a" });
    expect(subject.consumeLine(scoped("accepted", "b", 0))).toEqual({ kind: "accepted", requestId: "b" });
    expect(subject.consumeLine(scoped("started", "b", 1))).toEqual({ kind: "started", requestId: "b" });
    expect(subject.consumeLine(scoped("started", "a", 1))).toEqual({ kind: "started", requestId: "a" });
    expect(subject.consumeLine(scoped("token", "a", 2, { text: "A" }))).toEqual({ kind: "token", requestId: "a", text: "A" });
    expect(subject.consumeLine(scoped("content", "b", 2, { content: { text: "B" } })))
      .toEqual({ kind: "content", requestId: "b", content: { text: "B" } });
    expect(subject.consumeLine(scoped("prefill_progress", "a", 3, { completed: 2, total: 4 })))
      .toEqual({ kind: "prefill_progress", requestId: "a", done: 2, total: 4 });
    expect(subject.consumeLine(scoped("completed", "b", 3, { result: { kind: "generated", content: "B" } })))
      .toMatchObject({ kind: "completed", requestId: "b", result: { kind: "generated", content: "B" } });
    expect(subject.state("a")).toBe("started");
    expect(subject.state("b")).toBe("terminal");
  });

  test("does not advance request sequences for unsolicited telemetry", () => {
    const subject = tracker();
    subject.register("a");
    subject.consumeLine(scoped("accepted", "a", 0));
    expect(subject.consumeLine(json({ protocol: 1, type: "telemetry", telemetry: { active: 1 } })))
      .toEqual({ kind: "telemetry", telemetry: { active: 1 } });
    expect(subject.consumeLine(json({ protocol: 1, type: "diagnostic", level: "warning", message: "busy" })))
      .toEqual({ kind: "diagnostic", level: "warning", message: "busy" });
    expect(subject.consumeLine(scoped("started", "a", 1))).toEqual({ kind: "started", requestId: "a" });
  });

  test("rejects malformed known-ID payloads as request faults", () => {
    const subject = tracker();
    acceptAndStart(subject, "known");
    expect(fault(() => subject.consumeLine(scoped("token", "known", 2))))
      .toMatchObject({ code: "malformed_event", scope: "request", requestId: "known" });
  });

  test("classifies unknown and missing IDs as worker faults", () => {
    const subject = tracker();
    expect(fault(() => subject.consumeLine(scoped("accepted", "unknown", 0))))
      .toMatchObject({ code: "unknown_request_id", scope: "worker", requestId: undefined });
    expect(fault(() => subject.consumeLine(json({ protocol: 1, type: "accepted", sequence: 0 }))))
      .toMatchObject({ code: "malformed_event", scope: "worker", requestId: undefined });
  });

  test("rejects missing, wrong, gapped, and regressed sequences", () => {
    const missing = tracker();
    missing.register("a");
    expect(fault(() => missing.consumeLine(json({ protocol: 1, type: "accepted", request_id: "a" }))))
      .toMatchObject({ code: "malformed_event", scope: "request", requestId: "a" });

    for (const sequence of [-1, 1, 9]) {
      const subject = tracker();
      subject.register("a");
      expect(fault(() => subject.consumeLine(scoped("accepted", "a", sequence))))
        .toMatchObject({ code: sequence < 0 ? "malformed_event" : "sequence_violation", scope: "request", requestId: "a" });
    }

    const regressed = tracker();
    regressed.register("a");
    regressed.consumeLine(scoped("accepted", "a", 0));
    expect(fault(() => regressed.consumeLine(scoped("started", "a", 0))))
      .toMatchObject({ code: "sequence_violation", scope: "request", requestId: "a" });
  });

  test("enforces accepted, started, and terminal state progression", () => {
    const subject = tracker();
    subject.register("a");
    expect(fault(() => subject.consumeLine(scoped("started", "a", 0))))
      .toMatchObject({ code: "state_violation", scope: "request" });

    const duplicateAccepted = tracker();
    duplicateAccepted.register("a");
    duplicateAccepted.consumeLine(scoped("accepted", "a", 0));
    expect(fault(() => duplicateAccepted.consumeLine(scoped("accepted", "a", 1))))
      .toMatchObject({ code: "state_violation", scope: "request" });

    const terminal = tracker();
    acceptAndStart(terminal, "a");
    terminal.consumeLine(scoped("completed", "a", 2, { result: { kind: "generated", content: "done" } }));
    expect(fault(() => terminal.consumeLine(scoped("completed", "a", 3, { result: { kind: "generated", content: "again" } }))))
      .toMatchObject({ code: "state_violation", scope: "request", requestId: "a" });
    expect(fault(() => terminal.consumeLine(scoped("token", "a", 3, { text: "late" }))))
      .toMatchObject({ code: "state_violation", scope: "request", requestId: "a" });
  });

  test("maps structured failures and terminal events after acceptance", () => {
    const subject = tracker();
    subject.register("a");
    subject.consumeLine(scoped("accepted", "a", 0));
    expect(subject.consumeLine(scoped("failed", "a", 1, {
      error: { code: "overloaded", message: "busy", retryable: true, fatal: false },
    }))).toMatchObject({ kind: "failed", requestId: "a", error: { code: "overloaded", retryable: true, fatal: false } });
    expect(subject.state("a")).toBe("terminal");
  });

  test("bounds terminal tombstones", () => {
    const subject = tracker(1);
    for (const requestId of ["a", "b"]) {
      acceptAndStart(subject, requestId);
      subject.consumeLine(scoped("completed", requestId, 2, { result: { kind: "generated", content: requestId } }));
    }
    expect(subject.state("a")).toBeUndefined();
    expect(subject.state("b")).toBe("terminal");
    expect(fault(() => subject.consumeLine(scoped("token", "a", 3, { text: "late" }))))
      .toMatchObject({ code: "unknown_request_id", scope: "worker" });
  });
});
