import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  COMPLETED_RESULT_KINDS,
  ProtocolValidationError,
  WORKER_EVENT_TYPES,
  WORKER_PROTOCOL_VERSION,
  WORKER_REQUEST_TYPES,
  decodeWorkerEvent,
  decodeWorkerRequest,
  encodeWorkerEvent,
  encodeWorkerRequest,
} from "./index";

const fixtureRoot = join(import.meta.dir, "../fixtures/v1");

async function fixtureLines(kind: "requests" | "events") {
  const text = await readFile(join(fixtureRoot, kind, "all.jsonl"), "utf8");
  return text.trim().split("\n");
}

describe("worker protocol v1 fixtures", () => {
  test("parses all six request fixtures", async () => {
    const requests = (await fixtureLines("requests")).map(decodeWorkerRequest);
    expect(requests).toHaveLength(6);
    expect(requests.map(({ type }) => type)).toEqual([...WORKER_REQUEST_TYPES]);
    expect(requests.every(({ protocol }) => protocol === WORKER_PROTOCOL_VERSION)).toBe(true);
  });

  test("parses every event and completed result fixture", async () => {
    const events = (await fixtureLines("events")).map(decodeWorkerEvent);
    expect(new Set(events.map(({ type }) => type))).toEqual(new Set(WORKER_EVENT_TYPES));
    expect(events.filter((event) => event.type === "completed").map((event) => event.result.kind))
      .toEqual([...COMPLETED_RESULT_KINDS]);
  });
});

describe("worker protocol validation", () => {
  test("requires protocol exactly 1 and nonempty request IDs", () => {
    expect(() => decodeWorkerRequest({ protocol: 2, type: "shutdown", request_id: "req" })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ protocol: 1, type: "shutdown", request_id: "" })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ protocol: 1, type: "shutdown" })).toThrow(ProtocolValidationError);
  });

  test("requires nonnegative integer sequences on every scoped event", () => {
    for (const event of [
      { type: "accepted" }, { type: "started" }, { type: "token", text: "x" },
      { type: "content", content: "x" }, { type: "prefill_progress", completed: 1, total: 1 },
      { type: "completed", result: { kind: "shutdown" } },
      { type: "failed", error: { code: "x", message: "x", retryable: false, fatal: false } },
    ]) {
      expect(() => decodeWorkerEvent({ protocol: 1, request_id: "req", sequence: -1, ...event })).toThrow(ProtocolValidationError);
      expect(() => decodeWorkerEvent({ protocol: 1, request_id: "req", ...event })).toThrow(ProtocolValidationError);
    }
  });

  test("requires event-specific scoped payloads", () => {
    expect(() => decodeWorkerEvent({ protocol: 1, type: "content", request_id: "req", sequence: 0 })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerEvent({ protocol: 1, type: "token", request_id: "req", sequence: 0 })).toThrow(ProtocolValidationError);
  });

  test("forbids request scope on unsolicited events", () => {
    const unsolicited = [
      { type: "ready", worker_capabilities: {}, model_capabilities: {} },
      { type: "telemetry", telemetry: {} },
      { type: "diagnostic", level: "info", message: "hello" },
    ];
    for (const event of unsolicited) {
      expect(() => decodeWorkerEvent({ protocol: 1, request_id: "req", ...event })).toThrow(ProtocolValidationError);
      expect(() => decodeWorkerEvent({ protocol: 1, sequence: 0, ...event })).toThrow(ProtocolValidationError);
    }
  });

  test("rejects unknown discriminators but preserves unknown optional fields", () => {
    expect(() => decodeWorkerRequest({ protocol: 1, type: "future", request_id: "req" })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerEvent({ protocol: 1, type: "future" })).toThrow(ProtocolValidationError);
    expect(decodeWorkerRequest({ protocol: 1, type: "shutdown", request_id: "req", extension: true })).toHaveProperty("extension", true);
    expect(decodeWorkerEvent({ protocol: 1, type: "telemetry", telemetry: {}, extension: true })).toHaveProperty("extension", true);
  });

  test("rejects unknown completed result kinds and unstructured failures", () => {
    const base = { protocol: 1, type: "completed", request_id: "req", sequence: 0 };
    expect(() => decodeWorkerEvent({ ...base, result: { kind: "future" } })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerEvent({ protocol: 1, type: "failed", request_id: "req", sequence: 0, error: "failed" })).toThrow(ProtocolValidationError);
  });

  test("provides serializable structured errors and validated codecs", () => {
    try {
      decodeWorkerEvent("not json");
      throw new Error("expected validation failure");
    } catch (error) {
      expect(error).toBeInstanceOf(ProtocolValidationError);
      expect((error as ProtocolValidationError).toJSON()).toMatchObject({ code: "invalid_json", kind: "event" });
    }
    const request = { protocol: 1, type: "shutdown", request_id: "req" } as const;
    expect(decodeWorkerRequest(encodeWorkerRequest(request))).toEqual(request);
    const event = { protocol: 1, type: "telemetry", telemetry: { active: 1 } } as const;
    expect(decodeWorkerEvent(encodeWorkerEvent(event))).toEqual(event);
  });
});
