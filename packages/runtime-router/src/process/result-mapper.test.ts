import { describe, expect, test } from "bun:test";
import { mapWorkerResultPayload, mapWorkerTelemetryPayload } from "./result-mapper";

describe("worker result mapper", () => {
  test("maps terminal v1 results without duplicating streamed content", () => {
    expect(mapWorkerResultPayload({ content: "fallback", finish_reason: "stop" }, "req_1", false)).toEqual({
      content: "fallback",
      finish_reason: "stop",
      id: "req_1",
      done: true,
    });
    expect(mapWorkerResultPayload({ content: "duplicate", finish_reason: "stop" }, "req_1", true)).toEqual({
      finish_reason: "stop",
      id: "req_1",
      done: true,
    });
  });

  test("normalizes bare retention telemetry without changing wrapped payloads", () => {
    expect(mapWorkerTelemetryPayload({ active: 1 })).toEqual({ retention: { active: 1 } });
    const wrapped = { memory: { active_bytes: 1 }, retention: { active: 1 } };
    expect(mapWorkerTelemetryPayload(wrapped)).toBe(wrapped);
  });
});
