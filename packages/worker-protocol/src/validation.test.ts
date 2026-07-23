import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  COMPLETED_RESULT_KINDS,
  ProtocolValidationError,
  WORKER_EVENT_TYPES,
  WORKER_PROTOCOL_VERSION,
  WORKER_REQUEST_TYPES,
  type CacheIdentity,
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
  test("parses the canonical cross-backend cache identity vector", async () => {
    const fixture = JSON.parse(await readFile(join(fixtureRoot, "cache-identity-vector.json"), "utf8"));
    const decoded = decodeWorkerRequest({
      protocol: 1, type: "generate", request_id: "shared_vector", prompt: "hello",
      cache_identity: fixture.identity,
    });
    const identity = decoded.cache_identity as CacheIdentity;
    expect(identity.namespace_id).toBe(fixture.expected.namespace_u64);
    expect(identity.tenant_root.slice(0, 16)).toBe(fixture.expected.tenant_u64_hex);
    expect(identity.session_fingerprint?.slice(0, 16)).toBe(fixture.expected.session_u64_hex);
  });

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
  test("validates honest memory values and telemetry companions", () => {
    const event = (telemetry: Record<string, unknown>) => ({ protocol: 1, type: "telemetry", telemetry });
    expect(decodeWorkerEvent(event({ memory: {
      active_bytes: 1024, active_bytes_source: "measured", active_bytes_basis: "runtime_allocator",
      cache_bytes: 0, cache_bytes_source: "estimated", cache_bytes_basis: "configured_cache",
      peak_active_bytes: 2048,
    } }))).toMatchObject({ telemetry: { memory: { active_bytes_source: "measured" } } });
    expect(decodeWorkerEvent(event({ retention: {
      retained_bytes: null, retained_bytes_source: "unavailable", retained_bytes_basis: "not_reported",
      evicted_bytes: null, evicted_bytes_source: "unavailable", evicted_bytes_basis: "not_observed",
      estimated_retained_bytes: 4096, estimated_retained_bytes_source: "estimated",
      estimated_retained_bytes_basis: "context_configuration",
    } }))).toMatchObject({ telemetry: { retention: { retained_bytes: null } } });

    for (const memory of [
      { active_bytes: null, active_bytes_source: "measured", active_bytes_basis: "runtime_allocator", cache_bytes: 0, peak_active_bytes: 1 },
      { active_bytes: 1, active_bytes_source: "unavailable", active_bytes_basis: "not_reported", cache_bytes: 0, peak_active_bytes: 1 },
      { active_bytes: 0, active_bytes_source: "measured", active_bytes_basis: "runtime_allocator", cache_bytes: 0, peak_active_bytes: 1 },
      { active_bytes: 1, active_bytes_source: "measured", cache_bytes: 0, peak_active_bytes: 1 },
      { active_bytes: null, cache_bytes: 0, peak_active_bytes: 1 },
    ]) expect(() => decodeWorkerEvent(event({ memory }))).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerEvent(event({ retention: {
      retained_bytes: 0, retained_bytes_source: "measured", retained_bytes_basis: "runtime_allocator",
    } }))).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerEvent(event({ retention: {
      estimated_retained_bytes: 1, estimated_retained_bytes_source: "estimated",
      estimated_retained_bytes_basis: "runtime_allocator",
    } }))).toThrow(ProtocolValidationError);
  });

  test("requires a strict opaque cache identity on generate requests", () => {
    const identity = fixtureCacheIdentity();
    const base = { protocol: 1, type: "generate", request_id: "req", prompt: "hello" };
    expect(decodeWorkerRequest({ ...base, cache_identity: identity })).toMatchObject({ cache_identity: identity });
    expect(() => decodeWorkerRequest(base)).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ ...base, cache_identity: { ...identity, tenant_root: "ABC" } })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ ...base, cache_identity: { ...identity, namespace_id: "18446744073709551616" } })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ ...base, cache_identity: { ...identity, secret: "raw-secret" } })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ ...base, cache_identity: {
      ...identity, physical: { ...identity.physical, installation_key: "raw-secret" },
    } })).toThrow(ProtocolValidationError);
    expect(() => decodeWorkerRequest({ ...base, cache_identity: {
      ...identity, display: { namespace: "x".repeat(129) },
    } })).toThrow(ProtocolValidationError);
  });

  test("validates strict structured-output contracts and ready capabilities", () => {
    const base = {
      protocol: 1, type: "generate", request_id: "req", prompt: "hello",
      cache_identity: fixtureCacheIdentity(),
    } as const;
    expect(decodeWorkerRequest({
      ...base, structured_output: { kind: "json_object", strength: "best_effort" },
    })).toMatchObject({ structured_output: { kind: "json_object", strength: "best_effort" } });
    expect(decodeWorkerRequest({
      ...base, structured_output: {
        kind: "json_schema", strength: "required", schema: { type: "object" },
      },
    })).toMatchObject({ structured_output: { kind: "json_schema", schema: { type: "object" } } });
    for (const structured_output of [
      { kind: "json_schema", strength: "required" },
      { kind: "json_object", strength: "best_effort", schema: {} },
      { kind: "json_schema", strength: "strict", schema: {} },
      { kind: "grammar", strength: "required", schema: {} },
      { kind: "json_schema", strength: "required", schema: {}, extension: true },
    ]) expect(() => decodeWorkerRequest({ ...base, structured_output })).toThrow(ProtocolValidationError);

    const ready = { protocol: 1, type: "ready", worker_capabilities: {}, model_capabilities: {} } as const;
    expect(decodeWorkerEvent({
      ...ready,
      structured_output: {
        json_object: "native", json_schema: "post_validate", post_validation: true, max_schema_bytes: 65_536,
      },
    })).toMatchObject({ structured_output: { json_object: "native", max_schema_bytes: 65_536 } });
    for (const structured_output of [
      { json_object: "native", json_schema: "unsupported", post_validation: true },
      { json_object: "grammar", json_schema: "unsupported", post_validation: false, max_schema_bytes: 0 },
      { json_object: "native", json_schema: "unsupported", post_validation: false, max_schema_bytes: -1 },
      { json_object: "native", json_schema: "unsupported", post_validation: false, max_schema_bytes: 0, extension: true },
    ]) expect(() => decodeWorkerEvent({ ...ready, structured_output })).toThrow(ProtocolValidationError);
  });

function fixtureCacheIdentity() {
  const fingerprint = "a".repeat(64);
  return {
    version: 1,
    generation: "sec_fixture",
    tenant_root: fingerprint,
    scope: "tenant",
    scope_fingerprint: fingerprint,
    namespace_fingerprint: "b".repeat(64),
    namespace_id: "1",
    priority: "interactive",
    side_request: false,
    display: {},
    physical: {
      fingerprint: "c".repeat(64),
      backend: "llama",
      resolved_revision: "local:fixture",
      model_artifact_fingerprint: "d".repeat(64),
      tokenizer_fingerprint: "d".repeat(64),
      context_allocation: 4096,
      kv_format: "f16",
      unified_kv: true,
      layout_version: 1,
    },
  } as const;
}

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
