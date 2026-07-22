import { describe, expect, test } from "bun:test";
import {
  INSUFFICIENT_MODEL_MEMORY_CODE,
  InsufficientModelMemoryError,
  MAX_MEMORY_BYTES,
  assertMemoryValue,
  createResidencyPolicy,
  estimatedMemory,
  isMemoryValue,
  isResidencyPolicy,
  measuredMemory,
  saturatingAddMemoryBytes,
  unavailableMemory,
} from "./index";

describe("MemoryValue", () => {
  test("constructs explicit measured, estimated, and unavailable values", () => {
    expect(measuredMemory(1.2, "resident_rss")).toEqual({
      kind: "measured", bytes: 2, basis: "resident_rss",
    });
    expect(estimatedMemory(0, "model_artifacts")).toEqual({
      kind: "estimated", bytes: 0, basis: "model_artifacts",
    });
    expect(unavailableMemory("not_supported")).toEqual({
      kind: "unavailable", bytes: null, basis: "not_supported",
    });
  });

  test("rejects invalid and misleading measured values", () => {
    for (const bytes of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => measuredMemory(bytes, "runtime_allocator")).toThrow(RangeError);
    }
    expect(isMemoryValue({ kind: "measured", bytes: 0, basis: "resident_rss" })).toBe(false);
    expect(isMemoryValue({ kind: "unavailable", bytes: 0, basis: "not_observed" })).toBe(false);
    expect(isMemoryValue({ kind: "unavailable", bytes: null, basis: "not_observed" })).toBe(true);
    expect(() => assertMemoryValue({ kind: "estimated", bytes: -1, basis: "model_artifacts" })).toThrow(TypeError);
    expect(() => measuredMemory(1, "invented" as never)).toThrow(TypeError);
  });

  test("saturates oversized values and additions", () => {
    expect(estimatedMemory(Number.MAX_VALUE, "conservative_fallback").bytes).toBe(MAX_MEMORY_BYTES);
    expect(saturatingAddMemoryBytes(MAX_MEMORY_BYTES - 2, 10)).toBe(MAX_MEMORY_BYTES);
    expect(() => saturatingAddMemoryBytes(1, -1)).toThrow(RangeError);
  });

  test("serializes unavailable memory as null rather than a false zero", () => {
    expect(JSON.stringify(unavailableMemory("not_reported"))).toBe(
      '{"kind":"unavailable","bytes":null,"basis":"not_reported"}',
    );
  });
});

describe("ResidencyPolicy", () => {
  test("applies defaults and validates policy invariants", () => {
    const policy = createResidencyPolicy({ minimumHeadroomBytes: 1.1, maximumConcurrentLoads: 2 });
    expect(policy.minimumHeadroomBytes).toBe(2);
    expect(policy.maximumConcurrentLoads).toBe(2);
    expect(isResidencyPolicy(policy)).toBe(true);
    expect(() => createResidencyPolicy({ reservationTtlMs: 0 })).toThrow(RangeError);
    expect(() => createResidencyPolicy({ conservativeFallbackBytes: 0 })).toThrow(RangeError);
  });
});

describe("InsufficientModelMemoryError", () => {
  test("has a stable retryable shape and redacts caller-supplied identities", () => {
    const secretPath = "/Users/alice/private/models/secret.gguf";
    const apiKey = "sk-private-value";
    const details = {
      reason: "insufficient_available_memory" as const,
      requestedBytes: 12,
      availableBytes: 4,
      reservedBytes: 3,
      headroomBytes: 2,
      evictableModelCount: 0,
      modelPath: secretPath,
      apiKey,
    };
    const error = new InsufficientModelMemoryError(details, new Error(`${apiKey}: ${secretPath}`));
    const serialized = JSON.stringify(error);

    expect(error.code).toBe(INSUFFICIENT_MODEL_MEMORY_CODE);
    expect(error.retryable).toBe(true);
    expect(JSON.parse(serialized)).toEqual({
      name: "InsufficientModelMemoryError",
      code: "insufficient_model_memory",
      message: "Insufficient memory to load the requested model safely",
      retryable: true,
      details: {
        reason: "insufficient_available_memory",
        requestedBytes: 12,
        availableBytes: 4,
        reservedBytes: 3,
        headroomBytes: 2,
        evictableModelCount: 0,
      },
    });
    expect(serialized).not.toContain(secretPath);
    expect(serialized).not.toContain(apiKey);
  });

  test("rejects non-finite details rather than serializing them as null", () => {
    expect(() => new InsufficientModelMemoryError({
      reason: "memory_state_unavailable",
      requestedBytes: Number.NaN,
      availableBytes: null,
      reservedBytes: 0,
      headroomBytes: 0,
      evictableModelCount: 0,
    })).toThrow(RangeError);
  });
});
