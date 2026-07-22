import { describe, expect, test } from "bun:test";
import {
  CONSERVATIVE_KV_BYTES_PER_TOKEN,
  DEFAULT_CONTEXT_TOKENS,
  GIB,
  MAX_MEMORY_BYTES,
  MIB,
  ModelLoadEstimateHistory,
  UNKNOWN_MODEL_MIN_BYTES_ENV,
  estimateModelLoadMemory,
  modelLoadEstimateHistoryKey,
  type ResidencyModelDescriptor,
} from "./index";

function model(overrides: Partial<ResidencyModelDescriptor> = {}): ResidencyModelDescriptor {
  return {
    modelKey: "artifact-a",
    modelId: "owner/model",
    backend: "llama",
    artifactBytes: GIB,
    revision: "r1",
    ...overrides,
  };
}

describe("model load estimates", () => {
  test("estimates GGUF weights, margin, and metadata-derived context KV", () => {
    const descriptor = model({
      artifactBytes: GIB,
      context: 4_096,
      configuredContext: 2_048,
      kv: { type: "f16", bytesPerToken: 128 * 1024 },
    });
    const estimate = estimateModelLoadMemory(descriptor, undefined, { physicalMemoryBytes: 32 * GIB, env: {} });
    const weights = Math.ceil(GIB * 1.2);
    const kv = 2_048 * 128 * 1024;

    expect(estimate).toEqual({
      kind: "estimated",
      bytes: weights + 512 * MIB + kv,
      basis: "architecture_metadata",
    });
  });

  test("uses the conservative context and KV defaults for GGUF metadata gaps", () => {
    const estimate = estimateModelLoadMemory(model(), undefined, { physicalMemoryBytes: 32 * GIB, env: {} });
    expect(estimate.bytes).toBe(
      Math.ceil(GIB * 1.2) + 512 * MIB + DEFAULT_CONTEXT_TOKENS * CONSERVATIVE_KV_BYTES_PER_TOKEN,
    );
    expect(estimate.basis).toBe("model_artifacts");
  });

  test("estimates MLX weights, margin, and a machine-bounded configured cache", () => {
    const descriptor = model({ backend: "mlx", artifactBytes: 4 * GIB, cacheBudget: 20 * GIB });
    const estimate = estimateModelLoadMemory(descriptor, undefined, { physicalMemoryBytes: 16 * GIB, env: {} });

    expect(estimate).toEqual({
      kind: "estimated",
      bytes: Math.ceil(4 * GIB * 1.35) + GIB + 8 * GIB,
      basis: "configured_cache",
    });
  });

  test("uses conservative context KV when MLX has no configured cache budget", () => {
    const descriptor = model({ backend: "mlx", artifactBytes: 2 * GIB, configuredContext: 1_024 });
    const estimate = estimateModelLoadMemory(descriptor, undefined, { physicalMemoryBytes: 16 * GIB, env: {} });
    expect(estimate.bytes).toBe(
      Math.ceil(2 * GIB * 1.35) + GIB + 1_024 * CONSERVATIVE_KV_BYTES_PER_TOKEN,
    );
    expect(estimate.basis).toBe("model_artifacts");
  });

  test("takes the high-water observed RSS without allowing lower updates to regress it", () => {
    const descriptor = model();
    const history = new ModelLoadEstimateHistory();
    history.update(descriptor, 12 * GIB);
    history.update(descriptor, 10 * GIB);

    expect(history.get(descriptor)).toBe(12 * GIB);
    expect(estimateModelLoadMemory(descriptor, history, { physicalMemoryBytes: 32 * GIB, env: {} })).toEqual({
      kind: "estimated", bytes: 12 * GIB, basis: "prior_observation",
    });
    expect(() => history.update(descriptor, 0)).toThrow(RangeError);
  });

  test("keys history by backend, artifact, revision, context, KV, and layout", () => {
    const baseline = model({
      architecture: "arch-a",
      modelType: "type-a",
      quantization: "q4",
      configuredContext: 4_096,
      kv: { type: "f16", bytesPerToken: 128 },
    });
    const key = modelLoadEstimateHistoryKey(baseline);
    const changes: Array<Partial<ResidencyModelDescriptor>> = [
      { backend: "mlx" },
      { modelKey: "artifact-b" },
      { artifactBytes: 2 * GIB },
      { revision: "r2" },
      { configuredContext: 8_192 },
      { kv: { type: "q8", bytesPerToken: 128 } },
      { kv: { type: "f16", bytesPerToken: 256 } },
      { architecture: "arch-b" },
      { modelType: "type-b" },
      { quantization: "q8" },
    ];
    for (const change of changes) {
      expect(modelLoadEstimateHistoryKey({ ...baseline, ...change })).not.toBe(key);
    }
  });

  test("unknown models use max(environment minimum, 25% physical) and never zero", () => {
    const unknown = model({ backend: "other", artifactBytes: undefined });
    expect(estimateModelLoadMemory(unknown, undefined, {
      physicalMemoryBytes: 40 * GIB,
      env: { [UNKNOWN_MODEL_MIN_BYTES_ENV]: String(6 * GIB) },
    })).toEqual({ kind: "estimated", bytes: 10 * GIB, basis: "conservative_fallback" });
    expect(estimateModelLoadMemory(unknown, undefined, {
      physicalMemoryBytes: 8 * GIB,
      env: { [UNKNOWN_MODEL_MIN_BYTES_ENV]: String(6 * GIB) },
    }).bytes).toBe(6 * GIB);
    expect(estimateModelLoadMemory(unknown, undefined, {
      physicalMemoryBytes: 0,
      env: { [UNKNOWN_MODEL_MIN_BYTES_ENV]: "0" },
    }).bytes).toBe(1);
  });

  test("invalid unknown minimum falls back conservatively and arithmetic saturates", () => {
    const unknown = model({ artifactBytes: undefined });
    expect(estimateModelLoadMemory(unknown, undefined, {
      physicalMemoryBytes: 8 * GIB,
      env: { [UNKNOWN_MODEL_MIN_BYTES_ENV]: "NaN" },
    }).bytes).toBe(4 * GIB);

    const overflow = estimateModelLoadMemory(model({ artifactBytes: Number.MAX_VALUE }), undefined, {
      physicalMemoryBytes: Number.MAX_VALUE,
      env: {},
    });
    expect(overflow.bytes).toBe(MAX_MEMORY_BYTES);
    expect(Number.isFinite(overflow.bytes)).toBe(true);
  });
});
