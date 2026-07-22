import { totalmem } from "node:os";
import {
  MAX_MEMORY_BYTES,
  estimatedMemory,
  saturatingAddMemoryBytes,
  type EstimatedMemoryBasis,
  type MemoryValue,
  type ResidencyModelDescriptor,
} from "./types";

export const MIB = 1024 ** 2;
export const GIB = 1024 ** 3;
export const DEFAULT_CONTEXT_TOKENS = 8_192;
export const CONSERVATIVE_KV_BYTES_PER_TOKEN = 256 * 1024;
export const DEFAULT_UNKNOWN_MODEL_MIN_BYTES = 4 * GIB;
export const UNKNOWN_MODEL_MIN_BYTES_ENV = "CLAP_UNKNOWN_MODEL_MEMORY_MIN_BYTES";

export interface ModelLoadEstimateOptions {
  readonly physicalMemoryBytes?: number;
  readonly env?: Readonly<Record<string, string | undefined>>;
}

export class ModelLoadEstimateHistory {
  private readonly highwaterByKey = new Map<string, number>();

  update(model: ResidencyModelDescriptor, observedRssBytes: number): number {
    const observed = normalizePositiveBytes(observedRssBytes, "observed RSS");
    const key = modelLoadEstimateHistoryKey(model);
    const highwater = Math.max(this.highwaterByKey.get(key) ?? 0, observed);
    this.highwaterByKey.set(key, highwater);
    return highwater;
  }

  get(model: ResidencyModelDescriptor): number | undefined {
    return this.highwaterByKey.get(modelLoadEstimateHistoryKey(model));
  }

  clear(): void {
    this.highwaterByKey.clear();
  }
}

export function modelLoadEstimateHistoryKey(model: ResidencyModelDescriptor): string {
  const context = effectiveContext(model);
  return JSON.stringify([
    model.backend,
    model.modelKey,
    normalizeOptionalBytes(model.artifactBytes),
    model.revision ?? null,
    context,
    model.kv?.type ?? null,
    normalizeOptionalBytes(model.kv?.bytesPerToken),
    model.architecture ?? null,
    model.modelType ?? null,
    model.quantization ?? null,
  ]);
}

export function estimateModelLoadMemory(
  model: ResidencyModelDescriptor,
  history?: ModelLoadEstimateHistory,
  options: ModelLoadEstimateOptions = {},
): MemoryValue {
  const physicalMemoryBytes = normalizePhysicalMemory(options.physicalMemoryBytes ?? totalmem());
  const baseline = baselineEstimate(model, physicalMemoryBytes, options.env ?? process.env);
  const prior = history?.get(model);
  if (prior !== undefined && prior > baseline.bytes) {
    return estimatedMemory(prior, "prior_observation");
  }
  return estimatedMemory(baseline.bytes, baseline.basis);
}

function baselineEstimate(
  model: ResidencyModelDescriptor,
  physicalMemoryBytes: number,
  env: Readonly<Record<string, string | undefined>>,
): { bytes: number; basis: EstimatedMemoryBasis } {
  const artifactBytes = normalizeOptionalBytes(model.artifactBytes);
  if (artifactBytes === null || artifactBytes === 0) {
    return { bytes: unknownModelEstimate(physicalMemoryBytes, env), basis: "conservative_fallback" };
  }
  const backend = model.backend.toLowerCase();
  if (backend === "llama" || backend === "gguf") {
    const weights = multiplySaturating(artifactBytes, 120, 100);
    const margin = Math.max(512 * MIB, multiplySaturating(artifactBytes, 10, 100));
    const kv = contextKvEstimate(model);
    return {
      bytes: saturatingAddMemoryBytes(weights, margin, kv),
      basis: normalizeOptionalBytes(model.kv?.bytesPerToken) === null ? "model_artifacts" : "architecture_metadata",
    };
  }
  if (backend === "mlx") {
    const weights = multiplySaturating(artifactBytes, 135, 100);
    const margin = Math.max(GIB, multiplySaturating(artifactBytes, 15, 100));
    const cache = boundedMlxCacheEstimate(model, physicalMemoryBytes);
    return {
      bytes: saturatingAddMemoryBytes(weights, margin, cache),
      basis: normalizeOptionalBytes(model.cacheBudget) !== null
        ? "configured_cache"
        : normalizeOptionalBytes(model.kv?.bytesPerToken) === null ? "model_artifacts" : "architecture_metadata",
    };
  }
  return { bytes: unknownModelEstimate(physicalMemoryBytes, env), basis: "conservative_fallback" };
}

function contextKvEstimate(model: ResidencyModelDescriptor): number {
  const context = effectiveContext(model);
  const bytesPerToken = normalizeOptionalBytes(model.kv?.bytesPerToken) ?? CONSERVATIVE_KV_BYTES_PER_TOKEN;
  return multiplySaturating(context, bytesPerToken, 1);
}

function boundedMlxCacheEstimate(model: ResidencyModelDescriptor, physicalMemoryBytes: number): number {
  const configured = normalizeOptionalBytes(model.cacheBudget);
  const conservative = contextKvEstimate(model);
  if (configured === null) return conservative;
  const machineBound = Math.max(GIB, Math.ceil(physicalMemoryBytes / 2));
  return Math.min(configured, machineBound);
}

function unknownModelEstimate(
  physicalMemoryBytes: number,
  env: Readonly<Record<string, string | undefined>>,
): number {
  const configuredMinimum = parseUnknownMinimum(env[UNKNOWN_MODEL_MIN_BYTES_ENV]);
  return Math.max(1, configuredMinimum, Math.ceil(physicalMemoryBytes / 4));
}

function parseUnknownMinimum(value: string | undefined): number {
  if (value === undefined || value.trim() === "") return DEFAULT_UNKNOWN_MODEL_MIN_BYTES;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return DEFAULT_UNKNOWN_MODEL_MIN_BYTES;
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(parsed));
}

function effectiveContext(model: ResidencyModelDescriptor): number {
  return normalizePositiveInteger(model.configuredContext)
    ?? normalizePositiveInteger(model.context)
    ?? DEFAULT_CONTEXT_TOKENS;
}

function multiplySaturating(value: number, numerator: number, denominator: number): number {
  if (value === 0) return 0;
  if (value > (MAX_MEMORY_BYTES * denominator) / numerator) return MAX_MEMORY_BYTES;
  return Math.min(MAX_MEMORY_BYTES, Math.ceil((value * numerator) / denominator));
}

function normalizePhysicalMemory(value: number): number {
  if (!Number.isFinite(value) || value < 0) return 0;
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(value));
}

function normalizeOptionalBytes(value: number | undefined): number | null {
  if (value === undefined || !Number.isFinite(value) || value < 0) return null;
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(value));
}

function normalizePositiveBytes(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be finite and positive`);
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(value));
}

function normalizePositiveInteger(value: number | undefined): number | undefined {
  if (value === undefined || !Number.isFinite(value) || value <= 0) return undefined;
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(value));
}
