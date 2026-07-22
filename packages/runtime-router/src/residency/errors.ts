import type { AdmissionReason } from "./types";

export const INSUFFICIENT_MODEL_MEMORY_CODE = "insufficient_model_memory" as const;

export interface InsufficientModelMemoryDetails {
  readonly reason: Exclude<AdmissionReason, "within_budget" | "within_budget_after_eviction">;
  readonly requestedBytes: number | null;
  readonly availableBytes: number | null;
  readonly reservedBytes: number;
  readonly headroomBytes: number;
  readonly evictableModelCount: number;
}

/** A deliberately path- and identity-free error suitable for API serialization. */
export class InsufficientModelMemoryError extends Error {
  readonly name = "InsufficientModelMemoryError";
  readonly code = INSUFFICIENT_MODEL_MEMORY_CODE;
  readonly retryable = true;
  readonly details: InsufficientModelMemoryDetails;

  constructor(details: InsufficientModelMemoryDetails, cause?: unknown) {
    super("Insufficient memory to load the requested model safely", { cause });
    if (!FAILURE_REASONS.has(details.reason)) throw new TypeError("invalid memory admission failure reason");
    this.details = Object.freeze({
      reason: details.reason,
      requestedBytes: normalizeNullableBytes(details.requestedBytes),
      availableBytes: normalizeNullableBytes(details.availableBytes),
      reservedBytes: normalizeBytes(details.reservedBytes),
      headroomBytes: normalizeBytes(details.headroomBytes),
      evictableModelCount: normalizeCount(details.evictableModelCount),
    });
  }

  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      details: this.details,
    };
  }
}

const FAILURE_REASONS: ReadonlySet<unknown> = new Set<InsufficientModelMemoryDetails["reason"]>([
  "insufficient_available_memory", "memory_state_unavailable", "no_evictable_models",
]);

function normalizeNullableBytes(value: number | null): number | null {
  return value === null ? null : normalizeBytes(value);
}

function normalizeBytes(value: number): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError("memory bytes must be finite and nonnegative");
  return Math.min(Number.MAX_SAFE_INTEGER, Math.ceil(value));
}

function normalizeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError("model count must be a nonnegative safe integer");
  return value;
}
