import { MemoryValueSchema, type EstimatedMemoryBasis, type MeasuredMemoryBasis,
  type MemoryValue, type UnavailableMemoryBasis } from "@clap/worker-protocol";

export const MAX_MEMORY_BYTES = Number.MAX_SAFE_INTEGER;
export type { EstimatedMemoryBasis, MeasuredMemoryBasis, MemoryValue, UnavailableMemoryBasis };

function normalizeBytes(bytes: number, allowZero: boolean): number {
  if (!Number.isFinite(bytes) || bytes < 0 || (!allowZero && bytes === 0)) {
    throw new RangeError(`memory bytes must be finite and ${allowZero ? "nonnegative" : "positive"}`);
  }
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(bytes));
}

export function measuredMemory(bytes: number, basis: MeasuredMemoryBasis): MemoryValue {
  const value = { source: "measured" as const, bytes: normalizeBytes(bytes, false), basis };
  assertMemoryValue(value);
  return Object.freeze(value);
}

export function estimatedMemory(bytes: number, basis: EstimatedMemoryBasis): MemoryValue {
  const value = { source: "estimated" as const, bytes: normalizeBytes(bytes, true), basis };
  assertMemoryValue(value);
  return Object.freeze(value);
}

export function unavailableMemory(basis: UnavailableMemoryBasis): MemoryValue {
  const value = { source: "unavailable" as const, bytes: null, basis };
  assertMemoryValue(value);
  return Object.freeze(value);
}

export function isMemoryValue(value: unknown): value is MemoryValue {
  return MemoryValueSchema.safeParse(value).success;
}

export function assertMemoryValue(value: unknown): asserts value is MemoryValue {
  if (!isMemoryValue(value)) throw new TypeError("invalid memory value");
}

export function saturatingAddMemoryBytes(...values: number[]): number {
  let total = 0;
  for (const value of values) {
    const normalized = normalizeBytes(value, true);
    if (total >= MAX_MEMORY_BYTES - normalized) return MAX_MEMORY_BYTES;
    total += normalized;
  }
  return total;
}

export interface ResidencyModelDescriptor {
  readonly modelKey: string;
  readonly backend: string;
  readonly modelId: string;
  readonly revision?: string | null;
  readonly artifactBytes?: number;
  readonly architecture?: string;
  readonly modelType?: string;
  readonly quantization?: string;
  readonly context?: number;
  readonly configuredContext?: number;
  readonly kv?: Readonly<{ type?: string; bytesPerToken?: number }>;
  readonly cacheBudget?: number;
}

interface ResidentModelSnapshotBase {
  readonly model: ResidencyModelDescriptor;
  readonly workerId: string;
  readonly memory: MemoryValue;
  readonly pinned: boolean;
  readonly loadedAtMs: number;
  readonly lastUsedAtMs: number;
}

export type ResidentModelSnapshot =
  | (ResidentModelSnapshotBase & Readonly<{ state: "loading"; activeRequests: 0; reservationId: string }>)
  | (ResidentModelSnapshotBase & Readonly<{ state: "idle"; activeRequests: 0 }>)
  | (ResidentModelSnapshotBase & Readonly<{ state: "active"; activeRequests: number }>)
  | (ResidentModelSnapshotBase & Readonly<{ state: "stopping"; activeRequests: 0 }>);

export type LoadReservationState = "held" | "committed" | "released";

export interface LoadReservation {
  readonly reservationId: string;
  readonly model: ResidencyModelDescriptor;
  readonly bytes: number;
  readonly state: LoadReservationState;
  readonly createdAtMs: number;
  readonly expiresAtMs: number;
}

export type AdmissionReason =
  | "within_budget"
  | "within_budget_after_eviction"
  | "insufficient_available_memory"
  | "memory_state_unavailable"
  | "no_evictable_models";

export interface LoadAdmissionDecision {
  readonly admitted: boolean;
  readonly reason: AdmissionReason;
  readonly requested: MemoryValue;
  readonly available: MemoryValue;
  readonly reservedBytes: number;
  readonly headroomBytes: number;
  readonly evictedModelKeys: readonly string[];
  readonly decidedAtMs: number;
}

export interface LoadAdmissionTelemetry {
  readonly reservationId: string;
  readonly modelKey: string;
  readonly backend: string;
  readonly admitted: boolean;
  readonly reason: AdmissionReason;
  readonly requestedBytes: number | null;
  readonly availableBytes: number | null;
  readonly reservedBytes: number;
  readonly evictedModelCount: number;
  readonly decisionDurationMs: number;
}

export interface ResidencyPolicy {
  readonly minimumHeadroomBytes: number;
  readonly conservativeFallbackBytes: number;
  readonly reservationTtlMs: number;
  readonly maximumConcurrentLoads: number;
}

export const DEFAULT_RESIDENCY_POLICY: ResidencyPolicy = Object.freeze({
  minimumHeadroomBytes: 2 * 1024 ** 3,
  conservativeFallbackBytes: 8 * 1024 ** 3,
  reservationTtlMs: 5 * 60_000,
  maximumConcurrentLoads: 1,
});

export function createResidencyPolicy(overrides: Partial<ResidencyPolicy> = {}): ResidencyPolicy {
  const policy = {
    ...DEFAULT_RESIDENCY_POLICY,
    ...overrides,
  };
  policy.minimumHeadroomBytes = normalizeBytes(policy.minimumHeadroomBytes, true);
  policy.conservativeFallbackBytes = normalizeBytes(policy.conservativeFallbackBytes, false);
  policy.reservationTtlMs = positiveSafeInteger(policy.reservationTtlMs, "reservation TTL");
  policy.maximumConcurrentLoads = positiveSafeInteger(policy.maximumConcurrentLoads, "maximum concurrent loads");
  return Object.freeze(policy);
}

export function isResidencyPolicy(value: unknown): value is ResidencyPolicy {
  if (!value || typeof value !== "object") return false;
  const policy = value as Partial<ResidencyPolicy>;
  return isValidBytes(policy.minimumHeadroomBytes, true)
    && isValidBytes(policy.conservativeFallbackBytes, false)
    && isPositiveSafeInteger(policy.reservationTtlMs)
    && isPositiveSafeInteger(policy.maximumConcurrentLoads);
}

function isValidBytes(value: unknown, allowZero: boolean): value is number {
  return typeof value === "number"
    && Number.isSafeInteger(value)
    && value >= (allowZero ? 0 : 1)
    && value <= MAX_MEMORY_BYTES;
}

function positiveSafeInteger(value: number, label: string): number {
  if (!isPositiveSafeInteger(value)) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function isPositiveSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}
