import { randomUUID } from "node:crypto";
import type {
  IdleEvictionResult,
  LifecycleResidencySnapshot,
} from "../lifecycle";
import { InsufficientModelMemoryError } from "./errors";
import { selectIdleEvictionVictims } from "./eviction";
import {
  ModelLoadEstimateHistory,
  estimateModelLoadMemory,
  type ModelLoadEstimateOptions,
} from "./estimates";
import {
  MAX_MEMORY_BYTES,
  createResidencyPolicy,
  measuredMemory,
  saturatingAddMemoryBytes,
  unavailableMemory,
  type LoadAdmissionDecision,
  type LoadReservation,
  type MemoryValue,
  type ResidencyModelDescriptor,
  type ResidencyPolicy,
} from "./types";

export const DEFAULT_OS_HEADROOM_BYTES = 1024 ** 3;
export const DEFAULT_RUNTIME_HEADROOM_BYTES = 1024 ** 3;
export const OS_HEADROOM_BYTES_ENV = "CLAP_MODEL_OS_HEADROOM_BYTES";
export const RUNTIME_HEADROOM_BYTES_ENV = "CLAP_MODEL_RUNTIME_HEADROOM_BYTES";

export interface ResidencyMemorySnapshot {
  readonly available: MemoryValue;
  readonly physicalMemoryBytes: number;
}

export interface ResidencyLifecycleAdapter {
  snapshotForResidency(): LifecycleResidencySnapshot[];
  tryEvictIdle(snapshot: LifecycleResidencySnapshot): Promise<IdleEvictionResult>;
  setResidencyTransition(key: string, state: "starting" | "loading" | "closing"): void;
  clearResidencyTransition(key: string): void;
}

export interface ResidencyCoordinatorDependencies {
  readonly memorySnapshot: () => Promise<ResidencyMemorySnapshot>;
  readonly lifecycle: ResidencyLifecycleAdapter;
  readonly now?: () => number;
  readonly reservationId?: () => string;
  readonly history?: ModelLoadEstimateHistory;
  readonly policy?: Partial<ResidencyPolicy>;
  readonly osHeadroomBytes?: number;
  readonly runtimeHeadroomBytes?: number;
  readonly env?: Readonly<Record<string, string | undefined>>;
  readonly onDecision?: (decision: LoadAdmissionDecision, model: ResidencyModelDescriptor) => void;
  readonly onEvent?: (event: ResidencyCoordinatorEvent) => void;
}

export type ResidencyCoordinatorEventType = "model_load_reserved" | "model_load_started" | "model_load_committed"
  | "model_load_rolled_back" | "model_evicted_for_load" | "model_load_rejected";

export interface ResidencyCoordinatorEvent {
  readonly type: ResidencyCoordinatorEventType;
  readonly backend: string;
  readonly reason?: string;
  readonly reservationBytes: number;
  readonly activeReservations: number;
  readonly estimateBytes?: number;
  readonly observedRssBytes?: number;
}

export interface ResidencyLoadOperation<T> {
  readonly performLoad: () => Promise<T>;
  readonly observeRss?: (loaded: T) => Promise<number | undefined> | number | undefined;
  readonly stabilize?: (loaded: T) => Promise<void> | void;
  readonly shutdownPartial?: (loaded: T | undefined, cause: unknown) => Promise<void> | void;
}

export interface ResidencyLoadResult<T> {
  readonly value: T;
  readonly reservation: LoadReservation;
  readonly decision: LoadAdmissionDecision;
}

export class ResidencyCoordinator {
  readonly history: ModelLoadEstimateHistory;
  readonly policy: ResidencyPolicy;
  readonly osHeadroomBytes: number;
  readonly runtimeHeadroomBytes: number;

  private readonly activeReservations = new Map<string, LoadReservation>();
  private readonly inFlightByKey = new Map<string, Promise<ResidencyLoadResult<unknown>>>();
  private mutexTail: Promise<void> = Promise.resolve();
  private readonly now: () => number;
  private readonly reservationId: () => string;

  constructor(private readonly dependencies: ResidencyCoordinatorDependencies) {
    this.now = dependencies.now ?? (() => Date.now());
    this.reservationId = dependencies.reservationId ?? randomUUID;
    this.history = dependencies.history ?? new ModelLoadEstimateHistory();
    this.policy = createResidencyPolicy({ ...dependencies.policy, maximumConcurrentLoads: 1 });
    const env = dependencies.env ?? process.env;
    this.osHeadroomBytes = normalizeHeadroom(
      dependencies.osHeadroomBytes ?? parseHeadroom(env[OS_HEADROOM_BYTES_ENV], DEFAULT_OS_HEADROOM_BYTES),
    );
    this.runtimeHeadroomBytes = normalizeHeadroom(
      dependencies.runtimeHeadroomBytes ?? parseHeadroom(env[RUNTIME_HEADROOM_BYTES_ENV], DEFAULT_RUNTIME_HEADROOM_BYTES),
    );
  }

  reservations(): readonly LoadReservation[] {
    return [...this.activeReservations.values()].map((reservation) => Object.freeze({ ...reservation }));
  }

  load<T>(model: ResidencyModelDescriptor, operation: ResidencyLoadOperation<T>): Promise<ResidencyLoadResult<T>> {
    const existing = this.inFlightByKey.get(model.modelKey);
    if (existing) return existing as Promise<ResidencyLoadResult<T>>;
    const load = this.withMutex(() => this.loadSerialized(model, operation));
    this.inFlightByKey.set(model.modelKey, load as Promise<ResidencyLoadResult<unknown>>);
    void load.finally(() => {
      if (this.inFlightByKey.get(model.modelKey) === load) this.inFlightByKey.delete(model.modelKey);
    }).catch(() => {});
    return load;
  }

  private async loadSerialized<T>(
    model: ResidencyModelDescriptor,
    operation: ResidencyLoadOperation<T>,
  ): Promise<ResidencyLoadResult<T>> {
    const startedAt = this.now();
    let memory = await this.dependencies.memorySnapshot();
    const estimateOptions: ModelLoadEstimateOptions = {
      physicalMemoryBytes: memory.physicalMemoryBytes,
      env: this.dependencies.env,
    };
    const requested = estimateModelLoadMemory(model, this.history, estimateOptions);
    const reservationId = this.reservationId();
    let reservation: LoadReservation = Object.freeze({
      reservationId,
      model,
      bytes: requested.bytes ?? this.policy.conservativeFallbackBytes,
      state: "held",
      createdAtMs: startedAt,
      expiresAtMs: saturatingAddMemoryBytes(startedAt, this.policy.reservationTtlMs),
    });
    this.activeReservations.set(reservationId, reservation);
    this.emit("model_load_reserved", model, reservation.bytes, { estimateBytes: requested.bytes ?? undefined });
    let loaded: T | undefined;
    let loadStarted = false;
    const evictedModelKeys: string[] = [];
    try {
      let reason: LoadAdmissionDecision["reason"] = "within_budget";
      while (!this.fits(memory.available, reservation.bytes, reservationId)) {
        if (memory.available.source !== "measured") {
          this.emit("model_load_rejected", model, reservation.bytes, { reason: "memory_state_unavailable" });
          throw this.insufficient("memory_state_unavailable", requested, memory.available, reservationId, 0);
        }
        const snapshots = this.dependencies.lifecycle.snapshotForResidency();
        const victims = selectIdleEvictionVictims(snapshots, {
          targetKey: model.modelKey,
          reservedKeys: new Set([...this.activeReservations.values()].map((entry) => entry.model.modelKey)),
        });
        if (victims.length === 0) {
          this.emit("model_load_rejected", model, reservation.bytes, { reason: "no_evictable_models" });
          throw this.insufficient("no_evictable_models", requested, memory.available, reservationId, 0);
        }
        const result = await this.dependencies.lifecycle.tryEvictIdle(victims[0]!);
        if (result === "changed") continue;
        evictedModelKeys.push(victims[0]!.key);
        this.emit("model_evicted_for_load", model, reservation.bytes, { reason: "memory_admission" });
        reason = "within_budget_after_eviction";
        memory = await this.dependencies.memorySnapshot();
      }

      const decision: LoadAdmissionDecision = Object.freeze({
        admitted: true,
        reason,
        requested,
        available: memory.available,
        reservedBytes: this.reservedBytesExcept(reservationId),
        headroomBytes: this.totalHeadroomBytes(),
        evictedModelKeys: Object.freeze([...evictedModelKeys]),
        decidedAtMs: this.now(),
      });
      this.dependencies.onDecision?.(decision, model);
      this.dependencies.lifecycle.setResidencyTransition(model.modelKey, "loading");
      this.emit("model_load_started", model, reservation.bytes);
      loadStarted = true;
      loaded = await operation.performLoad();
      await operation.stabilize?.(loaded);
      const observed = await operation.observeRss?.(loaded);
      if (observed !== undefined) this.history.update(model, observed);
      reservation = Object.freeze({ ...reservation, state: "committed" });
      this.activeReservations.set(reservationId, reservation);
      this.emit("model_load_committed", model, reservation.bytes, {
        reason: decision.reason, estimateBytes: requested.bytes ?? undefined, observedRssBytes: observed,
      });
      return { value: loaded, reservation, decision };
    } catch (error) {
      if (!(error instanceof InsufficientModelMemoryError)) {
        this.emit("model_load_rolled_back", model, reservation.bytes, { reason: "load_failure" });
      }
      if (loadStarted) await operation.shutdownPartial?.(loaded, error);
      throw error;
    } finally {
      if (loadStarted) this.dependencies.lifecycle.clearResidencyTransition(model.modelKey);
      this.activeReservations.delete(reservationId);
    }
  }

  private fits(available: MemoryValue, requestedBytes: number, ownReservationId: string): boolean {
    if (available.source !== "measured") return false;
    const committed = saturatingAddMemoryBytes(
      requestedBytes,
      this.reservedBytesExcept(ownReservationId),
      this.totalHeadroomBytes(),
    );
    return available.bytes >= committed;
  }

  private insufficient(
    reason: "memory_state_unavailable" | "no_evictable_models" | "insufficient_available_memory",
    requested: MemoryValue,
    available: MemoryValue,
    ownReservationId: string,
    evictableModelCount: number,
  ): InsufficientModelMemoryError {
    return new InsufficientModelMemoryError({
      reason,
      requestedBytes: requested.bytes,
      availableBytes: available.bytes,
      reservedBytes: this.reservedBytesExcept(ownReservationId),
      headroomBytes: this.totalHeadroomBytes(),
      evictableModelCount,
    });
  }

  private reservedBytesExcept(reservationId: string): number {
    return saturatingAddMemoryBytes(...[...this.activeReservations.values()]
      .filter((reservation) => reservation.reservationId !== reservationId && reservation.state === "held")
      .map((reservation) => reservation.bytes));
  }

  private emit(type: ResidencyCoordinatorEventType, model: ResidencyModelDescriptor, reservationBytes: number,
    extra: Partial<Omit<ResidencyCoordinatorEvent, "type" | "backend" | "reservationBytes" | "activeReservations">> = {}): void {
    const terminal = type === "model_load_committed" || type === "model_load_rolled_back" || type === "model_load_rejected";
    this.dependencies.onEvent?.(Object.freeze({
      type, backend: model.backend, reservationBytes,
      activeReservations: Math.max(0, this.activeReservations.size - (terminal ? 1 : 0)), ...extra,
    }));
  }

  private totalHeadroomBytes(): number {
    return Math.max(
      this.policy.minimumHeadroomBytes,
      saturatingAddMemoryBytes(this.osHeadroomBytes, this.runtimeHeadroomBytes),
    );
  }

  private async withMutex<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.mutexTail;
    let release!: () => void;
    this.mutexTail = new Promise<void>((resolve) => { release = resolve; });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}

function parseHeadroom(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function normalizeHeadroom(value: number): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError("headroom must be finite and nonnegative");
  return Math.min(MAX_MEMORY_BYTES, Math.ceil(value));
}

export function measuredResidencyMemory(availableBytes: number, physicalMemoryBytes: number): ResidencyMemorySnapshot {
  return {
    available: availableBytes === 0 ? unavailableMemory("not_reported") : measuredMemory(availableBytes, "os_available"),
    physicalMemoryBytes,
  };
}
