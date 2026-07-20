export type MemoryPressure = "normal" | "warning" | "critical";

export type ActiveLimitWorker = {
  key: string;
  mode: "auto" | "fixed";
  requestedMax: number;
  currentMax: number;
  backendCeiling: number;
  hardwareCeiling: number;
  modelCeiling: number;
  retainedCeiling: number;
  perActiveReserveBytes: number;
  residentBytes: number;
  retainedBytes: number;
  retainedBudgetBytes: number;
  recentRetainedGrowthBytes: number;
  growthMinimumBytes?: number;
  growthReservePercent?: number;
};

export type GlobalActiveLimitInput = {
  physicalMemoryBytes: number;
  osReserveBytes: number;
  pressure: MemoryPressure;
  workers: ActiveLimitWorker[];
};

export type ActiveLimitPlan = {
  key: string;
  selectedMax: number;
  previousMax: number;
  retainedBytes: number;
  retainedGrowthReserveBytes: number;
  globalResidentBytes: number;
  pressure: MemoryPressure;
  reason: string;
  limitingReason: string;
};

const MIB = 1024 * 1024;

export function classifyMemoryPressure(availableBytes: number, physicalBytes: number,
  previous: MemoryPressure = "normal"): MemoryPressure {
  if (physicalBytes <= 0) return "critical";
  const availablePercent = Math.max(0, availableBytes) / physicalBytes;
  if (availablePercent <= 0.08) return "critical";
  if (availablePercent <= 0.15) return "warning";
  // Require extra recovery headroom before lowering pressure state.
  if (previous === "critical" && availablePercent < 0.12) return "critical";
  if (previous !== "normal" && availablePercent < 0.20) return "warning";
  return "normal";
}

export function retainedGrowthReserve(worker: ActiveLimitWorker): number {
  const remaining = Math.max(0, worker.retainedBudgetBytes - worker.retainedBytes);
  const percent = Math.max(0, Math.min(100, worker.growthReservePercent ?? 10));
  const fraction = Math.floor(remaining / 100) * percent;
  return Math.min(remaining, Math.max(
    worker.recentRetainedGrowthBytes,
    worker.growthMinimumBytes ?? 64 * MIB,
    fraction,
  ));
}

export function selectGlobalActiveLimits(input: GlobalActiveLimitInput): ActiveLimitPlan[] {
  if (input.workers.length === 0) return [];
  const globalResidentBytes = input.workers.reduce((sum, worker) =>
    sum + Math.max(worker.residentBytes, worker.retainedBytes), 0);
  const growth = new Map(input.workers.map((worker) => [worker.key, retainedGrowthReserve(worker)]));
  const growthTotal = [...growth.values()].reduce((sum, value) => sum + value, 0);
  const pressureReserve = input.pressure === "critical" ? Math.floor(input.physicalMemoryBytes * 0.1)
    : input.pressure === "warning" ? Math.floor(input.physicalMemoryBytes * 0.05) : 0;
  const activePool = Math.max(0, input.physicalMemoryBytes - input.osReserveBytes
    - pressureReserve - globalResidentBytes - growthTotal);
  const share = Math.floor(activePool / input.workers.length);

  return input.workers.map((worker) => {
    const memoryCeiling = Math.max(1, Math.floor(share / Math.max(1, worker.perActiveReserveBytes)));
    const safeCeiling = Math.max(1, Math.min(worker.backendCeiling, worker.hardwareCeiling,
      worker.modelCeiling, worker.retainedCeiling, memoryCeiling));
    const target = worker.mode === "fixed" ? Math.min(worker.requestedMax, safeCeiling)
      : Math.min(8, safeCeiling);
    return {
      key: worker.key,
      selectedMax: target,
      previousMax: worker.currentMax,
      retainedBytes: worker.retainedBytes,
      retainedGrowthReserveBytes: growth.get(worker.key)!,
      globalResidentBytes,
      pressure: input.pressure,
      reason: input.pressure === "critical" && target < worker.currentMax ? "critical_pressure"
        : target < worker.currentMax ? "global_memory_ceiling"
        : target > worker.currentMax ? "global_headroom_available" : "unchanged",
      limitingReason: target === memoryCeiling ? "global_memory_ceiling"
        : target === worker.modelCeiling ? "model_ceiling"
        : target === worker.hardwareCeiling ? "hardware_ceiling"
        : target === worker.retainedCeiling ? "retained_ceiling"
        : target === worker.backendCeiling ? "backend_ceiling"
        : worker.mode === "fixed" ? "explicit_override" : "bounded_backend_default",
    };
  });
}

export function shouldAdjustActiveLimit(plan: ActiveLimitPlan, nowMs: number,
  lastAdjustmentMs?: number, cooldownMs = 30_000): boolean {
  if (plan.selectedMax === plan.previousMax) return false;
  if (plan.pressure === "critical" && plan.selectedMax < plan.previousMax) return true;
  if (Math.abs(plan.selectedMax - plan.previousMax) < 2) return false;
  return lastAdjustmentMs === undefined || nowMs - lastAdjustmentMs >= cooldownMs;
}
