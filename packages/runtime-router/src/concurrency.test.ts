import { describe, expect, test } from "bun:test";
import { classifyMemoryPressure, retainedGrowthReserve, selectGlobalActiveLimits, shouldAdjustActiveLimit } from "./concurrency";

const gib = 1024 ** 3;
const worker = {
  key: "mlx:model",
  mode: "auto" as const,
  requestedMax: 8,
  currentMax: 1,
  backendCeiling: 16,
  hardwareCeiling: 16,
  modelCeiling: 16,
  retainedCeiling: 64,
  perActiveReserveBytes: 512 * 1024 ** 2,
  residentBytes: 8 * gib,
  retainedBytes: 71 * 1024 ** 2,
  retainedBudgetBytes: 1160 * 1024 ** 2,
  recentRetainedGrowthBytes: 16 * 1024 ** 2,
};

describe("global active limit policy", () => {
  test("uses physical retained bytes and a bounded growth reserve", () => {
    const reserve = retainedGrowthReserve(worker);
    expect(reserve).toBe(Math.floor((worker.retainedBudgetBytes - worker.retainedBytes) / 100) * 10);
    expect(reserve).toBeLessThan(worker.retainedBudgetBytes - worker.retainedBytes);
    const [plan] = selectGlobalActiveLimits({
      physicalMemoryBytes: 32 * gib,
      osReserveBytes: 4 * gib,
      pressure: "normal",
      workers: [worker],
    });
    expect(plan!.selectedMax).toBe(8);
    expect(plan!.retainedBytes).toBe(worker.retainedBytes);
  });

  test("coordinates resident workers without double-counting retained model memory", () => {
    const second = { ...worker, key: "llama:model", residentBytes: 7 * gib, retainedBytes: 0 };
    const plans = selectGlobalActiveLimits({
      physicalMemoryBytes: 32 * gib,
      osReserveBytes: 4 * gib,
      pressure: "warning",
      workers: [worker, second],
    });
    expect(plans[0]!.globalResidentBytes).toBe(15 * gib);
    expect(plans[1]!.globalResidentBytes).toBe(15 * gib);
    expect(plans.every((plan) => plan.selectedMax >= 1)).toBe(true);
  });

  test("applies cooldown and meaningful-change hysteresis except critical decreases", () => {
    const base = { ...selectGlobalActiveLimits({
      physicalMemoryBytes: 32 * gib,
      osReserveBytes: 4 * gib,
      pressure: "normal" as const,
      workers: [worker],
    })[0]!, previousMax: 7, selectedMax: 8 };
    expect(shouldAdjustActiveLimit(base, 40_000, 0)).toBe(false);
    expect(shouldAdjustActiveLimit({ ...base, previousMax: 4 }, 20_000, 0)).toBe(false);
    expect(shouldAdjustActiveLimit({ ...base, previousMax: 4 }, 40_000, 0)).toBe(true);
    expect(shouldAdjustActiveLimit({ ...base, pressure: "critical", previousMax: 8, selectedMax: 3 }, 1, 0)).toBe(true);
  });

  test("fixed limits stay fixed until the global safety ceiling clamps them", () => {
    const fixed = { ...worker, mode: "fixed" as const, requestedMax: 6, currentMax: 6 };
    const normal = selectGlobalActiveLimits({ physicalMemoryBytes: 32 * gib, osReserveBytes: 4 * gib,
      pressure: "normal", workers: [fixed] })[0]!;
    const critical = selectGlobalActiveLimits({ physicalMemoryBytes: 12 * gib, osReserveBytes: 3 * gib,
      pressure: "critical", workers: [fixed] })[0]!;
    expect(normal.selectedMax).toBe(6);
    expect(critical.selectedMax).toBeLessThan(6);
  });

  test("classifies pressure with recovery hysteresis", () => {
    expect(classifyMemoryPressure(2 * gib, 32 * gib)).toBe("critical");
    expect(classifyMemoryPressure(4 * gib, 32 * gib)).toBe("warning");
    expect(classifyMemoryPressure(5 * gib, 32 * gib, "warning")).toBe("warning");
    expect(classifyMemoryPressure(7 * gib, 32 * gib, "warning")).toBe("normal");
  });
});
