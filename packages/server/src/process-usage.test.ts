import { describe, expect, test } from "bun:test";
import { cpuPercentBetween, createSystemCpuPercentSampler, type CpuTimes } from "./process-usage";

describe("system CPU accounting", () => {
  test("reports utilization from aggregate CPU tick deltas", () => {
    expect(cpuPercentBetween(
      { cores: 2, idle: 1_000, total: 4_000 },
      { cores: 2, idle: 1_600, total: 5_000 },
    )).toBe(40);
  });

  test("rejects invalid samples instead of showing false spikes", () => {
    expect(cpuPercentBetween(
      { cores: 2, idle: 10, total: 10 },
      { cores: 2, idle: 10, total: 10 },
    )).toBeUndefined();
    expect(cpuPercentBetween(
      { cores: 2, idle: 10, total: 20 },
      { cores: 1, idle: 15, total: 30 },
    )).toBeUndefined();
    expect(cpuPercentBetween(
      { cores: 2, idle: 10, total: 20 },
      { cores: 2, idle: 5, total: 30 },
    )).toBeUndefined();
  });

  test("establishes and resets the sampling baseline", () => {
    const snapshots: CpuTimes[] = [
      { cores: 2, idle: 1_000, total: 4_000 },
      { cores: 2, idle: 1_600, total: 5_000 },
      { cores: 1, idle: 1_800, total: 5_400 },
      { cores: 1, idle: 2_000, total: 5_800 },
    ];
    const sample = createSystemCpuPercentSampler(() => snapshots.shift()!);

    expect(sample()).toBeUndefined();
    expect(sample()).toBe(40);
    expect(sample()).toBeUndefined();
    expect(sample()).toBe(50);
  });
});
