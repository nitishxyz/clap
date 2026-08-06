import { describe, expect, test } from "bun:test";
import { totalmem } from "node:os";
import { cgroupMemoryLimitBytes, effectiveTotalMemoryBytes } from "./container-memory";
import { systemMemoryBytes } from "./process-usage";

describe("container memory limits", () => {
  test("effective total never exceeds physical memory", () => {
    // A cgroup limit above host memory (the "unlimited" sentinel) must not
    // inflate the reported total.
    expect(effectiveTotalMemoryBytes()).toBeLessThanOrEqual(totalmem());
    expect(effectiveTotalMemoryBytes()).toBeGreaterThan(0);
  });

  test("systemMemoryBytes reports the effective limit, not raw host memory", () => {
    // Memory admission sizes model loads from this number. On a container it
    // must be the cgroup ceiling, otherwise clap admits models that cannot fit
    // and the kernel OOM killer stops the container instead of clap refusing
    // the load with a reason.
    expect(systemMemoryBytes()).toBe(effectiveTotalMemoryBytes());
  });

  test("a detected cgroup limit is strictly below host memory", () => {
    const limit = cgroupMemoryLimitBytes();
    // Undefined off Linux and on unconstrained hosts; when present it must be
    // a real constraint rather than the unlimited sentinel.
    if (limit !== undefined) {
      expect(limit).toBeGreaterThan(0);
      expect(limit).toBeLessThan(totalmem());
    }
  });

  test("non-linux platforms report physical memory unchanged", () => {
    if (process.platform !== "linux") {
      expect(cgroupMemoryLimitBytes()).toBeUndefined();
      expect(effectiveTotalMemoryBytes()).toBe(totalmem());
    }
  });
});
