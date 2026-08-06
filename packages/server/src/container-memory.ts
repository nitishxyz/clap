import { freemem, totalmem } from "node:os";
import { readFileSync } from "node:fs";

/**
 * Memory limits as seen from inside a container.
 *
 * `os.totalmem()` reads /proc/meminfo on Linux, which reports the *host*'s
 * memory even when the process runs under a cgroup limit. On a container with
 * 234 GiB allocated from a 2 TiB host that is an 8.6x overestimate, so memory
 * admission would happily accept models that cannot fit and the kernel OOM
 * killer would stop the container instead of clap refusing the load with a
 * clear reason.
 *
 * Read the cgroup limit when present and clamp against it. Both cgroup v2
 * (memory.max / memory.current) and v1 (memory.limit_in_bytes / usage_in_bytes)
 * are supported. A cgroup that reports no limit uses a sentinel far above any
 * real allocation, so anything at or above the host's own total is ignored.
 */

const CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max";
const CGROUP_V2_USAGE = "/sys/fs/cgroup/memory.current";
const CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
const CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes";

function readNumber(path: string): number | undefined {
  try {
    const text = readFileSync(path, "utf8").trim();
    if (!text || text === "max") return undefined;
    const value = Number(text);
    return Number.isFinite(value) && value > 0 ? value : undefined;
  } catch {
    return undefined;
  }
}

/** Cgroup memory ceiling in bytes, or undefined when unlimited/unavailable. */
export function cgroupMemoryLimitBytes(): number | undefined {
  if (process.platform !== "linux") return undefined;
  const limit = readNumber(CGROUP_V2_LIMIT) ?? readNumber(CGROUP_V1_LIMIT);
  if (limit === undefined) return undefined;
  // An unlimited cgroup reports a sentinel (commonly PAGE_COUNTER_MAX scaled
  // to bytes) that exceeds host memory; treat that as "no container limit".
  return limit < totalmem() ? limit : undefined;
}

function cgroupUsageBytes(): number | undefined {
  if (process.platform !== "linux") return undefined;
  return readNumber(CGROUP_V2_USAGE) ?? readNumber(CGROUP_V1_USAGE);
}

/**
 * Total memory this process may actually use: the cgroup limit when one
 * applies, otherwise physical memory.
 */
export function effectiveTotalMemoryBytes(): number {
  const limit = cgroupMemoryLimitBytes();
  const physical = totalmem();
  return limit === undefined ? physical : Math.min(limit, physical);
}

/**
 * Free memory within the effective limit. Inside a constrained cgroup the
 * host's free memory is meaningless, so derive it from cgroup usage; the
 * host figure is still an upper bound because the host can be full while the
 * cgroup is not.
 */
export function effectiveFreeMemoryBytes(): number {
  const limit = cgroupMemoryLimitBytes();
  const hostFree = freemem();
  if (limit === undefined) return hostFree;
  const usage = cgroupUsageBytes();
  const cgroupFree = usage === undefined ? limit : Math.max(0, limit - usage);
  return Math.min(cgroupFree, hostFree);
}
