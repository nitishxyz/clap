import { cpus, freemem, totalmem } from "node:os";

export type ProcessUsage = { rssBytes: number; cpuPercent: number };
export type SystemMemorySnapshot = { physicalBytes: number; usedBytes: number; availableBytes: number };

let cache: { at: number; usage: Map<number, ProcessUsage> } | undefined;

// One ps invocation per second at most; the dashboard polls every 2s.
export async function sampleProcessUsage(pids: number[]): Promise<Map<number, ProcessUsage>> {
  const unique = [...new Set(pids)].filter((pid) => Number.isInteger(pid) && pid > 0);
  if (!unique.length) return new Map();
  if (cache && Date.now() - cache.at < 1000 && unique.every((pid) => cache!.usage.has(pid))) {
    return cache.usage;
  }
  const usage = new Map<number, ProcessUsage>();
  try {
    const proc = Bun.spawn(["ps", "-o", "pid=,rss=,%cpu=", "-p", unique.join(",")], {
      stdout: "pipe",
      stderr: "ignore",
    });
    const text = await new Response(proc.stdout).text();
    await proc.exited;
    for (const line of text.split("\n")) {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 3) continue;
      const pid = Number(parts[0]);
      const rssKb = Number(parts[1]);
      const cpu = Number(parts[2]);
      if (!Number.isFinite(pid) || !Number.isFinite(rssKb)) continue;
      usage.set(pid, { rssBytes: rssKb * 1024, cpuPercent: Number.isFinite(cpu) ? cpu : 0 });
    }
  } catch {
    // ps unavailable; return what we have
  }
  cache = { at: Date.now(), usage };
  return usage;
}

export function systemMemoryBytes(): number {
  return totalmem();
}

export async function systemMemorySnapshot(): Promise<SystemMemorySnapshot> {
  const physicalBytes = systemMemoryBytes();
  const usedBytes = Math.min(physicalBytes, Math.max(0, await systemMemoryUsedBytes()));
  return { physicalBytes, usedBytes, availableBytes: Math.max(0, physicalBytes - usedBytes) };
}

export async function processRssBytes(pid: number): Promise<number | undefined> {
  if (!Number.isInteger(pid) || pid <= 0) return undefined;
  const rssBytes = (await sampleProcessUsage([pid])).get(pid)?.rssBytes;
  return rssBytes !== undefined && Number.isFinite(rssBytes) && rssBytes > 0 ? rssBytes : undefined;
}

let memCache: { at: number; used: number } | undefined;

// macOS keeps RAM filled with file cache, so totalmem-freemem reads ~100%
// forever. Use vm_stat pressure accounting (anonymous + wired + compressed,
// Activity Monitor's "Memory Used") on darwin; total-free elsewhere.
export async function systemMemoryUsedBytes(): Promise<number> {
  if (memCache && Date.now() - memCache.at < 1000) return memCache.used;
  let used = Math.max(0, totalmem() - freemem());
  if (process.platform === "darwin") {
    try {
      const proc = Bun.spawn(["vm_stat"], { stdout: "pipe", stderr: "ignore" });
      const text = await new Response(proc.stdout).text();
      await proc.exited;
      const pageSize = Number(text.match(/page size of (\d+) bytes/)?.[1] ?? 16384);
      const pages = (label: string) => Number(text.match(new RegExp(`${label}:\\s+(\\d+)`))?.[1] ?? 0);
      const anonymous = pages("Anonymous pages") - pages("Pages purgeable");
      const computed = (Math.max(0, anonymous) + pages("Pages wired down") + pages("Pages occupied by compressor")) * pageSize;
      if (computed > 0) used = Math.min(computed, totalmem());
    } catch {
      // fall back to total - free
    }
  }
  memCache = { at: Date.now(), used };
  return used;
}

export function cpuCoreCount(): number {
  return Math.max(1, cpus().length);
}

export type CpuTimes = { cores: number; idle: number; total: number };

function cpuTimes(): CpuTimes {
  let idle = 0;
  let total = 0;
  const processors = cpus();
  for (const cpu of processors) {
    idle += cpu.times.idle;
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq;
  }
  return { cores: processors.length, idle, total };
}

export function cpuPercentBetween(previous: CpuTimes, current: CpuTimes): number | undefined {
  const total = current.total - previous.total;
  const idle = current.idle - previous.idle;
  if (current.cores < 1 || current.cores !== previous.cores || total <= 0 || idle < 0 || idle > total) return undefined;
  return Math.max(0, Math.min(100, Math.round((1 - idle / total) * 100)));
}

export function createSystemCpuPercentSampler(read: () => CpuTimes = cpuTimes): () => number | undefined {
  let previous: CpuTimes | undefined;
  return () => {
    const current = read();
    if (!previous) {
      previous = current;
      return undefined;
    }
    const percent = cpuPercentBetween(previous, current);
    previous = current;
    return percent;
  };
}

const sampleSystemCpuPercent = createSystemCpuPercentSampler();
let cpuCache: { at: number; percent: number | undefined } | undefined;

// Aggregate CPU tick deltas across every core. Unlike load average, this is
// actual utilization during the dashboard polling interval and does not count
// runnable or blocked processes as if they were consuming CPU.
export function systemCpuPercent(): number | undefined {
  if (cpuCache && Date.now() - cpuCache.at < 500) return cpuCache.percent;
  const percent = sampleSystemCpuPercent();
  cpuCache = { at: Date.now(), percent };
  return percent;
}
