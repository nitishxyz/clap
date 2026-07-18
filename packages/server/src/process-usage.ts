import { cpus, freemem, loadavg, totalmem } from "node:os";

export type ProcessUsage = { rssBytes: number; cpuPercent: number };

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

// System-wide CPU utilization approximation: 1-minute load average over
// cores, capped at 100. Good enough for a dashboard pressure bar.
export function systemCpuPercent(): number {
  const load = loadavg()[0] ?? 0;
  return Math.min(100, Math.round((load / cpuCoreCount()) * 100));
}
