import { totalmem } from "node:os";

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
