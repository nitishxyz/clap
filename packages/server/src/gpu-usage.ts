// Best-effort GPU telemetry for the dashboard. NVIDIA via nvidia-smi (Linux
// and Windows); Apple Silicon via the IORegistry accelerator node. Absence of
// a GPU is normal — consumers must treat the result as optional.

import { systemMemoryBytes, systemMemoryUsedBytes } from "./process-usage";

export type GpuUsage = {
  vendor: "nvidia" | "apple";
  name: string;
  utilizationPercent?: number;
  memoryUsedBytes?: number;
  memoryTotalBytes?: number;
  processes?: Array<{ pid: number; memoryBytes: number }>;
};

let cache: { at: number; gpus: GpuUsage[] } | undefined;
let unavailable = false;

export async function sampleGpuUsage(): Promise<GpuUsage[]> {
  if (unavailable) return [];
  if (cache && Date.now() - cache.at < 1000) return cache.gpus;
  const gpus = process.platform === "darwin" ? await sampleAppleGpu() : await sampleNvidia();
  if (!gpus.length && !cache) unavailable = true;
  cache = { at: Date.now(), gpus };
  return gpus;
}

async function run(command: string[]): Promise<string | undefined> {
  try {
    const proc = Bun.spawn(command, { stdout: "pipe", stderr: "ignore" });
    const text = await new Response(proc.stdout).text();
    const code = await proc.exited;
    return code === 0 ? text : undefined;
  } catch {
    return undefined;
  }
}

async function sampleNvidia(): Promise<GpuUsage[]> {
  const query = await run([
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
  ]);
  if (!query) return [];
  const gpus: GpuUsage[] = [];
  for (const line of query.trim().split("\n")) {
    const [name, util, used, total] = line.split(",").map((part) => part.trim());
    if (!name) continue;
    gpus.push({
      vendor: "nvidia",
      name,
      utilizationPercent: toNumber(util),
      memoryUsedBytes: mibToBytes(toNumber(used)),
      memoryTotalBytes: mibToBytes(toNumber(total)),
    });
  }
  if (gpus.length) {
    const procs = await run([
      "nvidia-smi",
      "--query-compute-apps=pid,used_memory",
      "--format=csv,noheader,nounits",
    ]);
    if (procs) {
      const processes: Array<{ pid: number; memoryBytes: number }> = [];
      for (const line of procs.trim().split("\n")) {
        const [pid, memory] = line.split(",").map((part) => part.trim());
        const pidNumber = toNumber(pid);
        const memoryBytes = mibToBytes(toNumber(memory));
        if (pidNumber !== undefined && memoryBytes !== undefined) {
          processes.push({ pid: pidNumber, memoryBytes });
        }
      }
      if (processes.length && gpus[0]) gpus[0].processes = processes;
    }
  }
  return gpus;
}

async function sampleAppleGpu(): Promise<GpuUsage[]> {
  if (process.arch !== "arm64") return [];
  // Apple exposes GPU load in the IORegistry accelerator node
  // (PerformanceStatistics -> "Device Utilization %"), readable without
  // privileges. This is the same signal Activity Monitor's GPU history uses;
  // btop reads the equivalent data via the private IOReport framework.
  const [profile, ioreg] = await Promise.all([
    run(["sysctl", "-n", "machdep.cpu.brand_string"]),
    run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]),
  ]);
  const name = profile?.trim() || "Apple Silicon";
  const utilization = ioreg?.match(/"Device Utilization %"=(\d+)/)?.[1];
  const gpu: GpuUsage = { vendor: "apple", name };
  if (utilization !== undefined) {
    gpu.utilizationPercent = Math.min(100, Number(utilization));
    // Unified memory: the GPU shares system RAM, so report system used/total
    // as the memory pressure signal (what Activity Monitor implies too).
    gpu.memoryUsedBytes = await systemMemoryUsedBytes();
    gpu.memoryTotalBytes = systemMemoryBytes();
  }
  return [gpu];
}

function toNumber(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function mibToBytes(value: number | undefined): number | undefined {
  return value === undefined ? undefined : value * 1024 * 1024;
}
