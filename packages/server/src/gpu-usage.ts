// Best-effort GPU telemetry for the dashboard. NVIDIA via nvidia-smi (Linux
// and Windows); Apple Silicon has no unprivileged utilization API, so macOS
// reports availability only when a supported tool exists. Absence of a GPU is
// normal — consumers must treat the result as optional.

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
  // No unprivileged utilization API on macOS; report the device so the UI can
  // show unified-memory pressure via system memory instead of GPU util.
  const profile = await run(["sysctl", "-n", "machdep.cpu.brand_string"]);
  const name = profile?.trim() || "Apple Silicon";
  return [{ vendor: "apple", name }];
}

function toNumber(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function mibToBytes(value: number | undefined): number | undefined {
  return value === undefined ? undefined : value * 1024 * 1024;
}
