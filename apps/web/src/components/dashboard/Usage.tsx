import { useEffect, useRef, useState, type ReactNode } from "react";
import type { DashboardData, DashboardGpu, DashboardLoadedModel } from "@/lib/api";
import { fmtBytes } from "@/lib/format";

const SEGMENTS = 16;
const SWEEP_MS_PER_SEGMENT = 30;

function toneFor(pct: number): string {
  if (pct >= 85) return "bg-err";
  if (pct >= 60) return "bg-warn";
  return "bg-ok";
}

// VU-meter sweep: step the displayed segment count one at a time toward the
// target, retargeting from wherever the sweep currently is when a new value
// arrives mid-animation.
function useSweep(target: number): { displayed: number; rising: boolean } {
  const [displayed, setDisplayed] = useState(target);
  const displayedRef = useRef(target);
  const risingRef = useRef(false);
  displayedRef.current = displayed;

  useEffect(() => {
    if (displayedRef.current === target) {
      risingRef.current = false;
      return;
    }
    risingRef.current = target > displayedRef.current;
    const timer = setInterval(() => {
      setDisplayed((current) => {
        const next = current + Math.sign(target - current);
        if (next === target) clearInterval(timer);
        return next;
      });
    }, SWEEP_MS_PER_SEGMENT);
    return () => clearInterval(timer);
  }, [target]);

  return { displayed, rising: risingRef.current && displayed !== target };
}

export function BoxBar({ pct, segments = SEGMENTS, tone, className }: { pct?: number; segments?: number; tone?: string; className?: string }) {
  const na = pct === undefined;
  const clamped = Math.max(0, Math.min(100, pct ?? 0));
  const target = clamped > 0 ? Math.max(1, Math.round((clamped / 100) * segments)) : 0;
  const { displayed, rising } = useSweep(target);
  const filled = Math.max(0, Math.min(segments, displayed));
  // Color thresholds follow the displayed level so the meter passes through
  // green/amber on its way up, like an analog VU meter.
  const displayedPct = (filled / segments) * 100;
  const fill = tone ?? toneFor(displayedPct);
  return (
    <span
      className={`flex h-2.5 items-stretch gap-px ${na ? "opacity-35" : ""} ${className ?? ""}`}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped)}
    >
      {Array.from({ length: segments }, (_, index) => (
        <i
          key={index}
          className={`min-w-0 flex-1 ${index < filled ? `${fill} ${rising && index === filled - 1 ? "brightness-150" : ""}` : "bg-panel-strong"}`}
        />
      ))}
    </span>
  );
}

// Middle-ellipsis so long model ids keep their distinguishing head and tail;
// full value stays available via the title tooltip.
function midTruncate(text: string, max = 40): string {
  if (text.length <= max) return text;
  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${text.slice(0, head)}\u2026${text.slice(-tail)}`;
}

// Compact byte value so the fixed-width value slot never overflows: 14.1G, 21M.
function shortBytes(value: number): string {
  const gib = value / 2 ** 30;
  if (gib >= 10) return `${Math.round(gib)}G`;
  if (gib >= 1) return `${gib.toFixed(1)}G`;
  return `${Math.round(value / 2 ** 20)}M`;
}

// One geometry for every metric row: fixed label, flexible bar, fixed
// right-aligned value slot so nothing shifts as numbers change.
function Row({ label, pct, value, tone, title }: { label: string; pct?: number; value?: string; tone?: string; title?: string }) {
  return (
    <div className="grid grid-cols-[3.5rem_minmax(0,1fr)_5.25rem] items-center gap-x-2 px-3 py-1" title={title}>
      <span className="truncate text-[0.68rem] lowercase text-muted">{label}</span>
      <BoxBar pct={pct} tone={tone} />
      <span className={`truncate text-right text-[0.72rem] tabular-nums ${pct === undefined ? "text-muted" : ""}`}>
        {value ?? "n/a"}
      </span>
    </div>
  );
}

function Card({ title, meta, footer, children }: { title: string; meta?: ReactNode; footer?: ReactNode; children: ReactNode }) {
  return (
    <section className="flex w-full min-w-0 flex-col border border-border bg-panel">
      <div className="flex items-baseline justify-between gap-3 border-b border-border px-3 py-2">
        <span className="shrink-0 text-[0.72rem] uppercase tracking-[0.08em] text-muted">{title}</span>
        {meta ? <span className="truncate text-[0.68rem] text-muted">{meta}</span> : null}
      </div>
      <div className="flex-1 py-1.5">{children}</div>
      {footer ? <div className="truncate px-3 pb-2 text-[0.68rem] text-muted">{footer}</div> : null}
    </section>
  );
}

function GpuRows({ gpu }: { gpu: DashboardGpu }) {
  const util = gpu.utilizationPercent;
  const used = gpu.memoryUsedBytes;
  const total = gpu.memoryTotalBytes;
  const hasVram = used !== undefined && total !== undefined && total > 0;
  return (
    <>
      <Row label="util" pct={util} value={util !== undefined ? `${util.toFixed(0)}%` : undefined} />
      {gpu.vendor === "apple" ? (
        <Row label="memory" value="shared RAM" title="Apple GPU memory is the same unified system RAM shown in the System card, not additional VRAM" />
      ) : (
        <Row
          label="vram"
          pct={hasVram ? (used / total) * 100 : undefined}
          tone="bg-cache"
          value={hasVram ? `${shortBytes(used)}/${shortBytes(total)}` : undefined}
        />
      )}
    </>
  );
}

export function formatWorkerCpu(percent?: number): string | undefined {
  if (percent === undefined) return undefined;
  if (percent > 0 && percent < 1) return "<1%";
  return `${percent.toFixed(0)}%`;
}

export function WorkerMetricCell({ label, pct, value, tone, title }: { label: string; pct?: number; value?: string; tone?: string; title?: string }) {
  const available = value !== undefined;
  return (
    <div className={`min-w-0 border border-soft-border px-2 py-1.5 ${available ? "bg-panel" : "bg-panel/40"}`} title={title} data-worker-metric={label.toLowerCase()}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="truncate text-[0.62rem] uppercase tracking-[0.06em] text-muted">{label}</span>
        <span className={`shrink-0 text-[0.72rem] tabular-nums ${available ? "" : "text-muted"}`}>{value ?? "unavailable"}</span>
      </div>
      <BoxBar pct={pct} tone={tone} className="w-full" />
    </div>
  );
}

export const RSS_TITLE = "Resident Set Size: CPU-visible physical pages currently mapped by the process; Metal allocations can be tracked separately";
export const MLX_ACTIVE_TITLE = "Current live MLX Metal allocations: model weights, retained KV caches, and prefix anchors";
export const MLX_CACHE_TITLE = "Recyclable MLX allocator buffers retained for reuse; cleared whenever the scheduler becomes idle";

export function isUnifiedMlx(backend: string, platform?: string): boolean {
  return backend === "mlx" && platform === "darwin";
}

function WorkerRow({ entry, platform, systemMemoryBytes, cpuCount, gpuTotalBytes }: { entry: DashboardLoadedModel; platform?: string; systemMemoryBytes?: number; cpuCount?: number; gpuTotalBytes?: number }) {
  const usage = entry.usage;
  const unifiedMlx = isUnifiedMlx(entry.backend, platform);
  const mlxMemory = entry.worker.memory;
  const cpuPct = usage ? (cpuCount ? usage.cpuPercent / cpuCount : Math.min(100, usage.cpuPercent)) : undefined;
  const rssPct = usage && systemMemoryBytes ? (usage.rssBytes / systemMemoryBytes) * 100 : undefined;
  const gpuBytes = entry.gpuMemoryBytes;
  const gpuPct = gpuBytes !== undefined && gpuTotalBytes ? (gpuBytes / gpuTotalBytes) * 100 : undefined;
  const acceleratorBytes = unifiedMlx ? mlxMemory?.activeBytes : gpuBytes;
  const acceleratorPct = acceleratorBytes !== undefined && (unifiedMlx ? systemMemoryBytes : gpuTotalBytes)
    ? (acceleratorBytes / (unifiedMlx ? systemMemoryBytes! : gpuTotalBytes!)) * 100
    : undefined;
  const cachePct = mlxMemory && systemMemoryBytes ? (mlxMemory.cacheBytes / systemMemoryBytes) * 100 : undefined;
  return (
    <div className="border-b border-soft-border px-3 py-2 last:border-b-0">
      <div className="mb-1.5 flex min-w-0 items-baseline justify-between gap-3">
        <span className="truncate text-[0.74rem]" title={`${entry.id}\n${entry.localPath}`}>{midTruncate(entry.id)}</span>
        <span className="shrink-0 text-[0.68rem] tabular-nums text-muted">PID {entry.worker?.pid ?? "unavailable"}</span>
      </div>
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 xl:grid-cols-4">
        <WorkerMetricCell label="CPU" pct={cpuPct} value={usage ? formatWorkerCpu(usage.cpuPercent) : undefined} title="Process CPU sampled by ps; values below one percent are shown as <1%" />
        <WorkerMetricCell label="RSS" pct={rssPct} value={usage ? fmtBytes(usage.rssBytes) : undefined} title={RSS_TITLE} />
        <WorkerMetricCell label="Accelerator" pct={acceleratorPct} tone="bg-cache" value={acceleratorBytes !== undefined ? fmtBytes(acceleratorBytes) : undefined} title={unifiedMlx ? MLX_ACTIVE_TITLE : "Dedicated GPU memory used by this worker; unavailable when the platform cannot attribute GPU memory by process"} />
        <WorkerMetricCell label="Reclaimable" pct={cachePct} tone="bg-warn" value={mlxMemory ? fmtBytes(mlxMemory.cacheBytes) : undefined} title={unifiedMlx ? MLX_CACHE_TITLE : "Backend does not expose reclaimable allocator memory"} />
      </div>
    </div>
  );
}

export function UsagePanel({ data }: { data: DashboardData }) {
  const server = data.server;
  const gpus = data.gpus ?? [];
  const queue = data.queue;
  const memUsed = server.systemMemoryUsedBytes;
  const memTotal = server.systemMemoryBytes;
  const memPct = memUsed !== undefined && memTotal ? (memUsed / memTotal) * 100 : undefined;
  const sysCpu = server.systemCpuPercent;
  const gpuTotal = gpus[0]?.memoryTotalBytes;
  const systemFooter = [
    server.rssBytes !== undefined ? `server rss ${fmtBytes(server.rssBytes)}` : null,
    server.cpuCount ? `${server.cpuCount} cores` : null,
    `pid ${server.pid}`,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <>
      <div className="grid grid-cols-1 gap-2 lg:grid-cols-3">
        <Card title="system" footer={systemFooter}>
          <Row label="cpu" pct={sysCpu} value={sysCpu !== undefined ? `${sysCpu.toFixed(0)}%` : undefined} />
          <Row
            label="mem"
            pct={memPct}
            value={memUsed !== undefined && memTotal ? `${shortBytes(memUsed)}/${shortBytes(memTotal)}` : undefined}
          />
        </Card>
        <Card title="gpu" meta={gpus.length ? `${gpus[0].name} (${gpus[0].vendor})` : "no telemetry"}>
          {gpus.length ? (
            gpus.map((gpu, index) => <GpuRows key={`${gpu.vendor}-${gpu.name}-${index}`} gpu={gpu} />)
          ) : (
            <>
              <Row label="util" />
              <Row label="vram" />
            </>
          )}
        </Card>
        <Card title="queue" meta={queue ? `${queue.inflight + queue.queued} in system` : undefined}>
          <Row
            label="inflight"
            pct={queue && queue.maxInflight ? (queue.inflight / queue.maxInflight) * 100 : undefined}
            tone="bg-accent"
            value={queue ? `${queue.inflight}/${queue.maxInflight}` : undefined}
          />
          <Row
            label="waiting"
            pct={queue && queue.queueDepth ? (queue.queued / queue.queueDepth) * 100 : undefined}
            tone={queue?.queued ? "bg-warn" : "bg-accent"}
            value={queue ? `${queue.queued}/${queue.queueDepth}` : undefined}
          />
        </Card>
      </div>
      <Card title="workers" meta={data.loaded.length ? `${data.loaded.length} resident` : undefined}>
        {data.loaded.length ? (
          <div className="max-h-72 overflow-y-auto">
            {data.loaded.map((entry) => (
              <WorkerRow
                key={entry.key}
                entry={entry}
                platform={server.platform}
                systemMemoryBytes={memTotal}
                cpuCount={server.cpuCount}
                gpuTotalBytes={gpuTotal}
              />
            ))}
          </div>
        ) : (
          <div className="px-3 py-1 text-[0.72rem] text-muted">no resident workers</div>
        )}
      </Card>
    </>
  );
}
