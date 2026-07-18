import type { ReactNode } from "react";
import type { DashboardData, DashboardGpu, DashboardLoadedModel } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import { Panel } from "./Shared";

function toneFor(pct: number): string {
  if (pct >= 85) return "bg-err";
  if (pct >= 60) return "bg-warn";
  return "bg-ok";
}

export function BoxBar({ pct, segments = 20, tone, className }: { pct: number; segments?: number; tone?: string; className?: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const filled = clamped > 0 ? Math.max(1, Math.round((clamped / 100) * segments)) : 0;
  const fill = tone ?? toneFor(clamped);
  return (
    <span
      className={`flex h-2.5 items-stretch gap-px ${className ?? ""}`}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped)}
    >
      {Array.from({ length: segments }, (_, index) => (
        <i
          key={index}
          className={`min-w-0 flex-1 transition-colors duration-500 ${index < filled ? fill : "bg-panel-strong"}`}
        />
      ))}
    </span>
  );
}

// Every row shares this template so all bars start and end on the same
// columns: fixed label, flexible bar, fixed right-aligned value.
const ROW = "grid grid-cols-[3.75rem_minmax(0,1fr)_minmax(8rem,auto)] items-center gap-x-3 px-3 py-[3px]";

function Row({ label, pct, value, tone }: { label: string; pct?: number; value?: string; tone?: string }) {
  const na = pct === undefined && value === undefined;
  return (
    <div className={ROW}>
      <span className="truncate text-[0.68rem] uppercase tracking-[0.06em] text-muted">{label}</span>
      <BoxBar pct={pct ?? 0} tone={tone} className={pct === undefined ? "opacity-35" : ""} />
      <span className={`truncate text-right text-[0.72rem] tabular-nums ${na || pct === undefined ? "text-muted" : ""}`}>
        {value ?? "n/a"}
      </span>
    </div>
  );
}

function Section({ title, aside, children }: { title: ReactNode; aside?: ReactNode; children: ReactNode }) {
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-3 px-3 pb-1">
        <span className="shrink-0 text-[0.66rem] uppercase tracking-[0.08em] text-muted">{title}</span>
        {aside ? <span className="truncate text-[0.68rem] text-muted">{aside}</span> : null}
      </div>
      {children}
    </div>
  );
}

function GpuSection({ gpu }: { gpu: DashboardGpu }) {
  const util = gpu.utilizationPercent;
  const used = gpu.memoryUsedBytes;
  const total = gpu.memoryTotalBytes;
  const hasVram = used !== undefined && total !== undefined && total > 0;
  return (
    <Section title="gpu" aside={`${gpu.name} (${gpu.vendor})`}>
      <Row label="util" pct={util} value={util !== undefined ? `${util.toFixed(0)}%` : undefined} />
      <Row
        label="vram"
        pct={hasVram ? (used / total) * 100 : undefined}
        tone="bg-cache"
        value={hasVram ? `${fmtBytes(used)} / ${fmtBytes(total)}` : undefined}
      />
    </Section>
  );
}

function WorkerSection({ entry, systemMemoryBytes, cpuCount, gpuTotalBytes }: { entry: DashboardLoadedModel; systemMemoryBytes?: number; cpuCount?: number; gpuTotalBytes?: number }) {
  const usage = entry.usage;
  const cpuPct = usage ? (cpuCount ? usage.cpuPercent / cpuCount : Math.min(100, usage.cpuPercent)) : undefined;
  const rssPct = usage && systemMemoryBytes ? (usage.rssBytes / systemMemoryBytes) * 100 : undefined;
  const gpuBytes = entry.gpuMemoryBytes;
  const gpuPct = gpuBytes !== undefined && gpuTotalBytes ? (gpuBytes / gpuTotalBytes) * 100 : undefined;
  return (
    <Section title="worker" aside={<span title={entry.localPath}>{entry.id} · pid {entry.worker?.pid ?? "-"}</span>}>
      <Row label="cpu" pct={cpuPct} value={usage ? `${usage.cpuPercent.toFixed(0)}%` : undefined} />
      <Row label="mem" pct={rssPct} value={usage ? fmtBytes(usage.rssBytes) : undefined} />
      {gpuBytes !== undefined ? (
        <Row label="vram" pct={gpuPct} tone="bg-cache" value={fmtBytes(gpuBytes)} />
      ) : null}
    </Section>
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
  const workers = data.loaded.filter((entry) => entry.usage || entry.gpuMemoryBytes !== undefined);
  const systemAside = [
    server.rssBytes !== undefined ? `server rss ${fmtBytes(server.rssBytes)}` : null,
    server.cpuCount ? `${server.cpuCount} cores` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <Panel title="usage" count={queue ? `${queue.inflight}/${queue.maxInflight} inflight` : ""}>
      <div className="grid divide-y divide-soft-border lg:grid-cols-2 lg:divide-x lg:divide-y-0">
        <div className="pb-1">
          <Section title="system" aside={systemAside}>
            <Row label="cpu" pct={sysCpu} value={sysCpu !== undefined ? `${sysCpu.toFixed(0)}%` : undefined} />
            <Row
              label="mem"
              pct={memPct}
              value={memUsed !== undefined && memTotal ? `${fmtBytes(memUsed)} / ${fmtBytes(memTotal)}` : undefined}
            />
          </Section>
          <Section title="queue" aside={queue ? `${queue.inflight + queue.queued} in system` : undefined}>
            <Row
              label="inflight"
              pct={queue && queue.maxInflight ? (queue.inflight / queue.maxInflight) * 100 : undefined}
              tone="bg-accent"
              value={queue ? `${queue.inflight} / ${queue.maxInflight}` : undefined}
            />
            <Row
              label="waiting"
              pct={queue && queue.queueDepth ? (queue.queued / queue.queueDepth) * 100 : undefined}
              tone={queue?.queued ? "bg-warn" : "bg-accent"}
              value={queue ? `${queue.queued} / ${queue.queueDepth}` : undefined}
            />
          </Section>
        </div>
        <div className="pb-1">
          {gpus.length ? (
            gpus.map((gpu, index) => <GpuSection key={`${gpu.vendor}-${gpu.name}-${index}`} gpu={gpu} />)
          ) : (
            <Section title="gpu" aside="no telemetry">
              <Row label="util" />
              <Row label="vram" />
            </Section>
          )}
          {workers.map((entry) => (
            <WorkerSection
              key={entry.key}
              entry={entry}
              systemMemoryBytes={memTotal}
              cpuCount={server.cpuCount}
              gpuTotalBytes={gpuTotal}
            />
          ))}
          {workers.length === 0 ? (
            <Section title="worker" aside="none resident">
              <Row label="cpu" />
              <Row label="mem" />
            </Section>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}
