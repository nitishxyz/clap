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

function GroupLabel({ children }: { children: ReactNode }) {
  return <div className="px-3 pt-2 pb-0.5 text-[0.66rem] uppercase tracking-[0.08em] text-muted">{children}</div>;
}

function UsageRow({ label, pct, value, tone }: { label: string; pct?: number; value: string; tone?: string }) {
  return (
    <div className="grid grid-cols-[4.5rem_minmax(0,1fr)_auto] items-center gap-2 px-3 py-1">
      <span className="truncate text-[0.7rem] uppercase tracking-[0.06em] text-muted">{label}</span>
      {pct === undefined ? <span className="text-[0.72rem] text-muted">·</span> : <BoxBar pct={pct} tone={tone} />}
      <span className="min-w-[5.5rem] text-right text-[0.74rem] tabular-nums">{value}</span>
    </div>
  );
}

function MiniBar({ label, pct, value, tone }: { label: string; pct?: number; value: string; tone?: string }) {
  return (
    <span className="flex min-w-0 items-center gap-2 text-[0.72rem]">
      <span className="w-9 shrink-0 uppercase tracking-[0.04em] text-muted">{label}</span>
      {pct === undefined ? (
        <span className="min-w-0 flex-1 text-muted">·</span>
      ) : (
        <BoxBar pct={pct} segments={12} tone={tone} className="min-w-0 flex-1" />
      )}
      <span className="shrink-0 tabular-nums text-muted">{value}</span>
    </span>
  );
}

function GpuUsageRows({ gpu }: { gpu: DashboardGpu }) {
  const util = gpu.utilizationPercent;
  const used = gpu.memoryUsedBytes;
  const total = gpu.memoryTotalBytes;
  const hasVram = used !== undefined && total !== undefined && total > 0;
  return (
    <div>
      <div className="truncate px-3 pt-1 text-[0.74rem]" title={gpu.name}>
        {gpu.name} <span className="text-muted">({gpu.vendor})</span>
      </div>
      {util !== undefined ? <UsageRow label="util" pct={util} value={`${util.toFixed(0)}%`} /> : null}
      {hasVram ? <UsageRow label="vram" pct={(used / total) * 100} tone="bg-cache" value={`${fmtBytes(used)} / ${fmtBytes(total)}`} /> : null}
      {util === undefined && !hasVram ? (
        <div className="px-3 py-1 text-[0.72rem] text-muted">utilization not reported on this platform</div>
      ) : null}
    </div>
  );
}

function WorkerUsage({ entry, systemMemoryBytes, gpuTotalBytes }: { entry: DashboardLoadedModel; systemMemoryBytes?: number; gpuTotalBytes?: number }) {
  const usage = entry.usage;
  const rssPct = usage && systemMemoryBytes ? (usage.rssBytes / systemMemoryBytes) * 100 : undefined;
  const gpuPct = entry.gpuMemoryBytes !== undefined && gpuTotalBytes ? (entry.gpuMemoryBytes / gpuTotalBytes) * 100 : undefined;
  return (
    <div className="px-3 py-1.5">
      <div className="truncate text-[0.74rem]" title={entry.localPath}>
        {entry.id} <span className="text-muted">pid {entry.worker?.pid ?? "-"}</span>
      </div>
      <div className="mt-1 grid gap-x-6 gap-y-1 sm:grid-cols-2 xl:grid-cols-3">
        <MiniBar label="cpu" pct={usage?.cpuPercent} value={usage ? `${usage.cpuPercent.toFixed(0)}%` : "-"} />
        <MiniBar label="rss" pct={rssPct} value={usage ? fmtBytes(usage.rssBytes) : "-"} />
        {entry.gpuMemoryBytes !== undefined ? (
          <MiniBar label="vram" pct={gpuPct} tone="bg-cache" value={fmtBytes(entry.gpuMemoryBytes)} />
        ) : null}
      </div>
    </div>
  );
}

export function UsagePanel({ data }: { data: DashboardData }) {
  const server = data.server;
  const gpus = data.gpus ?? [];
  const queue = data.queue;
  const memPct = server.rssBytes !== undefined && server.systemMemoryBytes ? (server.rssBytes / server.systemMemoryBytes) * 100 : undefined;
  const gpuTotal = gpus[0]?.memoryTotalBytes;
  const workers = data.loaded.filter((entry) => entry.usage || entry.gpuMemoryBytes !== undefined);
  return (
    <Panel title="usage" count={queue ? `${queue.inflight}/${queue.maxInflight} inflight` : ""}>
      <div className="grid lg:grid-cols-2 lg:gap-x-4">
        <div className="pb-2">
          <GroupLabel>server</GroupLabel>
          <UsageRow label="cpu" pct={server.cpuPercent} value={server.cpuPercent !== undefined ? `${server.cpuPercent.toFixed(0)}%` : "-"} />
          <UsageRow label="mem" pct={memPct} value={`${fmtBytes(server.rssBytes)} / ${fmtBytes(server.systemMemoryBytes)}`} />
          {queue ? (
            <>
              <GroupLabel>queue</GroupLabel>
              <UsageRow
                label="inflight"
                pct={queue.maxInflight ? (queue.inflight / queue.maxInflight) * 100 : 0}
                tone="bg-accent"
                value={`${queue.inflight} / ${queue.maxInflight}`}
              />
              <UsageRow
                label="queued"
                pct={queue.queueDepth ? (queue.queued / queue.queueDepth) * 100 : 0}
                tone={queue.queued ? "bg-warn" : "bg-accent"}
                value={`${queue.queued} / ${queue.queueDepth}`}
              />
            </>
          ) : null}
        </div>
        <div className="pb-2 lg:border-l lg:border-soft-border">
          <GroupLabel>gpu</GroupLabel>
          {gpus.length ? (
            gpus.map((gpu, index) => <GpuUsageRows key={`${gpu.vendor}-${gpu.name}-${index}`} gpu={gpu} />)
          ) : (
            <div className="px-3 py-1.5 text-[0.74rem] text-muted">no gpu telemetry</div>
          )}
          {workers.length ? (
            <>
              <GroupLabel>workers</GroupLabel>
              {workers.map((entry) => (
                <WorkerUsage key={entry.key} entry={entry} systemMemoryBytes={server.systemMemoryBytes} gpuTotalBytes={gpuTotal} />
              ))}
            </>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}
