import { useState } from "react";
import { Events } from "@/components/dashboard/Events";
import { Downloads, LoadedModels, ModelCache } from "@/components/dashboard/ModelTables";
import { RequestDetailModal } from "@/components/dashboard/RequestDetail";
import { ActiveRequests, RecentRequests } from "@/components/dashboard/RequestTables";
import { Tiles } from "@/components/dashboard/Tiles";
import { UsagePanel } from "@/components/dashboard/Usage";
import { useActions } from "@/hooks/useActions";
import { useDashboard } from "@/hooks/useDashboard";
import { fmtBytes, fmtClock, fmtDuration } from "@/lib/format";

export function App() {
  const { data, connected, refreshedAt, mode } = useDashboard();
  const now = refreshedAt ?? Date.now();
  const [selectedRequest, setSelectedRequest] = useState<string>();
  const actions = useActions();

  return (
    <div className="mx-auto grid w-full max-w-[1280px] grid-cols-1 gap-4 p-4">
      <header className="flex flex-wrap items-baseline gap-3 border border-border bg-panel px-4 py-3">
        <h1 className="m-0 text-[1.05rem] uppercase tracking-[0.08em]">
          <span className={`mr-2 inline-block h-2 w-2 ${connected ? "animate-pulse bg-ok" : "bg-err"}`} />
          clap
        </h1>
        {data ? (
          <div className="flex flex-wrap gap-4 text-[0.78rem] text-muted">
            <span>v{data.server.version}</span>
            <span>up {fmtDuration(data.server.uptimeMs)}</span>
            <span>
              {data.server.platform}/{data.server.arch} bun {data.server.bunVersion}
            </span>
            <span>pid {data.server.pid}</span>
            {data.server.rssBytes ? <span>server {fmtBytes(data.server.rssBytes)}</span> : null}
            {data.server.systemMemoryBytes ? <span>sys {fmtBytes(data.server.systemMemoryBytes)}</span> : null}
          </div>
        ) : (
          <span className="text-[0.78rem] text-muted">{connected ? "loading..." : "connecting..."}</span>
        )}
      </header>

      {actions.error ? (
        <div className="flex items-start justify-between gap-3 border border-err/50 bg-panel px-3 py-2 text-[0.78rem] text-err">
          <span className="break-words">{actions.error}</span>
          <button type="button" onClick={actions.dismissError} className="cursor-pointer border border-err/50 px-1.5 text-[0.68rem] uppercase hover:bg-err hover:text-background">
            dismiss
          </button>
        </div>
      ) : null}

      {data ? (
        <>
          <Tiles data={data} />
          <UsagePanel data={data} />
          <LoadedModels models={data.loaded} now={now} actions={actions} platform={data.server.platform} systemMemoryBytes={data.server.systemMemoryBytes} cpuCount={data.server.cpuCount} />
          <ActiveRequests requests={data.active} now={now} onSelect={setSelectedRequest} />
          <RecentRequests requests={data.requests} onSelect={setSelectedRequest} />
          <Events events={data.events ?? []} />
          <Downloads downloads={data.downloads} actions={actions} />
          <ModelCache models={data.models} loaded={data.loaded} actions={actions} />
          <footer className="text-right text-[0.72rem] text-muted">
            refreshed {refreshedAt ? fmtClock(refreshedAt) : "-"} · {mode === "live" ? "live via sse" : "polling every 2s"}
          </footer>
        </>
      ) : null}

      {selectedRequest ? <RequestDetailModal id={selectedRequest} onClose={() => setSelectedRequest(undefined)} /> : null}
    </div>
  );
}
