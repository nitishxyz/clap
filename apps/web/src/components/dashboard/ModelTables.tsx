import { useState, type ReactNode } from "react";
import { cancelDownload, loadModel, pullModel, removeModel, resolveModel, unloadModel, type DashboardDownload, type DashboardLoadedModel, type DashboardModel, type ModelResolveOption, type ModelResolveResponse } from "@/lib/api";
import { fmtBytes, fmtDuration, fmtMaxOutputTokens, fmtTokens } from "@/lib/format";
import type { ActionState } from "@/hooks/useActions";
import { Empty, Panel, Table, Tag, Td } from "./Shared";
import { BoxBar, formatMemoryValue, formatWorkerCpu, isUnifiedMlx, MLX_ACTIVE_TITLE, MLX_CACHE_TITLE, RSS_TITLE } from "./Usage";

function ActionButton({ label, busy, danger, onClick }: { label: string; busy?: boolean; danger?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      disabled={busy}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
      className={`cursor-pointer border px-1.5 py-0 text-[0.68rem] uppercase tracking-[0.04em] disabled:cursor-wait disabled:opacity-40 ${
        danger
          ? "border-err/50 text-err hover:bg-err hover:text-background"
          : "border-soft-border text-muted hover:bg-foreground hover:text-background"
      }`}
    >
      {busy ? "…" : label}
    </button>
  );
}

// Fixed geometry: bar and value each occupy a constant width so cells in the
// same column align across rows no matter how wide the value string is.
function TableBarCell({ pct, value, tone, dim, title }: { pct?: number; value?: string; tone?: string; dim?: boolean; title?: string }) {
  return (
    <span className="inline-flex items-center gap-2" title={title}>
      <BoxBar pct={pct} tone={tone} className="w-20 shrink-0" />
      <span className={`w-14 shrink-0 truncate text-right tabular-nums ${dim || pct === undefined ? "text-muted" : ""}`}>{value ?? "-"}</span>
    </span>
  );
}

function DetailItem({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3">
      <span className="shrink-0 text-[0.66rem] uppercase tracking-[0.06em] text-muted">{label}</span>
      <span className="min-w-0 truncate text-right text-[0.74rem] tabular-nums">{children}</span>
    </div>
  );
}

function LoadedModelRow({ entry, now, actions, platform, systemMemoryBytes, cpuCount, open, onToggle }: { entry: DashboardLoadedModel; now: number; actions: ActionState; platform?: string; systemMemoryBytes?: number; cpuCount?: number; open: boolean; onToggle: () => void }) {
  const unifiedMlx = isUnifiedMlx(entry.backend, platform);
  const concurrency = entry.worker.retention;
  const residency = entry.worker.residency;
  const growthReserve = concurrency?.retainedGrowthReserveBytes
    ?? (typeof concurrency?.activePolicy.inputs.retained_growth_reserve_bytes === "number"
      ? concurrency.activePolicy.inputs.retained_growth_reserve_bytes : undefined);
  return (
    <div className="border-b border-soft-border last:border-b-0">
      <div
        className="flex cursor-pointer items-center gap-2 px-3 py-2 hover:bg-panel-strong"
        onClick={onToggle}
        role="button"
        aria-expanded={open}
      >
        <span className={`shrink-0 text-[0.66rem] text-muted transition-transform duration-200 ${open ? "rotate-90" : ""}`}>&#9656;</span>
        <span className="min-w-0 flex-1 truncate text-[0.78rem]" title={entry.localPath}>{entry.id}</span>
        <span className="hidden shrink-0 sm:inline"><Tag>{entry.backend}</Tag></span>
        <span className="shrink-0">{entry.state === "active" ? <Tag tone="ok">active</Tag> : <Tag>{entry.worker.loadState ?? entry.state}</Tag>}</span>
        <span className="shrink-0 tabular-nums" data-model-capacity="summary">
          <Tag tone={concurrency?.underPressure ? "warn" : undefined}>
            <span className="sm:hidden">{concurrency ? `${concurrency.active}/${concurrency.maxActive} · q${concurrency.queued ?? "?"}` : "n/a"}</span>
            <span className="hidden sm:inline">{concurrency ? `${concurrency.active}/${concurrency.maxActive} active · ${concurrency.queued ?? "?"} queued` : "capacity unavailable"}</span>
          </Tag>
        </span>
        {entry.pinned ? <span className="hidden shrink-0 sm:inline"><Tag tone="pin">pinned</Tag></span> : null}
        <span className="flex shrink-0 gap-1">
          {entry.pinned ? (
            <ActionButton
              label="unpin"
              busy={actions.busy[`pin:${entry.id}`]}
              onClick={() => actions.run(`pin:${entry.id}`, () => loadModel(entry.id, "5m"))}
            />
          ) : (
            <ActionButton
              label="pin"
              busy={actions.busy[`pin:${entry.id}`]}
              onClick={() => actions.run(`pin:${entry.id}`, () => loadModel(entry.id, "always"))}
            />
          )}
          <ActionButton
            label="unload"
            danger
            busy={actions.busy[`unload:${entry.id}`]}
            onClick={() => actions.run(`unload:${entry.id}`, () => unloadModel(entry.id))}
          />
        </span>
      </div>
      <div className={`grid transition-[grid-template-rows] duration-300 ease-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden">
          <div className="border-t border-soft-border px-3 py-2.5 pl-8">
            <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-2 xl:grid-cols-4" data-model-details="primary">
            <DetailItem label="memory">
              <TableBarCell
                pct={entry.usage && systemMemoryBytes ? (entry.usage.rssBytes / systemMemoryBytes) * 100 : undefined}
                value={entry.usage ? fmtBytes(entry.usage.rssBytes) : undefined}
                title={RSS_TITLE}
              />
            </DetailItem>
            <DetailItem label="load estimate">
              {residency?.estimateBytes != null
                ? `${fmtBytes(residency.estimateBytes)} estimated · ${residency.estimateSource?.replaceAll("_", " ") ?? "unknown source"}`
                : "unavailable"}
            </DetailItem>
            <DetailItem label="observed rss">
              {residency?.observedRssBytes != null && residency.observedRssSource
                ? `${fmtBytes(residency.observedRssBytes)} measured · RSS`
                : "not measured"}
            </DetailItem>
            <DetailItem label="reservation">{residency ? fmtBytes(residency.reservationBytes) : "unavailable"}</DetailItem>
            <DetailItem label="admission">
              {residency?.lastAdmissionReason?.replaceAll("_", " ") ?? "not admitted"}
            </DetailItem>
            <DetailItem label="eviction">
              {residency?.lastEvictionReason?.replaceAll("_", " ") ?? "none"}
            </DetailItem>
            <DetailItem label="cpu">
              <TableBarCell
                pct={entry.usage ? (cpuCount ? entry.usage.cpuPercent / cpuCount : Math.min(100, entry.usage.cpuPercent)) : undefined}
                value={entry.usage ? formatWorkerCpu(entry.usage.cpuPercent) : undefined}
              />
            </DetailItem>
            <DetailItem label="model active / max">{entry.activeRequests} / {concurrency?.maxActive ?? "unavailable"}</DetailItem>
            <DetailItem label="model queued">{concurrency?.queued ?? "unavailable"}</DetailItem>
            <DetailItem label="keep-alive">{entry.keepAlive}</DetailItem>
            <DetailItem label="expires">
              {entry.pinned ? "pinned" : entry.expiresAt ? fmtDuration(new Date(entry.expiresAt).getTime() - now) : "-"}
            </DetailItem>
            <DetailItem label="last used">{fmtDuration(now - new Date(entry.lastUsedAt).getTime())} ago</DetailItem>
            <DetailItem label="pid">{entry.worker?.pid ?? "-"}</DetailItem>
            <DetailItem label="backend">{entry.backend} · {entry.format}</DetailItem>
            <DetailItem label="context">{entry.worker.tokenCapabilities?.effectiveContextWindow ? fmtTokens(entry.worker.tokenCapabilities.effectiveContextWindow) : "unknown"}</DetailItem>
            <DetailItem label="max input">{entry.worker.tokenCapabilities?.maxInputTokens != null ? fmtTokens(entry.worker.tokenCapabilities.maxInputTokens) : "unknown"}</DetailItem>
            <DetailItem label="max output">
              <span title={entry.worker.tokenCapabilities?.maxOutputTokens == null && entry.worker.tokenCapabilities?.effectiveContextWindow != null
                ? "No fixed generation cap is declared; available output depends on prompt length."
                : undefined}>
                {fmtMaxOutputTokens(entry.worker.tokenCapabilities?.maxOutputTokens, entry.worker.tokenCapabilities?.effectiveContextWindow)}
              </span>
            </DetailItem>
            <DetailItem label="context source">
              {entry.worker.tokenCapabilities?.userConfiguredOverride
                ? `override ${fmtTokens(entry.worker.tokenCapabilities.userConfiguredOverride)}`
                : entry.worker.tokenCapabilities?.modelContextWindow
                  ? `model ${fmtTokens(entry.worker.tokenCapabilities.modelContextWindow)}`
                  : "unknown"}
            </DetailItem>
            </div>
            <div className="mt-2 border-t border-soft-border pt-2" data-model-details="concurrency">
              <div className="mb-1.5 flex items-center justify-between gap-2 text-[0.62rem] uppercase tracking-[0.08em] text-muted">
                <span>Worker Capacity</span>
                <Tag tone={concurrency?.underPressure ? "warn" : concurrency ? "ok" : undefined}>
                  {concurrency?.pressureState ?? (concurrency?.underPressure ? "pressure" : concurrency ? "normal" : "unavailable")}
                </Tag>
              </div>
              <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-2 xl:grid-cols-4">
                <DetailItem label="worker active / max">{concurrency ? `${concurrency.active} / ${concurrency.maxActive}` : "unavailable"}</DetailItem>
                <DetailItem label="worker queued">{concurrency?.queued ?? "unavailable"}</DetailItem>
                <DetailItem label="mode">{concurrency?.activePolicy.mode ?? "unavailable"}</DetailItem>
                <DetailItem label="selected limit">{concurrency?.activePolicy.selectedMax ?? "unavailable"}</DetailItem>
                <DetailItem label="previous limit">{concurrency?.previousMaxActive ?? "unavailable"}</DetailItem>
                <DetailItem label="backend ceiling">{concurrency?.activePolicy.backendCeiling ?? "unavailable"}</DetailItem>
                <DetailItem label="hardware ceiling">{concurrency?.activePolicy.hardwareCeiling ?? "unavailable"}</DetailItem>
                <DetailItem label="model ceiling">{concurrency?.activePolicy.modelCeiling ?? "unavailable"}</DetailItem>
                <DetailItem label="memory ceiling">{concurrency?.activePolicy.memoryCeiling ?? "unavailable"}</DetailItem>
                <DetailItem label="limiting reason">{concurrency?.activePolicy.reason.replaceAll("_", " ") ?? "unavailable"}</DetailItem>
                <DetailItem label="last adjustment">{concurrency?.lastAdjustmentReason?.replaceAll("_", " ") ?? "unavailable"}</DetailItem>
                <DetailItem label="adjusted">
                  {concurrency?.lastAdjustmentAt
                    ? <span title={concurrency.lastAdjustmentAt}>{fmtDuration(now - new Date(concurrency.lastAdjustmentAt).getTime())} ago</span>
                    : "unavailable"}
                </DetailItem>
                <DetailItem label="retained growth reserve">{growthReserve !== undefined ? fmtBytes(growthReserve) : "unavailable"}</DetailItem>
                <DetailItem label="current retained">{concurrency ? formatMemoryValue(concurrency.retainedBytes, concurrency.retainedBytesSource) ?? "unavailable" : "unavailable"}</DetailItem>
                <DetailItem label="global resident memory">{concurrency?.globalResidentMemoryBytes !== undefined ? fmtBytes(concurrency.globalResidentMemoryBytes) : "unavailable"}</DetailItem>
              </div>
            </div>
            <div className="mt-2 border-t border-soft-border pt-2" data-model-details="backend">
              <div className="mb-1.5 text-[0.62rem] uppercase tracking-[0.08em] text-muted">{entry.backend} memory</div>
              <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-3">
                {unifiedMlx ? (
                  <>
                    <DetailItem label="active"><span title={MLX_ACTIVE_TITLE}>{formatMemoryValue(entry.worker.memory?.activeBytes, entry.worker.memory?.activeBytesSource) ?? "unavailable"}</span></DetailItem>
                    <DetailItem label="cache"><span title={MLX_CACHE_TITLE}>{formatMemoryValue(entry.worker.memory?.cacheBytes, entry.worker.memory?.cacheBytesSource) ?? "unavailable"}</span></DetailItem>
                    <DetailItem label="peak">{formatMemoryValue(entry.worker.memory?.peakActiveBytes, entry.worker.memory?.peakActiveBytesSource) ?? "unavailable"}</DetailItem>
                  </>
                ) : (
                  <DetailItem label="vram">{entry.gpuMemoryBytes !== undefined ? fmtBytes(entry.gpuMemoryBytes) : "unavailable"}</DetailItem>
                )}
              </div>
            </div>
            {entry.worker.retention ? (
              <div className="mt-2 border-t border-soft-border pt-2" data-model-details="retained-cache">
                <div className="mb-1.5 flex items-center justify-between gap-2 text-[0.62rem] uppercase tracking-[0.08em] text-muted">
                  <span>Retained Cache</span>
                  <Tag tone={entry.worker.retention.underPressure ? "warn" : "ok"}>
                    {entry.worker.retention.underPressure ? "pressure" : "healthy"}
                  </Tag>
                </div>
                <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-2 xl:grid-cols-4">
                  <DetailItem label="startup available">
                    {typeof entry.worker.retention.activePolicy.inputs.startup_available_bytes === "number"
                      ? fmtBytes(entry.worker.retention.activePolicy.inputs.startup_available_bytes) : "unknown"}
                  </DetailItem>
                  <DetailItem label="active reserve">
                    {typeof entry.worker.retention.activePolicy.inputs.per_active_reserve_bytes === "number"
                      ? fmtBytes(entry.worker.retention.activePolicy.inputs.per_active_reserve_bytes) : "unknown"}
                  </DetailItem>
                  <DetailItem label="retained total">{entry.worker.retention.retainedTotal}</DetailItem>
                  <DetailItem label="sessions">{entry.worker.retention.retainedSessions}</DetailItem>
                  <DetailItem label="anchors">{entry.worker.retention.retainedAnchors}</DetailItem>
                  <DetailItem label="retained bytes">{formatMemoryValue(entry.worker.retention.retainedBytes, entry.worker.retention.retainedBytesSource) ?? "unavailable"}</DetailItem>
                  <DetailItem label="estimated retained">{formatMemoryValue(entry.worker.retention.estimatedRetainedBytes, entry.worker.retention.estimatedRetainedBytesSource) ?? "unavailable"}</DetailItem>
                  <DetailItem label="session bytes">{formatMemoryValue(entry.worker.retention.sessionBytes, entry.worker.retention.sessionBytesSource) ?? "unavailable"}</DetailItem>
                  <DetailItem label="anchor bytes">{formatMemoryValue(entry.worker.retention.anchorBytes, entry.worker.retention.anchorBytesSource) ?? "unavailable"}</DetailItem>
                  <DetailItem label="evicted bytes">{formatMemoryValue(entry.worker.retention.evictedBytes, entry.worker.retention.evictedBytesSource) ?? "unavailable"}</DetailItem>
                  <DetailItem label="automatic checkpoints">
                    {entry.worker.retention.automaticCheckpointCount ?? "unknown"}
                  </DetailItem>
                  <DetailItem label="automatic checkpoint bytes">
                    {entry.worker.retention.automaticCheckpointBytes === undefined
                      ? "unknown" : fmtBytes(entry.worker.retention.automaticCheckpointBytes)}
                  </DetailItem>
                  <DetailItem label="checkpoint policy">
                    {entry.worker.retention.automaticCheckpointsEnabled === undefined ? "unknown"
                      : entry.worker.retention.automaticCheckpointsEnabled
                        ? `${entry.worker.retention.automaticCheckpointMinimumTokens}/${entry.worker.retention.automaticCheckpointIntervalTokens}/${entry.worker.retention.automaticCheckpointMax}`
                        : "disabled"}
                  </DetailItem>
                  <DetailItem label="checkpoint budget">
                    {entry.worker.retention.automaticCheckpointBudgetBytes === undefined
                      ? "unknown" : fmtBytes(entry.worker.retention.automaticCheckpointBudgetBytes)}
                  </DetailItem>
                  <DetailItem label="budget">{fmtBytes(entry.worker.retention.budgetBytes)}</DetailItem>
                  <DetailItem label="high watermark">{fmtBytes(entry.worker.retention.highWatermarkBytes)}</DetailItem>
                  <DetailItem label="low watermark">{fmtBytes(entry.worker.retention.lowWatermarkBytes)}</DetailItem>
                  <DetailItem label="hard ceiling">{entry.worker.retention.hardCeiling}</DetailItem>
                  <DetailItem label="evictions">{entry.worker.retention.evictionCount}</DetailItem>
                  <DetailItem label="last reason">{entry.worker.retention.evictionReason ?? "none"}</DetailItem>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

export function LoadedModels({ models, now, actions, platform, systemMemoryBytes, cpuCount }: { models: DashboardLoadedModel[]; now: number; actions: ActionState; platform?: string; systemMemoryBytes?: number; cpuCount?: number }) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  return (
    <Panel title="loaded models" count={models.length || ""}>
      {models.length ? (
        <div>
          {models.map((entry) => (
            <LoadedModelRow
              key={entry.key}
              entry={entry}
              now={now}
              actions={actions}
              platform={platform}
              systemMemoryBytes={systemMemoryBytes}
              cpuCount={cpuCount}
              open={!!expanded[entry.key]}
              onToggle={() => setExpanded((previous) => ({ ...previous, [entry.key]: !previous[entry.key] }))}
            />
          ))}
        </div>
      ) : (
        <Empty>no models resident — send a request, `clap load &lt;model&gt;`, or press load below</Empty>
      )}
    </Panel>
  );
}

export function Downloads({ downloads, actions }: { downloads: DashboardDownload[]; actions: ActionState }) {
  const [pullInput, setPullInput] = useState("");
  const [resolving, setResolving] = useState(false);
  const [resolved, setResolved] = useState<ModelResolveResponse>();
  const [resolveError, setResolveError] = useState<string>();
  const active = downloads.filter((download) => download.status === "running" || download.status === "queued");
  const shown = active.length ? active : downloads.slice(-5);
  const submitResolve = () => {
    const model = pullInput.trim();
    if (!model || resolving) return;
    setResolving(true);
    setResolveError(undefined);
    resolveModel(model)
      .then((response) => {
        setResolved(response);
        setPullInput("");
      })
      .catch((cause) => {
        setResolved(undefined);
        setResolveError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => setResolving(false));
  };
  const pullOption = (option: ModelResolveOption) => {
    actions.run(`pull:${option.id}`, () => pullModel(option.repo, { file: option.file, backend: option.backend }));
    setResolved(undefined);
    setResolveError(undefined);
  };
  return (
    <Panel
      title="downloads"
      count={
        <span className="flex items-center gap-2 normal-case">
          <input
            value={pullInput}
            onChange={(event) => setPullInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitResolve();
            }}
            placeholder="org/repo or alias to pull…"
            className="w-56 border border-soft-border bg-background px-2 py-0.5 text-[0.72rem] text-foreground placeholder:text-muted focus:outline-none"
          />
          <ActionButton label="pull" busy={resolving} onClick={submitResolve} />
          {active.length ? <span className="text-foreground">{active.length} active</span> : null}
        </span>
      }
    >
      {resolveError ? (
        <div className="flex items-start justify-between gap-3 border-b border-soft-border px-3 py-2 text-[0.78rem] text-err">
          <span className="break-words">{resolveError}</span>
          <ActionButton label="dismiss" onClick={() => setResolveError(undefined)} />
        </div>
      ) : null}
      {resolved ? (
        <div className="border-b border-soft-border">
          <div className="flex items-center justify-between gap-3 px-3 py-1.5 text-[0.72rem] uppercase tracking-[0.06em] text-muted">
            <span>
              options for <span className="normal-case text-foreground">{resolved.model}</span>
            </span>
            <ActionButton label="clear" onClick={() => setResolved(undefined)} />
          </div>
          {resolved.options.length ? (
            <Table headers={["option", "backend", "quant", "size", "why", ""]}>
              {resolved.options.map((option) => (
                <tr key={option.id} className={option.supported ? "" : "text-muted"}>
                  <Td className="max-w-[280px] overflow-hidden text-ellipsis" title={option.repo}>
                    <span className="flex items-center gap-2">
                      <span className="overflow-hidden text-ellipsis">{option.file ?? `${option.repo} (${option.format})`}</span>
                      {option.recommended ? <Tag tone="ok">recommended</Tag> : null}
                      {option.supported ? null : <Tag tone="warn">unsupported</Tag>}
                    </span>
                  </Td>
                  <Td>{option.backend}</Td>
                  <Td>{option.quantization ?? "-"}</Td>
                  <Td>{option.sizeBytes ? fmtBytes(option.sizeBytes) : "-"}</Td>
                  <Td className="max-w-[340px] whitespace-normal text-muted">{option.supported ? option.reason : option.unsupportedReason ?? option.reason}</Td>
                  <Td>
                    {option.supported ? (
                      <ActionButton
                        label="pull"
                        busy={actions.busy[`pull:${option.id}`]}
                        onClick={() => pullOption(option)}
                      />
                    ) : null}
                  </Td>
                </tr>
              ))}
            </Table>
          ) : (
            <Empty>no runnable artifacts found for {resolved.model}</Empty>
          )}
          {resolved.options.some((option) => option.supported) ? null : (
            <div className="border-t border-soft-border px-3 py-2 text-[0.76rem] text-muted">
              This repo has no artifact Clap can run on this machine. Search Hugging Face for a{" "}
              <a
                className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
                href={`https://huggingface.co/models?search=${encodeURIComponent(`${resolved.model.split("/").pop() ?? resolved.model} GGUF`)}`}
                target="_blank"
                rel="noreferrer"
              >
                GGUF conversion
              </a>
              .
            </div>
          )}
        </div>
      ) : null}
      {shown.length ? (
        <Table headers={["model", "status", "progress", "size", ""]}>
          {shown.map((download) => {
            const pct = download.totalBytes
              ? Math.min(100, Math.round((download.bytesReceived / download.totalBytes) * 100))
              : null;
            const running = download.status === "running" || download.status === "queued";
            return (
              <tr key={download.id}>
                <Td className="max-w-[260px] overflow-hidden text-ellipsis" title={download.currentFile}>{download.model}</Td>
                <Td>
                  {download.status === "completed" ? (
                    <Tag tone="ok">completed</Tag>
                  ) : download.status === "failed" ? (
                    <Tag tone="err">failed</Tag>
                  ) : (
                    <Tag>{download.status}</Tag>
                  )}
                </Td>
                <Td>
                  {download.status === "failed" && download.error ? (
                    <span className="block max-w-[260px] overflow-hidden text-ellipsis text-err" title={download.error}>{download.error}</span>
                  ) : pct === null ? (
                    "-"
                  ) : (
                    <span className="flex items-center gap-2">
                      <span className="relative inline-block h-1.5 w-32 border border-soft-border bg-panel-strong">
                        <i className="absolute inset-y-0 left-0 bg-accent" style={{ width: `${pct}%` }} />
                      </span>
                      <span className="text-muted">{pct}%</span>
                    </span>
                  )}
                </Td>
                <Td>
                  {fmtBytes(download.bytesReceived)} / {fmtBytes(download.totalBytes)}
                </Td>
                <Td>
                  {running ? (
                    <ActionButton
                      label="cancel"
                      danger
                      busy={actions.busy[`cancel:${download.id}`]}
                      onClick={() => actions.run(`cancel:${download.id}`, () => cancelDownload(download.id))}
                    />
                  ) : null}
                </Td>
              </tr>
            );
          })}
        </Table>
      ) : (
        <Empty>no downloads — paste a HuggingFace repo above to pull one</Empty>
      )}
    </Panel>
  );
}

export function ModelCache({ models, loaded, actions }: { models: DashboardModel[]; loaded: DashboardLoadedModel[]; actions: ActionState }) {
  const cached = models.filter((model) => model.status === "available");
  const loadedIds = new Set(loaded.map((entry) => entry.id));
  const [confirmRemove, setConfirmRemove] = useState<string>();
  const [copiedModel, setCopiedModel] = useState<string>();
  const copyModelId = (modelId: string) => {
    navigator.clipboard.writeText(modelId).then(() => {
      setCopiedModel(modelId);
      window.setTimeout(() => setCopiedModel((current) => current === modelId ? undefined : current), 1500);
    }).catch(() => undefined);
  };
  return (
    <Panel title="model cache" count={`${cached.length} on disk`}>
      {cached.length ? (
        <Table headers={["client model id", "backend", "format", "quant", { label: "size", numeric: true }, { label: "context", numeric: true }, "capabilities", ""]}>
          {cached.map((model) => {
            const caps = [
              model.capabilities?.toolCall ? "tools" : null,
              model.capabilities?.reasoning ? "reasoning" : null,
              model.modalities?.input?.includes("image") ? "vision" : null,
            ]
              .filter(Boolean)
              .join(" · ");
            const isLoaded = loadedIds.has(model.id);
            return (
              <tr key={model.id}>
                <Td className="max-w-[360px] overflow-hidden text-ellipsis" title={model.id}>{model.id}</Td>
                <Td>{model.backend}</Td>
                <Td>{model.format}</Td>
                <Td>{model.quantization ?? "-"}</Td>
                <Td numeric>{model.sizeBytes !== undefined ? fmtBytes(model.sizeBytes) : "-"}</Td>
                <Td numeric>{model.limit?.context ? fmtTokens(model.limit.context) : "-"}</Td>
                <Td className="text-muted">{caps || "-"}</Td>
                <Td>
                  <span className="flex gap-1">
                    <ActionButton
                      label={copiedModel === model.id ? "copied" : "copy"}
                      onClick={() => copyModelId(model.id)}
                    />
                    {isLoaded ? (
                      <Tag tone="ok">loaded</Tag>
                    ) : (
                      <ActionButton
                        label="load"
                        busy={actions.busy[`load:${model.id}`]}
                        onClick={() => actions.run(`load:${model.id}`, () => loadModel(model.id))}
                      />
                    )}
                    {confirmRemove === model.id ? (
                      <>
                        <ActionButton
                          label="confirm rm"
                          danger
                          busy={actions.busy[`rm:${model.id}`]}
                          onClick={() => {
                            setConfirmRemove(undefined);
                            actions.run(`rm:${model.id}`, () => removeModel(model.id));
                          }}
                        />
                        <ActionButton label="keep" onClick={() => setConfirmRemove(undefined)} />
                      </>
                    ) : (
                      <ActionButton label="rm" danger onClick={() => setConfirmRemove(model.id)} />
                    )}
                  </span>
                </Td>
              </tr>
            );
          })}
        </Table>
      ) : (
        <Empty>no models cached — `clap pull &lt;model&gt;` or use the pull box above</Empty>
      )}
    </Panel>
  );
}
