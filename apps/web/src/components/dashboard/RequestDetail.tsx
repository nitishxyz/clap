import { useEffect, useState } from "react";
import { fetchRequestDetail, type DashboardRequest } from "@/lib/api";
import { fmtClock, fmtDuration, fmtTokens } from "@/lib/format";
import { CACHE_OUTCOME_LABELS, cacheOutcomeLabel, cacheReusePercent, HistoricalTag, IdentityTag } from "./RequestTables";
import { Tag } from "./Shared";

const roleColor: Record<string, string> = {
  system: "text-thinking",
  user: "text-accent",
  assistant: "text-ok",
  tool: "text-warn",
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border border-soft-border bg-panel-strong px-2 py-1.5">
      <div className="text-[0.62rem] uppercase tracking-[0.06em] text-muted">{label}</div>
      <div className="mt-0.5 text-[0.8rem]">{children}</div>
    </div>
  );
}

function fmtDecisionLatency(us: number): string {
  if (us < 1000) return `${Math.round(us)}µs`;
  return `${(us / 1000).toFixed(1)}ms`;
}

const unavailable = <span className="text-muted">unavailable</span>;

// Cache outcome is reported independently of request intent (side vs primary)
// and of final status: cancelled/error requests may still carry the decision.
export function CacheDecisionSection({ record }: { record: DashboardRequest }) {
  const pct = cacheReusePercent(record);
  const reused = record.reusedTokens ?? (record.cacheHit === false ? 0 : undefined);
  const diagnostics = record.cacheDiagnostics;
  const outcome = record.cacheOutcome;
  const skippedBoundaries = diagnostics?.stableBoundaries?.filter((boundary) => boundary.status === "skipped")
    ?? (outcome?.boundariesSkipped ? Array.from({ length: outcome.boundariesSkipped }) : []);
  return (
    <div>
      <div className="text-[0.68rem] uppercase tracking-[0.08em] text-muted">cache decision</div>
      <div className="mt-1.5 grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="outcome">
          {outcome ? (
            <span className="inline-flex flex-wrap items-center gap-1">
              <Tag
                tone={outcome.category === "hit" ? "hit" : outcome.category === "cache_error" || outcome.category === "unexplained_miss" ? "err" : outcome.category === "donor_busy" ? "warn" : outcome.category === "isolated" ? "pin" : "default"}
                title={outcome.reason}
                ariaLabel={`cache outcome ${cacheOutcomeLabel(outcome)}`}
              >
                {cacheOutcomeLabel(outcome)}
              </Tag>
              <HistoricalTag historical={record.historical} />
            </span>
          ) : record.cacheHit === true ? (
            <Tag tone="hit" title="prefix cache hit: reused tokens came from the KV cache instead of being re-prefilled">hit</Tag>
          ) : record.cacheHit === false ? (
            <Tag title="prefix cache miss: the full prompt was prefilled">miss</Tag>
          ) : (
            unavailable
          )}
          {record.status !== "ok" && record.status !== "active" ? (
            <span className="ml-2 text-muted">· from {record.status} request</span>
          ) : null}
        </Field>
        <Field label="raw decision">
          {record.cacheHit === true ? (
            <Tag tone="hit" title="raw telemetry: hit">hit</Tag>
          ) : record.cacheHit === false ? (
            <Tag title="raw telemetry: miss">miss</Tag>
          ) : (
            unavailable
          )}
        </Field>
        <Field label="intent">
          {record.sideRequest ? (
            <Tag tone="warn" title="side request: branched from primary work; intent is independent of cache hit/miss">side request</Tag>
          ) : (
            "primary"
          )}
        </Field>
        <Field label="reused tokens">
          {reused !== undefined ? `${fmtTokens(reused)} tok${pct !== undefined ? ` · ${pct}%` : ""}` : unavailable}
        </Field>
        <Field label="kind">{record.reuseKind ?? unavailable}</Field>
        <Field label="scope">{record.reuseScope ?? unavailable}</Field>
        <Field label="namespace">{record.cacheNamespace ?? unavailable}</Field>
        <Field label="planned / realized">
          {record.plannedReuseTokens === undefined && record.realizedReuseTokens === undefined
            ? unavailable
            : `${record.plannedReuseTokens !== undefined ? `${fmtTokens(record.plannedReuseTokens)} tok` : "unavailable"} / ${record.realizedReuseTokens !== undefined ? `${fmtTokens(record.realizedReuseTokens)} tok` : "unavailable"}`}
        </Field>
        <Field label="donor → target">
          {record.donorSlot === undefined && record.targetSlot === undefined
            ? unavailable
            : `${record.donorSlot !== undefined ? `s${record.donorSlot}` : "unavailable"} → ${record.targetSlot !== undefined ? `s${record.targetSlot}` : "unavailable"}`}
        </Field>
        <Field label="evictions">
          {record.evictedSlots === undefined
            ? unavailable
            : record.evictedSlots.length
              ? record.evictedSlots.map((slot) => `s${slot}`).join(", ")
              : "none"}
        </Field>
        <Field label="fallback">{record.cacheFallback ?? unavailable}</Field>
        <Field label="decision latency">
          {record.cacheDecisionUs !== undefined ? fmtDecisionLatency(record.cacheDecisionUs) : unavailable}
        </Field>
        <Field label="boundaries skipped">
          {outcome?.boundariesSkipped !== undefined
            ? outcome.boundariesSkipped
            : skippedBoundaries.length
              ? skippedBoundaries.length
              : diagnostics?.stableBoundaries
                ? 0
                : unavailable}
        </Field>
        <Field label="blocked prefix">
          {outcome?.maxBlockedPrefixTokens !== undefined
            ? `${fmtTokens(outcome.maxBlockedPrefixTokens)} tok`
            : unavailable}
        </Field>
      </div>
      {outcome ? (
        <div className="mt-2 border border-soft-border bg-panel-strong p-2">
          <div className="text-[0.64rem] uppercase tracking-[0.07em] text-muted">
            classification · {CACHE_OUTCOME_LABELS[outcome.category]}
            {outcome.hitKind ? ` · ${outcome.hitKind}` : ""}
          </div>
          <div className="mt-1 text-[0.74rem]" title={outcome.reason}>{outcome.reason}</div>
          {outcome.evidence.length ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {outcome.evidence.map((item) => (
                <Tag key={item} title={item} ariaLabel={item}>{item}</Tag>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
      {record.timing ? (
        <div className="mt-2">
          <div className="text-[0.68rem] uppercase tracking-[0.08em] text-muted">MLX phase timing</div>
          <div className="mt-1.5 grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="prefill compute">{record.timing.prefillMs !== undefined ? fmtDuration(record.timing.prefillMs) : unavailable}</Field>
            <Field label="scheduler wait">{record.timing.schedulerWaitMs !== undefined ? fmtDuration(record.timing.schedulerWaitMs) : unavailable}</Field>
            <Field label="cache materialize">{record.timing.cacheMaterializeMs !== undefined ? fmtDuration(record.timing.cacheMaterializeMs) : unavailable}</Field>
            <Field label="first decode">{record.timing.firstDecodeMs !== undefined ? fmtDuration(record.timing.firstDecodeMs) : unavailable}</Field>
            <Field label="template + tokenize">{record.timing.templateTokenizeMs !== undefined ? fmtDuration(record.timing.templateTokenizeMs) : unavailable}</Field>
            <Field label="received → admitted">{record.timing.receivedToAdmittedMs !== undefined ? fmtDuration(record.timing.receivedToAdmittedMs) : unavailable}</Field>
            <Field label="coordinator wait">{record.timing.coordinatorWaitMs !== undefined ? fmtDuration(record.timing.coordinatorWaitMs) : unavailable}</Field>
            <Field label="coordinator plan / apply">
              {record.timing.coordinatorPlanMs !== undefined || record.timing.coordinatorApplyMs !== undefined
                ? `${fmtDuration(record.timing.coordinatorPlanMs ?? 0)} / ${fmtDuration(record.timing.coordinatorApplyMs ?? 0)}`
                : unavailable}
            </Field>
            <Field label="residual prefill">
              {record.timing.residualPrefillTokens !== undefined
                ? `${fmtTokens(record.timing.residualPrefillTokens)} tok · ${record.timing.prefillChunks ?? 0} chunks`
                : unavailable}
            </Field>
            <Field label="prefill quantum">
              {record.timing.normalPrefillQuantum !== undefined && record.timing.contendedPrefillQuantum !== undefined
                ? `${record.timing.normalPrefillQuantum} alone / ${record.timing.contendedPrefillQuantum} contended`
                : unavailable}
            </Field>
          </div>
        </div>
      ) : null}
      {diagnostics ? (
        <div className="mt-2 border border-soft-border bg-panel-strong p-2">
          <div className="text-[0.64rem] uppercase tracking-[0.07em] text-muted">privacy-safe persisted diagnostics</div>
          <div className="mt-1 text-[0.72rem] text-muted">
            backend {diagnostics.backend ?? "unknown"} · server {diagnostics.serverLaunchId.slice(0, 8)}
            {diagnostics.workerLaunchId ? ` · worker ${diagnostics.workerLaunchId.slice(0, 8)}` : ""}
            {diagnostics.cache?.missReason ? ` · miss ${diagnostics.cache.missReason}` : ""}
            {diagnostics.errorCode ? ` · error ${diagnostics.errorCode}` : ""}
          </div>
          {diagnostics.stableBoundaries?.length ? (
            <div className="mt-1.5 grid gap-1">
              {diagnostics.stableBoundaries.map((boundary, index) => (
                <div key={`${boundary.tokenCount ?? "skipped"}-${index}`} className="text-[0.7rem]">
                  boundary {index + 1} · {boundary.kind}
                  {boundary.label ? ` · ${boundary.label}` : " · automatic"}
                  {boundary.tokenCount !== undefined ? ` · ${fmtTokens(boundary.tokenCount)} tok` : ""}
                  {boundary.requested ? " · requested" : ""}
                  {` · ${boundary.status}`}
                  {boundary.skipReason ? ` · ${boundary.skipReason}` : ""}
                  {boundary.materialized ? " · materialized" : ""}
                  {diagnostics.stableBoundaryTokenCount === boundary.tokenCount ? " · selected anchor" : ""}
                  {boundary.tokenHash ? ` · hash ${boundary.tokenHash.slice(0, 12)}` : ""}
                </div>
              ))}
            </div>
          ) : null}
          {diagnostics.cache?.candidates?.length ? (
            <div className="mt-1.5 grid gap-1">
              {diagnostics.cache.candidates.map((candidate, index) => (
                <div key={`${candidate.slot}-${candidate.generation ?? index}`} className="text-[0.7rem]">
                  slot {candidate.slot}
                  {candidate.generation !== undefined ? ` gen ${candidate.generation}` : ""}
                  {candidate.state ? ` · ${candidate.state}` : ""}
                  {` · shared ${fmtTokens(candidate.sharedPrefixTokens)} tok`}
                  {candidate.selected ? " · selected" : candidate.rejection
                    ? ` · rejected: ${candidate.rejection}` : " · eligible (lower rank)"}
                  {candidate.namespaceCompatible === false ? " · namespace mismatch" : ""}
                  {candidate.modelCompatible === false ? " · model mismatch" : ""}
                  {candidate.sessionCompatible === false ? " · session mismatch" : ""}
                  {candidate.materialized === false ? " · not materialized" : ""}
                  {candidate.trimEligible === false ? " · cannot trim" : ""}
                  {candidate.copyEligible === false ? " · cannot copy" : ""}
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-1 text-[0.7rem] text-muted">candidate diagnostics unavailable</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function MessageCard({ role, content, truncated, toolCalls }: { role: string; content: string; truncated?: boolean; toolCalls?: Array<{ name: string; arguments: string }> }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = content.length > 600;
  const shown = expanded || !isLong ? content : `${content.slice(0, 600)}…`;
  return (
    <div className="border border-soft-border">
      <div className="flex items-baseline justify-between border-b border-soft-border bg-panel-strong px-2 py-1">
        <span className={`text-[0.68rem] uppercase tracking-[0.06em] ${roleColor[role] ?? "text-muted"}`}>{role}</span>
        <span className="text-[0.65rem] text-muted">
          {content.length ? `${fmtTokens(content.length)} chars${truncated ? " (truncated)" : ""}` : ""}
        </span>
      </div>
      {content ? (
        <pre className="m-0 max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words px-2 py-1.5 text-[0.74rem] leading-relaxed">{shown}</pre>
      ) : null}
      {toolCalls?.length ? (
        <div className="border-t border-soft-border px-2 py-1.5">
          {toolCalls.map((call, index) => (
            <div key={index} className="text-[0.74rem]">
              <span className="text-warn">{call.name}</span>
              <span className="text-muted">({call.arguments})</span>
            </div>
          ))}
        </div>
      ) : null}
      {isLong ? (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="w-full cursor-pointer border-t border-soft-border bg-panel-strong px-2 py-1 text-left text-[0.65rem] uppercase tracking-[0.06em] text-muted hover:text-foreground"
        >
          {expanded ? "collapse" : `expand (${fmtTokens(content.length)} chars)`}
        </button>
      ) : null}
    </div>
  );
}

export function RequestDetailModal({ id, onClose }: { id: string; onClose: () => void }) {
  const [record, setRecord] = useState<DashboardRequest>();
  const [error, setError] = useState<string>();
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    let disposed = false;
    fetchRequestDetail(id)
      .then((data) => {
        if (!disposed) setRecord(data);
      })
      .catch((cause) => {
        if (!disposed) setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      disposed = true;
    };
  }, [id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const detail = record?.detail;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 sm:p-8" onClick={onClose}>
      <div className="w-full max-w-[860px] border border-border bg-panel" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-border px-3 py-2">
          <span className="text-[0.72rem] uppercase tracking-[0.08em] text-muted">
            request {id} {record ? `· ${fmtClock(record.startedAt)}` : ""}
          </span>
          <button type="button" onClick={onClose} className="cursor-pointer border border-soft-border px-2 text-[0.72rem] text-muted hover:text-foreground">
            esc
          </button>
        </div>

        {error ? <div className="p-3 text-[0.78rem] text-err">{error}</div> : null}
        {!record && !error ? <div className="p-3 text-[0.78rem] text-muted">loading…</div> : null}

        {record ? (
          <div className="grid gap-3 p-3">
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
              <Field label="model">{record.model}</Field>
              <Field label="status">
                {record.status === "ok" ? <Tag tone="ok">ok</Tag> : record.status === "error" ? <Tag tone="err">error</Tag> : <Tag tone="warn">{record.status}</Tag>}
                {record.finishReason ? <span className="ml-2 text-muted">{record.finishReason}</span> : null}
              </Field>
              <Field label="endpoint">{record.endpoint} · {record.stream ? "sse" : "json"}</Field>
              <Field label="timing">
                {record.loadMs !== undefined && record.loadMs > 500 ? `load ${fmtDuration(record.loadMs)} · ` : ""}
                ttft {fmtDuration(record.ttftMs)} · total {fmtDuration(record.durationMs)}
              </Field>
              <Field label="tokens">
                in {fmtTokens(record.promptTokens)} · out {fmtTokens(record.completionTokens)}
                {record.tokensPerSecond ? ` · ${record.tokensPerSecond} tok/s` : ""}
              </Field>
              <Field label="session / prefix">
                <IdentityTag request={record} />
              </Field>
              <Field label="intent">
                {record.sideRequest ? (
                  <Tag tone="warn" title="side request: branched from primary work; intent is independent of cache hit/miss">side request</Tag>
                ) : (
                  "primary"
                )}
              </Field>
              <Field label="structured output">
                {record.structuredOutput ? [
                  `${record.structuredOutput.kind} (${record.structuredOutput.requestedStrength})`,
                  record.structuredOutput.backendMode ? `mode ${record.structuredOutput.backendMode}` : null,
                  record.structuredOutput.outcome?.replaceAll("_", " "),
                  record.structuredOutput.repairApplied ? "repair applied" : null,
                  record.structuredOutput.selectedParser ? `parser ${record.structuredOutput.selectedParser}` : null,
                  record.structuredOutput.validationMs !== undefined ? `validation ${record.structuredOutput.validationMs}ms` : null,
                ].filter(Boolean).join(" · ") : "none"}
              </Field>
              <Field label="params">
                {detail ? [
                  detail.params.temperature !== undefined ? `temp ${detail.params.temperature}` : null,
                  detail.params.topP !== undefined ? `top_p ${detail.params.topP}` : null,
                  detail.params.maxTokens !== undefined ? `max ${fmtTokens(detail.params.maxTokens)}` : null,
                  detail.params.stop?.length ? `stop [${detail.params.stop.join(", ")}]` : null,
                ].filter(Boolean).join(" · ") || "-" : "-"}
              </Field>
              <Field label="tools offered">
                {detail?.toolNames.length ? `${detail.toolNames.length}: ${detail.toolNames.slice(0, 6).join(", ")}${detail.toolNames.length > 6 ? "…" : ""}` : "none"}
              </Field>
            </div>

            <CacheDecisionSection record={record} />

            {record.error ? (
              <div className="border border-err/40 bg-panel-strong p-2 text-[0.76rem] text-err">{record.error}</div>
            ) : null}

            {detail ? (
              <>
                <div className="text-[0.68rem] uppercase tracking-[0.08em] text-muted">
                  conversation ({record.messageCount ?? detail.messages.length} messages{detail.droppedMessages ? `, first ${detail.droppedMessages} hidden` : ""})
                </div>
                <div className="grid max-h-[400px] gap-1.5 overflow-y-auto">
                  {detail.messages.map((message, index) => (
                    <MessageCard key={index} {...message} />
                  ))}
                </div>

                {detail.response ? (
                  <>
                    <div className="text-[0.68rem] uppercase tracking-[0.08em] text-muted">response</div>
                    {detail.response.reasoning ? (
                      <MessageCard role="assistant" content={detail.response.reasoning} toolCalls={undefined} />
                    ) : null}
                    <MessageCard
                      role="assistant"
                      content={detail.response.content ?? ""}
                      toolCalls={detail.response.toolCalls}
                    />
                  </>
                ) : null}

                {detail.rawOutput ? (
                  <div>
                    <button
                      type="button"
                      onClick={() => setShowRaw((value) => !value)}
                      className="cursor-pointer border border-soft-border px-2 py-0.5 text-[0.68rem] uppercase tracking-[0.06em] text-muted hover:text-foreground"
                    >
                      {showRaw ? "hide" : "show"} raw model output
                    </button>
                    {showRaw ? (
                      <pre className="m-0 mt-1.5 max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words border border-soft-border bg-panel-strong px-2 py-1.5 text-[0.72rem]">{detail.rawOutput}</pre>
                    ) : null}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="text-[0.76rem] text-muted">no detail captured for this request</div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}
