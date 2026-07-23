import type { CacheOutcome, CacheOutcomeCategory, DashboardRequest } from "@/lib/api";
import { fmtClock, fmtDuration, fmtTokens } from "@/lib/format";
import { Empty, Panel, Table, Tag, Td } from "./Shared";

export const CACHE_OUTCOME_LABELS: Record<CacheOutcomeCategory, string> = {
  hit: "hit",
  cold: "cold",
  isolated: "isolated",
  below_checkpoint: "below checkpoint",
  no_shared_prefix: "no shared prefix",
  donor_busy: "donor busy",
  no_eligible_donor: "no eligible donor",
  fresh_by_policy: "fresh by policy",
  cache_error: "cache error",
  unexplained_miss: "unexplained",
  miss_reason_unavailable: "unavailable",
  unknown: "n/a",
};

const CACHE_OUTCOME_TONES: Record<CacheOutcomeCategory, "hit" | "ok" | "err" | "warn" | "pin" | "cold" | "isolated" | "busy" | "default"> = {
  hit: "hit",
  cold: "cold",
  isolated: "isolated",
  below_checkpoint: "default",
  no_shared_prefix: "default",
  donor_busy: "busy",
  no_eligible_donor: "warn",
  fresh_by_policy: "pin",
  cache_error: "err",
  unexplained_miss: "err",
  miss_reason_unavailable: "default",
  unknown: "default",
};

export const CACHE_OUTCOME_LEGEND: Array<{ category: CacheOutcomeCategory; blurb: string }> = [
  { category: "hit", blurb: "session/branch/checkpoint reuse from the KV cache" },
  { category: "cold", blurb: "first decision for model/worker with no resident donor" },
  { category: "isolated", blurb: "prefix exists but namespace/model isolation blocked it" },
  { category: "below_checkpoint", blurb: "prompt under automatic checkpoint minimum" },
  { category: "no_shared_prefix", blurb: "warm worker, no reusable token prefix" },
  { category: "donor_busy", blurb: "matching donor leased/busy; retry likely hits" },
  { category: "no_eligible_donor", blurb: "prefix exists but every candidate was rejected" },
  { category: "fresh_by_policy", blurb: "coordinator intentionally planned a fresh slot" },
  { category: "cache_error", blurb: "fallback or plan/realize mismatch" },
  { category: "unexplained_miss", blurb: "eligible candidates with no rejection reason" },
  { category: "miss_reason_unavailable", blurb: "miss without enough evidence to classify" },
  { category: "unknown", blurb: "no cache telemetry for this request" },
];

export function cacheOutcomeLabel(outcome?: CacheOutcome): string {
  if (!outcome) return "n/a";
  if (outcome.category === "hit" && outcome.hitKind) return `${outcome.hitKind} hit`;
  return CACHE_OUTCOME_LABELS[outcome.category];
}

function StatusTag({ status }: { status: DashboardRequest["status"] }) {
  if (status === "ok") return <Tag tone="ok">ok</Tag>;
  if (status === "error") return <Tag tone="err">error</Tag>;
  if (status === "cancelled") return <Tag tone="warn">cancelled</Tag>;
  return <Tag>{status}</Tag>;
}

function PhaseTag({ request }: { request: DashboardRequest }) {
  const { phase } = request;
  if (phase === "loading") return <Tag tone="warn">loading model</Tag>;
  if (phase === "queued") return <Tag>queued</Tag>;
  if (phase === "prefill") {
    const pct = request.prefillTotal
      ? ` ${Math.min(99, Math.round(((request.prefillDone ?? 0) / request.prefillTotal) * 100))}%`
      : "";
    return <Tag tone="pin">prefill{pct}</Tag>;
  }
  if (phase === "decode") return <Tag tone="ok">decoding</Tag>;
  return <Tag>{phase}</Tag>;
}

// Privacy-safe identity badge. cache_session uses the installation-keyed
// session fingerprint (short display, full fingerprint only in tooltip).
// prompt_prefix is explicitly not a session — same opening prompt collides.
export function IdentityTag({ request }: { request: DashboardRequest }) {
  const kind = request.sessionIdentityKind
    ?? (request.sessionFingerprint ? "cache_session" : request.conversation || request.sessionDisplayId ? "prompt_prefix" : undefined);
  const displayId = request.sessionDisplayId
    ?? (kind === "cache_session" ? request.sessionFingerprint?.slice(0, 8) : request.conversation);
  if (!kind || !displayId) return <span className="text-muted">-</span>;

  const hue = parseInt(displayId.replace(/[^0-9a-f]/gi, "").slice(0, 8) || "0", 16) % 360;
  if (kind === "cache_session") {
    const full = request.sessionFingerprint;
    return (
      <span
        className="inline-flex max-w-full items-center gap-1 border px-1.5 text-[0.7rem] tabular-nums"
        style={{ borderColor: `hsl(${hue} 45% 45%)`, color: `hsl(${hue} 65% 70%)` }}
        title={full
          ? `cache session fingerprint (privacy-safe, installation-keyed)\n${full}`
          : "cache session fingerprint (privacy-safe, installation-keyed)"}
        aria-label={`session ${displayId}`}
      >
        <span className="text-[0.58rem] uppercase tracking-[0.06em] text-muted">session</span>
        <span className="truncate">{displayId}</span>
      </span>
    );
  }

  return (
    <span
      className="inline-flex max-w-full items-center gap-1 border border-soft-border px-1.5 text-[0.7rem] tabular-nums text-muted"
      title="prompt-prefix grouping (model + system + first user message hash). Not a session identity — distinct cache.session values that share the same opening prompt will collide here."
      aria-label={`prefix ${displayId}`}
    >
      <span className="text-[0.58rem] uppercase tracking-[0.06em]">prefix</span>
      <span className="truncate">{displayId}</span>
    </span>
  );
}

const phaseOrder: Record<DashboardRequest["phase"], number> = { decode: 0, prefill: 1, loading: 2, queued: 3, done: 4 };

// Share of the prompt served from the KV cache, clamped to 0-100. Undefined
// when the prompt size is unknown so the UI never invents a percentage.
export function cacheReusePercent(request: Pick<DashboardRequest, "reusedTokens" | "promptTokens">): number | undefined {
  const { reusedTokens, promptTokens } = request;
  if (reusedTokens === undefined || promptTokens === undefined || promptTokens <= 0) return undefined;
  return Math.max(0, Math.min(100, Math.round((reusedTokens / promptTokens) * 100)));
}

// Intent is orthogonal to cache outcome: a side request can hit or miss.
export function IntentTag({ request }: { request: DashboardRequest }) {
  if (!request.sideRequest) return null;
  return (
    <Tag tone="warn" title="side request: branched from primary work (title generation, background checks); intent is independent of cache hit/miss">
      side request
    </Tag>
  );
}

export function HistoricalTag({ historical }: { historical?: boolean }) {
  if (!historical) return null;
  return (
    <Tag
      tone="default"
      title="historical: this decision is from a previous server or worker launch; its KV is no longer resident"
      ariaLabel="historical cache decision"
    >
      historical
    </Tag>
  );
}

export function CacheTag({ request }: { request: DashboardRequest }) {
  if (request.cacheEligibility === "no_intent") {
    return <Tag title="not cache-eligible: the request supplied no cache intent">cache n/a · no intent</Tag>;
  }
  if (request.cacheEligibility === "no_admission") {
    return <Tag title="not cache-eligible: the request ended before a cache admission decision">cache n/a · no admission</Tag>;
  }
  const outcome = request.cacheOutcome;
  const slot = request.slot !== undefined ? ` · s${request.slot}` : "";
  const skipped = outcome?.boundariesSkipped && outcome.boundariesSkipped > 0
    ? ` · ${outcome.boundariesSkipped} boundary skip${outcome.boundariesSkipped === 1 ? "" : "s"}`
    : "";
  const pct = cacheReusePercent(request);

  // Prefer server classification when present; fall back to raw hit/miss so
  // older payloads without cacheOutcome still render something honest.
  if (outcome) {
    const label = cacheOutcomeLabel(outcome);
    const detail = outcome.category === "hit"
      ? ` · ${fmtTokens(request.reusedTokens ?? 0)} tok${pct !== undefined ? ` · ${pct}%` : ""}${slot}`
      : request.cacheHit === false
        ? ` · ${fmtTokens(request.reusedTokens ?? 0)} tok${slot}`
        : "";
    return (
      <Tag
        tone={CACHE_OUTCOME_TONES[outcome.category]}
        title={outcome.reason}
        ariaLabel={`cache ${label}${detail}${skipped}`}
      >
        {`cache ${label}${detail}${skipped}`}
      </Tag>
    );
  }

  if (request.cacheHit === true) {
    return (
      <Tag tone="hit" title="prefix cache hit: reused tokens came from the KV cache instead of being re-prefilled" ariaLabel="cache hit">
        {`cache hit · ${fmtTokens(request.reusedTokens ?? 0)} tok${pct !== undefined ? ` · ${pct}%` : ""}${slot}`}
      </Tag>
    );
  }
  if (request.cacheHit === false) {
    return (
      <Tag title="prefix cache miss: the full prompt was prefilled" ariaLabel="cache miss">
        {`cache miss · ${fmtTokens(request.reusedTokens ?? 0)} tok${slot}`}
      </Tag>
    );
  }
  return <Tag title="cache telemetry unavailable for this request" ariaLabel="cache unavailable">cache n/a</Tag>;
}

export function CacheOutcomeLegend() {
  return (
    <details className="border-t border-soft-border px-3 py-2 text-[0.72rem] text-muted">
      <summary className="cursor-pointer select-none text-[0.66rem] uppercase tracking-[0.06em] hover:text-foreground">
        cache outcome legend
      </summary>
      <ul className="mt-1.5 grid gap-1 sm:grid-cols-2">
        {CACHE_OUTCOME_LEGEND.map((entry) => (
          <li key={entry.category} className="flex min-w-0 items-start gap-1.5">
            <Tag tone={CACHE_OUTCOME_TONES[entry.category]} ariaLabel={CACHE_OUTCOME_LABELS[entry.category]}>
              {CACHE_OUTCOME_LABELS[entry.category]}
            </Tag>
            <span className="min-w-0 leading-snug">{entry.blurb}</span>
          </li>
        ))}
        <li className="flex min-w-0 items-start gap-1.5 sm:col-span-2">
          <Tag tone="default" ariaLabel="historical">historical</Tag>
          <span className="min-w-0 leading-snug">decision from a previous server/worker launch; KV no longer resident</span>
        </li>
      </ul>
    </details>
  );
}

export function ActiveRequests({ requests, now, onSelect }: { requests: DashboardRequest[]; now: number; onSelect: (id: string) => void }) {
  const sorted = [...requests].sort((a, b) => (phaseOrder[a.phase] - phaseOrder[b.phase]) || a.startedAt - b.startedAt);
  return (
    <Panel title="active requests" count={requests.length || ""}>
      {requests.length ? (
        <Table headers={["started", "session / prefix", "model", "phase", "elapsed", { label: "msgs", numeric: true }, "endpoint", "stream"]}>
          {sorted.map((request) => (
            <tr key={request.id} className="cursor-pointer hover:bg-panel-strong" onClick={() => onSelect(request.id)}>
              <Td>{fmtClock(request.startedAt)}</Td>
              <Td><IdentityTag request={request} /></Td>
              <Td className="max-w-[260px] overflow-hidden text-ellipsis">{request.model}</Td>
              <Td><PhaseTag request={request} /></Td>
              <Td>{fmtDuration(now - request.startedAt)}</Td>
              <Td numeric>{request.messageCount ?? "-"}</Td>
              <Td>{request.endpoint}</Td>
              <Td>{request.stream ? "sse" : "json"}</Td>
            </tr>
          ))}
        </Table>
      ) : (
        <Empty>idle</Empty>
      )}
    </Panel>
  );
}

export function RecentRequests({ requests, onSelect }: { requests: DashboardRequest[]; onSelect: (id: string) => void }) {
  return (
    <Panel title="recent requests" count={requests.length ? `${requests.length} shown · click a row for detail` : ""}>
      {requests.length ? (
        <>
          <Table
            headers={[
              "time",
              "session / prefix",
              "model",
              "status",
              "priority",
              "structured output",
              { label: "queue", numeric: true },
              { label: "load", numeric: true },
              { label: "ttft", numeric: true },
              { label: "total", numeric: true },
              { label: "in", numeric: true },
              { label: "out", numeric: true },
              { label: "tok/s", numeric: true },
              "intent / cache",
              "finish",
              { label: "tools", numeric: true },
            ]}
          >
            {requests.map((request) => (
              <tr
                key={request.id}
                className="cursor-pointer hover:bg-panel-strong"
                onClick={() => onSelect(request.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(request.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-label={`request ${request.id} detail`}
              >
                <Td>{fmtClock(request.startedAt)}</Td>
                <Td><IdentityTag request={request} /></Td>
                <Td className="max-w-[160px] overflow-hidden text-ellipsis sm:max-w-[240px]" title={request.error}>
                  {request.model}
                </Td>
                <Td><StatusTag status={request.status} /></Td>
                <Td><Tag tone={request.priority === "interactive" ? "ok" : request.priority === "background" ? "warn" : undefined}>{request.priority ?? "normal"}</Tag></Td>
                {request.structuredOutput ? <Td className="max-w-[150px]">
                  <span title={`${request.structuredOutput.kind} · ${request.structuredOutput.requestedStrength}`}>
                    {request.structuredOutput.requestedStrength} · {request.structuredOutput.backendMode ?? "pending"}
                    {request.structuredOutput.outcome ? ` · ${request.structuredOutput.outcome.replaceAll("_", " ")}` : ""}
                  </span>
                </Td> : <Td>-</Td>}
                <Td numeric>{request.queuedMs !== undefined && request.queuedMs > 500 ? fmtDuration(request.queuedMs) : "-"}</Td>
                <Td numeric>{request.loadMs !== undefined && request.loadMs > 500 ? fmtDuration(request.loadMs) : "-"}</Td>
                <Td numeric>{fmtDuration(request.ttftMs)}</Td>
                <Td numeric>{fmtDuration(request.durationMs)}</Td>
                <Td numeric>{fmtTokens(request.promptTokens)}</Td>
                <Td numeric>{fmtTokens(request.completionTokens)}</Td>
                <Td numeric>{request.tokensPerSecond ?? "-"}</Td>
                <Td>
                  <div className="flex max-w-[280px] flex-wrap items-center gap-1 sm:max-w-none">
                    <IntentTag request={request} />
                    <HistoricalTag historical={request.historical} />
                    <CacheTag request={request} />
                  </div>
                </Td>
                <Td>{request.finishReason ?? "-"}</Td>
                <Td numeric>{request.toolCalls ?? "-"}</Td>
              </tr>
            ))}
          </Table>
          <CacheOutcomeLegend />
        </>
      ) : (
        <Empty>no requests yet</Empty>
      )}
    </Panel>
  );
}
