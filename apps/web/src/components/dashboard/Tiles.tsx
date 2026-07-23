import type { DashboardData } from "@/lib/api";
import { fmtTokens } from "@/lib/format";

function Tile({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="min-w-0 border border-border bg-panel px-3 py-2.5">
      <div className="truncate text-[0.68rem] uppercase tracking-[0.06em] text-muted">{label}</div>
      <div key={String(value)} className="mt-0.5 truncate animate-[value-pulse_0.7s_ease-out] text-xl">{value}</div>
      {sub ? <div className="truncate text-[0.72rem] text-muted" title={sub}>{sub}</div> : null}
    </div>
  );
}

export function Tiles({ data }: { data: DashboardData }) {
  const totals = data.totals;
  const eligible = totals.cacheEligible ?? totals.cacheHits + totals.cacheMisses;
  const hitRate = eligible ? `${Math.round((totals.cacheHits / eligible) * 100)}%` : "-";
  const cachedCount = data.models.filter((model) => model.status === "available").length;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
      <Tile label="requests" value={totals.requests} sub={`${totals.ok} ok · ${totals.errors} err · ${totals.cancelled} cancelled`} />
      <Tile label="active now" value={data.active.length} />
      <Tile label="tokens in" value={fmtTokens(totals.promptTokens)} />
      <Tile label="tokens out" value={fmtTokens(totals.completionTokens)} />
      <Tile label="kv cache" value={hitRate} sub={`${eligible} eligible · ${totals.cacheHits} hit · ${totals.cacheMisses} miss`} />
      <Tile label="kv reused" value={fmtTokens(totals.reusedTokens)} sub="prompt tokens skipped" />
      <Tile label="models loaded" value={data.loaded.length} />
      <Tile label="models cached" value={cachedCount} sub={`${data.models.length} known`} />
    </div>
  );
}
