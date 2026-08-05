import type { DashboardData, DashboardTotals } from "@/lib/api";
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

/**
 * KV reuse summary for the tiles. Physical totals cover every admitted
 * request, so stock OpenAI clients that never send cache intent are still
 * represented. Servers that predate those fields fall back to the
 * intent-gated KPI rather than silently reporting zero.
 */
export function cacheSummary(totals: DashboardTotals): {
  hitRate: string;
  detail: string;
  reusedTokens: number;
  reuseSub: string;
} {
  const physicalHits = totals.physicalCacheHits;
  const physicalMisses = totals.physicalCacheMisses;
  const hasPhysical = physicalHits !== undefined && physicalMisses !== undefined;
  const hits = hasPhysical ? physicalHits : totals.cacheHits;
  const misses = hasPhysical ? physicalMisses : totals.cacheMisses;
  const denominator = hasPhysical
    ? hits + misses
    : totals.cacheEligible ?? totals.cacheHits + totals.cacheMisses;
  const reusedTokens = hasPhysical
    ? totals.physicalReusedTokens ?? 0
    : totals.reusedTokens;
  const promptTokens = hasPhysical ? totals.physicalPromptTokens ?? 0 : 0;
  const reusePercent = promptTokens > 0
    ? Math.round((Math.min(reusedTokens, promptTokens) / promptTokens) * 100)
    : undefined;
  return {
    hitRate: denominator ? `${Math.round((hits / denominator) * 100)}%` : "-",
    detail: `${denominator} ${hasPhysical ? "admitted" : "eligible"} · ${hits} hit · ${misses} miss`,
    reusedTokens,
    reuseSub: reusePercent === undefined
      ? "prompt tokens skipped"
      : `${reusePercent}% of prompt tokens skipped`,
  };
}

export function Tiles({ data }: { data: DashboardData }) {
  const totals = data.totals;
  const cache = cacheSummary(totals);
  const cachedCount = data.models.filter((model) => model.status === "available").length;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
      <Tile label="requests" value={totals.requests} sub={`${totals.ok} ok · ${totals.errors} err · ${totals.cancelled} cancelled`} />
      <Tile label="active now" value={data.active.length} />
      <Tile label="tokens in" value={fmtTokens(totals.promptTokens)} />
      <Tile label="tokens out" value={fmtTokens(totals.completionTokens)} />
      <Tile label="kv cache" value={cache.hitRate} sub={cache.detail} />
      <Tile label="kv reused" value={fmtTokens(cache.reusedTokens)} sub={cache.reuseSub} />
      <Tile label="models loaded" value={data.loaded.length} />
      <Tile label="models cached" value={cachedCount} sub={`${data.models.length} known`} />
    </div>
  );
}
