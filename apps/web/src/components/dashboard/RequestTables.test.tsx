import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import type { DashboardRequest } from "@/lib/api";
import { CacheDecisionSection } from "./RequestDetail";
import { cacheReusePercent, RecentRequests } from "./RequestTables";

const base: DashboardRequest = {
  id: "r1",
  startedAt: new Date("2026-07-20T11:00:00.000Z").getTime(),
  model: "mlx/model",
  endpoint: "/v1/chat/completions",
  stream: true,
  status: "ok",
  phase: "done",
};

function row(request: DashboardRequest): string {
  return renderToStaticMarkup(<RecentRequests requests={[request]} onSelect={() => undefined} />);
}

describe("recent request intent and cache badges", () => {
  test("side request with cache hit shows intent and cache as separate badges", () => {
    const html = row({ ...base, sideRequest: true, cacheHit: true, reusedTokens: 64, promptTokens: 74 });
    expect(html).toContain("side request");
    expect(html).toContain("cache hit · 64 tok · 86%");
    expect(html).not.toContain(">cache miss");
  });

  test("side request with cache miss still reports the miss", () => {
    const html = row({ ...base, sideRequest: true, cacheHit: false, reusedTokens: 0, promptTokens: 74 });
    expect(html).toContain("side request");
    expect(html).toContain("cache miss · 0 tok");
    expect(html).not.toContain(">cache hit");
  });

  test("non-side hit shows only the cache badge with reuse percentage", () => {
    const html = row({ ...base, cacheHit: true, reusedTokens: 30, promptTokens: 100 });
    expect(html).not.toContain("side request");
    expect(html).toContain("cache hit · 30 tok · 30%");
  });

  test("absent telemetry renders an explicit cache n/a badge", () => {
    const html = row({ ...base });
    expect(html).toContain("cache n/a");
    expect(html).not.toContain(">cache hit");
    expect(html).not.toContain(">cache miss");
  });

  test("cancelled request with retained telemetry still shows the decision", () => {
    const html = row({ ...base, status: "cancelled", cacheHit: true, reusedTokens: 64, promptTokens: 74 });
    expect(html).toContain("cancelled");
    expect(html).toContain("cache hit · 64 tok · 86%");
  });

  test("errored request with retained telemetry still shows the decision", () => {
    const html = row({ ...base, status: "error", error: "boom", cacheHit: false, reusedTokens: 0, promptTokens: 74 });
    expect(html).toContain("error");
    expect(html).toContain("cache miss · 0 tok");
  });
});

describe("cacheReusePercent", () => {
  test("computes reusedTokens / promptTokens as a rounded percentage", () => {
    expect(cacheReusePercent({ reusedTokens: 64, promptTokens: 74 })).toBe(86);
    expect(cacheReusePercent({ reusedTokens: 1, promptTokens: 3 })).toBe(33);
  });

  test("clamps out-of-range ratios", () => {
    expect(cacheReusePercent({ reusedTokens: 120, promptTokens: 100 })).toBe(100);
    expect(cacheReusePercent({ reusedTokens: 0, promptTokens: 100 })).toBe(0);
  });

  test("is undefined when the prompt count is missing or zero", () => {
    expect(cacheReusePercent({ reusedTokens: 64, promptTokens: 0 })).toBeUndefined();
    expect(cacheReusePercent({ reusedTokens: 64 })).toBeUndefined();
    expect(cacheReusePercent({ promptTokens: 74 })).toBeUndefined();
  });
});

describe("cache decision section", () => {
  test("groups the full decision telemetry", () => {
    const html = renderToStaticMarkup(
      <CacheDecisionSection
        record={{
          ...base,
          sideRequest: true,
          cacheHit: true,
          reusedTokens: 64,
          promptTokens: 74,
          reuseKind: "anchor",
          reuseScope: "conversation",
          cacheNamespace: "mlx/model:conv:ab12cd",
          plannedReuseTokens: 64,
          realizedReuseTokens: 64,
          donorSlot: 1,
          targetSlot: 2,
          evictedSlots: [3],
          cacheFallback: "namespace-miss",
          cacheDecisionUs: 420,
        }}
      />,
    );
    expect(html).toContain("cache decision");
    expect(html).toContain("hit");
    expect(html).toContain("side request");
    expect(html).toContain("64 tok · 86%");
    expect(html).toContain("anchor");
    expect(html).toContain("conversation");
    expect(html).toContain("mlx/model:conv:ab12cd");
    expect(html).toContain("64 tok / 64 tok");
    expect(html).toContain("s1 → s2");
    expect(html).toContain("s3");
    expect(html).toContain("namespace-miss");
    expect(html).toContain("420µs");
  });

  test("marks missing telemetry as unavailable instead of a dash", () => {
    const html = renderToStaticMarkup(<CacheDecisionSection record={{ ...base }} />);
    expect(html).toContain("primary");
    expect(html).not.toContain(">-<");
    expect(html.match(/unavailable/g)?.length).toBeGreaterThan(5);
  });

  test("keeps the decision visible for a cancelled request", () => {
    const html = renderToStaticMarkup(
      <CacheDecisionSection record={{ ...base, status: "cancelled", cacheHit: true, reusedTokens: 64, promptTokens: 74 }} />,
    );
    expect(html).toContain("hit");
    expect(html).toContain("from cancelled request");
    expect(html).toContain("64 tok · 86%");
  });

  test("keeps the decision visible for an errored request", () => {
    const html = renderToStaticMarkup(
      <CacheDecisionSection record={{ ...base, status: "error", error: "boom", cacheHit: false, reusedTokens: 0, promptTokens: 74 }} />,
    );
    expect(html).toContain("miss");
    expect(html).toContain("from error request");
    expect(html).toContain("0 tok");
  });
});
