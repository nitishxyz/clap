import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import type { CacheOutcome, CacheOutcomeCategory, DashboardRequest } from "@/lib/api";
import { CacheDecisionSection } from "./RequestDetail";
import {
  CACHE_OUTCOME_LEGEND,
  cacheOutcomeLabel,
  cacheReusePercent,
  IdentityTag,
  RecentRequests,
} from "./RequestTables";

const base: DashboardRequest = {
  id: "r1",
  startedAt: new Date("2026-07-20T11:00:00.000Z").getTime(),
  model: "mlx/model",
  endpoint: "/v1/chat/completions",
  stream: true,
  status: "ok",
  phase: "done",
};

function outcome(category: CacheOutcomeCategory, extra: Partial<CacheOutcome> = {}): CacheOutcome {
  return {
    category,
    reason: `reason for ${category}`,
    evidence: [`category=${category}`],
    ...extra,
  };
}

function row(request: DashboardRequest): string {
  return renderToStaticMarkup(<RecentRequests requests={[request]} onSelect={() => undefined} />);
}

describe("recent request intent and cache badges", () => {
  test("shows structured constraint mode and outcome without schema content", () => {
    const html = row({ ...base, structuredOutput: {
      kind: "json_schema", requestedStrength: "required", backendMode: "native",
      outcome: "native_validated", repairApplied: false, selectedParser: "structured",
      validationMs: 1.2, schemaFingerprint: "abc123", schemaSize: 42,
    } });
    expect(html).toContain("required · native · native validated");
    expect(html).not.toContain("properties");
  });
  test("side request with cache hit shows intent and cache as separate badges", () => {
    const html = row({
      ...base,
      sideRequest: true,
      cacheHit: true,
      reusedTokens: 64,
      promptTokens: 74,
      cacheOutcome: outcome("hit", { hitKind: "session" }),
    });
    expect(html).toContain("side request");
    expect(html).toContain("cache session hit · 64 tok · 86%");
    expect(html).not.toContain(">cache miss");
  });

  test("side request with cache miss still reports the miss", () => {
    const html = row({
      ...base,
      sideRequest: true,
      cacheHit: false,
      reusedTokens: 0,
      promptTokens: 74,
      cacheOutcome: outcome("cold"),
    });
    expect(html).toContain("side request");
    expect(html).toContain("cache cold · 0 tok");
    expect(html).not.toContain(">cache hit");
  });

  test("non-side hit shows only the cache badge with reuse percentage", () => {
    const html = row({
      ...base,
      cacheHit: true,
      reusedTokens: 30,
      promptTokens: 100,
      cacheOutcome: outcome("hit", { hitKind: "branch" }),
    });
    expect(html).not.toContain("side request");
    expect(html).toContain("cache branch hit · 30 tok · 30%");
  });

  test("absent telemetry renders an explicit cache n/a badge", () => {
    const html = row({ ...base });
    expect(html).toContain("cache n/a");
    expect(html).not.toContain(">cache hit");
    expect(html).not.toContain(">cache miss");
  });

  test("cancelled request with retained telemetry still shows the decision", () => {
    const html = row({
      ...base,
      status: "cancelled",
      cacheHit: true,
      reusedTokens: 64,
      promptTokens: 74,
      cacheOutcome: outcome("hit", { hitKind: "checkpoint" }),
    });
    expect(html).toContain("cancelled");
    expect(html).toContain("cache checkpoint hit · 64 tok · 86%");
  });

  test("errored request with retained telemetry still shows the decision", () => {
    const html = row({
      ...base,
      status: "error",
      error: "boom",
      cacheHit: false,
      reusedTokens: 0,
      promptTokens: 74,
      cacheOutcome: outcome("cache_error"),
    });
    expect(html).toContain("error");
    expect(html).toContain("cache cache error · 0 tok");
  });

  test("renders each classified outcome category as a badge", () => {
    const categories: CacheOutcomeCategory[] = [
      "hit",
      "cold",
      "isolated",
      "below_checkpoint",
      "no_shared_prefix",
      "donor_busy",
      "no_eligible_donor",
      "fresh_by_policy",
      "cache_error",
      "unexplained_miss",
      "miss_reason_unavailable",
      "unknown",
    ];
    for (const category of categories) {
      const html = row({
        ...base,
        cacheHit: category === "hit" ? true : category === "unknown" ? undefined : false,
        reusedTokens: category === "hit" ? 10 : 0,
        promptTokens: 100,
        cacheOutcome: outcome(category, category === "hit" ? { hitKind: "session" } : {}),
      });
      expect(html).toContain(`cache ${cacheOutcomeLabel(outcome(category, category === "hit" ? { hitKind: "session" } : {}))}`);
    }
  });

  test("marks historical rows and skipped boundaries", () => {
    const html = row({
      ...base,
      historical: true,
      cacheHit: false,
      reusedTokens: 0,
      cacheOutcome: outcome("no_shared_prefix", { boundariesSkipped: 2 }),
    });
    expect(html).toContain("historical");
    expect(html).toContain("2 boundary skips");
    expect(html).toContain("cache outcome legend");
  });

  test("falls back to raw hit/miss when outcome is absent on old records", () => {
    const hit = row({ ...base, cacheHit: true, reusedTokens: 64, promptTokens: 74 });
    expect(hit).toContain("cache hit · 64 tok · 86%");
    const miss = row({ ...base, cacheHit: false, reusedTokens: 0 });
    expect(miss).toContain("cache miss · 0 tok");
  });
});

describe("session / prefix identity badges", () => {
  test("four sessions with the same prompt prefix render four distinct session ids", () => {
    const prefix = "116deb";
    const fingerprints = [
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    ];
    const htmls = fingerprints.map((sessionFingerprint, index) =>
      row({
        ...base,
        id: `s${index + 1}`,
        conversation: prefix,
        sessionDisplayId: sessionFingerprint.slice(0, 8),
        sessionIdentityKind: "cache_session",
        sessionFingerprint,
      }));
    const ids = fingerprints.map((fp) => fp.slice(0, 8));
    expect(new Set(ids).size).toBe(4);
    for (const [index, html] of htmls.entries()) {
      expect(html).toContain("session");
      expect(html).toContain(ids[index]!);
      expect(html).not.toContain(`>${prefix}<`);
      expect(html).toContain("cache session fingerprint");
    }
  });

  test("no-session rows show PREFIX not session", () => {
    const html = row({
      ...base,
      conversation: "116deb",
      sessionDisplayId: "116deb",
      sessionIdentityKind: "prompt_prefix",
    });
    expect(html).toContain("prefix");
    expect(html).toContain("116deb");
    expect(html).toContain("prompt-prefix grouping");
    expect(html).not.toContain("aria-label=\"session");
  });

  test("IdentityTag prefers session fingerprint over conversation", () => {
    const html = renderToStaticMarkup(
      <IdentityTag
        request={{
          ...base,
          conversation: "116deb",
          sessionDisplayId: "feedface",
          sessionIdentityKind: "cache_session",
          sessionFingerprint: "feedface0123456789abcdef0123456789abcdef0123456789abcdef01234567",
        }}
      />,
    );
    expect(html).toContain("feedface");
    expect(html).toContain("session");
    expect(html).not.toContain("116deb");
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
          cacheOutcome: outcome("hit", { hitKind: "checkpoint", evidence: ["reusedTokens=64"] }),
        }}
      />,
    );
    expect(html).toContain("cache decision");
    expect(html).toContain("checkpoint hit");
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
    expect(html).toContain("classification");
    expect(html).toContain("reusedTokens=64");
  });

  test("marks missing telemetry as unavailable instead of a dash", () => {
    const html = renderToStaticMarkup(<CacheDecisionSection record={{ ...base }} />);
    expect(html).toContain("primary");
    expect(html).not.toContain(">-<");
    expect(html.match(/unavailable/g)?.length).toBeGreaterThan(5);
  });

  test("keeps the decision visible for a cancelled request", () => {
    const html = renderToStaticMarkup(
      <CacheDecisionSection
        record={{
          ...base,
          status: "cancelled",
          cacheHit: true,
          reusedTokens: 64,
          promptTokens: 74,
          cacheOutcome: outcome("hit", { hitKind: "session" }),
        }}
      />,
    );
    expect(html).toContain("session hit");
    expect(html).toContain("from cancelled request");
    expect(html).toContain("64 tok · 86%");
  });

  test("keeps the decision visible for an errored request", () => {
    const html = renderToStaticMarkup(
      <CacheDecisionSection
        record={{
          ...base,
          status: "error",
          error: "boom",
          cacheHit: false,
          reusedTokens: 0,
          promptTokens: 74,
          cacheOutcome: outcome("donor_busy", { maxBlockedPrefixTokens: 42 }),
          historical: true,
        }}
      />,
    );
    expect(html).toContain("donor busy");
    expect(html).toContain("historical");
    expect(html).toContain("from error request");
    expect(html).toContain("0 tok");
    expect(html).toContain("42 tok");
  });

  test("legend covers every category used by the UI", () => {
    const expected: CacheOutcomeCategory[] = [
      "below_checkpoint",
      "cache_error",
      "cold",
      "donor_busy",
      "fresh_by_policy",
      "hit",
      "isolated",
      "miss_reason_unavailable",
      "no_eligible_donor",
      "no_shared_prefix",
      "unexplained_miss",
      "unknown",
    ];
    expect([...CACHE_OUTCOME_LEGEND.map((entry) => entry.category)].sort()).toEqual([...expected].sort());
  });
});
