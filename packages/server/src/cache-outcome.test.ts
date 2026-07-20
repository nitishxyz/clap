import { describe, expect, test } from "bun:test";
import { classifyCacheOutcome, type CacheOutcomeCategory } from "./cache-outcome";
import type { CacheCandidateDiagnostic } from "./cache-event-store";

const candidate = (overrides: Partial<CacheCandidateDiagnostic> & { slot: number; sharedPrefixTokens: number }): CacheCandidateDiagnostic => ({
  ...overrides,
});

function expectCategory(
  input: Parameters<typeof classifyCacheOutcome>[0],
  category: CacheOutcomeCategory,
  options?: Parameters<typeof classifyCacheOutcome>[1],
) {
  const outcome = classifyCacheOutcome(input, options);
  expect(outcome.category).toBe(category);
  expect(outcome.reason.length).toBeGreaterThan(0);
  expect(outcome.evidence.length).toBeGreaterThan(0);
  return outcome;
}

describe("classifyCacheOutcome", () => {
  test("hit session/branch/checkpoint kinds", () => {
    expect(expectCategory({ hit: true, reusedTokens: 64, reuseKind: "slot", donorSlot: 1 }, "hit").hitKind).toBe("session");
    expect(expectCategory({ hit: true, reusedTokens: 32, reuseKind: "branch", donorSlot: 2 }, "hit").hitKind).toBe("branch");
    expect(expectCategory({ hit: true, reusedTokens: 128, reuseKind: "anchor", donorSlot: 0 }, "hit").hitKind).toBe("checkpoint");
  });

  test("unknown when hit telemetry is absent", () => {
    expectCategory({}, "unknown");
  });

  test("cold only with authoritative first decision and empty candidates", () => {
    expectCategory({
      hit: false,
      candidates: [],
      promptTokenCount: 4096,
      isFirstDecisionForWorkerModelDomain: true,
    }, "cold");
  });

  test("empty candidates without first-decision authority is unavailable, not cold", () => {
    expectCategory({
      hit: false,
      candidates: [],
      promptTokenCount: 4096,
    }, "miss_reason_unavailable");
    expectCategory({
      hit: false,
      candidates: [],
      promptTokenCount: 4096,
      isFirstDecisionForWorkerModelDomain: false,
    }, "miss_reason_unavailable");
  });

  test("empty candidates with min_prefix miss reason is no_shared_prefix without guessing cold", () => {
    expectCategory({
      hit: false,
      candidates: [],
      missReason: "min_prefix",
    }, "no_shared_prefix");
  });

  test("below checkpoint when first decision and prompt under minimum", () => {
    expectCategory({
      hit: false,
      candidates: [],
      promptTokenCount: 512,
      isFirstDecisionForWorkerModelDomain: true,
    }, "below_checkpoint", { checkpointMinimumTokens: 2048 });
  });

  test("isolated when shared prefix exists only across namespace/model domain", () => {
    expectCategory({
      hit: false,
      candidates: [
        candidate({ slot: 1, sharedPrefixTokens: 400, namespaceCompatible: false, rejection: "namespace" }),
        candidate({ slot: 2, sharedPrefixTokens: 200, modelCompatible: false, rejection: "model_domain" }),
      ],
    }, "isolated");
  });

  test("no shared prefix when warm candidates share zero tokens", () => {
    expectCategory({
      hit: false,
      candidates: [
        candidate({ slot: 0, sharedPrefixTokens: 0 }),
        candidate({ slot: 1, sharedPrefixTokens: 0 }),
      ],
    }, "no_shared_prefix");
  });

  test("donor busy when all relevant candidates are busy/leased", () => {
    expectCategory({
      hit: false,
      candidates: [
        candidate({ slot: 3, sharedPrefixTokens: 80, rejection: "busy_lease", busyEligible: false }),
        candidate({ slot: 4, sharedPrefixTokens: 40, leaseEligible: false, rejection: "busy_lease" }),
      ],
    }, "donor_busy");
  });

  test("no eligible donor for other rejections", () => {
    expectCategory({
      hit: false,
      candidates: [
        candidate({ slot: 1, sharedPrefixTokens: 90, rejection: "capacity" }),
        candidate({ slot: 2, sharedPrefixTokens: 70, rejection: "generation" }),
      ],
    }, "no_eligible_donor");
  });

  test("fresh by policy", () => {
    expectCategory({ hit: false, missReason: "fresh_by_policy", candidates: [] }, "fresh_by_policy");
    expectCategory({ hit: false, missReason: "policy_fresh" }, "fresh_by_policy");
  });

  test("cache error from fallback or plan/realize mismatch", () => {
    expectCategory({ hit: false, fallback: "copy_retry", candidates: [] }, "cache_error");
    expectCategory({ hit: false, plannedTokens: 100, realizedTokens: 0, candidates: [] }, "cache_error");
    expectCategory({
      hit: false,
      candidates: [candidate({ slot: 2, sharedPrefixTokens: 50, selected: true })],
      plannedTokens: 50,
      realizedTokens: 0,
    }, "cache_error");
  });

  test("unexplained miss when eligible candidates lack rejection", () => {
    expectCategory({
      hit: false,
      candidates: [candidate({ slot: 1, sharedPrefixTokens: 64, eligible: true })],
    }, "unexplained_miss");
  });

  test("miss reason unavailable when candidates field is missing", () => {
    expectCategory({ hit: false }, "miss_reason_unavailable");
    expectCategory({ hit: false, missReason: "busy_lease" }, "miss_reason_unavailable");
  });

  test("attaches skipped boundary count without inventing categories", () => {
    const outcome = expectCategory({
      hit: true,
      reusedTokens: 10,
      reuseKind: "slot",
      stableBoundaries: [
        { status: "resolved" },
        { status: "skipped" },
        { status: "skipped" },
      ],
    }, "hit");
    expect(outcome.boundariesSkipped).toBe(2);
  });

  test("reproducibly classifies old persisted-shaped records without rewriting them", () => {
    // Pre-outcome records: only raw cache fields. Empty candidates must not
    // become cold without first-decision authority.
    const oldEmpty = classifyCacheOutcome({
      hit: false,
      reusedTokens: 0,
      candidates: [],
      promptTokenCount: 3000,
    });
    expect(oldEmpty.category).toBe("miss_reason_unavailable");

    const oldBusy = classifyCacheOutcome({
      hit: false,
      candidates: [candidate({ slot: 3, sharedPrefixTokens: 42, rejection: "busy_lease" })],
      missReason: "busy_lease",
    });
    expect(oldBusy.category).toBe("donor_busy");

    const oldHit = classifyCacheOutcome({
      hit: true,
      reusedTokens: 12,
      reuseKind: "branch",
    });
    expect(oldHit).toMatchObject({ category: "hit", hitKind: "branch" });
  });
});
