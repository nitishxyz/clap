import type { CacheCandidateDiagnostic } from "./cache-event-store";

// Evidence-based cache outcome classification. The classifier is pure and
// backend-neutral: it derives a category only from persisted/live telemetry
// fields (launch IDs aside, which are handled by the caller), candidate
// diagnostics and rejection reasons, prompt size, the checkpoint policy
// minimum, plan/fallback fields, and boundary status. It never infers test
// intent and never guesses from session, project, or model names.

export type CacheOutcomeCategory =
  | "hit"
  | "cold"
  | "isolated"
  | "below_checkpoint"
  | "no_shared_prefix"
  | "donor_busy"
  | "no_eligible_donor"
  | "fresh_by_policy"
  | "cache_error"
  | "unexplained_miss"
  | "miss_reason_unavailable"
  | "unknown";

export type CacheOutcome = {
  category: CacheOutcomeCategory;
  reason: string;
  hitKind?: "session" | "branch" | "checkpoint";
  maxBlockedPrefixTokens?: number;
  boundariesSkipped?: number;
  evidence: string[];
};

export type CacheOutcomeInput = {
  hit?: boolean;
  reusedTokens?: number;
  reuseKind?: "slot" | "branch" | "anchor";
  plannedTokens?: number;
  realizedTokens?: number;
  donorSlot?: number;
  fallback?: string;
  missReason?: string;
  promptTokenCount?: number;
  candidates?: CacheCandidateDiagnostic[];
  stableBoundaries?: Array<{ status: string }>;
  // True only when the caller has authoritative evidence that this is the first
  // cache decision for the same model + worker launch (domain). Empty candidates
  // alone must never be guessed as cold. Derived from ordered list construction
  // (timestamp + workerLaunchId + model), never guessed from empty candidates.
  isFirstDecisionForWorkerModelDomain?: boolean;
};

export type CacheOutcomeOptions = {
  checkpointMinimumTokens?: number;
};

const DEFAULT_CHECKPOINT_MINIMUM_TOKENS = 2048;

// Rejections that mean the candidate is intentionally unreachable for this
// request's namespace/model domain: isolation working as designed.
const ISOLATION_REJECTIONS = new Set(["namespace", "model_domain", "capability"]);

function isIsolationRejected(candidate: CacheCandidateDiagnostic): boolean {
  return candidate.namespaceCompatible === false
    || candidate.modelCompatible === false
    || (candidate.rejection !== undefined && ISOLATION_REJECTIONS.has(candidate.rejection));
}

function isBusyRejected(candidate: CacheCandidateDiagnostic): boolean {
  return candidate.rejection === "busy_lease"
    || candidate.busyEligible === false
    || candidate.leaseEligible === false;
}

function skippedBoundaryCount(boundaries: Array<{ status: string }> | undefined): number {
  return (boundaries ?? []).filter((boundary) => boundary.status === "skipped").length;
}

function withBoundaries(outcome: CacheOutcome, input: CacheOutcomeInput): CacheOutcome {
  const skipped = skippedBoundaryCount(input.stableBoundaries);
  if (skipped > 0) outcome.boundariesSkipped = skipped;
  return outcome;
}

function fmtCount(value: number): string {
  return value.toLocaleString("en-US");
}

// Classifies one request's cache outcome. Returns category "unknown" when no
// cache telemetry exists at all; a miss with no candidate evidence stays
// "miss_reason_unavailable" rather than being guessed as cold.
export function classifyCacheOutcome(input: CacheOutcomeInput, options: CacheOutcomeOptions = {}): CacheOutcome {
  const checkpointMinimum = options.checkpointMinimumTokens && options.checkpointMinimumTokens > 0
    ? options.checkpointMinimumTokens
    : DEFAULT_CHECKPOINT_MINIMUM_TOKENS;

  if (input.hit === true) {
    const hitKind = input.reuseKind === "slot" ? "session"
      : input.reuseKind === "branch" ? "branch"
      : input.reuseKind === "anchor" ? "checkpoint"
      : undefined;
    const kindLabel = hitKind ?? "prefix";
    return withBoundaries({
      category: "hit",
      hitKind,
      reason: `${kindLabel} reuse: ${fmtCount(input.reusedTokens ?? 0)} tokens came from the KV cache instead of being re-prefilled`,
      evidence: [
        `reusedTokens=${input.reusedTokens ?? 0}`,
        input.reuseKind ? `reuseKind=${input.reuseKind}` : "reuseKind=unavailable",
        input.donorSlot !== undefined ? `donorSlot=s${input.donorSlot}` : "donorSlot=unavailable",
      ],
    }, input);
  }

  if (input.hit !== false) {
    return withBoundaries({
      category: "unknown",
      reason: "cache telemetry unavailable for this request",
      evidence: ["hit=unavailable"],
    }, input);
  }

  // From here on: a realized miss (hit === false).

  if (input.fallback) {
    return withBoundaries({
      category: "cache_error",
      reason: `cache coordinator fell back (${input.fallback}); the request completed without reuse`,
      evidence: [`fallback=${input.fallback}`],
    }, input);
  }

  const planned = input.plannedTokens ?? 0;
  const realized = input.realizedTokens ?? 0;
  if (planned > 0 && realized === 0) {
    return withBoundaries({
      category: "cache_error",
      reason: `planned reuse of ${fmtCount(planned)} tokens but realized 0; plan/realize mismatch`,
      evidence: [`plannedTokens=${planned}`, `realizedTokens=${realized}`],
    }, input);
  }

  if (input.missReason === "fresh_by_policy" || input.missReason === "policy_fresh") {
    return withBoundaries({
      category: "fresh_by_policy",
      reason: `coordinator intentionally planned a fresh slot (${input.missReason})`,
      evidence: [`missReason=${input.missReason}`],
    }, input);
  }

  const candidates = input.candidates;
  if (candidates === undefined) {
    return withBoundaries({
      category: "miss_reason_unavailable",
      reason: "cache miss; candidate diagnostics unavailable, reason not persisted",
      evidence: [input.missReason ? `missReason=${input.missReason}` : "missReason=unavailable", "candidates=unavailable"],
    }, input);
  }

  if (candidates.length === 0) {
    // Empty candidates is cold only with isFirstDecisionForWorkerModelDomain.
    // Never guess cold from an empty list alone (old records or missing launch).
    if (input.isFirstDecisionForWorkerModelDomain === true) {
      if (input.promptTokenCount !== undefined && input.promptTokenCount < checkpointMinimum) {
        return withBoundaries({
          category: "below_checkpoint",
          reason: `prompt of ${fmtCount(input.promptTokenCount)} tokens is below the automatic checkpoint minimum of ${fmtCount(checkpointMinimum)} tokens and no donor was resident; exact/session cache can still hit for repeated identical prompts`,
          evidence: [
            `promptTokens=${input.promptTokenCount}`,
            `checkpointMinimum=${checkpointMinimum}`,
            "candidates=0",
            "isFirstDecisionForWorkerModelDomain=true",
          ],
        }, input);
      }
      return withBoundaries({
        category: "cold",
        reason: "first decision for this model/worker domain had no resident donor; zero cache candidates at decision time",
        evidence: [
          "candidates=0",
          "isFirstDecisionForWorkerModelDomain=true",
          input.promptTokenCount !== undefined ? `promptTokens=${input.promptTokenCount}` : "promptTokens=unavailable",
        ],
      }, input);
    }

    // Without first-decision authority, empty candidates is not enough to claim
    // cold. Prefer miss_reason_unavailable unless a miss reason already points
    // at a prefix/policy rejection (no_shared_prefix).
    if (
      input.missReason === "min_prefix"
      || input.missReason === "no_shared_prefix"
      || input.missReason === "absent_anchor"
    ) {
      return withBoundaries({
        category: "no_shared_prefix",
        reason: `cache miss with zero candidates and miss reason ${input.missReason}; shared-prefix reuse was not available`,
        evidence: [`candidates=0`, `missReason=${input.missReason}`, "isFirstDecisionForWorkerModelDomain=false"],
      }, input);
    }

    return withBoundaries({
      category: "miss_reason_unavailable",
      reason: "cache miss with zero candidates but no authoritative first-decision evidence for this model/worker domain; not classified as cold",
      evidence: [
        "candidates=0",
        "isFirstDecisionForWorkerModelDomain=false",
        input.missReason ? `missReason=${input.missReason}` : "missReason=unavailable",
        input.promptTokenCount !== undefined ? `promptTokens=${input.promptTokenCount}` : "promptTokens=unavailable",
      ],
    }, input);
  }

  const prefixCandidates = candidates.filter((candidate) => candidate.sharedPrefixTokens > 0);
  const maxShared = prefixCandidates.reduce((max, candidate) => Math.max(max, candidate.sharedPrefixTokens), 0);

  if (prefixCandidates.length === 0) {
    return withBoundaries({
      category: "no_shared_prefix",
      reason: `worker is warm with ${candidates.length} resident slot(s) but none shares a token prefix with this prompt`,
      evidence: [`candidates=${candidates.length}`, "maxSharedPrefix=0"],
    }, input);
  }

  const relevant = prefixCandidates.filter((candidate) => !isIsolationRejected(candidate));
  if (relevant.length === 0) {
    return withBoundaries({
      category: "isolated",
      reason: `namespace/model isolation held: ${prefixCandidates.length} candidate(s) share up to ${fmtCount(maxShared)} tokens but belong to a different namespace or model domain; not a performance failure`,
      maxBlockedPrefixTokens: maxShared,
      evidence: [`isolatedCandidates=${prefixCandidates.length}`, `maxSharedPrefix=${maxShared}`],
    }, input);
  }

  const selected = relevant.filter((candidate) => candidate.selected);
  if (selected.length > 0) {
    return withBoundaries({
      category: "cache_error",
      reason: `a donor was selected (slot ${selected[0].slot}) but 0 tokens were realized; plan/realize mismatch`,
      evidence: [`selectedSlot=${selected[0].slot}`, `plannedTokens=${planned}`, `realizedTokens=${realized}`],
    }, input);
  }

  const unselected = relevant;
  const withoutReason = unselected.filter((candidate) => candidate.rejection === undefined || candidate.eligible === true);
  if (withoutReason.length > 0) {
    return withBoundaries({
      category: "unexplained_miss",
      reason: `${withoutReason.length} reuse-eligible candidate(s) with up to ${fmtCount(maxShared)} shared tokens were not selected and carry no normalized rejection reason`,
      maxBlockedPrefixTokens: maxShared,
      evidence: [`eligibleUnselected=${withoutReason.length}`, `maxSharedPrefix=${maxShared}`, input.missReason ? `missReason=${input.missReason}` : "missReason=unavailable"],
    }, input);
  }

  const busy = unselected.filter(isBusyRejected);
  if (busy.length > 0) {
    const blocked = busy.reduce((max, candidate) => Math.max(max, candidate.sharedPrefixTokens), 0);
    return withBoundaries({
      category: "donor_busy",
      reason: `${busy.length} candidate(s) share up to ${fmtCount(blocked)} tokens but all were busy serving or leased to another request; retry would likely hit`,
      maxBlockedPrefixTokens: blocked,
      evidence: [`busyCandidates=${busy.length}`, `maxBlockedPrefix=${blocked}`],
    }, input);
  }

  if (unselected.every((candidate) => candidate.rejection === "min_prefix")) {
    return withBoundaries({
      category: "no_shared_prefix",
      reason: `exact shared prefix tops out at ${fmtCount(maxShared)} tokens, below the minimum reusable prefix; nothing eligible to reuse`,
      maxBlockedPrefixTokens: maxShared,
      evidence: [`rejection=min_prefix`, `maxSharedPrefix=${maxShared}`],
    }, input);
  }

  const reasons = [...new Set(
    unselected
      .map((candidate) => candidate.rejection)
      .filter((reason): reason is NonNullable<CacheCandidateDiagnostic["rejection"]> => reason !== undefined)
      .map((reason) => String(reason)),
  )];
  return withBoundaries({
    category: "no_eligible_donor",
    reason: `shared prefix of up to ${fmtCount(maxShared)} tokens exists but every candidate was rejected (${reasons.join(", ") || "reason unavailable"})`,
    maxBlockedPrefixTokens: maxShared,
    evidence: [`rejections=${reasons.join("+") || "unavailable"}`, `maxBlockedPrefix=${maxShared}`],
  }, input);
}
