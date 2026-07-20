import { classifyCacheOutcome, type CacheOutcome, type CacheOutcomeInput } from "./cache-outcome";
import type { CacheCandidateDiagnostic, PersistedCacheDecision } from "./cache-event-store";

export type SessionIdentityKind = "cache_session" | "prompt_prefix";

export type SessionDisplayIdentity = {
  // Privacy-safe short id shown in the table. Never the raw session string.
  sessionDisplayId?: string;
  sessionIdentityKind?: SessionIdentityKind;
  // Full installation-keyed fingerprint for tooltips only (cache_session).
  sessionFingerprint?: string;
  // Prompt-prefix grouping id (legacy `conversation` field). Not a session.
  promptPrefixId?: string;
};

const SESSION_DISPLAY_CHARS = 8;

// Builds a privacy-safe dashboard identity. Explicit cache.session maps to the
// installation HMAC fingerprint (already stored). Without a session, fall back
// to the prompt-prefix grouping id and label it as such — never call it a session.
export function sessionDisplayIdentity(input: {
  sessionFingerprint?: string;
  promptPrefixId?: string;
}): SessionDisplayIdentity {
  const sessionFingerprint = input.sessionFingerprint && input.sessionFingerprint.length > 0
    ? input.sessionFingerprint
    : undefined;
  const promptPrefixId = input.promptPrefixId && input.promptPrefixId.length > 0
    ? input.promptPrefixId
    : undefined;

  if (sessionFingerprint) {
    return {
      sessionFingerprint,
      sessionDisplayId: sessionFingerprint.slice(0, SESSION_DISPLAY_CHARS),
      sessionIdentityKind: "cache_session",
      promptPrefixId,
    };
  }
  if (promptPrefixId) {
    return {
      sessionDisplayId: promptPrefixId,
      sessionIdentityKind: "prompt_prefix",
      promptPrefixId,
    };
  }
  return {};
}

export type FirstDecisionSubject = {
  id: string;
  timestamp: number;
  model: string;
  workerLaunchId?: string;
};

// Authoritative first-decision set for worker+model domains. Requires both a
// worker launch id and stable ordering (timestamp, then id). Missing launch or
// ordering evidence never marks a row as first — so empty candidates cannot
// become cold without that evidence.
export function firstDecisionIdsForWorkerModelDomain(subjects: FirstDecisionSubject[]): Set<string> {
  const ordered = [...subjects].sort((a, b) =>
    a.timestamp - b.timestamp
    || a.id.localeCompare(b.id));
  const seenDomains = new Set<string>();
  const first = new Set<string>();
  for (const subject of ordered) {
    if (!subject.workerLaunchId || !subject.model) continue;
    const domain = `${subject.workerLaunchId}::${subject.model}`;
    if (seenDomains.has(domain)) continue;
    seenDomains.add(domain);
    first.add(subject.id);
  }
  return first;
}

export function classifyPersistedCacheOutcome(
  event: PersistedCacheDecision,
  options: {
    checkpointMinimumTokens?: number;
    isFirstDecisionForWorkerModelDomain?: boolean;
  } = {},
): CacheOutcome | undefined {
  if (event.cache?.hit === undefined) return event.cacheOutcome;
  // Always re-derive at query time so cold authority can be corrected from
  // ordered domain evidence without rewriting the on-disk event.
  const input: CacheOutcomeInput = {
    hit: event.cache.hit,
    reusedTokens: event.cache.reusedTokens,
    reuseKind: event.cache.kind === "slot" || event.cache.kind === "branch" || event.cache.kind === "anchor"
      ? event.cache.kind
      : undefined,
    plannedTokens: event.cache.plannedTokens,
    realizedTokens: event.cache.realizedTokens,
    donorSlot: event.cache.donorSlot,
    fallback: event.cache.fallback,
    missReason: event.cache.missReason,
    promptTokenCount: event.promptTokenCount,
    candidates: event.cache.candidates as CacheCandidateDiagnostic[] | undefined,
    stableBoundaries: event.stableBoundaries,
    isFirstDecisionForWorkerModelDomain: options.isFirstDecisionForWorkerModelDomain === true,
  };
  return classifyCacheOutcome(input, { checkpointMinimumTokens: options.checkpointMinimumTokens });
}

export function dashboardIdentityFromPersisted(event: PersistedCacheDecision): SessionDisplayIdentity {
  // Never recompute fingerprints from raw content (unavailable on disk). Use
  // only already-persisted installation-keyed session fingerprint and/or the
  // stored prompt-prefix id.
  return sessionDisplayIdentity({
    sessionFingerprint: event.sessionFingerprint,
    promptPrefixId: event.promptPrefixId,
  });
}
