import { describe, expect, test } from "bun:test";
import {
  classifyPersistedCacheOutcome,
  firstDecisionIdsForWorkerModelDomain,
  sessionDisplayIdentity,
} from "./dashboard-identity";
import type { PersistedCacheDecision } from "./cache-event-store";

function event(
  id: string,
  overrides: Partial<PersistedCacheDecision> = {},
): PersistedCacheDecision {
  return {
    schemaVersion: 2,
    source: "persisted",
    requestId: id,
    timestamp: Date.now(),
    serverLaunchId: "server",
    model: "model-a",
    status: "ok",
    cache: { hit: false, reusedTokens: 0, candidates: [] },
    promptTokenCount: 4096,
    ...overrides,
  };
}

describe("sessionDisplayIdentity", () => {
  test("four distinct session fingerprints with the same prompt prefix render distinct display ids", () => {
    const promptPrefixId = "116deb";
    const sessions = [
      "a1b2c3d4e5f60718293a4b5c6d7e8f90aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "b2c3d4e5f60718293a4b5c6d7e8f90a1bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "c3d4e5f60718293a4b5c6d7e8f90a1b2cccccccccccccccccccccccccccccccc",
      "d4e5f60718293a4b5c6d7e8f90a1b2c3dddddddddddddddddddddddddddddddd",
    ];
    const displays = sessions.map((sessionFingerprint) =>
      sessionDisplayIdentity({ sessionFingerprint, promptPrefixId }));
    expect(new Set(displays.map((item) => item.sessionDisplayId)).size).toBe(4);
    for (const item of displays) {
      expect(item.sessionIdentityKind).toBe("cache_session");
      expect(item.sessionDisplayId).toHaveLength(8);
      expect(item.promptPrefixId).toBe("116deb");
      expect(item.sessionFingerprint).toHaveLength(64);
      // Never expose a raw session string — only the fingerprint.
      expect(item.sessionDisplayId).not.toMatch(/^S[1-4]$/);
    }
  });

  test("no-session rows fall back to prompt-prefix kind and id", () => {
    const identity = sessionDisplayIdentity({ promptPrefixId: "116deb" });
    expect(identity).toEqual({
      sessionDisplayId: "116deb",
      sessionIdentityKind: "prompt_prefix",
      promptPrefixId: "116deb",
    });
  });

  test("session fingerprint wins over prompt prefix for display", () => {
    const identity = sessionDisplayIdentity({
      sessionFingerprint: "abcdef0123456789ffffffffffffffffffffffffffffffffffffffffffffffff",
      promptPrefixId: "116deb",
    });
    expect(identity.sessionIdentityKind).toBe("cache_session");
    expect(identity.sessionDisplayId).toBe("abcdef01");
  });
});

describe("firstDecisionIdsForWorkerModelDomain", () => {
  test("marks only the earliest decision per worker+model domain", () => {
    const first = firstDecisionIdsForWorkerModelDomain([
      { id: "r2", timestamp: 200, model: "m", workerLaunchId: "w1" },
      { id: "r1", timestamp: 100, model: "m", workerLaunchId: "w1" },
      { id: "r3", timestamp: 150, model: "m", workerLaunchId: "w2" },
      { id: "r4", timestamp: 50, model: "other", workerLaunchId: "w1" },
    ]);
    expect(first.has("r1")).toBe(true);
    expect(first.has("r2")).toBe(false);
    expect(first.has("r3")).toBe(true);
    expect(first.has("r4")).toBe(true);
  });

  test("never marks cold authority without worker launch id", () => {
    const first = firstDecisionIdsForWorkerModelDomain([
      { id: "r1", timestamp: 100, model: "m" },
      { id: "r2", timestamp: 200, model: "m", workerLaunchId: "" },
    ]);
    expect(first.size).toBe(0);
  });
});

describe("classifyPersistedCacheOutcome cold authority", () => {
  test("cold only when isFirstDecisionForWorkerModelDomain is true", () => {
    const cold = classifyPersistedCacheOutcome(event("first", {
      workerLaunchId: "w1",
      cache: { hit: false, candidates: [] },
    }), { isFirstDecisionForWorkerModelDomain: true });
    expect(cold?.category).toBe("cold");

    const later = classifyPersistedCacheOutcome(event("later", {
      workerLaunchId: "w1",
      cache: { hit: false, candidates: [] },
    }), { isFirstDecisionForWorkerModelDomain: false });
    expect(later?.category).toBe("miss_reason_unavailable");

    const noLaunch = classifyPersistedCacheOutcome(event("orphan", {
      cache: { hit: false, candidates: [] },
    }), { isFirstDecisionForWorkerModelDomain: false });
    expect(noLaunch?.category).toBe("miss_reason_unavailable");
  });

  test("ordered domain list yields one cold then non-cold empty candidates", () => {
    const events = [
      event("second", { timestamp: 200, workerLaunchId: "w1", model: "m" }),
      event("first", { timestamp: 100, workerLaunchId: "w1", model: "m" }),
      event("other-worker", { timestamp: 150, workerLaunchId: "w2", model: "m" }),
    ];
    const firstIds = firstDecisionIdsForWorkerModelDomain(events.map((item) => ({
      id: item.requestId,
      timestamp: item.timestamp,
      model: item.model,
      workerLaunchId: item.workerLaunchId,
    })));
    const outcomes = Object.fromEntries(events.map((item) => [
      item.requestId,
      classifyPersistedCacheOutcome(item, {
        isFirstDecisionForWorkerModelDomain: firstIds.has(item.requestId),
      })?.category,
    ]));
    expect(outcomes.first).toBe("cold");
    expect(outcomes.second).toBe("miss_reason_unavailable");
    expect(outcomes["other-worker"]).toBe("cold");
  });
});
