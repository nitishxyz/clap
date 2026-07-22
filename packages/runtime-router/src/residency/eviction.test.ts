import { describe, expect, test } from "bun:test";
import type { LifecycleResidencySnapshot, LifecycleResidencyState } from "../lifecycle";
import { estimatedMemory, measuredMemory, unavailableMemory } from "./types";
import { selectIdleEvictionVictims } from "./eviction";

function snapshot(
  key: string,
  overrides: Partial<LifecycleResidencySnapshot> = {},
): LifecycleResidencySnapshot {
  return {
    key,
    state: "idle",
    activeRequests: 0,
    pinned: false,
    always: false,
    loadedAtMs: 10,
    lastUsedAtMs: 20,
    lifecycleVersion: 1,
    retainedValueScore: 0,
    memory: unavailableMemory("not_observed"),
    ...overrides,
  };
}

describe("idle residency eviction", () => {
  test("orders by oldest use, retained value, measured RSS, load time, then key", () => {
    const victims = selectIdleEvictionVictims([
      snapshot("newer", { lastUsedAtMs: 21 }),
      snapshot("valuable", { retainedValueScore: 2 }),
      snapshot("estimated-large", { memory: estimatedMemory(10_000, "model_artifacts") }),
      snapshot("measured-small", { memory: measuredMemory(100, "resident_rss") }),
      snapshot("measured-large", { memory: measuredMemory(200, "resident_rss") }),
      snapshot("loaded-later", { loadedAtMs: 11 }),
      snapshot("key-b"),
      snapshot("key-a"),
    ]);

    expect(victims.map((entry) => entry.key)).toEqual([
      "measured-large", "measured-small", "estimated-large", "key-a", "key-b",
      "loaded-later", "valuable", "newer",
    ]);
  });

  test("excludes unsafe states, active, pinned, always, target, and reserved models", () => {
    const excludedStates: LifecycleResidencyState[] = ["starting", "loading", "active", "closing"];
    const candidates = [
      snapshot("eligible"),
      ...excludedStates.map((state) => snapshot(state, { state })),
      snapshot("active-count", { activeRequests: 1 }),
      snapshot("pinned", { pinned: true }),
      snapshot("always", { always: true }),
      snapshot("target"),
      snapshot("reserved"),
    ];

    expect(selectIdleEvictionVictims(candidates, {
      targetKey: "target",
      reservedKeys: new Set(["reserved"]),
    }).map((entry) => entry.key)).toEqual(["eligible"]);
  });
});
