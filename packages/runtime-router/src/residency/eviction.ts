import type { LifecycleResidencySnapshot } from "../lifecycle";

export interface IdleEvictionSelectionOptions {
  readonly targetKey?: string;
  readonly reservedKeys?: ReadonlySet<string>;
}

export function selectIdleEvictionVictims(
  snapshots: readonly LifecycleResidencySnapshot[],
  options: IdleEvictionSelectionOptions = {},
): LifecycleResidencySnapshot[] {
  return snapshots
    .filter((snapshot) => isEligible(snapshot, options))
    .sort(compareVictims);
}

export function compareIdleEvictionVictims(
  left: LifecycleResidencySnapshot,
  right: LifecycleResidencySnapshot,
): number {
  return compareVictims(left, right);
}

function isEligible(snapshot: LifecycleResidencySnapshot, options: IdleEvictionSelectionOptions): boolean {
  return snapshot.state === "idle"
    && snapshot.activeRequests === 0
    && !snapshot.pinned
    && !snapshot.always
    && snapshot.key !== options.targetKey
    && !options.reservedKeys?.has(snapshot.key);
}

function compareVictims(left: LifecycleResidencySnapshot, right: LifecycleResidencySnapshot): number {
  return left.lastUsedAtMs - right.lastUsedAtMs
    || left.retainedValueScore - right.retainedValueScore
    || measuredBytes(right) - measuredBytes(left)
    || left.loadedAtMs - right.loadedAtMs
    || left.key.localeCompare(right.key);
}

function measuredBytes(snapshot: LifecycleResidencySnapshot): number {
  return snapshot.memory.source === "measured" ? snapshot.memory.bytes : -1;
}
