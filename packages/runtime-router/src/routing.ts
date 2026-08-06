import { createHash } from "node:crypto";

export type RoutingWorkerState = "ready" | "draining" | "unhealthy";
export type RoutingLocationKind = "session" | "system" | "tools" | "prefix";

export type RoutingWorkerHeartbeat = {
  workerId: string;
  generation: string;
  modelDomain: string;
  state: RoutingWorkerState;
  observedAt: number;
  activeRequests: number;
  queueDepth: number;
  estimatedQueueWaitMs: number;
  pressure: number;
  loaded: boolean;
  estimatedColdLoadMs: number;
  prefillTokensPerSecond: number;
};

export type RoutingLocation = {
  modelDomain: string;
  shareDomain: string;
  kind: RoutingLocationKind;
  fingerprint: string;
  workerId: string;
  workerGeneration: string;
  tokenCount: number;
  lastSeenAt: number;
};

export type RoutingPrefix = {
  kind: Exclude<RoutingLocationKind, "session">;
  fingerprint: string;
};

export type RoutingQuery = {
  modelDomain: string;
  shareDomain: string;
  sessionFingerprint?: string;
  prefixes?: readonly RoutingPrefix[];
  promptTokens: number;
  affinityFingerprint: string;
  excludedWorkerIds?: ReadonlySet<string>;
  now?: number;
};

export type RoutingCandidate = {
  workerId: string;
  workerGeneration: string;
  locationKind?: RoutingLocationKind;
  matchedTokens: number;
  queueWaitMs: number;
  missingPrefillMs: number;
  pressurePenaltyMs: number;
  coldLoadMs: number;
  totalCostMs: number;
  stickyTieBreak: number;
};

export type RoutingDecision = {
  workerId: string;
  workerGeneration: string;
  reason: "session_locality" | "prefix_locality" | "lowest_cost" | "sticky_fallback";
  directoryHit: boolean;
  locationKind?: RoutingLocationKind;
  matchedTokens: number;
  promptTokens: number;
  queueWaitMs: number;
  missingPrefillMs: number;
  pressurePenaltyMs: number;
  coldLoadMs: number;
  totalCostMs: number;
  candidateCount: number;
  staleLocationsIgnored: number;
  candidates: RoutingCandidate[];
};

export type RoutingDirectoryOptions = {
  now?: () => number;
  workerTtlMs?: number;
  locationTtlMs?: number;
  maxWorkers?: number;
  maxLocations?: number;
  pressurePenaltyMs?: number;
};

export type RoutingDirectorySnapshot = {
  workers: RoutingWorkerHeartbeat[];
  locations: RoutingLocation[];
  limits: {
    workerTtlMs: number;
    locationTtlMs: number;
    maxWorkers: number;
    maxLocations: number;
  };
};

const DEFAULT_WORKER_TTL_MS = 15_000;
const DEFAULT_LOCATION_TTL_MS = 15 * 60_000;
const DEFAULT_MAX_WORKERS = 1_024;
const DEFAULT_MAX_LOCATIONS = 50_000;
const DEFAULT_PRESSURE_PENALTY_MS = 1_000;

function positiveInteger(value: number | undefined, fallback: number): number {
  return Number.isSafeInteger(value) && value! > 0 ? value! : fallback;
}

function finiteNonnegative(value: number, fallback = 0): number {
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function normalizedPressure(value: number): number {
  return Math.min(1, finiteNonnegative(value));
}

function locationKey(location: Pick<RoutingLocation,
  "modelDomain" | "shareDomain" | "kind" | "fingerprint" | "workerId">): string {
  return [location.modelDomain, location.shareDomain, location.kind,
    location.fingerprint, location.workerId].join("\0");
}

function stickyTieBreak(affinityFingerprint: string, workerId: string): number {
  const digest = createHash("sha256").update(affinityFingerprint).update("\0").update(workerId).digest();
  return digest.readUInt32BE(0);
}

export function stableRoutingWorkerId(nodeId: string, modelDomain: string, replica: number): string {
  if (!nodeId) throw new Error("routing node id must not be empty");
  if (!modelDomain) throw new Error("routing model domain must not be empty");
  if (!Number.isSafeInteger(replica) || replica < 0) throw new RangeError("routing replica must be nonnegative");
  return `w_${createHash("sha256").update(nodeId).update("\0").update(modelDomain)
    .update("\0").update(String(replica)).digest("hex").slice(0, 20)}`;
}

export class CacheAwareRoutingDirectory {
  private readonly workers = new Map<string, RoutingWorkerHeartbeat>();
  private readonly locations = new Map<string, RoutingLocation>();
  private readonly now: () => number;
  private readonly workerTtlMs: number;
  private readonly locationTtlMs: number;
  private readonly maxWorkers: number;
  private readonly maxLocations: number;
  private readonly pressurePenaltyMs: number;

  constructor(options: RoutingDirectoryOptions = {}) {
    this.now = options.now ?? Date.now;
    this.workerTtlMs = positiveInteger(options.workerTtlMs, DEFAULT_WORKER_TTL_MS);
    this.locationTtlMs = positiveInteger(options.locationTtlMs, DEFAULT_LOCATION_TTL_MS);
    this.maxWorkers = positiveInteger(options.maxWorkers, DEFAULT_MAX_WORKERS);
    this.maxLocations = positiveInteger(options.maxLocations, DEFAULT_MAX_LOCATIONS);
    this.pressurePenaltyMs = finiteNonnegative(options.pressurePenaltyMs ?? DEFAULT_PRESSURE_PENALTY_MS,
      DEFAULT_PRESSURE_PENALTY_MS);
  }

  heartbeat(input: RoutingWorkerHeartbeat): void {
    if (!input.workerId || !input.generation || !input.modelDomain) {
      throw new Error("routing heartbeat requires worker, generation, and model domain");
    }
    const heartbeat: RoutingWorkerHeartbeat = {
      ...input,
      observedAt: finiteNonnegative(input.observedAt, this.now()),
      activeRequests: Math.floor(finiteNonnegative(input.activeRequests)),
      queueDepth: Math.floor(finiteNonnegative(input.queueDepth)),
      estimatedQueueWaitMs: finiteNonnegative(input.estimatedQueueWaitMs),
      pressure: normalizedPressure(input.pressure),
      estimatedColdLoadMs: finiteNonnegative(input.estimatedColdLoadMs),
      prefillTokensPerSecond: Math.max(1, finiteNonnegative(input.prefillTokensPerSecond, 1)),
    };
    this.workers.set(heartbeat.workerId, heartbeat);
    this.trimWorkers();
  }

  publishLocation(input: RoutingLocation): void {
    if (!input.modelDomain || !input.shareDomain || !input.fingerprint || !input.workerId
      || !input.workerGeneration) {
      throw new Error("routing location is missing an identity field");
    }
    if (!Number.isSafeInteger(input.tokenCount) || input.tokenCount <= 0) {
      throw new RangeError("routing location token count must be positive");
    }
    const location = { ...input, lastSeenAt: finiteNonnegative(input.lastSeenAt, this.now()) };
    this.locations.delete(locationKey(location));
    this.locations.set(locationKey(location), location);
    this.trimLocations();
  }

  invalidateWorker(workerId: string, generation?: string): void {
    const heartbeat = this.workers.get(workerId);
    if (!generation || heartbeat?.generation === generation) this.workers.delete(workerId);
    for (const [key, location] of this.locations) {
      if (location.workerId === workerId && (!generation || location.workerGeneration === generation)) {
        this.locations.delete(key);
      }
    }
  }

  cleanup(now = this.now()): { workers: number; locations: number } {
    let workers = 0;
    let locations = 0;
    for (const [workerId, heartbeat] of this.workers) {
      if (now - heartbeat.observedAt > this.workerTtlMs) {
        this.workers.delete(workerId);
        workers += 1;
      }
    }
    for (const [key, location] of this.locations) {
      const heartbeat = this.workers.get(location.workerId);
      if (now - location.lastSeenAt > this.locationTtlMs || !heartbeat
        || heartbeat.generation !== location.workerGeneration) {
        this.locations.delete(key);
        locations += 1;
      }
    }
    return { workers, locations };
  }

  select(query: RoutingQuery): RoutingDecision | undefined {
    const now = query.now ?? this.now();
    let staleLocationsIgnored = 0;
    const liveWorkers = [...this.workers.values()].filter((worker) => {
      if (query.excludedWorkerIds?.has(worker.workerId)) return false;
      return worker.modelDomain === query.modelDomain && worker.state === "ready"
        && now - worker.observedAt <= this.workerTtlMs;
    });
    if (!liveWorkers.length) return undefined;

    const requestedLocations = new Map<string, RoutingLocationKind>();
    if (query.sessionFingerprint) requestedLocations.set(query.sessionFingerprint, "session");
    for (const prefix of query.prefixes ?? []) requestedLocations.set(prefix.fingerprint, prefix.kind);

    const bestLocationByWorker = new Map<string, RoutingLocation>();
    for (const location of this.locations.values()) {
      const requestedKind = requestedLocations.get(location.fingerprint);
      if (!requestedKind || requestedKind !== location.kind || location.modelDomain !== query.modelDomain
        || location.shareDomain !== query.shareDomain) continue;
      const heartbeat = this.workers.get(location.workerId);
      const stale = now - location.lastSeenAt > this.locationTtlMs || !heartbeat
        || now - heartbeat.observedAt > this.workerTtlMs || heartbeat.state !== "ready"
        || heartbeat.generation !== location.workerGeneration
        || heartbeat.modelDomain !== query.modelDomain;
      if (stale) {
        staleLocationsIgnored += 1;
        continue;
      }
      const existing = bestLocationByWorker.get(location.workerId);
      if (!existing || location.tokenCount > existing.tokenCount
        || (location.tokenCount === existing.tokenCount && location.kind === "session")) {
        bestLocationByWorker.set(location.workerId, location);
      }
    }

    const promptTokens = Math.max(0, Math.floor(finiteNonnegative(query.promptTokens)));
    const candidates = liveWorkers.map((worker): RoutingCandidate => {
      const location = bestLocationByWorker.get(worker.workerId);
      const matchedTokens = Math.min(promptTokens, location?.tokenCount ?? 0);
      const missingTokens = Math.max(0, promptTokens - matchedTokens);
      const missingPrefillMs = (missingTokens / worker.prefillTokensPerSecond) * 1_000;
      const pressurePenaltyMs = worker.pressure * this.pressurePenaltyMs;
      const coldLoadMs = worker.loaded ? 0 : worker.estimatedColdLoadMs;
      const totalCostMs = worker.estimatedQueueWaitMs + missingPrefillMs
        + pressurePenaltyMs + coldLoadMs;
      return {
        workerId: worker.workerId,
        workerGeneration: worker.generation,
        locationKind: location?.kind,
        matchedTokens,
        queueWaitMs: worker.estimatedQueueWaitMs,
        missingPrefillMs,
        pressurePenaltyMs,
        coldLoadMs,
        totalCostMs,
        stickyTieBreak: stickyTieBreak(query.affinityFingerprint, worker.workerId),
      };
    }).sort((left, right) => left.totalCostMs - right.totalCostMs
      || right.matchedTokens - left.matchedTokens
      || right.stickyTieBreak - left.stickyTieBreak
      || left.workerId.localeCompare(right.workerId));

    const selected = candidates[0]!;
    const reason = selected.locationKind === "session" ? "session_locality"
      : selected.locationKind ? "prefix_locality"
      : candidates.every((candidate) => candidate.totalCostMs === selected.totalCostMs)
        ? "sticky_fallback" : "lowest_cost";
    return {
      workerId: selected.workerId,
      workerGeneration: selected.workerGeneration,
      reason,
      directoryHit: selected.locationKind !== undefined,
      locationKind: selected.locationKind,
      matchedTokens: selected.matchedTokens,
      promptTokens,
      queueWaitMs: selected.queueWaitMs,
      missingPrefillMs: selected.missingPrefillMs,
      pressurePenaltyMs: selected.pressurePenaltyMs,
      coldLoadMs: selected.coldLoadMs,
      totalCostMs: selected.totalCostMs,
      candidateCount: candidates.length,
      staleLocationsIgnored,
      candidates,
    };
  }

  snapshot(now = this.now()): RoutingDirectorySnapshot {
    this.cleanup(now);
    return {
      workers: [...this.workers.values()].sort((a, b) => a.workerId.localeCompare(b.workerId)),
      locations: [...this.locations.values()].sort((a, b) => a.lastSeenAt - b.lastSeenAt),
      limits: {
        workerTtlMs: this.workerTtlMs,
        locationTtlMs: this.locationTtlMs,
        maxWorkers: this.maxWorkers,
        maxLocations: this.maxLocations,
      },
    };
  }

  private trimWorkers(): void {
    while (this.workers.size > this.maxWorkers) {
      const oldest = [...this.workers.values()].sort((a, b) => a.observedAt - b.observedAt
        || a.workerId.localeCompare(b.workerId))[0];
      if (!oldest) return;
      this.invalidateWorker(oldest.workerId);
    }
  }

  private trimLocations(): void {
    while (this.locations.size > this.maxLocations) {
      const oldest = this.locations.entries().next().value as [string, RoutingLocation] | undefined;
      if (!oldest) return;
      this.locations.delete(oldest[0]);
    }
  }
}
