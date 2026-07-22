export const WORKER_LAUNCH_METADATA_VERSION = 1 as const;

export interface WorkerLaunchIdentity {
  backend: string;
  modelId: string;
  revision?: string | null;
  modelPath: string;
}

export type WorkerLoadState = "not_started" | "starting" | "loading" | "resident" | "closing";

export interface WorkerLoadStateEvent {
  readonly key: string;
  readonly backend: string;
  readonly loadState: WorkerLoadState;
  readonly pid?: number;
  readonly atMs: number;
}

export type WorkerRssSampler = (pid: number) => Promise<number | null | undefined>;

export interface WorkerModelDescriptor {
  modelId: string;
  revision?: string | null;
  artifactBytes?: number;
  architecture?: string;
  modelType?: string;
  quantization?: string;
  context?: number;
  configuredContext?: number;
  kv?: { type?: string; bytesPerToken?: number };
  cacheBudget?: number;
}

export interface WorkerLaunchPaths {
  clapHome: string;
  backend: string;
  modelHash: string;
  launchId: string;
  directory: string;
  stderrPath: string;
  metadataPath: string;
  modelPathFingerprint: string;
}

export interface WorkerLaunchMetadata {
  version: typeof WORKER_LAUNCH_METADATA_VERSION;
  launchId: string;
  modelId: string;
  modelPathFingerprint: string;
  backend: string;
  pid: number;
  command: string[];
  protocolVersion: string;
  startedAt: string;
  readyAt?: string;
  endedAt?: string;
  exitStatus?: number | null;
  crashClassification?: string | null;
}

export type WorkerRequestPhase = "handshake" | "load" | "prefill" | "decode" | "idle";
export type WorkerCrashClassification = WorkerRequestPhase | "protocol_fault" | "unexpected_exit_0" | "expected_exit" | "spawn_failure";

export interface WorkerLaunchContext {
  paths: WorkerLaunchPaths;
  metadata: WorkerLaunchMetadata;
  phase: WorkerRequestPhase;
  protocolFault: boolean;
  releaseActive: () => void;
}

export interface LaunchRetentionLimits {
  maxLaunchesPerModel: number;
  maxBytesPerBackend: number;
}
export type ResidentCacheInfo = {
  hit?: boolean;
  reusedTokens?: number;
  reuseKind?: "slot" | "branch" | "anchor";
  reuseScope?: "system" | "conversation" | "session" | "agent" | "project" | "harness" | "tenant";
  sideRequest?: boolean;
  slot?: number;
  namespace?: string;
  donorSlot?: number;
  targetSlot?: number;
  evictedSlots?: number[];
  decisionUs?: number;
  plannedReuseTokens?: number;
  realizedReuseTokens?: number;
  fallback?: string;
  missReason?: string;
  workerLaunchId?: string;
  donorGeneration?: number;
  targetGeneration?: number;
  evictions?: Array<{ slot: number; reason?: string }>;
  candidates?: Array<{
    slot: number;
    generation?: number;
    state?: string;
    sharedPrefixTokens: number;
    namespaceCompatible?: boolean;
    modelCompatible?: boolean;
    sessionCompatible?: boolean;
    generationCompatible?: boolean;
    busyEligible?: boolean;
    leaseEligible?: boolean;
    materialized?: boolean;
    trimEligible?: boolean;
    copyEligible?: boolean;
    eligible?: boolean;
    selected?: boolean;
    rejection?: "namespace" | "model_domain" | "generation" | "capability" | "busy_lease" | "materialization" | "session" | "nontrim" | "min_prefix" | "capacity" | "absent_anchor" | "lower_rank";
  }>;
  systemTokenHash?: string;
  systemTokenCount?: number;
  toolsTokenHash?: string;
  toolsTokenCount?: number;
  stableBoundaryTokenHash?: string;
  stableBoundaryTokenCount?: number;
  stableBoundaryKind?: string;
  stableBoundaries?: Array<{
    tokenHash?: string;
    tokenCount?: number;
    kind: string;
    label?: string;
    requested: boolean;
    status: "resolved" | "authorized" | "skipped";
    skipReason?: "unsupported_template_boundary" | "non_prefix_template_boundary";
    materialized?: boolean;
  }>;
  promptTokenHash?: string;
  promptTokenCount?: number;
  prefillMs?: number;
};
