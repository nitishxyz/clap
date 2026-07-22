import { isGgufModel, llamaBackendStatus } from "@clap/runtime-llama";
import { isMlxModelDirectorySync, mlxBackendStatus } from "@clap/runtime-mlx";

export { ModelLifecycleManager, normalizeKeepAlive, parseKeepAliveMs, modelLifecycleKey,
  type IdleEvictionResult, type LifecycleRemovalHook, type LifecycleResidencySnapshot,
  type LifecycleResidencySnapshotOptions, type LifecycleResidencyState } from "./lifecycle";
export { ResidentWorkerProcess, ResidentWorkerRegistry, type ResidentChatOptions, type ResidentChatResult, type ResidentUsage,
  type RegistryResidencyConfiguration, type ResidentWorkerHandle } from "./resident";
export type { CacheIdentity } from "@clap/worker-protocol";
export { WorkerProtocolFault, protocolFault, type ProtocolFaultCode, type ProtocolFaultScope } from "./protocol/errors";
export { MAX_WORKER_PROTOCOL_LINE_BYTES, V1WorkerProtocolDecoder } from "./protocol/v1-decoder";
export { V1RequestTracker, type ResidentProtocolFact, type TrackedRequestState } from "./protocol/request-tracker";
export { classifyMemoryPressure, retainedGrowthReserve, selectGlobalActiveLimits, shouldAdjustActiveLimit,
  type ActiveLimitPlan, type ActiveLimitWorker, type GlobalActiveLimitInput,
  type MemoryPressure } from "./concurrency";
export { canonicalModelPath, createWorkerLaunchPaths, fingerprintModelPath, hashModelIdentity,
  resolveClapHome } from "./process/launch-paths";
export { DEFAULT_MAX_BYTES_PER_BACKEND, DEFAULT_MAX_LAUNCHES_PER_MODEL, LaunchLogStore, WorkerLaunchLogStore,
  launchRetentionLimits, pruneLaunchLogs, RETENTION_BYTES_ENV, RETENTION_COUNT_ENV,
  writeLaunchMetadataAtomic } from "./process/launch-log-store";
export { WORKER_LAUNCH_METADATA_VERSION, type LaunchRetentionLimits, type WorkerLaunchIdentity,
  type WorkerLaunchMetadata, type WorkerLaunchPaths, type WorkerLaunchContext,
  type WorkerLoadState, type WorkerLoadStateEvent, type WorkerModelDescriptor, type WorkerRssSampler,
  type WorkerRequestPhase, type WorkerCrashClassification } from "./process/types";
export { classifyWorkerCrash, classifyWorkerExit, classifyWorkerExitPhase, type WorkerExitClassification,
  type WorkerExitPhase } from "./process/crash-classification";
export { mapWorkerResultPayload, mapWorkerTelemetryPayload } from "./process/result-mapper";
export * from "./residency";

export function listBackends() {
  return [llamaBackendStatus(), mlxBackendStatus()];
}

export function selectBackendForModel(model: string) {
  if (isGgufModel(model)) return llamaBackendStatus();
  if (isMlxModelDirectorySync(model)) return mlxBackendStatus();
  return llamaBackendStatus();
}
