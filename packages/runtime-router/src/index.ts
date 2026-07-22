import { isGgufModel, llamaBackendStatus } from "@clap/runtime-llama";
import { isMlxModelDirectorySync, mlxBackendStatus } from "@clap/runtime-mlx";

export { ModelLifecycleManager, normalizeKeepAlive, parseKeepAliveMs, modelLifecycleKey } from "./lifecycle";
export { ResidentWorkerProcess, ResidentWorkerRegistry, type ResidentChatResult, type ResidentUsage,
  type ResidentWorkerHandle, type ResidentWorkerProtocolMode } from "./resident";
export { LegacyWorkerProtocol, allowsLegacyStartupFallback, type LegacyWorkerLine } from "./protocol/legacy-worker-protocol";
export { WorkerProtocolFault, protocolFault, type ProtocolFaultCode, type ProtocolFaultScope } from "./protocol/errors";
export { MAX_WORKER_PROTOCOL_LINE_BYTES, V1WorkerProtocolDecoder } from "./protocol/v1-decoder";
export { V1RequestTracker, type ResidentProtocolFact, type TrackedRequestState } from "./protocol/request-tracker";
export { classifyMemoryPressure, retainedGrowthReserve, selectGlobalActiveLimits, shouldAdjustActiveLimit,
  type ActiveLimitPlan, type ActiveLimitWorker, type GlobalActiveLimitInput,
  type MemoryPressure } from "./concurrency";

export function listBackends() {
  return [llamaBackendStatus(), mlxBackendStatus()];
}

export function selectBackendForModel(model: string) {
  if (isGgufModel(model)) return llamaBackendStatus();
  if (isMlxModelDirectorySync(model)) return mlxBackendStatus();
  return llamaBackendStatus();
}
