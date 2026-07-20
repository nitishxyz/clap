import { isGgufModel, llamaBackendStatus } from "@clap/runtime-llama";
import { isMlxModelDirectorySync, mlxBackendStatus } from "@clap/runtime-mlx";

export { ModelLifecycleManager, normalizeKeepAlive, parseKeepAliveMs, modelLifecycleKey } from "./lifecycle";
export { ResidentWorkerProcess, ResidentWorkerRegistry, type ResidentChatResult, type ResidentUsage, type ResidentWorkerHandle } from "./resident";
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
