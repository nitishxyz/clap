import type { ChatCompletionRequest, LoadedModel, ModelTokenCapabilities } from "@clap/api";
import { WorkerRetentionTelemetrySchema, type CacheIdentity, type MemoryBasis, type MemorySource,
  type StructuredOutputCapabilities, type StructuredOutputContract,
  type WorkerCapabilities } from "@clap/worker-protocol";
import { freemem, totalmem } from "node:os";
import { classifyMemoryPressure, selectGlobalActiveLimits, shouldAdjustActiveLimit,
  type MemoryPressure } from "./concurrency";
import { ResidentWorkerProcess } from "./process/resident-worker-process";
import type { ResidentCacheInfo } from "./process/types";
import { measuredMemory, unavailableMemory, type MemoryValue } from "./residency/types";
import type { WorkerLoadState, WorkerLoadStateEvent, WorkerRssSampler } from "./process/types";
import { ResidencyCoordinator, measuredResidencyMemory,
  type ResidencyCoordinatorDependencies, type ResidencyLifecycleAdapter } from "./residency/coordinator";
import type { LoadReservation, ResidencyModelDescriptor } from "./residency/types";
import { selectIdleEvictionVictims } from "./residency/eviction";

export { ResidentWorkerProcess } from "./process/resident-worker-process";
export type { ResidentCacheInfo } from "./process/types";
export type ResidentBackend = LoadedModel["backend"];

export type EffectiveCapabilities = {
  cache: {
    partialSuffixTrim: boolean;
    partialPrefixBranch: boolean;
    wholeStateCopy: boolean;
    promptBoundarySnapshots: boolean;
    quantizedKv: boolean;
  };
  generation: {
    structuredOutput: StructuredOutputCapabilities;
    toolTemplateSupport: boolean;
  };
  modalities: { input: ["text"]; output: ["text"] };
};

export type ResidentWorkerInfo = {
  pid?: number;
  launchId?: string;
  stderrLogPath?: string; launchMetadataPath?: string; crashClassification?: string;
  state: LoadedModel["worker"]["state"];
  loadState: WorkerLoadState;
  residency?: ResidentWorkerResidencyInfo;
  limitation?: string;
  crashes?: number;
  lastCrashAt?: string;
  memory?: ResidentMlxMemory;
  retention?: ResidentMlxRetention;
  tokenCapabilities?: ModelTokenCapabilities;
  workerCapabilities?: WorkerCapabilities;
  effectiveCapabilities?: EffectiveCapabilities;
  structuredOutputCapabilities?: StructuredOutputCapabilities;
};

export type ResidentWorkerResidencyInfo = {
  estimateBytes: number | null;
  estimateSource: import("./residency/types").EstimatedMemoryBasis | null;
  observedRssBytes: number | null;
  observedRssSource: "resident_rss" | null;
  reservationBytes: number;
  lastAdmissionReason: import("./residency/types").AdmissionReason | null;
  lastEvictionReason: "memory_admission" | null;
};

export type ResidentMlxMemory = {
  activeBytes: number | null;
  activeBytesSource?: MemorySource;
  activeBytesBasis?: MemoryBasis;
  cacheBytes: number | null;
  cacheBytesSource?: MemorySource;
  cacheBytesBasis?: MemoryBasis;
  peakActiveBytes: number | null;
  peakActiveBytesSource?: MemorySource;
  peakActiveBytesBasis?: MemoryBasis;
};

export type ResidentMlxRetention = {
  maxActive: number;
  queued?: number;
  previousMaxActive?: number;
  lastAdjustmentReason?: string;
  lastAdjustmentAt?: string;
  retainedGrowthReserveBytes?: number;
  globalResidentMemoryBytes?: number;
  pressureState?: "normal" | "warning" | "critical";
  activePolicy: {
    mode: "auto" | "fixed";
    selectedMax: number;
    backendCeiling: number;
    hardwareCeiling: number;
    modelCeiling: number;
    memoryCeiling: number;
    reason: string;
    inputs: Record<string, string | number | boolean | null>;
  };
  active: number;
  retainedTotal: number;
  retainedSessions: number;
  retainedAnchors: number;
  retainedBytes: number | null;
  retainedBytesSource?: MemorySource;
  retainedBytesBasis?: MemoryBasis;
  sessionBytes: number | null;
  sessionBytesSource?: MemorySource;
  sessionBytesBasis?: MemoryBasis;
  anchorBytes: number | null;
  anchorBytesSource?: MemorySource;
  anchorBytesBasis?: MemoryBasis;
  evictedBytes?: number | null;
  evictedBytesSource?: MemorySource;
  evictedBytesBasis?: MemoryBasis;
  estimatedRetainedBytes?: number | null;
  estimatedRetainedBytesSource?: MemorySource;
  estimatedRetainedBytesBasis?: MemoryBasis;
  automaticCheckpointCount?: number;
  automaticCheckpointBytes?: number;
  automaticCheckpointBudgetBytes?: number;
  automaticCheckpointsEnabled?: boolean;
  automaticCheckpointMinimumTokens?: number;
  automaticCheckpointIntervalTokens?: number;
  automaticCheckpointMax?: number;
  budgetBytes: number;
  highWatermarkBytes: number;
  lowWatermarkBytes: number;
  underPressure: boolean;
  hardCeiling: number;
  evictionReason?: string;
  evictionCount: number;
};

export type ResidentUsage = {
  promptTokens?: number;
  completionTokens?: number;
};

export type ResidentTiming = {
  receivedToAdmittedMs?: number;
  templateTokenizeMs?: number;
  coordinatorWaitMs?: number;
  coordinatorPlanMs?: number;
  coordinatorApplyMs?: number;
  schedulerWaitMs?: number;
  cacheMaterializeMs?: number;
  prefillMs?: number;
  residualPrefillTokens?: number;
  prefillTokens?: number;
  prefillChunks?: number;
  firstDecodeMs?: number;
  firstEmitMs?: number;
  normalPrefillQuantum?: number;
  contendedPrefillQuantum?: number;
};

export type ResidentChatResult = {
  content: string;
  usage?: ResidentUsage;
  finishReason?: "stop" | "length" | "cancel";
  cache?: ResidentCacheInfo;
  timing?: ResidentTiming;
  tokenCapabilities?: ModelTokenCapabilities;
  effectiveCapabilities?: EffectiveCapabilities;
};

export type ResidentProgress = (done: number, total: number) => void;
export type ResidentChatOptions = { cacheIdentity: CacheIdentity; structuredOutput?: StructuredOutputContract };
export type ResidentWorkerHandle = {
  key: string;
  backend: ResidentBackend;
  modelPath: string;
  info(): ResidentWorkerInfo;
  onLoadState?(listener: (event: WorkerLoadStateEvent) => void): () => void;
  observeRss?(sampler: WorkerRssSampler): Promise<MemoryValue>;
  load(): Promise<ResidentWorkerInfo>;
  chat(request: ChatCompletionRequest, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void, options?: ResidentChatOptions): Promise<ResidentChatResult>;
  setMaxActive?(maxActive: number, telemetry?: ActiveLimitTelemetry): Promise<void>;
  unload(): Promise<void>;
  shutdown(): void;
  shutdownAsync?(): Promise<void>;
  drainAndShutdownAsync?(): Promise<void>;
};

export async function evictOneIdleForCriticalPressure(input: {
  lifecycle: ResidencyLifecycleAdapter;
  reservedKeys?: ReadonlySet<string>;
  onEvicted?: (snapshot: import("./lifecycle").LifecycleResidencySnapshot) => void | Promise<void>;
  resample?: () => void | Promise<void>;
}): Promise<import("./lifecycle").LifecycleResidencySnapshot | undefined> {
  const attempted = new Set<string>();
  while (true) {
    const victim = selectIdleEvictionVictims(input.lifecycle.snapshotForResidency(), {
      reservedKeys: input.reservedKeys,
    }).find((snapshot) => !attempted.has(snapshot.key));
    if (!victim) return undefined;
    attempted.add(victim.key);
    if (await input.lifecycle.tryEvictIdle(victim) === "changed") continue;
    await input.onEvicted?.(victim);
    await input.resample?.();
    return victim;
  }
}

export function shouldEvictForCriticalPressure(
  pressure: MemoryPressure,
  workers: readonly Pick<ResidentMlxRetention, "maxActive" | "activePolicy">[],
): boolean {
  return pressure === "critical" && workers.length > 0 && workers.every((retention) =>
    retention.maxActive <= 1 || retention.activePolicy.memoryCeiling <= 1);
}

export type RegistryResidencyConfiguration = Omit<ResidencyCoordinatorDependencies,
  "memorySnapshot" | "lifecycle"> & {
  lifecycle: ResidencyLifecycleAdapter;
};

class RegistryResidentWorkerHandle implements ResidentWorkerHandle {
  retired = false;
  residency?: ResidentWorkerResidencyInfo;

  constructor(
    readonly process: ResidentWorkerProcess,
    private readonly registry: ResidentWorkerRegistry,
    readonly descriptor: ResidencyModelDescriptor,
  ) {}

  get key() { return this.process.key; }
  get backend() { return this.process.backend; }
  get modelPath() { return this.process.modelPath; }
  info() { return { ...this.process.info(), residency: this.residency }; }
  onLoadState(listener: (event: WorkerLoadStateEvent) => void) { return this.process.onLoadState(listener); }
  observeRss(sampler: WorkerRssSampler) { return this.process.observeRss(sampler); }
  load() { return this.registry.loadHandle(this); }
  async chat(...args: Parameters<ResidentWorkerProcess["chat"]>) {
    await this.load();
    return this.process.chat(...args);
  }
  async setMaxActive(...args: Parameters<ResidentWorkerProcess["setMaxActive"]>) {
    await this.load();
    return this.process.setMaxActive(...args);
  }
  unload() { return this.process.unload(); }
  shutdown() { this.process.shutdown(); }
  shutdownAsync() { return this.process.shutdownAsync(); }
  drainAndShutdownAsync() { return this.process.drainAndShutdownAsync(); }
}

class RegistryOperationGate {
  private active = 0;
  private exclusive = false;
  private exclusiveTail: Promise<void> = Promise.resolve();
  private waiters: Array<() => void> = [];

  async shared<T>(operation: () => Promise<T>): Promise<T> {
    while (this.exclusive) await new Promise<void>((resolve) => this.waiters.push(resolve));
    this.active += 1;
    try { return await operation(); } finally {
      this.active -= 1;
      this.wake();
    }
  }

  async exclusiveOperation<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.exclusiveTail;
    let releaseExclusive!: () => void;
    this.exclusiveTail = new Promise<void>((resolve) => { releaseExclusive = resolve; });
    await previous;
    this.exclusive = true;
    while (this.active > 0) await new Promise<void>((resolve) => this.waiters.push(resolve));
    try { return await operation(); } finally {
      this.exclusive = false;
      releaseExclusive();
      this.wake();
    }
  }

  private wake(): void {
    for (const resolve of this.waiters.splice(0)) resolve();
  }
}

export type ActiveLimitTelemetry = {
  previousMaxActive: number;
  limitingReason: string;
  lastAdjustmentReason?: string;
  lastAdjustmentAt?: string;
  retainedGrowthReserveBytes: number;
  globalResidentMemoryBytes: number;
  pressureState: MemoryPressure;
};

export type ResidentCrashListener = (info: {
  key: string;
  backend: ResidentBackend;
  exitCode: number;
  consecutiveCrashes: number;
  launchId?: string;
  logPath?: string;
  metadataPath?: string;
  classification?: string;
}) => void;

export function parseWorkerRetention(value: unknown): ResidentMlxRetention | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  if (!WorkerRetentionTelemetrySchema.safeParse(raw).success) return undefined;
  const integer = (key: string): number | undefined => {
    const entry = raw[key];
    return typeof entry === "number" && Number.isSafeInteger(entry) && entry >= 0 ? entry : undefined;
  };
  const nullableInteger = (key: string): number | null | undefined => raw[key] === null ? null : integer(key);
  const maxActive = integer("max_active");
  const queued = integer("queued");
  const previousMaxActive = integer("previous_max_active");
  const retainedGrowthReserveBytes = integer("retained_growth_reserve_bytes");
  const globalResidentMemoryBytes = integer("global_resident_memory_bytes");
  const policyRaw = raw.active_policy;
  if (!policyRaw || typeof policyRaw !== "object") return undefined;
  const policy = policyRaw as Record<string, unknown>;
  const policyInteger = (key: string): number | undefined => {
    const entry = policy[key];
    return typeof entry === "number" && Number.isSafeInteger(entry) && entry >= 0 ? entry : undefined;
  };
  const selectedMax = policyInteger("selected_max");
  const backendCeiling = policyInteger("backend_ceiling");
  const hardwareCeiling = policyInteger("hardware_ceiling");
  const modelCeiling = policyInteger("model_ceiling");
  const memoryCeiling = policyInteger("memory_ceiling");
  const mode = policy.mode === "auto" || policy.mode === "fixed" ? policy.mode : undefined;
  const reason = typeof policy.reason === "string" && policy.reason.length ? policy.reason : undefined;
  const inputsRaw = policy.inputs;
  if (!inputsRaw || typeof inputsRaw !== "object" || Array.isArray(inputsRaw)) return undefined;
  const inputs = inputsRaw as Record<string, unknown>;
  if (Object.values(inputs).some((entry) => entry !== null &&
    !["string", "number", "boolean"].includes(typeof entry))) return undefined;
  const active = integer("active");
  const retainedTotal = integer("retained_total");
  const retainedSessions = integer("retained_sessions");
  const retainedAnchors = integer("retained_anchors");
  const retainedBytes = nullableInteger("retained_bytes");
  const sessionBytes = nullableInteger("session_bytes");
  const anchorBytes = nullableInteger("anchor_bytes");
  const evictedBytes = nullableInteger("evicted_bytes");
  const estimatedRetainedBytes = nullableInteger("estimated_retained_bytes");
  const automaticCheckpointCount = integer("automatic_checkpoint_count");
  const automaticCheckpointBytes = integer("automatic_checkpoint_bytes");
  const automaticCheckpointBudgetBytes = integer("automatic_checkpoint_budget_bytes");
  const automaticCheckpointMinimumTokens = integer("automatic_checkpoint_minimum_tokens");
  const automaticCheckpointIntervalTokens = integer("automatic_checkpoint_interval_tokens");
  const automaticCheckpointMax = integer("automatic_checkpoint_max");
  const budgetBytes = integer("budget_bytes");
  const highWatermarkBytes = integer("high_watermark_bytes");
  const lowWatermarkBytes = integer("low_watermark_bytes");
  const hardCeiling = integer("hard_ceiling");
  const evictionCount = integer("eviction_count");
  const values = [maxActive, active, retainedTotal, retainedSessions, retainedAnchors, retainedBytes,
    sessionBytes, anchorBytes, budgetBytes, highWatermarkBytes, lowWatermarkBytes, hardCeiling, evictionCount,
    selectedMax, backendCeiling, hardwareCeiling, modelCeiling, memoryCeiling];
  if (values.some((entry) => entry === undefined) || typeof raw.under_pressure !== "boolean" ||
    mode === undefined || reason === undefined || selectedMax !== maxActive) return undefined;
  const evictionReason = raw.eviction_reason === null ? undefined
    : typeof raw.eviction_reason === "string" ? raw.eviction_reason : undefined;
  const lastAdjustmentReason = typeof raw.last_adjustment_reason === "string" && raw.last_adjustment_reason.length
    ? raw.last_adjustment_reason : undefined;
  const lastAdjustmentAt = typeof raw.last_adjustment_at === "string" && raw.last_adjustment_at.length
    ? raw.last_adjustment_at : undefined;
  const pressureState = raw.pressure_state === "normal" || raw.pressure_state === "warning"
    || raw.pressure_state === "critical" ? raw.pressure_state : undefined;
  if (raw.eviction_reason !== null && raw.eviction_reason !== undefined && evictionReason === undefined) return undefined;
  return {
    maxActive: maxActive!, ...(queued !== undefined ? { queued } : {}),
    ...(previousMaxActive !== undefined ? { previousMaxActive } : {}),
    ...(lastAdjustmentReason ? { lastAdjustmentReason } : {}),
    ...(lastAdjustmentAt ? { lastAdjustmentAt } : {}),
    ...(retainedGrowthReserveBytes !== undefined ? { retainedGrowthReserveBytes } : {}),
    ...(globalResidentMemoryBytes !== undefined ? { globalResidentMemoryBytes } : {}),
    ...(pressureState ? { pressureState } : {}),
    activePolicy: { mode, selectedMax: selectedMax!, backendCeiling: backendCeiling!,
      hardwareCeiling: hardwareCeiling!, modelCeiling: modelCeiling!, memoryCeiling: memoryCeiling!,
      reason, inputs: inputs as Record<string, string | number | boolean | null> },
    active: active!, retainedTotal: retainedTotal!, retainedSessions: retainedSessions!,
    retainedAnchors: retainedAnchors!, retainedBytes: retainedBytes!,
    ...memoryCompanions(raw, "retained_bytes", "retainedBytes"),
    sessionBytes: sessionBytes!, ...memoryCompanions(raw, "session_bytes", "sessionBytes"),
    anchorBytes: anchorBytes!, ...memoryCompanions(raw, "anchor_bytes", "anchorBytes"),
    ...(evictedBytes !== undefined ? { evictedBytes, ...memoryCompanions(raw, "evicted_bytes", "evictedBytes") } : {}),
    ...(estimatedRetainedBytes !== undefined ? { estimatedRetainedBytes,
      ...memoryCompanions(raw, "estimated_retained_bytes", "estimatedRetainedBytes") } : {}),
    budgetBytes: budgetBytes!, highWatermarkBytes: highWatermarkBytes!,
    ...(automaticCheckpointCount !== undefined ? { automaticCheckpointCount } : {}),
    ...(automaticCheckpointBytes !== undefined ? { automaticCheckpointBytes } : {}),
    ...(automaticCheckpointBudgetBytes !== undefined ? { automaticCheckpointBudgetBytes } : {}),
    ...(typeof raw.automatic_checkpoints_enabled === "boolean"
      ? { automaticCheckpointsEnabled: raw.automatic_checkpoints_enabled } : {}),
    ...(automaticCheckpointMinimumTokens !== undefined ? { automaticCheckpointMinimumTokens } : {}),
    ...(automaticCheckpointIntervalTokens !== undefined ? { automaticCheckpointIntervalTokens } : {}),
    ...(automaticCheckpointMax !== undefined ? { automaticCheckpointMax } : {}),
    lowWatermarkBytes: lowWatermarkBytes!, underPressure: raw.under_pressure, hardCeiling: hardCeiling!,
    ...(evictionReason ? { evictionReason } : {}), evictionCount: evictionCount!,
  };
}

function memoryCompanions(
  raw: Record<string, unknown>,
  wireName: string,
  propertyName: string,
): Record<string, MemorySource | MemoryBasis> {
  const source = raw[`${wireName}_source`] as MemorySource | undefined;
  const basis = raw[`${wireName}_basis`] as MemoryBasis | undefined;
  return source && basis
    ? { [`${propertyName}Source`]: source, [`${propertyName}Basis`]: basis }
    : {};
}

export function parseWorkerTokenCapabilities(value: unknown): ModelTokenCapabilities | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  const nullablePositive = (key: string): number | null | undefined => {
    const entry = raw[key];
    if (entry === null) return null;
    return typeof entry === "number" && Number.isInteger(entry) && entry > 0 ? entry : undefined;
  };
  const modelContextWindow = nullablePositive("model_context_window");
  const effectiveContextWindow = nullablePositive("effective_context_window");
  const maxOutputTokens = nullablePositive("max_output_tokens");
  const backendAllocationCap = nullablePositive("backend_allocation_cap");
  const userConfiguredOverride = nullablePositive("user_configured_override");
  const nullableSource = (key: string): string | null | undefined => {
    const entry = raw[key];
    if (entry === null) return null;
    return typeof entry === "string" && entry.length > 0 ? entry : undefined;
  };
  const modelContextWindowSource = nullableSource("model_context_window_source");
  const maxOutputTokensSource = nullableSource("max_output_tokens_source");
  const rawInput = raw.max_input_tokens;
  const maxInputTokens = rawInput === null ? null
    : typeof rawInput === "number" && Number.isInteger(rawInput) && rawInput >= 0 ? rawInput : undefined;
  if ([modelContextWindow, effectiveContextWindow, maxInputTokens, maxOutputTokens, backendAllocationCap, userConfiguredOverride].some((entry) => entry === undefined)) return undefined;
  return {
    modelContextWindow: modelContextWindow!,
    effectiveContextWindow: effectiveContextWindow!,
    maxInputTokens: maxInputTokens!,
    maxOutputTokens: maxOutputTokens!,
    ...(modelContextWindowSource !== undefined ? { modelContextWindowSource } : {}),
    ...(maxOutputTokensSource !== undefined ? { maxOutputTokensSource } : {}),
    backendAllocationCap: backendAllocationCap!,
    userConfiguredOverride: userConfiguredOverride!,
  };
}

export function deriveTokenCapabilities(input: {
  modelContextWindow?: number | null;
  backendAllocationCap?: number | null;
  maxOutputTokens?: number | null;
  userConfiguredOverride?: number | null;
}): ModelTokenCapabilities {
  const positive = (value: number | null | undefined) => typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
  const modelContextWindow = positive(input.modelContextWindow);
  const backendAllocationCap = positive(input.backendAllocationCap);
  const userConfiguredOverride = positive(input.userConfiguredOverride);
  const caps = [modelContextWindow, backendAllocationCap, userConfiguredOverride].filter((value): value is number => value !== null);
  const effectiveContextWindow = caps.length ? Math.min(...caps) : null;
  return {
    modelContextWindow,
    effectiveContextWindow,
    maxInputTokens: effectiveContextWindow === null ? null : Math.max(0, effectiveContextWindow - 1),
    maxOutputTokens: positive(input.maxOutputTokens),
    backendAllocationCap,
    userConfiguredOverride,
  };
}

export type TokenBudgetResult = { maxTokens: number } | { code: "context_length_exceeded" | "max_output_tokens_exceeded" | "token_capability_unknown"; message: string };

export function validateTokenBudget(capabilities: ModelTokenCapabilities, promptTokens: number, requestedMaxTokens?: number): TokenBudgetResult {
  if (capabilities.maxInputTokens !== null && promptTokens > capabilities.maxInputTokens) {
    return { code: "context_length_exceeded", message: `prompt_tokens=${promptTokens} exceeds max_input_tokens=${capabilities.maxInputTokens}` };
  }
  if (requestedMaxTokens !== undefined && capabilities.maxOutputTokens !== null && requestedMaxTokens > capabilities.maxOutputTokens) {
    return { code: "max_output_tokens_exceeded", message: `requested_output_tokens=${requestedMaxTokens} exceeds max_output_tokens=${capabilities.maxOutputTokens}` };
  }
  if (capabilities.effectiveContextWindow === null && requestedMaxTokens === undefined && capabilities.maxOutputTokens === null) {
    return { code: "token_capability_unknown", message: "max_tokens is required because this model does not declare token limits" };
  }
  const contextRemaining = capabilities.effectiveContextWindow === null ? null : capabilities.effectiveContextWindow - promptTokens;
  const maxTokens = requestedMaxTokens ?? (capabilities.maxOutputTokens !== null
    ? (contextRemaining === null ? capabilities.maxOutputTokens : Math.min(capabilities.maxOutputTokens, contextRemaining))
    : contextRemaining);
  if (maxTokens === null) {
    return { code: "token_capability_unknown", message: "max_tokens is required because this model does not declare token limits" };
  }
  if (contextRemaining !== null && maxTokens > contextRemaining) {
    return { code: "context_length_exceeded", message: `prompt_tokens=${promptTokens} plus requested_output_tokens=${maxTokens} exceeds effective_context_window=${capabilities.effectiveContextWindow}` };
  }
  return { maxTokens };
}

export class ResidentWorkerRegistry {
  private readonly workers = new Map<string, ResidentWorkerProcess>();
  private readonly handles = new Map<string, RegistryResidentWorkerHandle>();
  private readonly workerEnvironments = new Map<string, Record<string, string>>();
  private readonly recentRetainedGrowth = new Map<string, { bytes: number; at: number }>();
  private readonly lastAdjustments = new Map<string, { at: number; reason: string; previous: number }>();
  private pressureState: MemoryPressure = "normal";
  private pressureTimer?: ReturnType<typeof setInterval>;
  private rebalanceScheduled = false;
  private rebalancing = false;
  private readonly operationGate = new RegistryOperationGate();
  private residencyCoordinator?: ResidencyCoordinator;
  private residencyLifecycle?: ResidencyLifecycleAdapter;
  private residencyEvent?: ResidencyCoordinatorDependencies["onEvent"];
  onCrash?: ResidentCrashListener;
  memorySnapshot: (pids: number[]) => Promise<{
    physicalMemoryBytes: number;
    availableMemoryBytes: number;
    residentBytesByPid: Map<number, number>;
  }> = async () => ({
    physicalMemoryBytes: totalmem(),
    availableMemoryBytes: freemem(),
    residentBytesByPid: new Map(),
  });
  // Per-model worker environment (e.g. [models."x"] config sections);
  // consulted once when the worker handle is first created.
  workerEnv?: (modelPath: string, backend: ResidentBackend) => Record<string, string> | undefined;
  rssSampler: WorkerRssSampler = async () => undefined;
  configureResidency(configuration: RegistryResidencyConfiguration): ResidencyCoordinator {
    const lifecycle: ResidencyLifecycleAdapter = {
      snapshotForResidency: () => configuration.lifecycle.snapshotForResidency(),
      tryEvictIdle: (snapshot) => configuration.lifecycle.tryEvictIdle(snapshot),
      setResidencyTransition: (key, state) => configuration.lifecycle.setResidencyTransition(key, state),
      clearResidencyTransition: (key) => configuration.lifecycle.clearResidencyTransition(key),
    };
    this.residencyLifecycle = lifecycle;
    this.residencyEvent = configuration.onEvent;
    this.residencyCoordinator = new ResidencyCoordinator({
      ...configuration,
      lifecycle,
      memorySnapshot: async () => {
        const pids = [...this.workers.values()].map((worker) => worker.info().pid)
          .filter((pid): pid is number => pid !== undefined);
        const snapshot = await this.memorySnapshot(pids);
        return measuredResidencyMemory(snapshot.availableMemoryBytes, snapshot.physicalMemoryBytes);
      },
    });
    return this.residencyCoordinator;
  }

  residencyReservations(): readonly LoadReservation[] {
    return this.residencyCoordinator?.reservations() ?? [];
  }

  getOrCreate(key: string, backend: ResidentBackend, modelPath: string,
    descriptor: Partial<import("./process/types").WorkerModelDescriptor> = {}): ResidentWorkerHandle {
    const existing = this.handles.get(key);
    if (existing) return existing;
    const environment = this.workerEnv?.(modelPath, backend) ?? {};
    const worker = new ResidentWorkerProcess(key, backend, modelPath,
      (info) => this.onCrash?.(info), environment,
      (source, previous, current) => this.handleTelemetry(source, previous, current),
      { ...descriptor, modelId: descriptor.modelId ?? key });
    this.workers.set(key, worker);
    const handle = new RegistryResidentWorkerHandle(worker, this, {
      modelKey: key,
      backend,
      modelId: descriptor.modelId ?? key,
      revision: descriptor.revision,
      artifactBytes: descriptor.artifactBytes,
      architecture: descriptor.architecture,
      modelType: descriptor.modelType,
      quantization: descriptor.quantization,
      context: descriptor.context,
      configuredContext: descriptor.configuredContext,
      kv: descriptor.kv,
      cacheBudget: descriptor.cacheBudget,
    });
    this.handles.set(key, handle);
    this.workerEnvironments.set(key, environment);
    if (!this.pressureTimer) {
      this.pressureTimer = setInterval(() => this.scheduleRebalance("pressure_poll"), 5_000);
      this.pressureTimer.unref?.();
    }
    return handle;
  }

  async loadHandle(handle: RegistryResidentWorkerHandle): Promise<ResidentWorkerInfo> {
    if (handle.retired) throw new Error("resident worker handle is retired");
    if (handle.process.info().loadState === "resident" || !this.residencyCoordinator) {
      return handle.process.load();
    }
    return this.operationGate.shared(async () => {
      if (handle.retired) throw new Error("resident worker handle is retired");
      if (handle.process.info().loadState === "resident") return handle.process.info();
      const result = await this.residencyCoordinator!.load(handle.descriptor, {
        performLoad: () => handle.process.load(),
        stabilize: async () => { await Bun.sleep(0); },
        observeRss: async () => {
          const observed = await handle.process.observeRss(this.rssSampler);
          return observed.source === "measured" ? observed.bytes : undefined;
        },
        shutdownPartial: () => handle.process.shutdownAsync(),
      });
      const observed = await handle.process.observeRss(this.rssSampler);
      handle.residency = {
        estimateBytes: result.decision.requested.bytes,
        estimateSource: result.decision.requested.source === "estimated" ? result.decision.requested.basis : null,
        observedRssBytes: observed.source === "measured" ? observed.bytes : null,
        observedRssSource: observed.source === "measured" ? "resident_rss" : null,
        reservationBytes: result.reservation.bytes,
        lastAdmissionReason: result.decision.reason,
        lastEvictionReason: result.decision.evictedModelKeys.length > 0 ? "memory_admission" : null,
      };
      return handle.info();
    });
  }

  async observeWorkerRss(key: string): Promise<MemoryValue> {
    const worker = this.workers.get(key);
    if (!worker) return unavailableMemory("not_observed");
    if (worker.observeRss) return worker.observeRss(this.rssSampler);
    const pid = worker.info().pid;
    return pid === undefined ? unavailableMemory("not_observed") : sampleRss(pid, this.rssSampler);
  }

  get(key: string): ResidentWorkerHandle | undefined {
    return this.handles.get(key);
  }

  async unload(key: string): Promise<void> {
    const worker = this.workers.get(key);
    const handle = this.handles.get(key);
    if (handle) handle.retired = true;
    this.workers.delete(key);
    this.handles.delete(key);
    this.workerEnvironments.delete(key);
    this.recentRetainedGrowth.delete(key);
    this.lastAdjustments.delete(key);
    await worker?.unload();
    this.scheduleRebalance("model_unloaded");
  }

  shutdown(key: string): void {
    void this.shutdownAsync(key);
  }

  async shutdownAsync(key?: string): Promise<void> {
    if (key === undefined) {
      return this.operationGate.exclusiveOperation(async () => {
        const keys = [...this.workers.keys()];
        await Promise.all(keys.map((entry) => this.shutdownKey(entry)));
        if (this.pressureTimer) clearInterval(this.pressureTimer);
        this.pressureTimer = undefined;
      });
    }
    await this.shutdownKey(key);
  }

  private async shutdownKey(key: string): Promise<void> {
    const worker = this.workers.get(key);
    const handle = this.handles.get(key);
    if (handle) handle.retired = true;
    this.workers.delete(key);
    this.handles.delete(key);
    this.workerEnvironments.delete(key);
    this.recentRetainedGrowth.delete(key);
    this.lastAdjustments.delete(key);
    if (worker?.shutdownAsync) await worker.shutdownAsync();
    else worker?.shutdown();
    this.scheduleRebalance("model_shutdown");
  }

  shutdownAll(): void {
    void this.shutdownAsync();
  }

  async shutdownAllAsync(): Promise<void> {
    await this.shutdownAsync();
  }

  /** Drains every old-generation worker after the new secret is durable. */
  async rotateCacheIdentityGeneration(): Promise<number> {
    return this.operationGate.exclusiveOperation(async () => {
      const workers = [...this.workers.values()];
      const cleared = workers.length;
      for (const handle of this.handles.values()) handle.retired = true;
      this.workers.clear();
      this.handles.clear();
      this.workerEnvironments.clear();
      this.recentRetainedGrowth.clear();
      this.lastAdjustments.clear();
      await Promise.all(workers.map((worker) => worker.drainAndShutdownAsync?.() ?? worker.shutdownAsync?.()));
      if (this.pressureTimer) clearInterval(this.pressureTimer);
      this.pressureTimer = undefined;
      return cleared;
    });
  }

  async rebalance(reason = "manual"): Promise<void> {
    if (this.rebalancing) return;
    this.rebalancing = true;
    try {
      await this.operationGate.exclusiveOperation(() => this.rebalanceExclusive(reason));
    } finally {
      this.rebalancing = false;
    }
  }

  private async rebalanceExclusive(reason: string): Promise<void> {
    let resident = [...this.workers.values()].filter((worker) => worker.info().retention);
    if (!resident.length) return;
    let pids = resident.map((worker) => worker.info().pid).filter((pid): pid is number => pid !== undefined);
    let snapshot = await this.memorySnapshot(pids);
    let physical = Math.max(1, snapshot.physicalMemoryBytes);
    let pressure = classifyMemoryPressure(snapshot.availableMemoryBytes, physical, this.pressureState);
    const topologyChanged = pressure !== this.pressureState;
    this.pressureState = pressure;
    if (shouldEvictForCriticalPressure(pressure,
      resident.flatMap((worker) => worker.info().retention ?? []))) {
      const eviction = await this.evictOneForCriticalPressure();
      if (eviction) {
        resident = [...this.workers.values()].filter((worker) => worker.info().retention);
        if (!resident.length) return;
        pids = resident.map((worker) => worker.info().pid).filter((pid): pid is number => pid !== undefined);
        snapshot = await this.memorySnapshot(pids);
        physical = Math.max(1, snapshot.physicalMemoryBytes);
        pressure = classifyMemoryPressure(snapshot.availableMemoryBytes, physical, pressure);
        this.pressureState = pressure;
      }
    }
    const now = Date.now();
    const inputWorkers = resident.flatMap((worker) => {
      const info = worker.info();
      const retention = info.retention;
      if (!retention) return [];
      const inputs = retention.activePolicy.inputs;
      const inputNumber = (name: string) => typeof inputs[name] === "number" ? inputs[name] : undefined;
      const environment = this.workerEnvironments.get(worker.key) ?? {};
      const configured = Number(environment.CLAP_MAX_ACTIVE ?? process.env.CLAP_MAX_ACTIVE ?? 0);
      const growth = this.recentRetainedGrowth.get(worker.key);
      const configuredMinimum = Number(environment.CLAP_MLX_RETAINED_GROWTH_MIN_BYTES);
      const configuredPercent = Number(environment.CLAP_MLX_RETAINED_GROWTH_RESERVE_PERCENT);
      const fallbackResident = info.memory && info.memory.activeBytes !== null && info.memory.cacheBytes !== null
        ? info.memory.activeBytes + info.memory.cacheBytes
        : inputNumber("model_active_bytes") ?? inputNumber("model_file_bytes") ?? 0;
      return [{
        key: worker.key,
        mode: retention.activePolicy.mode,
        requestedMax: retention.activePolicy.mode === "fixed" && configured > 0
          ? configured : retention.activePolicy.selectedMax,
        currentMax: retention.maxActive,
        backendCeiling: retention.activePolicy.backendCeiling,
        hardwareCeiling: retention.activePolicy.hardwareCeiling,
        modelCeiling: retention.activePolicy.modelCeiling,
        retainedCeiling: retention.hardCeiling,
        perActiveReserveBytes: inputNumber("per_active_reserve_bytes") ?? 1024 ** 3,
        residentBytes: info.pid ? snapshot.residentBytesByPid.get(info.pid) ?? fallbackResident : fallbackResident,
        retainedBytes: retention.retainedBytes ?? retention.estimatedRetainedBytes ?? 0,
        retainedBudgetBytes: retention.budgetBytes,
        recentRetainedGrowthBytes: growth && now - growth.at <= 60_000 ? growth.bytes : 0,
        growthMinimumBytes: Number.isFinite(configuredMinimum) && configuredMinimum >= 0
          ? configuredMinimum : 64 * 1024 ** 2,
        growthReservePercent: Number.isFinite(configuredPercent) && configuredPercent >= 0
          ? configuredPercent : 10,
      }];
    });
    const osReserve = Math.min(physical, Math.max(2 * 1024 ** 3, Math.floor(physical / 10)));
    const plans = selectGlobalActiveLimits({
      physicalMemoryBytes: physical,
      osReserveBytes: osReserve,
      pressure,
      workers: inputWorkers,
    });
    for (const plan of plans) {
      const worker = this.workers.get(plan.key);
      if (!worker) continue;
      const previousAdjustment = this.lastAdjustments.get(plan.key);
      const adjust = shouldAdjustActiveLimit(plan, now, previousAdjustment?.at);
      if (adjust) this.lastAdjustments.set(plan.key, {
        at: now,
        reason: plan.reason,
        previous: plan.previousMax,
      });
      const adjustment = this.lastAdjustments.get(plan.key);
      await worker.setMaxActive(adjust ? plan.selectedMax : plan.previousMax, {
        previousMaxActive: adjustment?.previous ?? plan.previousMax,
        limitingReason: plan.limitingReason,
        lastAdjustmentReason: adjust ? plan.reason : adjustment?.reason
          ?? (topologyChanged ? `pressure_${pressure}` : reason),
        lastAdjustmentAt: adjustment ? new Date(adjustment.at).toISOString() : undefined,
        retainedGrowthReserveBytes: plan.retainedGrowthReserveBytes,
        globalResidentMemoryBytes: plan.globalResidentBytes,
        pressureState: pressure,
      });
    }
  }

  private async evictOneForCriticalPressure(): Promise<boolean> {
    const lifecycle = this.residencyLifecycle;
    if (!lifecycle) return false;
    const reservedKeys = new Set(this.residencyReservations().map((reservation) => reservation.model.modelKey));
    const backendByKey = new Map([...this.handles].map(([key, handle]) => [key, handle.backend]));
    const victim = await evictOneIdleForCriticalPressure({
      lifecycle,
      reservedKeys,
      onEvicted: (snapshot) => this.residencyEvent?.({
        type: "model_evicted_for_pressure",
        backend: backendByKey.get(snapshot.key) ?? "unknown",
        reason: "critical_memory_pressure",
        reservationBytes: 0,
        activeReservations: this.residencyReservations().length,
      }),
    });
    return victim !== undefined;
  }

  private handleTelemetry(worker: ResidentWorkerProcess, previous: ResidentMlxRetention | undefined,
    current: ResidentMlxRetention): void {
    const currentBytes = current.retainedBytes ?? current.estimatedRetainedBytes ?? 0;
    const previousBytes = previous ? previous.retainedBytes ?? previous.estimatedRetainedBytes ?? 0 : 0;
    const growth = previous ? Math.max(0, currentBytes - previousBytes) : 0;
    if (growth > 0) {
      const recent = this.recentRetainedGrowth.get(worker.key);
      this.recentRetainedGrowth.set(worker.key, {
        bytes: Math.max(growth, recent && Date.now() - recent.at <= 60_000 ? recent.bytes : 0),
        at: Date.now(),
      });
    }
    const threshold = Math.max(32 * 1024 ** 2, Math.floor(previousBytes / 10));
    const material = !previous || Math.abs(currentBytes - previousBytes) >= threshold
      || current.underPressure !== previous.underPressure;
    if (material) this.scheduleRebalance(previous ? "retained_threshold" : "model_loaded");
  }

  private scheduleRebalance(reason: string): void {
    if (this.rebalanceScheduled || this.rebalancing || !this.workers.size) return;
    this.rebalanceScheduled = true;
    setTimeout(() => {
      this.rebalanceScheduled = false;
      void this.rebalance(reason).catch(() => {});
    }, 0);
  }
}

async function sampleRss(pid: number, sampler: WorkerRssSampler): Promise<MemoryValue> {
  const bytes = await sampler(pid);
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes <= 0) {
    return unavailableMemory("not_reported");
  }
  return measuredMemory(bytes, "resident_rss");
}
