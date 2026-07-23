export type DashboardServer = {
  version: string;
  uptimeMs: number;
  platform: string;
  arch: string;
  bunVersion: string;
  pid: number;
  rssBytes?: number;
  cpuPercent?: number;
  systemMemoryBytes?: number;
  systemMemoryUsedBytes?: number;
  systemCpuPercent?: number;
  cpuCount?: number;
};

export type DashboardGpu = {
  vendor: "nvidia" | "apple";
  name: string;
  utilizationPercent?: number;
  memoryUsedBytes?: number;
  memoryTotalBytes?: number;
  processes?: Array<{ pid: number; memoryBytes: number }>;
};

export type DashboardQueue = {
  inflight: number;
  queued: number;
  maxInflight: number;
  queueDepth: number;
  inflightByPriority: Record<"interactive" | "normal" | "background", number>;
  waitingByPriority: Record<"interactive" | "normal" | "background", number>;
  outcomesByPriority: Record<"interactive" | "normal" | "background",
    Record<"admitted" | "rejected" | "aborted", number>>;
};

export type DashboardTotals = {
  requests: number;
  ok: number;
  errors: number;
  cancelled: number;
  promptTokens: number;
  completionTokens: number;
  cacheHits: number;
  cacheMisses: number;
  reusedTokens: number;
};

export type DetailMessage = {
  role: string;
  content: string;
  truncated?: boolean;
  toolCalls?: Array<{ name: string; arguments: string }>;
};

export type StructuredOutputFacts = {
  kind: "json_object" | "json_schema";
  requestedStrength: "best_effort" | "required";
  backendMode?: "native" | "post_validate";
  outcome?: "native_validated" | "validated" | "repaired_validated" | "invalid" | "capability_rejected";
  selectedParser?: string;
  repairApplied?: boolean;
  validationMs?: number;
  schemaFingerprint?: string;
  schemaSize?: number;
};

export type RequestDetail = {
  params: {
    temperature?: number;
    topP?: number;
    maxTokens?: number;
    stop?: string[];
    responseFormat?: string;
  };
  toolNames: string[];
  messages: DetailMessage[];
  droppedMessages: number;
  response?: {
    content?: string;
    reasoning?: string;
    toolCalls?: Array<{ name: string; arguments: string }>;
  };
  rawOutput?: string;
};

export type CacheOutcomeCategory =
  | "hit"
  | "cold"
  | "isolated"
  | "below_checkpoint"
  | "no_shared_prefix"
  | "donor_busy"
  | "no_eligible_donor"
  | "fresh_by_policy"
  | "cache_error"
  | "unexplained_miss"
  | "miss_reason_unavailable"
  | "unknown";

export type CacheOutcome = {
  category: CacheOutcomeCategory;
  reason: string;
  hitKind?: "session" | "branch" | "checkpoint";
  maxBlockedPrefixTokens?: number;
  boundariesSkipped?: number;
  evidence: string[];
};

export type DashboardRequest = {
  source?: "live" | "persisted";
  id: string;
  startedAt: number;
  endedAt?: number;
  durationMs?: number;
  ttftMs?: number;
  loadMs?: number;
  queuedMs?: number;
  priority: "interactive" | "normal" | "background";
  model: string;
  endpoint: string;
  stream: boolean;
  status: "active" | "ok" | "error" | "cancelled";
  phase: "queued" | "loading" | "prefill" | "decode" | "done";
  prefillDone?: number;
  prefillTotal?: number;
  // Legacy prompt-prefix id; prefer sessionDisplayId + sessionIdentityKind.
  conversation?: string;
  // Privacy-safe identity: installation-keyed session fingerprint (short form)
  // when cache.session was set, else prompt-prefix grouping. Never raw session.
  sessionDisplayId?: string;
  sessionIdentityKind?: "cache_session" | "prompt_prefix";
  sessionFingerprint?: string;
  promptTokens?: number;
  completionTokens?: number;
  tokensPerSecond?: number;
  cacheHit?: boolean;
  reuseKind?: "slot" | "branch" | "anchor";
  reuseScope?: "system" | "conversation" | "session" | "agent" | "project" | "harness" | "tenant";
  reusedTokens?: number;
  sideRequest?: boolean;
  slot?: number;
  cacheNamespace?: string;
  donorSlot?: number;
  targetSlot?: number;
  evictedSlots?: number[];
  cacheDecisionUs?: number;
  plannedReuseTokens?: number;
  realizedReuseTokens?: number;
  cacheFallback?: string;
  finishReason?: string;
  // Evidence-based classification from the server. Preserve raw telemetry
  // (cacheHit/reusedTokens/etc.) separately; never invent a category client-side.
  cacheOutcome?: CacheOutcome;
  timing?: {
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
  // True when the record belongs to a previous server/worker launch whose KV
  // is no longer resident. Orthogonal to the outcome category.
  historical?: boolean;
  toolCalls?: number;
  messageCount?: number;
  error?: string;
  structuredOutput?: StructuredOutputFacts;
  detail?: RequestDetail;
  cacheDiagnostics?: {
    serverLaunchId: string;
    workerLaunchId?: string;
    backend?: string;
    namespaceFingerprint?: string;
    sessionIdentitySource?: string;
    sessionFingerprint?: string;
    projectFingerprint?: string;
    agentFingerprint?: string;
    harnessFingerprint?: string;
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
      status: "resolved" | "skipped";
      skipReason?: "unsupported_template_boundary" | "non_prefix_template_boundary";
      materialized?: boolean;
    }>;
    promptTokenHash?: string;
    promptTokenCount?: number;
    errorCode?: string;
    prefillMs?: number;
    timing?: DashboardRequest["timing"];
    cache?: {
      missReason?: string;
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
        materialized?: boolean;
        trimEligible?: boolean;
        copyEligible?: boolean;
        selected?: boolean;
        rejection?: string;
      }>;
    };
  };
};

export type ServerEvent = {
  id: string;
  at: number;
  type: "server" | "load" | "unload" | "expire" | "error" | "download"
    | "model_load_reserved" | "model_load_started" | "model_load_committed" | "model_load_rolled_back"
    | "model_evicted_for_load" | "model_load_rejected";
  message: string;
  model?: string;
  durationMs?: number;
  backend?: string;
  reason?: string;
  reservationBytes?: number;
  activeReservations?: number;
};

export type ModelTokenCapabilities = {
  modelContextWindow: number | null;
  effectiveContextWindow: number | null;
  maxInputTokens: number | null;
  maxOutputTokens: number | null;
  backendAllocationCap: number | null;
  userConfiguredOverride: number | null;
};

export type DashboardLoadedModel = {
  key: string;
  id: string;
  backend: "llama" | "mlx";
  format: "gguf" | "mlx";
  localPath: string;
  state: "warm" | "active" | "unloading";
  activeRequests: number;
  loadedAt: string;
  lastUsedAt: string;
  keepAlive: string;
  expiresAt: string | null;
  pinned: boolean;
  worker: {
    pid?: number;
    state: string;
    loadState?: "not_started" | "starting" | "loading" | "resident" | "closing";
    residency?: {
      estimateBytes: number | null;
      estimateSource: "prior_observation" | "model_artifacts" | "architecture_metadata" | "configured_cache" | "conservative_fallback" | null;
      observedRssBytes: number | null;
      observedRssSource: "resident_rss" | null;
      reservationBytes: number;
      lastAdmissionReason: "within_budget" | "within_budget_after_eviction" | "insufficient_available_memory" | "memory_state_unavailable" | "no_evictable_models" | null;
      lastEvictionReason: "memory_admission" | null;
    };
    memory?: {
      activeBytes: number | null; activeBytesSource?: "measured" | "estimated" | "unavailable"; activeBytesBasis?: string;
      cacheBytes: number | null; cacheBytesSource?: "measured" | "estimated" | "unavailable"; cacheBytesBasis?: string;
      peakActiveBytes: number | null; peakActiveBytesSource?: "measured" | "estimated" | "unavailable"; peakActiveBytesBasis?: string;
    };
    retention?: {
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
      retainedBytes: number | null; retainedBytesSource?: "measured" | "estimated" | "unavailable"; retainedBytesBasis?: string;
      sessionBytes: number | null; sessionBytesSource?: "measured" | "estimated" | "unavailable"; sessionBytesBasis?: string;
      anchorBytes: number | null; anchorBytesSource?: "measured" | "estimated" | "unavailable"; anchorBytesBasis?: string;
      evictedBytes?: number | null; evictedBytesSource?: "measured" | "estimated" | "unavailable"; evictedBytesBasis?: string;
      estimatedRetainedBytes?: number | null; estimatedRetainedBytesSource?: "measured" | "estimated" | "unavailable"; estimatedRetainedBytesBasis?: string;
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
    tokenCapabilities?: ModelTokenCapabilities;
    effectiveCapabilities?: {
      cache: {
        partialSuffixTrim: boolean;
        partialPrefixBranch: boolean;
        wholeStateCopy: boolean;
        promptBoundarySnapshots: boolean;
        quantizedKv: boolean;
      };
      generation: {
        structuredOutput: {
          json_object: "native" | "post_validate" | "unsupported";
          json_schema: "native" | "post_validate" | "unsupported";
          post_validation: boolean;
          max_schema_bytes: number;
        };
        toolTemplateSupport: boolean;
      };
      modalities: { input: ["text"]; output: ["text"] };
    };
  };
  usage?: { rssBytes: number; cpuPercent: number };
  gpuMemoryBytes?: number;
};

export type DashboardModel = {
  id: string;
  displayName: string;
  backend: "llama" | "mlx";
  format: string;
  status: "available" | "not_downloaded" | "unsupported";
  quantization?: string;
  sizeBytes?: number;
  limit?: { context?: number | null; output?: number | null };
  capabilities?: { toolCall?: boolean; reasoning?: boolean };
  modalities?: { input?: string[] };
};

export type DashboardDownload = {
  id: string;
  model: string;
  file?: string;
  backend?: "gguf" | "mlx";
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  bytesReceived: number;
  totalBytes?: number;
  currentFile?: string;
  error?: string;
};

export type ModelResolveOption = {
  id: string;
  model: string;
  backend: "gguf" | "mlx";
  format: "gguf" | "mlx" | "safetensors";
  repo: string;
  file?: string;
  sizeBytes?: number;
  quantization?: string;
  supported: boolean;
  unsupportedReason?: string;
  recommended: boolean;
  reason: string;
};

export type ModelResolveResponse = {
  model: string;
  repo: string;
  options: ModelResolveOption[];
  selected?: ModelResolveOption;
};

export type DashboardData = {
  server: DashboardServer;
  gpus?: DashboardGpu[];
  totals: DashboardTotals;
  queue?: DashboardQueue;
  active: DashboardRequest[];
  requests: DashboardRequest[];
  events: ServerEvent[];
  loaded: DashboardLoadedModel[];
  models: DashboardModel[];
  downloads: DashboardDownload[];
};

export async function fetchDashboard(): Promise<DashboardData> {
  const response = await fetch("/clap/v1/dashboard");
  if (!response.ok) throw new Error(`dashboard fetch failed: ${response.status}`);
  return response.json() as Promise<DashboardData>;
}

export async function fetchRequestDetail(id: string): Promise<DashboardRequest> {
  const response = await fetch(`/clap/v1/dashboard/requests/${id}`);
  if (!response.ok) throw new Error(`request detail fetch failed: ${response.status}`);
  return response.json() as Promise<DashboardRequest>;
}

async function post(path: string, body: Record<string, unknown>): Promise<unknown> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => undefined) as { error?: { message?: string } } | undefined;
  if (!response.ok) {
    throw new Error(payload?.error?.message ?? `${path} failed: ${response.status}`);
  }
  return payload;
}

export function loadModel(model: string, keepAlive?: string): Promise<unknown> {
  return post("/clap/v1/models/load", keepAlive ? { model, keepAlive } : { model });
}

export function unloadModel(model: string): Promise<unknown> {
  return post("/clap/v1/models/unload", { model });
}

export function removeModel(model: string): Promise<unknown> {
  return post("/clap/v1/models/remove", { model });
}

export function pullModel(model: string, opts?: { file?: string; backend?: "gguf" | "mlx" }): Promise<unknown> {
  return post("/clap/v1/models/pull", { model, ...(opts?.file ? { file: opts.file } : {}), ...(opts?.backend ? { backend: opts.backend } : {}) });
}

export function resolveModel(model: string): Promise<ModelResolveResponse> {
  return post("/clap/v1/models/resolve", { model }) as Promise<ModelResolveResponse>;
}

export function cancelDownload(id: string): Promise<unknown> {
  return post(`/clap/v1/downloads/${id}/cancel`, {});
}
