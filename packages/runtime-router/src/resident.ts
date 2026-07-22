import type { ChatCompletionRequest, LoadedModel, ModelTokenCapabilities } from "@clap/api";
import { getLlamaWorkerStatus, LlamaWorkerError } from "@clap/runtime-llama";
import { getMlxWorkerStatus, MlxWorkerError } from "@clap/runtime-mlx";
import { mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { freemem, totalmem } from "node:os";
import { dirname } from "node:path";
import { classifyMemoryPressure, selectGlobalActiveLimits, shouldAdjustActiveLimit,
  type MemoryPressure } from "./concurrency";
import { LegacyWorkerProtocol, allowsLegacyStartupFallback } from "./protocol/legacy-worker-protocol";
import { V1RequestTracker, type ResidentProtocolFact } from "./protocol/request-tracker";
import { WorkerProtocolFault } from "./protocol/errors";

export type ResidentWorkerProtocolMode = "legacy" | "v1" | "auto";

export type ResidentBackend = LoadedModel["backend"];

export type ResidentWorkerInfo = {
  pid?: number;
  launchId?: string;
  state: LoadedModel["worker"]["state"];
  limitation?: string;
  crashes?: number;
  lastCrashAt?: string;
  memory?: ResidentMlxMemory;
  retention?: ResidentMlxRetention;
  tokenCapabilities?: ModelTokenCapabilities;
};

export type ResidentMlxMemory = {
  activeBytes: number;
  cacheBytes: number;
  peakActiveBytes: number;
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
  retainedBytes: number;
  sessionBytes: number;
  anchorBytes: number;
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

export type ResidentChatResult = {
  content: string;
  usage?: ResidentUsage;
  finishReason?: "stop" | "length" | "cancel";
  cache?: ResidentCacheInfo;
  timing?: ResidentTiming;
  tokenCapabilities?: ModelTokenCapabilities;
};

export type ResidentProgress = (done: number, total: number) => void;

export type ResidentWorkerHandle = {
  key: string;
  backend: ResidentBackend;
  modelPath: string;
  info(): ResidentWorkerInfo;
  load(): Promise<ResidentWorkerInfo>;
  chat(request: ChatCompletionRequest, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult>;
  setMaxActive?(maxActive: number, telemetry?: ActiveLimitTelemetry): Promise<void>;
  unload(): Promise<void>;
  shutdown(): void;
};

export type ActiveLimitTelemetry = {
  previousMaxActive: number;
  limitingReason: string;
  lastAdjustmentReason?: string;
  lastAdjustmentAt?: string;
  retainedGrowthReserveBytes: number;
  globalResidentMemoryBytes: number;
  pressureState: MemoryPressure;
};

type Pending = {
  content: string[];
  resolve: (result: ResidentChatResult) => void;
  reject: (error: Error) => void;
  onToken?: (token: string) => void;
  onProgress?: ResidentProgress;
  onDispatch?: () => void;
  usage?: ResidentUsage;
  finishReason?: "stop" | "length" | "cancel";
  cache?: ResidentCacheInfo;
  timing?: ResidentTiming;
  tokenCapabilities?: ModelTokenCapabilities;
};

export type ResidentCrashListener = (info: {
  key: string;
  backend: ResidentBackend;
  exitCode: number;
  consecutiveCrashes: number;
}) => void;

export class ResidentWorkerProcess implements ResidentWorkerHandle {
  private proc?: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
  private readonly pending = new Map<string, Pending>();
  private loaded = false;
  private logPath?: string;
  private crashes = 0;
  private consecutiveCrashes = 0;
  private lastCrashAt?: number;
  private expectedExit = false;
  private memory?: ResidentMlxMemory;
  private retention?: ResidentMlxRetention;
  private tokenCapabilities?: ModelTokenCapabilities;
  private workerLaunchId?: string;
  private readonly legacyProtocol = new LegacyWorkerProtocol();
  private v1Tracker?: V1RequestTracker;
  private activeProtocol: "legacy" | "v1" = "v1";
  private handshake?: Promise<void>;
  private resolveHandshake?: () => void;
  private rejectHandshake?: (error: Error) => void;
  private handshakeTimer?: ReturnType<typeof setTimeout>;
  private startupBytes = false;

  constructor(
    public readonly key: string,
    public readonly backend: ResidentBackend,
    public readonly modelPath: string,
    private readonly onCrash?: ResidentCrashListener,
    private readonly envOverrides?: Record<string, string>,
    private readonly onTelemetry?: (worker: ResidentWorkerProcess,
      previous: ResidentMlxRetention | undefined, current: ResidentMlxRetention) => void,
    private readonly protocolMode: ResidentWorkerProtocolMode = "auto",
  ) {}

  info(): ResidentWorkerInfo {
    return {
      pid: this.proc?.pid,
      launchId: this.workerLaunchId,
      state: this.proc ? "resident" : "not_started",
      crashes: this.crashes,
      lastCrashAt: this.lastCrashAt ? new Date(this.lastCrashAt).toISOString() : undefined,
      memory: this.memory,
      retention: this.retention,
      tokenCapabilities: this.tokenCapabilities,
    };
  }

  async load(): Promise<ResidentWorkerInfo> {
    await this.awaitRestartBackoff();
    this.ensureStarted();
    if (!this.loaded) {
      const result = await this.sendControl("load", { model: this.modelPath });
      this.tokenCapabilities = result.tokenCapabilities;
      this.loaded = true;
      this.consecutiveCrashes = 0;
    }
    return this.info();
  }

  async chat(request: ChatCompletionRequest, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult> {
    await this.load();
    return this.sendControl("chat", request, onToken, signal, onProgress, onDispatch);
  }

  async setMaxActive(maxActive: number, telemetry?: ActiveLimitTelemetry): Promise<void> {
    if (!Number.isInteger(maxActive) || maxActive < 1) throw new RangeError("maxActive must be positive");
    await this.load();
    await this.sendControl("set_max_active", {
      max_active: maxActive,
      ...(telemetry ? {
        previous_max_active: telemetry.previousMaxActive,
        limiting_reason: telemetry.limitingReason,
        last_adjustment_reason: telemetry.lastAdjustmentReason,
        last_adjustment_at: telemetry.lastAdjustmentAt,
        retained_growth_reserve_bytes: telemetry.retainedGrowthReserveBytes,
        global_resident_memory_bytes: telemetry.globalResidentMemoryBytes,
        pressure_state: telemetry.pressureState,
      } : {}),
    });
  }

  async unload(): Promise<void> {
    if (!this.proc) return;
    try {
      await this.sendControl("unload", { model: this.modelPath });
    } catch {
      // The process may already be gone; shutdown below is authoritative.
    }
    this.loaded = false;
    this.shutdown();
  }

  shutdown(): void {
    for (const pending of this.pending.values()) pending.reject(new Error("resident worker shut down"));
    this.pending.clear();
    try {
      if (this.proc && this.proc.exitCode === null) {
        // Deliberate termination (unload/server shutdown): the SIGTERM exit
        // (143) must not count as a crash or trigger restart backoff.
        this.expectedExit = true;
        if (this.activeProtocol === "v1") {
          const id = `cmd_${crypto.randomUUID()}`;
          this.v1Tracker?.register(id);
          this.write({ protocol: 1, type: "shutdown", request_id: id });
        } else this.write({ type: "shutdown" });
        this.proc.kill();
      }
    } catch {
      // ignore shutdown races
    }
    this.proc = undefined;
    this.loaded = false;
    this.memory = undefined;
    this.retention = undefined;
    this.tokenCapabilities = undefined;
    this.clearHandshake();
  }

  private ensureStarted(): void {
    if (this.proc && this.proc.exitCode === null) return;
    this.expectedExit = false;
    this.memory = undefined;
    this.retention = undefined;
    this.tokenCapabilities = undefined;
    const status = this.backend === "mlx" ? getMlxWorkerStatus() : getLlamaWorkerStatus();
    if (!status.command) {
      const message = status.reason ?? `${this.backend} worker is not available`;
      if (this.backend === "mlx") throw new MlxWorkerError(message, "worker_not_found");
      throw new LlamaWorkerError(message, "worker_not_found");
    }
    mkdirSync(dirname(status.logPath), { recursive: true });
    // Bun's file-backed stderr starts writing at offset zero but does not
    // truncate a longer prior file. Rotate and unlink first so stale trailing
    // bytes cannot survive the next launch. Attribution lives in a sidecar
    // because any header in the stderr file would be overwritten at offset 0.
    try {
      renameSync(status.logPath, `${status.logPath}.previous`);
    } catch {
      // No prior launch log (or it was already removed).
    }
    rmSync(status.logPath, { force: true });
    this.workerLaunchId = crypto.randomUUID();
    writeFileSync(`${status.logPath}.launch.json`, JSON.stringify({
      launchId: this.workerLaunchId,
      startedAt: new Date().toISOString(),
      backend: this.backend,
      key: this.key,
      command: status.command,
    }, null, 2) + "\n");
    const environment = { ...process.env, ...this.envOverrides };
    if (status.source === "bundled") delete environment.CLAP_WORKER_PROTOCOL;
    else if (this.protocolMode === "legacy" || this.protocolMode === "v1") {
      environment.CLAP_WORKER_PROTOCOL = this.protocolMode;
    }
    const proc = Bun.spawn(status.command, {
      stdin: "pipe",
      stdout: "pipe",
      stderr: Bun.file(status.logPath),
      env: environment,
    });
    this.proc = proc;
    this.logPath = status.logPath;
    this.startProtocol(status.source);
    void this.readLoop(proc);
  }

  private startProtocol(source: "configured" | "bundled" | "missing"): void {
    this.clearHandshake();
    this.startupBytes = false;
    if (this.protocolMode === "legacy" && source === "configured") {
      this.activeProtocol = "legacy";
      this.handshake = Promise.resolve();
      return;
    }
    this.activeProtocol = "v1";
    this.v1Tracker = new V1RequestTracker();
    this.handshake = new Promise<void>((resolve, reject) => {
      this.resolveHandshake = resolve;
      this.rejectHandshake = reject;
    });
    // During migration, only explicitly configured workers may fall back, and
    // only when they emit no startup bytes. Bundled workers never downgrade.
    if (allowsLegacyStartupFallback(this.protocolMode, source)) {
      this.handshakeTimer = setTimeout(() => {
        if (this.startupBytes || this.activeProtocol !== "v1") return;
        this.activeProtocol = "legacy";
        this.v1Tracker = undefined;
        this.resolveHandshake?.();
        this.resolveHandshake = undefined;
        this.rejectHandshake = undefined;
      }, 1_000);
    }
  }

  private clearHandshake(): void {
    if (this.handshakeTimer) clearTimeout(this.handshakeTimer);
    this.handshakeTimer = undefined;
    this.handshake = undefined;
    this.resolveHandshake = undefined;
    this.rejectHandshake = undefined;
    this.v1Tracker = undefined;
  }

  private async readLoop(proc: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for await (const chunk of proc.stdout) {
        this.startupBytes = true;
        buffer += decoder.decode(chunk, { stream: true });
        let newlineIndex: number;
        while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (line) this.handleProtocolLine(line);
        }
      }
      const tail = buffer.trim();
      if (tail) this.handleProtocolLine(tail);
    } catch (error) {
      this.rejectAll(error instanceof Error ? error : new Error(String(error)));
    }
    const exitCode = await proc.exited;
    if (this.proc === proc) this.proc = undefined;
    this.loaded = false;
    const phase = this.resolveHandshake ? "during protocol handshake" : this.pending.size ? "during request" : "while idle";
    const exitError = this.workerError(`${this.backend} resident worker exited ${phase} with code ${exitCode}`);
    if (this.resolveHandshake) {
      this.rejectHandshake?.(exitError);
      this.rejectHandshake = undefined;
      this.resolveHandshake = undefined;
    }
    if (!this.expectedExit) {
      if (exitCode !== 0) {
        this.crashes += 1;
        this.consecutiveCrashes += 1;
        this.lastCrashAt = Date.now();
        this.onCrash?.({ key: this.key, backend: this.backend, exitCode, consecutiveCrashes: this.consecutiveCrashes });
      }
      this.rejectAll(exitError);
    }
  }

  // Exponential backoff between restarts after crashes (1s, 2s, 4s... capped
  // at 30s) so a model that dies on load cannot hot-loop expensive reloads.
  // Requests wait out the window instead of failing; a persistent crash loop
  // fails fast with an actionable error.
  private async awaitRestartBackoff(): Promise<void> {
    if (this.consecutiveCrashes === 0 || !this.lastCrashAt) return;
    if (this.proc && this.proc.exitCode === null) return;
    if (this.consecutiveCrashes >= 5) {
      throw this.workerError(
        `${this.backend} resident worker crashed ${this.consecutiveCrashes} times in a row; not restarting automatically. Check the worker log, then retry or restart the server.`,
        "worker_crash_loop",
      );
    }
    const delay = Math.min(1000 * 2 ** (this.consecutiveCrashes - 1), 30000);
    const remaining = this.lastCrashAt + delay - Date.now();
    if (remaining > 0) await Bun.sleep(remaining);
  }

  private handleProtocolLine(line: string): void {
    if (this.activeProtocol === "v1") {
      try {
        const fact = this.v1Tracker!.consumeLine(line);
        if (fact.kind === "ready") {
          this.resolveHandshake?.();
          this.resolveHandshake = undefined;
          this.rejectHandshake = undefined;
          if (this.handshakeTimer) clearTimeout(this.handshakeTimer);
          this.handshakeTimer = undefined;
        } else this.handleV1Fact(fact);
      } catch (error) {
        this.protocolUnhealthy(error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    const decoded = this.legacyProtocol.decode(line);
    if (decoded.kind === "text") {
      const pending = this.firstPending();
      if (pending) {
        pending.content.push(line);
        pending.onToken?.(line);
      }
      return;
    }
    this.handleLegacyMessage(decoded.message);
  }

  private handleLegacyMessage(message: Record<string, unknown>): void {
    const id = typeof message.id === "string" ? message.id : undefined;
    if (message.memory && typeof message.memory === "object") {
      const memory = message.memory as Record<string, unknown>;
      if (typeof memory.active_bytes === "number" && typeof memory.cache_bytes === "number" && typeof memory.peak_active_bytes === "number") {
        this.memory = {
          activeBytes: memory.active_bytes,
          cacheBytes: memory.cache_bytes,
          peakActiveBytes: memory.peak_active_bytes,
        };
      }
    }
    const parsedRetention = parseWorkerRetention(message.retention);
    if (parsedRetention) {
      const previous = this.retention;
      this.retention = parsedRetention;
      this.onTelemetry?.(this, previous, parsedRetention);
    }
    const parsedCapabilities = parseWorkerTokenCapabilities(message.token_capabilities);
    if (parsedCapabilities) this.tokenCapabilities = parsedCapabilities;
    const pending = id ? this.pending.get(id) : this.firstPending();
    if (!pending) return;
    if (message.started === true) {
      const onDispatch = pending.onDispatch;
      pending.onDispatch = undefined;
      onDispatch?.();
    }
    if (message.error) {
      this.pending.delete(id ?? "");
      const code = typeof message.code === "string" ? message.code : undefined;
      pending.reject(this.workerError(String(message.error), code));
      return;
    }
    if (typeof message.token === "string") {
      pending.content.push(message.token);
      pending.onToken?.(message.token);
    }
    if (message.prefill && typeof message.prefill === "object") {
      const prefill = message.prefill as Record<string, unknown>;
      if (typeof prefill.done === "number" && typeof prefill.total === "number") {
        pending.onProgress?.(prefill.done, prefill.total);
      }
    }
    if (typeof message.content === "string") {
      pending.content.push(message.content);
      pending.onToken?.(message.content);
    }
    if (message.usage && typeof message.usage === "object") {
      const usage = message.usage as Record<string, unknown>;
      pending.usage = {
        promptTokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : undefined,
        completionTokens: typeof usage.completion_tokens === "number" ? usage.completion_tokens : undefined,
      };
    }
    if (message.timing && typeof message.timing === "object") {
      const timing = message.timing as Record<string, unknown>;
      const number = (key: string) => typeof timing[key] === "number" ? timing[key] as number : undefined;
      pending.timing = {
        receivedToAdmittedMs: number("received_to_admitted_ms"),
        templateTokenizeMs: number("template_tokenize_ms"),
        coordinatorWaitMs: number("coordinator_wait_ms"),
        coordinatorPlanMs: number("coordinator_plan_ms"),
        coordinatorApplyMs: number("coordinator_apply_ms"),
        schedulerWaitMs: number("scheduler_wait_ms"),
        cacheMaterializeMs: number("cache_materialize_ms"),
        prefillMs: number("prefill_ms"),
        residualPrefillTokens: number("residual_prefill_tokens"),
        prefillTokens: number("prefill_tokens"),
        prefillChunks: number("prefill_chunks"),
        firstDecodeMs: number("first_decode_ms"),
        firstEmitMs: number("first_emit_ms"),
        normalPrefillQuantum: number("normal_prefill_quantum"),
        contendedPrefillQuantum: number("contended_prefill_quantum"),
      };
    }
    if (message.cache && typeof message.cache === "object") {
      const cache = message.cache as Record<string, unknown>;
      const rejectionReasons = new Set(["namespace", "model_domain", "generation", "capability", "busy_lease", "materialization", "session", "nontrim", "min_prefix", "capacity", "absent_anchor", "lower_rank"]);
      const candidates = Array.isArray(cache.candidates) ? cache.candidates.flatMap((value) => {
        if (!value || typeof value !== "object") return [];
        const candidate = value as Record<string, unknown>;
        if (typeof candidate.slot !== "number" || typeof candidate.shared_prefix_tokens !== "number") return [];
        return [{
          slot: candidate.slot,
          generation: typeof candidate.generation === "number" ? candidate.generation : undefined,
          state: typeof candidate.state === "string" ? candidate.state : undefined,
          sharedPrefixTokens: candidate.shared_prefix_tokens,
          namespaceCompatible: typeof candidate.namespace_compatible === "boolean" ? candidate.namespace_compatible : undefined,
          modelCompatible: typeof candidate.model_compatible === "boolean" ? candidate.model_compatible : undefined,
          sessionCompatible: typeof candidate.session_compatible === "boolean" ? candidate.session_compatible : undefined,
          generationCompatible: typeof candidate.generation_compatible === "boolean" ? candidate.generation_compatible : undefined,
          busyEligible: typeof candidate.busy_eligible === "boolean" ? candidate.busy_eligible : undefined,
          leaseEligible: typeof candidate.lease_eligible === "boolean" ? candidate.lease_eligible : undefined,
          materialized: typeof candidate.materialized === "boolean" ? candidate.materialized : undefined,
          trimEligible: typeof candidate.trim_eligible === "boolean" ? candidate.trim_eligible : undefined,
          copyEligible: typeof candidate.copy_eligible === "boolean" ? candidate.copy_eligible : undefined,
          eligible: typeof candidate.eligible === "boolean" ? candidate.eligible : undefined,
          selected: typeof candidate.selected === "boolean" ? candidate.selected : undefined,
          rejection: typeof candidate.rejection === "string" && rejectionReasons.has(candidate.rejection)
            ? candidate.rejection as NonNullable<ResidentCacheInfo["candidates"]>[number]["rejection"]
            : undefined,
        }];
      }) : undefined;
      pending.cache = {
        hit: typeof cache.hit === "boolean" ? cache.hit : undefined,
        reusedTokens: typeof cache.reused_tokens === "number" ? cache.reused_tokens : undefined,
        reuseKind: cache.reuse_kind === "slot" || cache.reuse_kind === "branch" || cache.reuse_kind === "anchor" ? cache.reuse_kind : undefined,
        reuseScope: cache.reuse_scope === "system" || cache.reuse_scope === "conversation" || cache.reuse_scope === "session" || cache.reuse_scope === "agent" || cache.reuse_scope === "project" || cache.reuse_scope === "harness" || cache.reuse_scope === "tenant" ? cache.reuse_scope : undefined,
        sideRequest: typeof cache.side_request === "boolean" ? cache.side_request : undefined,
        slot: typeof cache.slot === "number" ? cache.slot : undefined,
        namespace: typeof cache.namespace === "string" ? cache.namespace : undefined,
        donorSlot: typeof cache.donor_slot === "number" ? cache.donor_slot : undefined,
        targetSlot: typeof cache.target_slot === "number" ? cache.target_slot : undefined,
        evictedSlots: Array.isArray(cache.evicted_slots) && cache.evicted_slots.every((slot) => typeof slot === "number") ? cache.evicted_slots as number[] : undefined,
        decisionUs: typeof cache.decision_us === "number" ? cache.decision_us : undefined,
        plannedReuseTokens: typeof cache.planned_reuse_tokens === "number" ? cache.planned_reuse_tokens : undefined,
        realizedReuseTokens: typeof cache.realized_reuse_tokens === "number" ? cache.realized_reuse_tokens : undefined,
        fallback: typeof cache.fallback === "string" ? cache.fallback : undefined,
        missReason: typeof cache.miss_reason === "string" ? cache.miss_reason : undefined,
        workerLaunchId: typeof cache.worker_launch_id === "string" ? cache.worker_launch_id : undefined,
        donorGeneration: typeof cache.donor_generation === "number" ? cache.donor_generation : undefined,
        targetGeneration: typeof cache.target_generation === "number" ? cache.target_generation : undefined,
        evictions: Array.isArray(cache.evictions) ? cache.evictions.flatMap((value) => {
          if (!value || typeof value !== "object") return [];
          const eviction = value as Record<string, unknown>;
          return typeof eviction.slot === "number" ? [{ slot: eviction.slot, reason: typeof eviction.reason === "string" ? eviction.reason : undefined }] : [];
        }) : undefined,
        candidates,
        systemTokenHash: typeof cache.system_token_hash === "string" ? cache.system_token_hash : undefined,
        systemTokenCount: typeof cache.system_token_count === "number" ? cache.system_token_count : undefined,
        toolsTokenHash: typeof cache.tools_token_hash === "string" ? cache.tools_token_hash : undefined,
        toolsTokenCount: typeof cache.tools_token_count === "number" ? cache.tools_token_count : undefined,
        ...(typeof cache.stable_boundary_token_hash === "string" && cache.stable_boundary_token_hash.length > 0
          && typeof cache.stable_boundary_token_count === "number" && Number.isInteger(cache.stable_boundary_token_count)
          && cache.stable_boundary_token_count > 0
          && typeof cache.stable_boundary_kind === "string" && cache.stable_boundary_kind.length > 0 ? {
            stableBoundaryTokenHash: cache.stable_boundary_token_hash,
            stableBoundaryTokenCount: cache.stable_boundary_token_count,
            stableBoundaryKind: cache.stable_boundary_kind,
          } : {}),
        stableBoundaries: Array.isArray(cache.stable_boundaries) ? cache.stable_boundaries.flatMap((value) => {
          if (!value || typeof value !== "object") return [];
          const boundary = value as Record<string, unknown>;
          if (typeof boundary.kind !== "string" ||
              (boundary.status !== "resolved" && boundary.status !== "authorized" && boundary.status !== "skipped")) return [];
          const resolved = (boundary.status === "resolved" || boundary.status === "authorized") && typeof boundary.token_hash === "string" &&
              typeof boundary.token_count === "number" && typeof boundary.materialized === "boolean";
          if ((boundary.status === "resolved" || boundary.status === "authorized") && !resolved) return [];
          const label = typeof boundary.label === "string" && boundary.label.length <= 64 &&
              /^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(boundary.label) ? boundary.label : undefined;
          const skipReason = boundary.skip_reason === "unsupported_template_boundary" ||
              boundary.skip_reason === "non_prefix_template_boundary" ? boundary.skip_reason : undefined;
          return [{
            tokenHash: resolved ? boundary.token_hash as string : undefined,
            tokenCount: resolved ? boundary.token_count as number : undefined,
            kind: boundary.kind,
            label,
            requested: boundary.requested === true,
            status: boundary.status as "resolved" | "authorized" | "skipped",
            skipReason,
            materialized: resolved ? boundary.materialized as boolean : undefined,
          }];
        }) : undefined,
        promptTokenHash: typeof cache.prompt_token_hash === "string" ? cache.prompt_token_hash : undefined,
        promptTokenCount: typeof cache.prompt_token_count === "number" ? cache.prompt_token_count : undefined,
        prefillMs: typeof cache.prefill_ms === "number" ? cache.prefill_ms : undefined,
      };
    }
    if (message.finish_reason === "stop" || message.finish_reason === "length" || message.finish_reason === "cancel") {
      pending.finishReason = message.finish_reason;
    }
    if (parsedCapabilities) pending.tokenCapabilities = parsedCapabilities;
    if (message.loaded === true || message.unloaded === true || message.done === true) {
      if (id) this.pending.delete(id);
      else this.deleteFirstPending();
      if (pending.cache && !pending.cache.workerLaunchId) pending.cache.workerLaunchId = this.workerLaunchId;
      pending.resolve({ content: pending.content.join(""), usage: pending.usage, finishReason: pending.finishReason, cache: pending.cache, timing: pending.timing, tokenCapabilities: pending.tokenCapabilities });
    }
  }

  private handleV1Fact(fact: ResidentProtocolFact): void {
    if (fact.kind === "telemetry") {
      const telemetry = fact.telemetry;
      this.handleLegacyMessage("memory" in telemetry || "retention" in telemetry
        ? telemetry : { retention: telemetry });
      return;
    }
    if (fact.kind === "ready" || fact.kind === "diagnostic" || fact.kind === "accepted") return;
    const pending = this.pending.get(fact.requestId);
    if (!pending) return;
    if (fact.kind === "started") {
      const onDispatch = pending.onDispatch;
      pending.onDispatch = undefined;
      onDispatch?.();
      return;
    }
    if (fact.kind === "token") { pending.content.push(fact.text); pending.onToken?.(fact.text); return; }
    if (fact.kind === "content") {
      if (typeof fact.content === "string") { pending.content.push(fact.content); pending.onToken?.(fact.content); }
      return;
    }
    if (fact.kind === "prefill_progress") { pending.onProgress?.(fact.done, fact.total); return; }
    if (fact.kind === "failed") {
      this.pending.delete(fact.requestId);
      pending.reject(this.workerError(fact.error.message, fact.error.code));
      return;
    }
    const result = fact.result as Record<string, unknown>;
    const { content: resultContent, ...resultWithoutContent } = result;
    const content = pending.content.length === 0 && typeof resultContent === "string"
      ? { content: resultContent } : {};
    this.handleLegacyMessage({ ...resultWithoutContent, ...content, id: fact.requestId, done: true });
  }

  private protocolUnhealthy(cause: Error): void {
    const detail = cause instanceof WorkerProtocolFault ? `worker protocol ${cause.code}: ${cause.message}` : `worker protocol failure: ${cause.message}`;
    const error = this.workerError(detail, "worker_protocol_error");
    this.rejectHandshake?.(error);
    this.rejectHandshake = undefined;
    this.resolveHandshake = undefined;
    this.rejectAll(error);
    this.loaded = false;
    this.expectedExit = false;
    try { this.proc?.kill(); } catch { /* process already exited */ }
    this.proc = undefined;
  }

  private async sendControl(type: string, body: Record<string, unknown>, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult> {
    this.ensureStarted();
    await this.handshake;
    const id = `req_${crypto.randomUUID()}`;
    const promise = new Promise<ResidentChatResult>((resolve, reject) => {
      this.pending.set(id, { content: [], resolve, reject, onToken, onProgress, onDispatch });
    });
    if (signal) {
      const cancel = () => {
        if (!this.pending.has(id)) return;
        try {
          if (this.activeProtocol === "v1") {
            const cancelId = `cmd_${crypto.randomUUID()}`;
            this.v1Tracker!.register(cancelId);
            this.write({ protocol: 1, type: "cancel", request_id: cancelId, target_request_id: id });
          } else this.write({ id, type: "cancel" });
        } catch {
          // worker already gone; pending will be rejected by shutdown paths
        }
      };
      if (signal.aborted) queueMicrotask(cancel);
      else signal.addEventListener("abort", cancel, { once: true });
    }
    if (this.activeProtocol === "v1") {
      this.v1Tracker!.register(id);
      const requestType = type === "chat" ? "generate" : type;
      const envelope = requestType === "generate"
        ? { protocol: 1, type: requestType, request_id: id, prompt: JSON.stringify(body), request: body }
        : { protocol: 1, type: requestType, request_id: id, ...body };
      this.write(envelope);
    } else this.write({ id, type, ...body });
    return promise;
  }

  private write(message: Record<string, unknown>): void {
    if (!this.proc) throw new Error("resident worker is not started");
    this.proc.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private firstPending(): Pending | undefined {
    return this.pending.values().next().value;
  }

  private deleteFirstPending(): void {
    const key = this.pending.keys().next().value;
    if (key) this.pending.delete(key);
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private workerError(message: string, code = "resident_worker_error"): Error {
    const detail = code === "resident_worker_error" ? enrichWorkerError(message, this.backend, this.logPath) : message;
    if (this.backend === "mlx") return new MlxWorkerError(detail, code);
    return new LlamaWorkerError(detail, code);
  }
}

export function parseWorkerRetention(value: unknown): ResidentMlxRetention | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  const integer = (key: string): number | undefined => {
    const entry = raw[key];
    return typeof entry === "number" && Number.isSafeInteger(entry) && entry >= 0 ? entry : undefined;
  };
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
  const retainedBytes = integer("retained_bytes");
  const sessionBytes = integer("session_bytes");
  const anchorBytes = integer("anchor_bytes");
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
    retainedAnchors: retainedAnchors!, retainedBytes: retainedBytes!, sessionBytes: sessionBytes!,
    anchorBytes: anchorBytes!, budgetBytes: budgetBytes!, highWatermarkBytes: highWatermarkBytes!,
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

function enrichWorkerError(message: string, backend: ResidentBackend, logPath?: string): string {
  const logHint = logPath ? ` See ${logPath}.` : "";
  if (backend === "llama" && /llama_decode|outofmemory|out of memory|metal|gpu/i.test(message)) {
    return `${message}${logHint} For GGUF/llama.cpp Metal failures, try a smaller quant such as Q4_K_M, reduce CLAP_LLAMA_CONTEXT/CLAP_LLAMA_BATCH/CLAP_LLAMA_UBATCH, lower CLAP_LLAMA_GPU_LAYERS, or use CPU fallback with CLAP_LLAMA_GPU_LAYERS=0.`;
  }
  return `${message}${logHint}`;
}

export class ResidentWorkerRegistry {
  private readonly workers = new Map<string, ResidentWorkerProcess>();
  private readonly workerEnvironments = new Map<string, Record<string, string>>();
  private readonly recentRetainedGrowth = new Map<string, { bytes: number; at: number }>();
  private readonly lastAdjustments = new Map<string, { at: number; reason: string; previous: number }>();
  private pressureState: MemoryPressure = "normal";
  private pressureTimer?: ReturnType<typeof setInterval>;
  private rebalanceScheduled = false;
  private rebalancing = false;
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
  /** Configured workers may coexist in auto/legacy mode; bundled workers always require v1. */
  workerProtocolMode: ResidentWorkerProtocolMode = "auto";

  getOrCreate(key: string, backend: ResidentBackend, modelPath: string): ResidentWorkerHandle {
    const existing = this.workers.get(key);
    if (existing) return existing;
    const environment = this.workerEnv?.(modelPath, backend) ?? {};
    const worker = new ResidentWorkerProcess(key, backend, modelPath,
      (info) => this.onCrash?.(info), environment,
      (source, previous, current) => this.handleTelemetry(source, previous, current),
      this.workerProtocolMode);
    this.workers.set(key, worker);
    this.workerEnvironments.set(key, environment);
    if (!this.pressureTimer) {
      this.pressureTimer = setInterval(() => this.scheduleRebalance("pressure_poll"), 5_000);
      this.pressureTimer.unref?.();
    }
    return worker;
  }

  get(key: string): ResidentWorkerHandle | undefined {
    return this.workers.get(key);
  }

  async unload(key: string): Promise<void> {
    const worker = this.workers.get(key);
    this.workers.delete(key);
    this.workerEnvironments.delete(key);
    this.recentRetainedGrowth.delete(key);
    this.lastAdjustments.delete(key);
    await worker?.unload();
    this.scheduleRebalance("model_unloaded");
  }

  shutdown(key: string): void {
    const worker = this.workers.get(key);
    this.workers.delete(key);
    this.workerEnvironments.delete(key);
    this.recentRetainedGrowth.delete(key);
    this.lastAdjustments.delete(key);
    worker?.shutdown();
    this.scheduleRebalance("model_shutdown");
  }

  shutdownAll(): void {
    for (const key of [...this.workers.keys()]) this.shutdown(key);
    if (this.pressureTimer) clearInterval(this.pressureTimer);
    this.pressureTimer = undefined;
  }

  async rebalance(reason = "manual"): Promise<void> {
    if (this.rebalancing) return;
    const resident = [...this.workers.values()].filter((worker) => worker.info().retention);
    if (!resident.length) return;
    this.rebalancing = true;
    try {
      const pids = resident.map((worker) => worker.info().pid).filter((pid): pid is number => pid !== undefined);
      const snapshot = await this.memorySnapshot(pids);
      const physical = Math.max(1, snapshot.physicalMemoryBytes);
      const pressure = classifyMemoryPressure(snapshot.availableMemoryBytes, physical, this.pressureState);
      const topologyChanged = pressure !== this.pressureState;
      this.pressureState = pressure;
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
        const fallbackResident = info.memory
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
          retainedBytes: retention.retainedBytes,
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
    } finally {
      this.rebalancing = false;
    }
  }

  private handleTelemetry(worker: ResidentWorkerProcess, previous: ResidentMlxRetention | undefined,
    current: ResidentMlxRetention): void {
    const growth = previous ? Math.max(0, current.retainedBytes - previous.retainedBytes) : 0;
    if (growth > 0) {
      const recent = this.recentRetainedGrowth.get(worker.key);
      this.recentRetainedGrowth.set(worker.key, {
        bytes: Math.max(growth, recent && Date.now() - recent.at <= 60_000 ? recent.bytes : 0),
        at: Date.now(),
      });
    }
    const threshold = Math.max(32 * 1024 ** 2, Math.floor((previous?.retainedBytes ?? 0) / 10));
    const material = !previous || Math.abs(current.retainedBytes - previous.retainedBytes) >= threshold
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
