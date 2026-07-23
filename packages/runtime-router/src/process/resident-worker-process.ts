import type { ChatCompletionRequest, ModelTokenCapabilities } from "@clap/api";
import type { StructuredOutputContract, WorkerCapabilities } from "@clap/worker-protocol";
import { getLlamaWorkerStatus, LlamaWorkerError } from "@clap/runtime-llama";
import { getMlxWorkerStatus, MlxWorkerError } from "@clap/runtime-mlx";
import { V1RequestTracker, type ResidentProtocolFact } from "../protocol/request-tracker";
import { WorkerProtocolFault } from "../protocol/errors";
import { classifyWorkerCrash, classifyWorkerExitPhase } from "./crash-classification";
import { WorkerLaunchLogStore } from "./launch-log-store";
import { applyWorkerPayload, mapWorkerResultPayload, type PendingWorkerResult } from "./result-mapper";
import { type ActiveLimitTelemetry, type ResidentBackend, type ResidentChatResult,
  type ResidentCrashListener, type ResidentMlxMemory, type ResidentMlxRetention,
  type EffectiveCapabilities, type ResidentChatOptions, type ResidentProgress, type ResidentWorkerHandle,
  type ResidentWorkerInfo } from "../resident";
import type { WorkerLaunchContext, WorkerLaunchPaths, WorkerLoadState, WorkerLoadStateEvent,
  WorkerModelDescriptor, WorkerRssSampler } from "./types";
import { measuredMemory, unavailableMemory, type MemoryValue } from "../residency/types";

type WorkerProcess = Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
type CloseReason = "shutdown" | "unload";

type ActiveLaunch = {
  context: WorkerLaunchContext;
  proc: WorkerProcess;
  tracker: V1RequestTracker;
  handshake: Promise<void>;
  resolveHandshake?: () => void;
  rejectHandshake?: (error: Error) => void;
  handshakeTimer?: ReturnType<typeof setTimeout>;
  lastDiagnostic?: string;
  expectedExit: boolean;
  shutdownId?: string;
  resolveShutdown?: () => void;
  closePromise?: Promise<void>;
  workerCapabilities?: WorkerCapabilities;
  exited: Promise<void>;
  resolveExited: () => void;
};

export class ResidentWorkerProcess implements ResidentWorkerHandle {
  private readonly pending = new Map<string, PendingWorkerResult>();
  private active?: ActiveLaunch;
  private starting?: Promise<ActiveLaunch>;
  private loadPromise?: Promise<ResidentWorkerInfo>;
  private loaded = false;
  private loadState: WorkerLoadState = "not_started";
  private readonly loadStateListeners = new Set<(event: WorkerLoadStateEvent) => void>();
  private crashes = 0;
  private consecutiveCrashes = 0;
  private lastCrashAt?: number;
  private memory?: ResidentMlxMemory;
  private retention?: ResidentMlxRetention;
  private tokenCapabilities?: ModelTokenCapabilities;
  private effectiveCapabilities?: EffectiveCapabilities;
  private workerLaunchId?: string;
  private lastLaunchPaths?: WorkerLaunchPaths;
  private crashClassification?: string;

  constructor(
    public readonly key: string,
    public readonly backend: ResidentBackend,
    public readonly modelPath: string,
    private readonly onCrash?: ResidentCrashListener,
    private readonly envOverrides?: Record<string, string>,
    private readonly onTelemetry?: (worker: ResidentWorkerProcess,
      previous: ResidentMlxRetention | undefined, current: ResidentMlxRetention) => void,
    private readonly descriptor: WorkerModelDescriptor = { modelId: key },
    private readonly launchLogs = new WorkerLaunchLogStore(),
    private readonly spawnWorker: typeof Bun.spawn = Bun.spawn,
    private readonly onLoadStateEvent?: (event: WorkerLoadStateEvent) => void,
  ) {}

  info(): ResidentWorkerInfo {
    return {
      pid: this.active?.proc.pid,
      launchId: this.active?.context.paths.launchId ?? this.workerLaunchId,
      stderrLogPath: this.active?.context.paths.stderrPath ?? this.lastLaunchPaths?.stderrPath,
      launchMetadataPath: this.active?.context.paths.metadataPath ?? this.lastLaunchPaths?.metadataPath,
      crashClassification: this.crashClassification,
      state: this.active ? "resident" : "not_started",
      loadState: this.loadState,
      crashes: this.crashes,
      lastCrashAt: this.lastCrashAt ? new Date(this.lastCrashAt).toISOString() : undefined,
      memory: this.memory,
      retention: this.retention,
      tokenCapabilities: this.tokenCapabilities,
      workerCapabilities: this.active?.workerCapabilities,
      effectiveCapabilities: this.effectiveCapabilities,
      structuredOutputCapabilities: this.effectiveCapabilities?.generation.structuredOutput,
    };
  }

  async observeRss(sampler: WorkerRssSampler): Promise<MemoryValue> {
    const pid = this.active?.proc.pid;
    if (pid === undefined) return unavailableMemory("not_observed");
    const bytes = await sampler(pid);
    if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes <= 0) {
      return unavailableMemory("not_reported");
    }
    return measuredMemory(bytes, "resident_rss");
  }

  onLoadState(listener: (event: WorkerLoadStateEvent) => void): () => void {
    this.loadStateListeners.add(listener);
    return () => this.loadStateListeners.delete(listener);
  }

  load(): Promise<ResidentWorkerInfo> {
    if (this.loaded && this.active && !this.active.closePromise) return Promise.resolve(this.info());
    if (this.loadPromise) return this.loadPromise;
    if (!this.active) this.setLoadState("starting");
    const load = this.loadOnce();
    this.loadPromise = load;
    void load.finally(() => { if (this.loadPromise === load) this.loadPromise = undefined; }).catch(() => {});
    return load;
  }

  private async loadOnce(): Promise<ResidentWorkerInfo> {
    try {
      await this.awaitRestartBackoff();
      const launch = await this.ensureStarted();
      if (launch.closePromise) throw this.workerError("resident worker is shutting down");
      if (!this.loaded) {
        this.setLoadState("loading", launch);
        const result = await this.sendControl("load", { model: this.modelPath }, undefined, undefined,
          undefined, undefined, launch);
        if (this.active !== launch || launch.closePromise) throw this.workerError("resident worker shut down");
        this.tokenCapabilities = result.tokenCapabilities;
        this.effectiveCapabilities = result.effectiveCapabilities;
        this.loaded = true;
        this.setLoadState("resident", launch);
        this.consecutiveCrashes = 0;
      }
      return this.info();
    } catch (error) {
      if (this.loadState !== "closing") this.setLoadState(this.active ? "starting" : "not_started", this.active);
      throw error;
    }
  }

  async chat(request: ChatCompletionRequest, onToken?: (token: string) => void, signal?: AbortSignal,
    onProgress?: ResidentProgress, onDispatch?: () => void, options?: ResidentChatOptions): Promise<ResidentChatResult> {
    await this.load();
    if (!options?.cacheIdentity) throw new Error("resident chat requires an internal cache identity");
    const { cache_identity: _untrustedIdentity, structured_output: _untrustedContract, ...publicRequest } =
      request as ChatCompletionRequest & { cache_identity?: unknown; structured_output?: unknown };
    const structuredOutput = options.structuredOutput;
    this.requireStructuredOutputCapability(structuredOutput);
    return this.sendControl("chat", publicRequest, onToken, signal, onProgress, onDispatch,
      undefined, options.cacheIdentity, structuredOutput);
  }

  private requireStructuredOutputCapability(contract?: StructuredOutputContract): void {
    if (!contract) return;
    const capabilities = this.effectiveCapabilities?.generation.structuredOutput;
    const mode = capabilities?.[contract.kind] ?? "unsupported";
    const supported = contract.strength === "required"
      ? mode === "native"
      : mode === "native" || mode === "post_validate";
    const schemaBytes = contract.kind === "json_schema"
      ? new TextEncoder().encode(JSON.stringify(contract.schema)).byteLength : 0;
    if (supported && schemaBytes <= (capabilities?.max_schema_bytes ?? 0)) return;
    throw this.workerError(
      `active worker cannot satisfy ${contract.strength} ${contract.kind} structured output`,
      "structured_output_capability_required");
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
    const launch = this.active;
    if (!launch) return;
    try {
      await this.sendControl("unload", { model: this.modelPath }, undefined, undefined,
        undefined, undefined, launch);
    } catch {
      // Closing below is authoritative when unload races an exit.
    }
    this.loaded = false;
    await this.close(launch, "unload");
  }

  shutdown(): void {
    void this.shutdownAsync();
  }

  async drainAndShutdownAsync(): Promise<void> {
    // The server's identity barrier prevents new chats while rotation owns the
    // writer lock. Let already-dispatched old-generation work reach a terminal
    // event before asking the worker to exit.
    while (this.pending.size > 0) await Bun.sleep(5);
    await this.shutdownAsync();
  }

  shutdownAsync(): Promise<void> {
    const starting = this.starting;
    if (starting) return starting.then((launch) => this.close(launch, "shutdown"), () => {});
    const launch = this.active;
    return launch ? this.close(launch, "shutdown") : Promise.resolve();
  }

  private async ensureStarted(): Promise<ActiveLaunch> {
    if (this.active && this.active.proc.exitCode === null && !this.active.closePromise) return this.active;
    if (this.starting) return this.starting;
    const starting = this.startWorker();
    this.starting = starting;
    try { return await starting; } finally { if (this.starting === starting) this.starting = undefined; }
  }

  private async startWorker(): Promise<ActiveLaunch> {
    this.setLoadState("starting");
    this.loaded = false;
    this.memory = undefined;
    this.retention = undefined;
    this.tokenCapabilities = undefined;
    this.effectiveCapabilities = undefined;
    const status = this.backend === "mlx" ? getMlxWorkerStatus() : getLlamaWorkerStatus();
    if (!status.command) {
      const message = status.reason ?? `${this.backend} worker is not available`;
      if (this.backend === "mlx") throw new MlxWorkerError(message, "worker_not_found");
      throw new LlamaWorkerError(message, "worker_not_found");
    }
    const context = await this.launchLogs.prepareLaunch({ backend: this.backend,
      modelId: this.descriptor.modelId, revision: this.descriptor.revision, modelPath: this.modelPath }, status.command);
    let proc: WorkerProcess;
    try {
      proc = this.spawnWorker(status.command, { stdin: "pipe", stdout: "pipe",
        stderr: Bun.file(context.paths.stderrPath), env: { ...process.env, ...this.envOverrides } });
    } catch (error) {
      this.setLoadState("not_started");
      await this.ignoreMetadataFailure(() => this.launchLogs.finalize(context, null, "spawn_failure"));
      throw error;
    }
    let resolveHandshake!: () => void;
    let rejectHandshake!: (error: Error) => void;
    let resolveExited!: () => void;
    const launch: ActiveLaunch = {
      context, proc, tracker: new V1RequestTracker(), expectedExit: false,
      handshake: new Promise<void>((resolve, reject) => { resolveHandshake = resolve; rejectHandshake = reject; }),
      resolveHandshake, rejectHandshake,
      exited: new Promise<void>((resolve) => { resolveExited = resolve; }), resolveExited,
    };
    this.active = launch;
    this.workerLaunchId = context.paths.launchId;
    this.lastLaunchPaths = context.paths;
    this.crashClassification = undefined;
    await this.ignoreMetadataFailure(() => this.launchLogs.markSpawned(context, proc.pid));
    launch.handshakeTimer = setTimeout(() => {
      if (this.active === launch && launch.resolveHandshake) this.protocolUnhealthy(launch,
        new WorkerProtocolFault("handshake_timeout", "Worker did not send ready within 1000ms", "worker"));
    }, 1_000);
    void this.readLoop(launch);
    return launch;
  }

  private async readLoop(launch: ActiveLaunch): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for await (const chunk of launch.proc.stdout) {
        buffer += decoder.decode(chunk, { stream: true });
        let newlineIndex: number;
        while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (line) this.handleProtocolLine(launch, line);
        }
      }
      const tail = buffer.trim();
      if (tail) this.handleProtocolLine(launch, tail);
    } catch (error) {
      if (this.active === launch) this.rejectLaunchPending(launch,
        error instanceof Error ? error.message : String(error));
    }
    const exitCode = await launch.proc.exited;
    await this.handleExit(launch, exitCode);
  }

  private async handleExit(launch: ActiveLaunch, exitCode: number): Promise<void> {
    const current = this.active === launch;
    const launchPending = [...this.pending.values()].filter((item) =>
      item.launchPaths?.launchId === launch.context.paths.launchId).length;
    const phase = classifyWorkerExitPhase(Boolean(launch.resolveHandshake), launchPending);
    const classification = classifyWorkerCrash({ protocolFault: launch.context.protocolFault,
      expectedExit: launch.expectedExit, exitCode, phase: launch.context.phase });
    if (current) this.crashClassification = classification;
    const diagnostic = launch.lastDiagnostic ? `. Last worker diagnostic: ${launch.lastDiagnostic}` : "";
    const exitMessage = `${this.backend} resident worker exited ${phase} with code ${exitCode}${diagnostic}`;
    if (launch.resolveHandshake) {
      launch.rejectHandshake?.(this.workerError(exitMessage, "resident_worker_error", launch.context.paths.stderrPath));
      this.clearHandshake(launch);
    }
    if (classification !== "expected_exit") {
      this.crashes += 1;
      this.consecutiveCrashes += 1;
      this.lastCrashAt = Date.now();
      this.onCrash?.({ key: this.key, backend: this.backend, exitCode,
        consecutiveCrashes: this.consecutiveCrashes, launchId: launch.context.paths.launchId,
        logPath: launch.context.paths.stderrPath, metadataPath: launch.context.paths.metadataPath, classification });
      this.rejectLaunchPending(launch, exitMessage);
    }
    if (current) {
      this.active = undefined;
      this.loaded = false;
      this.memory = undefined;
      this.retention = undefined;
      this.tokenCapabilities = undefined;
      this.effectiveCapabilities = undefined;
      this.setLoadState("not_started");
    }
    await this.ignoreMetadataFailure(() => this.launchLogs.finalize(launch.context, exitCode, classification));
    launch.resolveExited();
  }

  private handleProtocolLine(launch: ActiveLaunch, line: string): void {
    try {
      const fact = launch.tracker.consumeLine(line);
      if (fact.kind === "ready") {
        if (this.active === launch) launch.workerCapabilities = fact.workerCapabilities;
        launch.context.phase = "idle";
        void this.ignoreMetadataFailure(() => this.launchLogs.markReady(launch.context));
        launch.resolveHandshake?.();
        this.clearHandshake(launch, false);
      } else this.handleV1Fact(launch, fact);
    } catch (error) {
      this.protocolUnhealthy(launch, error instanceof Error ? error : new Error(String(error)));
    }
  }

  private clearHandshake(launch: ActiveLaunch, reject = true): void {
    if (launch.handshakeTimer) clearTimeout(launch.handshakeTimer);
    launch.handshakeTimer = undefined;
    if (reject) launch.rejectHandshake = undefined;
    launch.resolveHandshake = undefined;
  }

  private handleV1Fact(launch: ActiveLaunch, fact: ResidentProtocolFact): void {
    if (fact.kind === "telemetry") {
      if (this.active === launch) this.applyWorkerPayload(launch, fact.telemetry);
      return;
    }
    if (fact.kind === "diagnostic") { launch.lastDiagnostic = fact.message; return; }
    if (fact.kind === "ready" || fact.kind === "accepted") return;
    if (fact.requestId === launch.shutdownId && fact.kind === "completed") {
      launch.resolveShutdown?.();
      launch.resolveShutdown = undefined;
      return;
    }
    const pending = this.pending.get(fact.requestId);
    if (!pending || pending.launchPaths?.launchId !== launch.context.paths.launchId) return;
    if (fact.kind === "started") {
      pending.phase = pending.phase === "load" ? "load" : "prefill";
      launch.context.phase = pending.phase;
      const callback = pending.onDispatch; pending.onDispatch = undefined; callback?.(); return;
    }
    if (fact.kind === "token") {
      pending.phase = "decode"; launch.context.phase = "decode";
      pending.content.push(fact.text); pending.onToken?.(fact.text); return;
    }
    if (fact.kind === "content") {
      if (typeof fact.content === "string") { pending.content.push(fact.content); pending.onToken?.(fact.content); }
      return;
    }
    if (fact.kind === "prefill_progress") {
      pending.phase = "prefill"; launch.context.phase = "prefill";
      pending.onProgress?.(fact.done, fact.total); return;
    }
    if (fact.kind === "failed") {
      this.pending.delete(fact.requestId); pending.cleanup?.();
      if (pending.phase === "load" && this.active === launch && !launch.closePromise) {
        this.tokenCapabilities = undefined;
        this.effectiveCapabilities = undefined;
        this.setLoadState("starting", launch);
      }
      pending.reject(this.workerError(fact.error.message, fact.error.code, pending.launchPaths?.stderrPath)); return;
    }
    this.applyWorkerPayload(launch, mapWorkerResultPayload(fact.result as Record<string, unknown>,
      fact.requestId, pending.content.length > 0));
    if (![...this.pending.values()].some((item) => item.launchPaths?.launchId === launch.context.paths.launchId)) {
      launch.context.phase = "idle";
    }
  }

  private applyWorkerPayload(launch: ActiveLaunch, message: Record<string, unknown>): void {
    applyWorkerPayload(message, {
      pending: this.pending, retention: this.retention, workerLaunchId: launch.context.paths.launchId,
      setMemory: (memory) => { if (this.active === launch) this.memory = memory; },
      setRetention: (retention) => { if (this.active === launch) this.retention = retention; },
      setTokenCapabilities: (capabilities) => { if (this.active === launch) this.tokenCapabilities = capabilities; },
      setEffectiveCapabilities: (capabilities) => {
        if (this.active === launch) this.effectiveCapabilities = capabilities;
      },
      onRetention: (previous, current) => { if (this.active === launch) this.onTelemetry?.(this, previous, current); },
      workerError: (detail, code) => this.workerError(detail, code, launch.context.paths.stderrPath),
    });
  }

  private protocolUnhealthy(launch: ActiveLaunch, cause: Error): void {
    launch.context.protocolFault = true;
    if (this.active === launch) this.crashClassification = "protocol_fault";
    const diagnostic = launch.lastDiagnostic ? `. Last worker diagnostic: ${launch.lastDiagnostic}` : "";
    const detail = cause instanceof WorkerProtocolFault
      ? `worker protocol ${cause.code}: ${cause.message}${diagnostic}`
      : `worker protocol failure: ${cause.message}${diagnostic}`;
    launch.rejectHandshake?.(this.workerError(detail, "worker_protocol_error", launch.context.paths.stderrPath));
    this.clearHandshake(launch);
    this.rejectLaunchPending(launch, detail, "worker_protocol_error");
    if (this.active === launch) {
      this.active = undefined;
      this.loaded = false;
      this.memory = undefined;
      this.retention = undefined;
      this.tokenCapabilities = undefined;
      this.effectiveCapabilities = undefined;
      this.setLoadState("not_started");
    }
    try { launch.proc.kill(); } catch { /* process already exited */ }
  }

  private async sendControl(type: string, body: Record<string, unknown>, onToken?: (token: string) => void,
    signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void,
    expectedLaunch?: ActiveLaunch, cacheIdentity?: ResidentChatOptions["cacheIdentity"],
    structuredOutput?: StructuredOutputContract): Promise<ResidentChatResult> {
    const launch = expectedLaunch ?? await this.ensureStarted();
    if (this.active !== launch || launch.closePromise) throw this.workerError("resident worker is shutting down");
    await launch.handshake;
    if (this.active !== launch || launch.closePromise) throw this.workerError("resident worker shut down");
    const id = `req_${crypto.randomUUID()}`;
    let cleanup: (() => void) | undefined;
    const promise = new Promise<ResidentChatResult>((resolve, reject) => {
      const phase = type === "load" ? "load" : type === "chat" ? "prefill" : "idle";
      this.pending.set(id, { content: [], resolve, reject, onToken, onProgress, onDispatch,
        launchPaths: launch.context.paths, phase, cleanup: () => cleanup?.() });
      launch.context.phase = phase;
    });
    if (signal) {
      const cancel = () => {
        if (!this.pending.has(id) || this.active !== launch) return;
        try {
          const cancelId = `cmd_${crypto.randomUUID()}`;
          launch.tracker.register(cancelId);
          this.write(launch, { protocol: 1, type: "cancel", request_id: cancelId, target_request_id: id });
        } catch { /* exit path rejects pending work */ }
      };
      cleanup = () => signal.removeEventListener("abort", cancel);
      if (signal.aborted) queueMicrotask(cancel);
      else signal.addEventListener("abort", cancel, { once: true });
    }
    launch.tracker.register(id);
    const requestType = type === "chat" ? "generate" : type;
    this.write(launch, requestType === "generate"
      ? { protocol: 1, type: requestType, request_id: id, prompt: JSON.stringify(body), request: body,
          cache_identity: cacheIdentity, ...(structuredOutput ? { structured_output: structuredOutput } : {}) }
      : { protocol: 1, type: requestType, request_id: id, ...body });
    return promise;
  }

  private close(launch: ActiveLaunch, _reason: CloseReason): Promise<void> {
    if (launch.closePromise) return launch.closePromise;
    this.setLoadState("closing", launch);
    const closing = this.closeOnce(launch);
    launch.closePromise = closing;
    return closing;
  }

  private async closeOnce(launch: ActiveLaunch): Promise<void> {
    launch.expectedExit = true;
    if (launch.resolveHandshake) {
      launch.rejectHandshake?.(this.workerError("resident worker shut down", "resident_worker_error",
        launch.context.paths.stderrPath));
      this.clearHandshake(launch);
    }
    this.rejectLaunchPending(launch, "resident worker shut down");
    if (launch.proc.exitCode === null) {
      const id = `cmd_${crypto.randomUUID()}`;
      launch.shutdownId = id;
      launch.tracker.register(id);
      const terminal = new Promise<void>((resolve) => { launch.resolveShutdown = resolve; });
      try { this.write(launch, { protocol: 1, type: "shutdown", request_id: id }); } catch { /* exited */ }
      const timeout = Number(process.env.CLAP_WORKER_SHUTDOWN_TIMEOUT_MS ?? 500);
      await Promise.race([terminal, launch.proc.exited.then(() => {}), Bun.sleep(timeout)]);
      if (launch.proc.exitCode === null) {
        try { launch.proc.kill(); } catch { /* exited */ }
      }
    }
    await launch.exited;
  }

  private write(launch: ActiveLaunch, message: Record<string, unknown>): void {
    if (launch.proc.exitCode !== null) throw new Error("resident worker is not started");
    launch.proc.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private rejectLaunchPending(launch: ActiveLaunch, message: string, code = "resident_worker_error"): void {
    for (const [id, pending] of this.pending) {
      if (pending.launchPaths?.launchId !== launch.context.paths.launchId) continue;
      this.pending.delete(id); pending.cleanup?.();
      pending.reject(this.workerError(message, code, pending.launchPaths.stderrPath));
    }
  }

  private async ignoreMetadataFailure(operation: () => Promise<void>): Promise<void> {
    try { await operation(); } catch { /* diagnostics persistence must not replace worker outcomes */ }
  }

  private setLoadState(loadState: WorkerLoadState, launch?: ActiveLaunch): void {
    if (launch && this.active !== launch && loadState !== "not_started") return;
    if (this.loadState === loadState) return;
    this.loadState = loadState;
    const event: WorkerLoadStateEvent = Object.freeze({
      key: this.key,
      backend: this.backend,
      loadState,
      pid: launch?.proc.pid ?? this.active?.proc.pid,
      atMs: Date.now(),
    });
    this.onLoadStateEvent?.(event);
    for (const listener of this.loadStateListeners) listener(event);
  }

  private async awaitRestartBackoff(): Promise<void> {
    if (this.consecutiveCrashes === 0 || !this.lastCrashAt || this.active) return;
    if (this.consecutiveCrashes >= 5) throw this.workerError(
      `${this.backend} resident worker crashed ${this.consecutiveCrashes} times in a row; not restarting automatically. Check the worker log, then retry or restart the server.`,
      "worker_crash_loop");
    const remaining = this.lastCrashAt + Math.min(1000 * 2 ** (this.consecutiveCrashes - 1), 30000) - Date.now();
    if (remaining > 0) await Bun.sleep(remaining);
  }

  private workerError(message: string, code = "resident_worker_error", logPath = this.active?.context.paths.stderrPath): Error {
    const detail = code === "worker_protocol_error" || code === "context_length_exceeded"
      ? `${message}${logPath ? ` See ${logPath}.` : ""}` : enrichWorkerError(message, this.backend, logPath);
    if (this.backend === "mlx") return new MlxWorkerError(detail, code);
    return new LlamaWorkerError(detail, code);
  }
}

function enrichWorkerError(message: string, backend: ResidentBackend, logPath?: string): string {
  const logHint = logPath ? ` See ${logPath}.` : "";
  if (backend === "llama" && /llama_decode|outofmemory|out of memory|metal|gpu/i.test(message)) {
    return `${message}${logHint} For GGUF/llama.cpp Metal failures, try a smaller quant such as Q4_K_M, reduce CLAP_LLAMA_CONTEXT/CLAP_LLAMA_BATCH/CLAP_LLAMA_UBATCH, lower CLAP_LLAMA_GPU_LAYERS, or use CPU fallback with CLAP_LLAMA_GPU_LAYERS=0.`;
  }
  return `${message}${logHint}`;
}
