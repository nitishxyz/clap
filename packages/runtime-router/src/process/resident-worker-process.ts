import type { ChatCompletionRequest, ModelTokenCapabilities } from "@clap/api";
import { getLlamaWorkerStatus, LlamaWorkerError } from "@clap/runtime-llama";
import { getMlxWorkerStatus, MlxWorkerError } from "@clap/runtime-mlx";
import { V1RequestTracker, type ResidentProtocolFact } from "../protocol/request-tracker";
import { WorkerProtocolFault } from "../protocol/errors";
import { classifyWorkerCrash, classifyWorkerExitPhase } from "./crash-classification";
import { WorkerLaunchLogStore } from "./launch-log-store";
import { applyWorkerPayload, mapWorkerResultPayload, mapWorkerTelemetryPayload,
  type PendingWorkerResult } from "./result-mapper";
import { type ActiveLimitTelemetry, type ResidentBackend, type ResidentChatResult,
  type ResidentCrashListener, type ResidentMlxMemory, type ResidentMlxRetention,
  type ResidentProgress, type ResidentWorkerHandle, type ResidentWorkerInfo } from "../resident";
import type { WorkerLaunchContext, WorkerLaunchPaths, WorkerModelDescriptor } from "./types";

export class ResidentWorkerProcess implements ResidentWorkerHandle {
  private proc?: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
  private readonly pending = new Map<string, PendingWorkerResult>();
  private loaded = false;
  private launch?: WorkerLaunchContext;
  private starting?: Promise<void>;
  private crashes = 0;
  private consecutiveCrashes = 0;
  private lastCrashAt?: number;
  private expectedExit = false;
  private memory?: ResidentMlxMemory;
  private retention?: ResidentMlxRetention;
  private tokenCapabilities?: ModelTokenCapabilities;
  private workerLaunchId?: string;
  private v1Tracker?: V1RequestTracker;
  private handshake?: Promise<void>;
  private resolveHandshake?: () => void;
  private rejectHandshake?: (error: Error) => void;
  private handshakeTimer?: ReturnType<typeof setTimeout>;
  private lastDiagnostic?: string;

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
    await this.ensureStarted();
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
    for (const pending of this.pending.values()) {
      pending.reject(this.workerError("resident worker shut down", "resident_worker_error",
        pending.launchPaths?.stderrPath));
    }
    this.pending.clear();
    try {
      if (this.proc && this.proc.exitCode === null) {
        // Deliberate termination (unload/server shutdown): the SIGTERM exit
        // (143) must not count as a crash or trigger restart backoff.
        this.expectedExit = true;
        const id = `cmd_${crypto.randomUUID()}`;
        this.v1Tracker?.register(id);
        this.write({ protocol: 1, type: "shutdown", request_id: id });
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
    this.lastDiagnostic = undefined;
    this.clearHandshake();
  }

  private async ensureStarted(): Promise<void> {
    if (this.proc && this.proc.exitCode === null) return;
    if (this.starting) return this.starting;
    this.starting = this.startWorker();
    try { await this.starting; } finally { this.starting = undefined; }
  }

  private async startWorker(): Promise<void> {
    if (this.proc && this.proc.exitCode === null) return;
    this.expectedExit = false;
    this.memory = undefined;
    this.retention = undefined;
    this.tokenCapabilities = undefined;
    this.lastDiagnostic = undefined;
    const status = this.backend === "mlx" ? getMlxWorkerStatus() : getLlamaWorkerStatus();
    if (!status.command) {
      const message = status.reason ?? `${this.backend} worker is not available`;
      if (this.backend === "mlx") throw new MlxWorkerError(message, "worker_not_found");
      throw new LlamaWorkerError(message, "worker_not_found");
    }
    const launch = await this.launchLogs.prepareLaunch({
      backend: this.backend,
      modelId: this.descriptor.modelId,
      revision: this.descriptor.revision,
      modelPath: this.modelPath,
    }, status.command);
    const environment = { ...process.env, ...this.envOverrides };
    let proc: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
    try {
      proc = Bun.spawn(status.command, {
        stdin: "pipe",
        stdout: "pipe",
        stderr: Bun.file(launch.paths.stderrPath),
        env: environment,
      });
    } catch (error) {
      await this.launchLogs.finalize(launch, null, "spawn_failure");
      throw error;
    }
    this.proc = proc;
    this.launch = launch;
    this.workerLaunchId = launch.paths.launchId;
    await this.launchLogs.markSpawned(launch, proc.pid);
    this.startProtocol();
    void this.readLoop(proc, launch);
  }

  private startProtocol(): void {
    this.clearHandshake();
    this.v1Tracker = new V1RequestTracker();
    this.handshake = new Promise<void>((resolve, reject) => {
      this.resolveHandshake = resolve;
      this.rejectHandshake = reject;
    });
    this.handshakeTimer = setTimeout(() => {
      if (!this.resolveHandshake) return;
      this.protocolUnhealthy(new WorkerProtocolFault("handshake_timeout",
        "Worker did not send ready within 1000ms", "worker"));
    }, 1_000);
  }

  private clearHandshake(): void {
    if (this.handshakeTimer) clearTimeout(this.handshakeTimer);
    this.handshakeTimer = undefined;
    this.handshake = undefined;
    this.resolveHandshake = undefined;
    this.rejectHandshake = undefined;
    this.v1Tracker = undefined;
  }

  private async readLoop(proc: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>,
    launch: WorkerLaunchContext): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for await (const chunk of proc.stdout) {
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
    const phase = classifyWorkerExitPhase(Boolean(this.resolveHandshake), this.pending.size);
    const classification = classifyWorkerCrash({ protocolFault: launch.protocolFault,
      expectedExit: this.expectedExit, exitCode, phase: launch.phase });
    const diagnostic = this.lastDiagnostic ? `. Last worker diagnostic: ${this.lastDiagnostic}` : "";
    const exitMessage = `${this.backend} resident worker exited ${phase} with code ${exitCode}${diagnostic}`;
    const exitError = this.workerError(exitMessage, "resident_worker_error", launch.paths.stderrPath);
    if (this.resolveHandshake) {
      this.rejectHandshake?.(exitError);
      this.rejectHandshake = undefined;
      this.resolveHandshake = undefined;
    }
    if (classification !== "expected_exit") {
      this.crashes += 1;
      this.consecutiveCrashes += 1;
      this.lastCrashAt = Date.now();
      this.onCrash?.({ key: this.key, backend: this.backend, exitCode,
        consecutiveCrashes: this.consecutiveCrashes, launchId: launch.paths.launchId,
        logPath: launch.paths.stderrPath, metadataPath: launch.paths.metadataPath, classification });
      this.rejectAllWithLaunch(exitMessage, launch.paths);
    }
    await this.launchLogs.finalize(launch, exitCode, classification);
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
    try {
      const fact = this.v1Tracker!.consumeLine(line);
      if (fact.kind === "ready") {
        if (this.launch) {
          this.launch.phase = "idle";
          void this.launchLogs.markReady(this.launch);
        }
        this.resolveHandshake?.();
        this.resolveHandshake = undefined;
        this.rejectHandshake = undefined;
        if (this.handshakeTimer) clearTimeout(this.handshakeTimer);
        this.handshakeTimer = undefined;
      } else this.handleV1Fact(fact);
    } catch (error) {
      this.protocolUnhealthy(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private applyWorkerPayload(message: Record<string, unknown>): void {
    applyWorkerPayload(message, {
      pending: this.pending,
      retention: this.retention,
      workerLaunchId: this.workerLaunchId,
      setMemory: (memory) => { this.memory = memory; },
      setRetention: (retention) => { this.retention = retention; },
      setTokenCapabilities: (capabilities) => { this.tokenCapabilities = capabilities; },
      onRetention: (previous, current) => this.onTelemetry?.(this, previous, current),
      workerError: (detail, code) => this.workerError(detail, code),
    });
  }

  private handleV1Fact(fact: ResidentProtocolFact): void {
    if (fact.kind === "telemetry") {
      const telemetry = fact.telemetry;
      this.applyWorkerPayload(mapWorkerTelemetryPayload(telemetry));
      return;
    }
    if (fact.kind === "diagnostic") { this.lastDiagnostic = fact.message; return; }
    if (fact.kind === "ready" || fact.kind === "accepted") return;
    const pending = this.pending.get(fact.requestId);
    if (!pending) return;
    if (fact.kind === "started") {
      pending.phase = pending.phase === "load" ? "load" : "prefill";
      if (this.launch) this.launch.phase = pending.phase;
      const onDispatch = pending.onDispatch;
      pending.onDispatch = undefined;
      onDispatch?.();
      return;
    }
    if (fact.kind === "token") {
      pending.phase = "decode";
      if (this.launch) this.launch.phase = "decode";
      pending.content.push(fact.text); pending.onToken?.(fact.text); return;
    }
    if (fact.kind === "content") {
      if (typeof fact.content === "string") { pending.content.push(fact.content); pending.onToken?.(fact.content); }
      return;
    }
    if (fact.kind === "prefill_progress") {
      pending.phase = "prefill";
      if (this.launch) this.launch.phase = "prefill";
      pending.onProgress?.(fact.done, fact.total); return;
    }
    if (fact.kind === "failed") {
      this.pending.delete(fact.requestId);
      pending.reject(this.workerError(fact.error.message, fact.error.code, pending.launchPaths?.stderrPath));
      return;
    }
    this.applyWorkerPayload(mapWorkerResultPayload(
      fact.result as Record<string, unknown>,
      fact.requestId,
      pending.content.length > 0,
    ));
    if (this.pending.size === 0 && this.launch) this.launch.phase = "idle";
  }

  private protocolUnhealthy(cause: Error): void {
    if (this.launch) this.launch.protocolFault = true;
    const diagnostic = this.lastDiagnostic ? `. Last worker diagnostic: ${this.lastDiagnostic}` : "";
    const detail = cause instanceof WorkerProtocolFault
      ? `worker protocol ${cause.code}: ${cause.message}${diagnostic}`
      : `worker protocol failure: ${cause.message}${diagnostic}`;
    const error = this.workerError(detail, "worker_protocol_error", this.launch?.paths.stderrPath);
    this.rejectHandshake?.(error);
    this.rejectHandshake = undefined;
    this.resolveHandshake = undefined;
    this.rejectAllWithLaunch(detail, this.launch?.paths, "worker_protocol_error");
    this.loaded = false;
    this.expectedExit = false;
    try { this.proc?.kill(); } catch { /* process already exited */ }
    this.proc = undefined;
  }

  private async sendControl(type: string, body: Record<string, unknown>, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult> {
    await this.ensureStarted();
    await this.handshake;
    const id = `req_${crypto.randomUUID()}`;
    const promise = new Promise<ResidentChatResult>((resolve, reject) => {
      const phase = type === "load" ? "load" : type === "chat" ? "prefill" : "idle";
      this.pending.set(id, { content: [], resolve, reject, onToken, onProgress, onDispatch,
        launchPaths: this.launch?.paths, phase });
      if (this.launch) this.launch.phase = phase;
    });
    if (signal) {
      const cancel = () => {
        if (!this.pending.has(id)) return;
        try {
          const cancelId = `cmd_${crypto.randomUUID()}`;
          this.v1Tracker!.register(cancelId);
          this.write({ protocol: 1, type: "cancel", request_id: cancelId, target_request_id: id });
        } catch {
          // worker already gone; pending will be rejected by shutdown paths
        }
      };
      if (signal.aborted) queueMicrotask(cancel);
      else signal.addEventListener("abort", cancel, { once: true });
    }
    this.v1Tracker!.register(id);
    const requestType = type === "chat" ? "generate" : type;
    const envelope = requestType === "generate"
      ? { protocol: 1, type: requestType, request_id: id, prompt: JSON.stringify(body), request: body }
      : { protocol: 1, type: requestType, request_id: id, ...body };
    this.write(envelope);
    return promise;
  }

  private write(message: Record<string, unknown>): void {
    if (!this.proc) throw new Error("resident worker is not started");
    this.proc.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private rejectAllWithLaunch(message: string, fallback?: WorkerLaunchPaths, code = "resident_worker_error"): void {
    for (const pending of this.pending.values()) {
      pending.reject(this.workerError(message, code, pending.launchPaths?.stderrPath ?? fallback?.stderrPath));
    }
    this.pending.clear();
  }

  private workerError(message: string, code = "resident_worker_error", logPath = this.launch?.paths.stderrPath): Error {
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
