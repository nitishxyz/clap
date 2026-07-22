import type { ChatCompletionRequest, ModelTokenCapabilities } from "@clap/api";
import { getLlamaWorkerStatus, LlamaWorkerError } from "@clap/runtime-llama";
import { getMlxWorkerStatus, MlxWorkerError } from "@clap/runtime-mlx";
import { mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { V1RequestTracker, type ResidentProtocolFact } from "../protocol/request-tracker";
import { WorkerProtocolFault } from "../protocol/errors";
import { classifyWorkerExit, classifyWorkerExitPhase } from "./crash-classification";
import { applyWorkerPayload, mapWorkerResultPayload, mapWorkerTelemetryPayload,
  type PendingWorkerResult } from "./result-mapper";
import { type ActiveLimitTelemetry, type ResidentBackend, type ResidentChatResult,
  type ResidentCrashListener, type ResidentMlxMemory, type ResidentMlxRetention,
  type ResidentProgress, type ResidentWorkerHandle, type ResidentWorkerInfo } from "../resident";

export class ResidentWorkerProcess implements ResidentWorkerHandle {
  private proc?: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
  private readonly pending = new Map<string, PendingWorkerResult>();
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

  private ensureStarted(): void {
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
    const proc = Bun.spawn(status.command, {
      stdin: "pipe",
      stdout: "pipe",
      stderr: Bun.file(status.logPath),
      env: environment,
    });
    this.proc = proc;
    this.logPath = status.logPath;
    this.startProtocol();
    void this.readLoop(proc);
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

  private async readLoop(proc: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>): Promise<void> {
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
    const diagnostic = this.lastDiagnostic ? `. Last worker diagnostic: ${this.lastDiagnostic}` : "";
    const exitError = this.workerError(`${this.backend} resident worker exited ${phase} with code ${exitCode}${diagnostic}`);
    if (this.resolveHandshake) {
      this.rejectHandshake?.(exitError);
      this.rejectHandshake = undefined;
      this.resolveHandshake = undefined;
    }
    if (classifyWorkerExit(this.expectedExit, exitCode) !== "expected_exit") {
      if (classifyWorkerExit(this.expectedExit, exitCode) === "crash") {
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
    this.applyWorkerPayload(mapWorkerResultPayload(
      fact.result as Record<string, unknown>,
      fact.requestId,
      pending.content.length > 0,
    ));
  }

  private protocolUnhealthy(cause: Error): void {
    const diagnostic = this.lastDiagnostic ? `. Last worker diagnostic: ${this.lastDiagnostic}` : "";
    const detail = cause instanceof WorkerProtocolFault
      ? `worker protocol ${cause.code}: ${cause.message}${diagnostic}`
      : `worker protocol failure: ${cause.message}${diagnostic}`;
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

  private workerError(message: string, code = "resident_worker_error"): Error {
    const detail = code === "worker_protocol_error" || code === "context_length_exceeded"
      ? message : enrichWorkerError(message, this.backend, this.logPath);
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
