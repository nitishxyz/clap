import type { ChatCompletionRequest, LoadedModel } from "@clap/api";
import { getLlamaWorkerStatus, LlamaWorkerError } from "@clap/runtime-llama";
import { getMlxWorkerStatus, MlxWorkerError } from "@clap/runtime-mlx";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

export type ResidentBackend = LoadedModel["backend"];

export type ResidentWorkerInfo = {
  pid?: number;
  state: LoadedModel["worker"]["state"];
  limitation?: string;
};

export type ResidentUsage = {
  promptTokens?: number;
  completionTokens?: number;
};

export type ResidentCacheInfo = {
  hit?: boolean;
  reusedTokens?: number;
  sideRequest?: boolean;
  slot?: number;
};

export type ResidentChatResult = {
  content: string;
  usage?: ResidentUsage;
  finishReason?: "stop" | "length" | "cancel";
  cache?: ResidentCacheInfo;
};

export type ResidentProgress = (done: number, total: number) => void;

export type ResidentWorkerHandle = {
  key: string;
  backend: ResidentBackend;
  modelPath: string;
  info(): ResidentWorkerInfo;
  load(): Promise<ResidentWorkerInfo>;
  chat(request: ChatCompletionRequest, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult>;
  unload(): Promise<void>;
  shutdown(): void;
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
};

export class ResidentWorkerProcess implements ResidentWorkerHandle {
  private proc?: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
  private readonly pending = new Map<string, Pending>();
  private loaded = false;
  private logPath?: string;

  constructor(
    public readonly key: string,
    public readonly backend: ResidentBackend,
    public readonly modelPath: string,
  ) {}

  info(): ResidentWorkerInfo {
    return {
      pid: this.proc?.pid,
      state: this.proc ? "resident" : "not_started",
    };
  }

  async load(): Promise<ResidentWorkerInfo> {
    this.ensureStarted();
    if (!this.loaded) {
      await this.sendControl("load", { model: this.modelPath });
      this.loaded = true;
    }
    return this.info();
  }

  async chat(request: ChatCompletionRequest, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult> {
    await this.load();
    return this.sendControl("chat", request, onToken, signal, onProgress, onDispatch);
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
        this.write({ type: "shutdown" });
        this.proc.kill();
      }
    } catch {
      // ignore shutdown races
    }
    this.proc = undefined;
    this.loaded = false;
  }

  private ensureStarted(): void {
    if (this.proc && this.proc.exitCode === null) return;
    const status = this.backend === "mlx" ? getMlxWorkerStatus() : getLlamaWorkerStatus();
    if (!status.command) {
      const message = status.reason ?? `${this.backend} worker is not available`;
      if (this.backend === "mlx") throw new MlxWorkerError(message, "worker_not_found");
      throw new LlamaWorkerError(message, "worker_not_found");
    }
    mkdirSync(dirname(status.logPath), { recursive: true });
    const proc = Bun.spawn(status.command, {
      stdin: "pipe",
      stdout: "pipe",
      stderr: Bun.file(status.logPath),
      env: process.env,
    });
    this.proc = proc;
    this.logPath = status.logPath;
    void this.readLoop(proc);
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
          if (line) this.handleLine(line);
        }
      }
      const tail = buffer.trim();
      if (tail) this.handleLine(tail);
    } catch (error) {
      this.rejectAll(error instanceof Error ? error : new Error(String(error)));
    }
    const exitCode = await proc.exited;
    if (this.proc === proc) this.proc = undefined;
    if (exitCode !== 0) this.rejectAll(this.workerError(`${this.backend} resident worker exited with code ${exitCode}`));
  }

  private handleLine(line: string): void {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      const pending = this.firstPending();
      if (pending) {
        pending.content.push(line);
        pending.onToken?.(line);
      }
      return;
    }
    const id = typeof message.id === "string" ? message.id : undefined;
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
    if (message.cache && typeof message.cache === "object") {
      const cache = message.cache as Record<string, unknown>;
      pending.cache = {
        hit: typeof cache.hit === "boolean" ? cache.hit : undefined,
        reusedTokens: typeof cache.reused_tokens === "number" ? cache.reused_tokens : undefined,
        sideRequest: typeof cache.side_request === "boolean" ? cache.side_request : undefined,
        slot: typeof cache.slot === "number" ? cache.slot : undefined,
      };
    }
    if (message.finish_reason === "stop" || message.finish_reason === "length" || message.finish_reason === "cancel") {
      pending.finishReason = message.finish_reason;
    }
    if (message.loaded === true || message.unloaded === true || message.done === true) {
      if (id) this.pending.delete(id);
      else this.deleteFirstPending();
      pending.resolve({ content: pending.content.join(""), usage: pending.usage, finishReason: pending.finishReason, cache: pending.cache });
    }
  }

  private sendControl(type: string, body: Record<string, unknown>, onToken?: (token: string) => void, signal?: AbortSignal, onProgress?: ResidentProgress, onDispatch?: () => void): Promise<ResidentChatResult> {
    this.ensureStarted();
    const id = `req_${crypto.randomUUID()}`;
    const promise = new Promise<ResidentChatResult>((resolve, reject) => {
      this.pending.set(id, { content: [], resolve, reject, onToken, onProgress, onDispatch });
    });
    if (signal) {
      const cancel = () => {
        if (!this.pending.has(id)) return;
        try {
          this.write({ id, type: "cancel" });
        } catch {
          // worker already gone; pending will be rejected by shutdown paths
        }
      };
      if (signal.aborted) queueMicrotask(cancel);
      else signal.addEventListener("abort", cancel, { once: true });
    }
    this.write({ id, type, ...body });
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

function enrichWorkerError(message: string, backend: ResidentBackend, logPath?: string): string {
  const logHint = logPath ? ` See ${logPath}.` : "";
  if (backend === "llama" && /llama_decode|outofmemory|out of memory|metal|gpu/i.test(message)) {
    return `${message}${logHint} For GGUF/llama.cpp Metal failures, try a smaller quant such as Q4_K_M, reduce CLAP_LLAMA_CONTEXT/CLAP_LLAMA_BATCH/CLAP_LLAMA_UBATCH, lower CLAP_LLAMA_GPU_LAYERS, or use CPU fallback with CLAP_LLAMA_GPU_LAYERS=0.`;
  }
  return `${message}${logHint}`;
}

export class ResidentWorkerRegistry {
  private readonly workers = new Map<string, ResidentWorkerHandle>();

  getOrCreate(key: string, backend: ResidentBackend, modelPath: string): ResidentWorkerHandle {
    const existing = this.workers.get(key);
    if (existing) return existing;
    const worker = new ResidentWorkerProcess(key, backend, modelPath);
    this.workers.set(key, worker);
    return worker;
  }

  get(key: string): ResidentWorkerHandle | undefined {
    return this.workers.get(key);
  }

  async unload(key: string): Promise<void> {
    const worker = this.workers.get(key);
    this.workers.delete(key);
    await worker?.unload();
  }

  shutdown(key: string): void {
    const worker = this.workers.get(key);
    this.workers.delete(key);
    worker?.shutdown();
  }

  shutdownAll(): void {
    for (const key of [...this.workers.keys()]) this.shutdown(key);
  }
}
