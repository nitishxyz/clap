import type { ChatCompletionRequest, LoadedModel } from "@clap/api";
import { getLlamaWorkerStatus, LlamaWorkerError } from "@clap/runtime-llama";
import { getMlxWorkerStatus, MlxWorkerError } from "@clap/runtime-mlx";

export type ResidentBackend = LoadedModel["backend"];

export type ResidentWorkerInfo = {
  pid?: number;
  state: LoadedModel["worker"]["state"];
  limitation?: string;
};

export type ResidentWorkerHandle = {
  key: string;
  backend: ResidentBackend;
  modelPath: string;
  info(): ResidentWorkerInfo;
  load(): Promise<ResidentWorkerInfo>;
  chat(request: ChatCompletionRequest): Promise<string>;
  unload(): Promise<void>;
  shutdown(): void;
};

type Pending = {
  content: string[];
  resolve: (content: string) => void;
  reject: (error: Error) => void;
};

export class ResidentWorkerProcess implements ResidentWorkerHandle {
  private proc?: Bun.Subprocess<"pipe", "pipe", ReturnType<typeof Bun.file>>;
  private readonly pending = new Map<string, Pending>();
  private loaded = false;

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

  async chat(request: ChatCompletionRequest): Promise<string> {
    await this.load();
    return this.sendControl("chat", request);
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
    const proc = Bun.spawn(status.command, {
      stdin: "pipe",
      stdout: "pipe",
      stderr: Bun.file(status.logPath),
      env: process.env,
    });
    this.proc = proc;
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
    if (exitCode !== 0) this.rejectAll(new Error(`${this.backend} resident worker exited with code ${exitCode}`));
  }

  private handleLine(line: string): void {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      const pending = this.firstPending();
      pending?.content.push(line);
      return;
    }
    const id = typeof message.id === "string" ? message.id : undefined;
    const pending = id ? this.pending.get(id) : this.firstPending();
    if (!pending) return;
    if (message.error) {
      this.pending.delete(id ?? "");
      pending.reject(new Error(String(message.error)));
      return;
    }
    if (typeof message.token === "string") pending.content.push(message.token);
    if (typeof message.content === "string") pending.content.push(message.content);
    if (message.loaded === true || message.unloaded === true || message.done === true) {
      if (id) this.pending.delete(id);
      else this.deleteFirstPending();
      pending.resolve(pending.content.join(""));
    }
  }

  private sendControl(type: string, body: Record<string, unknown>): Promise<string> {
    this.ensureStarted();
    const id = `req_${crypto.randomUUID()}`;
    const promise = new Promise<string>((resolve, reject) => {
      this.pending.set(id, { content: [], resolve, reject });
    });
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
