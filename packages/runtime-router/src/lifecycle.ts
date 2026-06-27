import type { BackendOverride, ResolvedModel } from "@clap/models";
import type { LoadedModel } from "@clap/api";

export type LoadModelOptions = {
  keepAlive?: string;
  worker?: LoadedModel["worker"];
};

export type ModelLifecycleEntry = LoadedModel;
export type LifecycleRemoveReason = "unload" | "expire" | "cleanup";

const DEFAULT_KEEP_ALIVE = process.env.CLAP_KEEP_ALIVE ?? "5m";
export class ModelLifecycleManager {
  private readonly entries = new Map<string, LoadedModel>();
  private readonly timers = new Map<string, ReturnType<typeof setTimeout>>();

  constructor(
    private readonly now = () => Date.now(),
    private readonly onRemove?: (entry: LoadedModel, reason: LifecycleRemoveReason) => void,
  ) {}

  load(resolved: ResolvedModel, options: LoadModelOptions = {}): LoadedModel {
    const localPath = resolved.modelPath ?? resolved.input;
    const key = modelLifecycleKey(resolved.id, resolved.backend, localPath);
    const existing = this.entries.get(key);
    const keepAlive = normalizeKeepAlive(options.keepAlive ?? existing?.keepAlive ?? DEFAULT_KEEP_ALIVE);
    const timestamp = iso(this.now());
    const expiresAt = expiresAtFor(this.now(), keepAlive);
    if (existing) {
      existing.keepAlive = keepAlive;
      existing.pinned = keepAlive === "always";
      existing.always = keepAlive === "always";
      if (options.worker) existing.worker = options.worker;
      existing.lastUsedAt = timestamp;
      existing.expiresAt = expiresAt ? iso(expiresAt) : null;
      existing.state = existing.activeRequests > 0 ? "active" : "warm";
      this.scheduleExpiry(existing);
      return existing;
    }

    const entry: LoadedModel = {
      key,
      id: resolved.id,
      backend: resolved.backend,
      format: resolved.format,
      localPath,
      state: "warm",
      activeRequests: 0,
      loadedAt: timestamp,
      lastUsedAt: timestamp,
      keepAlive,
      expiresAt: expiresAt ? iso(expiresAt) : null,
      pinned: keepAlive === "always",
      always: keepAlive === "always",
      worker: options.worker ?? { state: "not_started" },
    };
    this.entries.set(key, entry);
    this.scheduleExpiry(entry);
    return entry;
  }

  unload(resolved: ResolvedModel): { unloaded: boolean; model?: LoadedModel } {
    const localPath = resolved.modelPath ?? resolved.input;
    const key = modelLifecycleKey(resolved.id, resolved.backend, localPath);
    const entry = this.entries.get(key);
    if (!entry) return { unloaded: false };
    if (entry.activeRequests > 0) {
      entry.state = "unloading";
      entry.keepAlive = "0ms";
      entry.pinned = false;
      entry.always = false;
      entry.expiresAt = iso(this.now());
      return { unloaded: false, model: entry };
    }
    this.delete(key, "unload");
    return { unloaded: true, model: entry };
  }

  list(): LoadedModel[] {
    this.expireIdle();
    return [...this.entries.values()].sort((a, b) => a.loadedAt.localeCompare(b.loadedAt));
  }

  async withUsage<T>(resolved: ResolvedModel, fn: (entry: LoadedModel) => Promise<T>, options: LoadModelOptions = {}): Promise<T> {
    const entry = this.beginUsage(resolved, options);
    try {
      return await fn(entry);
    } finally {
      this.finishUsage(entry);
    }
  }

  beginUsage(resolved: ResolvedModel, options: LoadModelOptions = {}): LoadedModel {
    const entry = this.load(resolved, options);
    entry.activeRequests += 1;
    entry.state = "active";
    this.touch(entry);
    return entry;
  }

  finishUsage(entry: LoadedModel): void {
    entry.activeRequests = Math.max(0, entry.activeRequests - 1);
    this.touch(entry);
    if (entry.activeRequests === 0) {
      if (entry.state === "unloading") {
        this.delete(entry.key, "unload");
      } else {
        entry.state = "warm";
        this.scheduleExpiry(entry);
      }
    }
  }

  touch(entry: LoadedModel): void {
    const timestamp = this.now();
    entry.lastUsedAt = iso(timestamp);
    const expiresAt = expiresAtFor(timestamp, entry.keepAlive);
    entry.expiresAt = expiresAt ? iso(expiresAt) : null;
  }

  cleanup(): void {
    for (const timer of this.timers.values()) clearTimeout(timer);
    this.timers.clear();
    for (const entry of this.entries.values()) this.onRemove?.(entry, "cleanup");
    this.entries.clear();
  }

  private expireIdle(): void {
    const timestamp = this.now();
    for (const entry of this.entries.values()) {
      if (entry.activeRequests > 0 || entry.always || !entry.expiresAt) continue;
      if (Date.parse(entry.expiresAt) <= timestamp) this.delete(entry.key, "expire");
    }
  }

  private scheduleExpiry(entry: LoadedModel): void {
    const existing = this.timers.get(entry.key);
    if (existing) clearTimeout(existing);
    this.timers.delete(entry.key);
    if (entry.always || !entry.expiresAt) return;
    const delay = Math.max(0, Date.parse(entry.expiresAt) - this.now());
    this.timers.set(entry.key, setTimeout(() => {
      if (entry.activeRequests === 0 && entry.expiresAt && Date.parse(entry.expiresAt) <= this.now()) {
        this.delete(entry.key, "unload");
      }
    }, delay));
  }

  private delete(key: string, reason: LifecycleRemoveReason): void {
    const timer = this.timers.get(key);
    if (timer) clearTimeout(timer);
    this.timers.delete(key);
    const entry = this.entries.get(key);
    if (entry) this.onRemove?.(entry, reason);
    this.entries.delete(key);
  }
}

export function modelLifecycleKey(id: string, backend: LoadedModel["backend"], localPath: string): string {
  return JSON.stringify([id, backend, localPath]);
}

export function normalizeKeepAlive(value: string): string {
  if (value === "always") return value;
  parseKeepAliveMs(value);
  return value;
}

export function parseKeepAliveMs(value: string): number | null {
  if (value === "always") return null;
  const match = value.match(/^(\d+)(ms|s|m|h|d)$/);
  if (!match) throw new Error(`invalid keep-alive duration: ${value}. Use 30s, 15m, 1h, or always.`);
  const amount = Number(match[1]);
  const unit = match[2];
  const multipliers: Record<string, number> = { ms: 1, s: 1_000, m: 60_000, h: 3_600_000, d: 86_400_000 };
  return amount * multipliers[unit]!;
}

function expiresAtFor(timestamp: number, keepAlive: string): number | null {
  const duration = parseKeepAliveMs(keepAlive);
  return duration === null ? null : timestamp + duration;
}

function iso(timestamp: number): string {
  return new Date(timestamp).toISOString();
}

export function backendOverrideFromResolved(resolved: ResolvedModel): BackendOverride {
  return resolved.format;
}
