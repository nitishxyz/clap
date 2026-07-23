import type { BackendOverride, ResolvedModel } from "@clap/models";
import type { LoadedModel } from "@clap/api";
import { unavailableMemory, type MemoryValue } from "./residency/types";

export type LoadModelOptions = {
  keepAlive?: string;
  worker?: LoadedModel["worker"];
};

export type ModelLifecycleEntry = LoadedModel;
export type LifecycleRemoveReason = "unload" | "expire" | "cleanup";
export type LifecycleResidencyState = "starting" | "loading" | "idle" | "active" | "closing";
export type LifecycleRemovalHook = (entry: LoadedModel, reason: LifecycleRemoveReason) => void | Promise<void>;

export interface LifecycleResidencySnapshot {
  readonly key: string;
  readonly state: LifecycleResidencyState;
  readonly activeRequests: number;
  readonly pinned: boolean;
  readonly always: boolean;
  readonly loadedAtMs: number;
  readonly lastUsedAtMs: number;
  readonly lifecycleVersion: number;
  readonly retainedValueScore: number;
  readonly memory: MemoryValue;
}

export interface LifecycleResidencySnapshotOptions {
  readonly memoryByKey?: ReadonlyMap<string, MemoryValue>;
  readonly retainedValueByKey?: ReadonlyMap<string, number>;
}

export type IdleEvictionResult = "evicted" | "changed";

const DEFAULT_KEEP_ALIVE = process.env.CLAP_KEEP_ALIVE ?? "15m";
export class ModelLifecycleManager {
  private readonly entries = new Map<string, LoadedModel>();
  private readonly timers = new Map<string, ReturnType<typeof setTimeout>>();
  private readonly residencyStates = new Map<string, {
    state: Exclude<LifecycleResidencyState, "idle" | "active">;
    at: number;
  }>();
  private readonly residencyVersions = new Map<string, number>();
  // Optional secondary listener (e.g. metrics/event log) attached after construction.
  removeListener?: LifecycleRemovalHook;

  constructor(
    private readonly now = () => Date.now(),
    private readonly onRemove?: LifecycleRemovalHook,
  ) {}

  load(resolved: ResolvedModel, options: LoadModelOptions = {}): LoadedModel {
    const localPath = resolved.modelPath ?? resolved.input;
    const key = modelLifecycleKey(resolved.id, resolved.backend, localPath);
    const existing = this.entries.get(key);
    this.residencyStates.delete(key);
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
      this.bumpResidencyVersion(key);
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
    this.bumpResidencyVersion(key);
    this.scheduleExpiry(entry);
    return entry;
  }

  unload(resolved: ResolvedModel): { unloaded: boolean; model?: LoadedModel } {
    const localPath = resolved.modelPath ?? resolved.input;
    const key = modelLifecycleKey(resolved.id, resolved.backend, localPath);
    const entry = this.entries.get(key);
    if (!entry) return { unloaded: false };
    if (entry.activeRequests > 0) {
      this.residencyStates.set(key, { state: "closing", at: this.now() });
      this.bumpResidencyVersion(key);
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

  async unloadAsync(resolved: ResolvedModel): Promise<{ unloaded: boolean; model?: LoadedModel }> {
    const localPath = resolved.modelPath ?? resolved.input;
    const key = modelLifecycleKey(resolved.id, resolved.backend, localPath);
    const entry = this.entries.get(key);
    if (!entry || entry.activeRequests > 0) return this.unload(resolved);
    await this.deleteAsync(key, "unload");
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
        this.residencyStates.delete(entry.key);
        this.delete(entry.key, "unload");
      } else {
        entry.state = "warm";
        this.scheduleExpiry(entry);
      }
    }
  }

  setResidencyTransition(key: string, state: "starting" | "loading" | "closing"): void {
    this.residencyStates.set(key, { state, at: this.now() });
    this.bumpResidencyVersion(key);
  }

  clearResidencyTransition(key: string): void {
    this.residencyStates.delete(key);
    this.bumpResidencyVersion(key);
  }

  snapshotForResidency(options: LifecycleResidencySnapshotOptions = {}): LifecycleResidencySnapshot[] {
    const entries: LifecycleResidencySnapshot[] = [...this.entries.values()].map((entry) => ({
      key: entry.key,
      state: this.residencyStates.get(entry.key)?.state ?? (entry.activeRequests > 0 ? "active" : "idle"),
      activeRequests: entry.activeRequests,
      pinned: entry.pinned,
      always: entry.always,
      loadedAtMs: Date.parse(entry.loadedAt),
      lastUsedAtMs: Date.parse(entry.lastUsedAt),
      lifecycleVersion: this.residencyVersions.get(entry.key) ?? 0,
      retainedValueScore: finiteNonnegative(options.retainedValueByKey?.get(entry.key)),
      memory: options.memoryByKey?.get(entry.key) ?? unavailableMemory("not_observed"),
    }));
    for (const [key, transition] of this.residencyStates) {
      if (this.entries.has(key)) continue;
      entries.push({
        key,
        state: transition.state,
        activeRequests: 0,
        pinned: false,
        always: false,
        loadedAtMs: transition.at,
        lastUsedAtMs: transition.at,
        lifecycleVersion: this.residencyVersions.get(key) ?? 0,
        retainedValueScore: finiteNonnegative(options.retainedValueByKey?.get(key)),
        memory: options.memoryByKey?.get(key) ?? unavailableMemory("not_observed"),
      });
    }
    return entries;
  }

  async tryEvictIdle(expected: LifecycleResidencySnapshot): Promise<IdleEvictionResult> {
    const entry = this.entries.get(expected.key);
    if (!entry
      || expected.state !== "idle"
      || expected.activeRequests !== 0
      || expected.pinned
      || expected.always
      || this.residencyStates.has(expected.key)
      || entry.activeRequests !== 0
      || entry.pinned
      || entry.always
      || entry.state !== "warm"
      || Date.parse(entry.lastUsedAt) !== expected.lastUsedAtMs
      || this.residencyVersions.get(expected.key) !== expected.lifecycleVersion) {
      return "changed";
    }
    this.residencyStates.set(expected.key, { state: "closing", at: this.now() });
    this.bumpResidencyVersion(expected.key);
    await this.deleteAsync(expected.key, "unload");
    return "evicted";
  }

  touch(entry: LoadedModel): void {
    const timestamp = this.now();
    entry.lastUsedAt = iso(timestamp);
    const expiresAt = expiresAtFor(timestamp, entry.keepAlive);
    entry.expiresAt = expiresAt ? iso(expiresAt) : null;
    this.bumpResidencyVersion(entry.key);
  }

  cleanup(): void {
    for (const timer of this.timers.values()) clearTimeout(timer);
    this.timers.clear();
    for (const entry of this.entries.values()) {
      void this.invokeRemovalHooks(entry, "cleanup");
    }
    this.entries.clear();
    this.residencyStates.clear();
    this.residencyVersions.clear();
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
    void this.deleteAsync(key, reason);
  }

  private async deleteAsync(key: string, reason: LifecycleRemoveReason): Promise<void> {
    const timer = this.timers.get(key);
    if (timer) clearTimeout(timer);
    this.timers.delete(key);
    const entry = this.entries.get(key);
    this.entries.delete(key);
    this.residencyStates.delete(key);
    this.residencyVersions.delete(key);
    if (entry) await this.invokeRemovalHooks(entry, reason);
  }

  private async invokeRemovalHooks(entry: LoadedModel, reason: LifecycleRemoveReason): Promise<void> {
    await this.onRemove?.(entry, reason);
    await this.removeListener?.(entry, reason);
  }

  private bumpResidencyVersion(key: string): void {
    this.residencyVersions.set(key, (this.residencyVersions.get(key) ?? 0) + 1);
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

function finiteNonnegative(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) && value >= 0 ? value : 0;
}

function iso(timestamp: number): string {
  return new Date(timestamp).toISOString();
}

export function backendOverrideFromResolved(resolved: ResolvedModel): BackendOverride {
  return resolved.format;
}
