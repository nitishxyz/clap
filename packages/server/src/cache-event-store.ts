import { createHmac, randomBytes } from "node:crypto";
import {
  appendFileSync,
  chmodSync,
  closeSync,
  existsSync,
  fdatasyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  truncateSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import type { CacheOutcome } from "./cache-outcome";
import type { StructuredOutputFacts } from "./metrics";

export const CACHE_EVENT_SCHEMA_VERSION = 2;
export const DEFAULT_CACHE_EVENT_MAX_BYTES = 32 * 1024 * 1024;
export const DEFAULT_CACHE_EVENT_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;
const DEFAULT_SEGMENT_BYTES = 2 * 1024 * 1024;
const FILE_PREFIX = "cache-decisions";
const CURRENT_FILE = `${FILE_PREFIX}.current.jsonl`;
const KEY_FILE = "telemetry.hmac-key";

export type CacheCandidateDiagnostic = {
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
};

export type PersistedCacheDecision = {
  schemaVersion: number;
  source: "persisted";
  requestId: string;
  timestamp: number;
  serverLaunchId: string;
  workerLaunchId?: string;
  model: string;
  backend?: string;
  endpoint?: string;
  namespaceFingerprint?: string;
  sessionIdentitySource?: string;
  sessionFingerprint?: string;
  // Privacy-safe prompt-prefix grouping id (not a session). Present when the
  // request had no explicit cache.session; never raw prompt content.
  promptPrefixId?: string;
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
    status: "resolved" | "authorized" | "skipped";
    skipReason?: "unsupported_template_boundary" | "non_prefix_template_boundary";
    materialized?: boolean;
  }>;
  promptTokenHash?: string;
  promptTokenCount?: number;
  status: "ok" | "error" | "cancelled";
  finishReason?: string;
  errorCode?: string;
  structuredOutput?: StructuredOutputFacts;
  // Classification derived at finish time and persisted with the event. Older
  // records lack it; readers re-derive it reproducibly via classifyCacheOutcome.
  cacheOutcome?: CacheOutcome;
  side?: boolean;
  priority?: "interactive" | "background";
  cache?: {
    hit?: boolean;
    missReason?: string;
    plannedTokens?: number;
    realizedTokens?: number;
    reusedTokens?: number;
    kind?: string;
    scope?: string;
    donorSlot?: number;
    donorGeneration?: number;
    targetSlot?: number;
    targetGeneration?: number;
    evictions?: Array<{ slot: number; reason?: string }>;
    fallback?: string;
    decisionUs?: number;
    candidates?: CacheCandidateDiagnostic[];
  };
  ttftMs?: number;
  prefillMs?: number;
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
  durationMs?: number;
};

export type CacheDecisionFilters = {
  model?: string;
  backend?: string;
  status?: PersistedCacheDecision["status"];
  hit?: boolean;
  requestId?: string;
  since?: number;
  until?: number;
};

export type CacheDecisionPage = {
  source: "persisted";
  items: PersistedCacheDecision[];
  nextCursor?: string;
};

export type CacheEventStoreOptions = {
  directory: string;
  enabled?: boolean;
  maxBytes?: number;
  maxAgeMs?: number;
  segmentBytes?: number;
  now?: () => number;
};

function normalizeStableBoundary(raw: Record<string, unknown>): void {
  const valid = typeof raw.stableBoundaryTokenHash === "string" && raw.stableBoundaryTokenHash.length > 0
    && typeof raw.stableBoundaryTokenCount === "number" && Number.isInteger(raw.stableBoundaryTokenCount)
    && raw.stableBoundaryTokenCount > 0
    && typeof raw.stableBoundaryKind === "string" && raw.stableBoundaryKind.length > 0;
  if (valid) return;
  delete raw.stableBoundaryTokenHash;
  delete raw.stableBoundaryTokenCount;
  delete raw.stableBoundaryKind;
}

function normalizeEvent(value: unknown): PersistedCacheDecision | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  if (typeof raw.requestId !== "string" || typeof raw.timestamp !== "number" || typeof raw.model !== "string") return undefined;
  if (raw.schemaVersion === 1) {
    raw.schemaVersion = CACHE_EVENT_SCHEMA_VERSION;
    raw.source = "persisted";
  }
  if (raw.schemaVersion !== CACHE_EVENT_SCHEMA_VERSION || raw.source !== "persisted") return undefined;
  normalizeStableBoundary(raw);
  return raw as PersistedCacheDecision;
}

function eventFiles(directory: string): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory)
    .filter((name) => name === CURRENT_FILE || (name.startsWith(`${FILE_PREFIX}.`) && name.endsWith(".jsonl")))
    .sort((a, b) => a === CURRENT_FILE ? 1 : b === CURRENT_FILE ? -1 : a.localeCompare(b));
}

export class CacheEventStore {
  readonly enabled: boolean;
  readonly directory: string;
  private readonly maxBytes: number;
  private readonly maxAgeMs: number;
  private readonly segmentBytes: number;
  private readonly now: () => number;
  private readonly key?: Buffer;
  private rotationSequence = 0;

  constructor(options: CacheEventStoreOptions) {
    this.enabled = options.enabled !== false;
    this.directory = options.directory;
    this.maxBytes = Math.max(64 * 1024, options.maxBytes ?? DEFAULT_CACHE_EVENT_MAX_BYTES);
    this.maxAgeMs = Math.max(60_000, options.maxAgeMs ?? DEFAULT_CACHE_EVENT_MAX_AGE_MS);
    this.segmentBytes = Math.max(16 * 1024, Math.min(options.segmentBytes ?? DEFAULT_SEGMENT_BYTES, this.maxBytes));
    this.now = options.now ?? Date.now;
    if (!this.enabled) return;
    mkdirSync(this.directory, { recursive: true, mode: 0o700 });
    chmodSync(this.directory, 0o700);
    this.key = this.loadOrCreateKey();
    this.recoverCurrentTail();
    this.enforceRetention();
  }

  fingerprint(value: unknown): string | undefined {
    if (!this.key || value === undefined || value === null || value === "") return undefined;
    const encoded = typeof value === "string" ? value : JSON.stringify(value);
    return createHmac("sha256", this.key).update(encoded).digest("hex");
  }

  tokenFingerprintKey(): string | undefined {
    return this.key?.toString("hex");
  }

  append(event: PersistedCacheDecision): void {
    if (!this.enabled) return;
    const normalized = { ...event, schemaVersion: CACHE_EVENT_SCHEMA_VERSION, source: "persisted" };
    normalizeStableBoundary(normalized);
    const line = `${JSON.stringify(normalized)}\n`;
    const current = join(this.directory, CURRENT_FILE);
    if (existsSync(current) && statSync(current).size + Buffer.byteLength(line) > this.segmentBytes) this.rotateCurrent();
    const fd = openSync(current, "a", 0o600);
    try {
      appendFileSync(fd, line, "utf8");
      fdatasyncSync(fd);
    } finally {
      closeSync(fd);
    }
    chmodSync(current, 0o600);
    this.enforceRetention();
  }

  get(requestId: string): PersistedCacheDecision | undefined {
    return this.readAll().reverse().find((event) => event.requestId === requestId);
  }

  list(filters: CacheDecisionFilters = {}, limit = 50, cursor?: string): CacheDecisionPage {
    const capped = Math.max(1, Math.min(200, limit));
    const offset = cursor ? Math.max(0, Number.parseInt(Buffer.from(cursor, "base64url").toString("utf8"), 10) || 0) : 0;
    const filtered = this.readAll().reverse().filter((event) => {
      if (filters.requestId && event.requestId !== filters.requestId) return false;
      if (filters.model && event.model !== filters.model) return false;
      if (filters.backend && event.backend !== filters.backend) return false;
      if (filters.status && event.status !== filters.status) return false;
      if (filters.hit !== undefined && event.cache?.hit !== filters.hit) return false;
      if (filters.since !== undefined && event.timestamp < filters.since) return false;
      if (filters.until !== undefined && event.timestamp > filters.until) return false;
      return true;
    });
    const items = filtered.slice(offset, offset + capped);
    const next = offset + items.length;
    return {
      source: "persisted",
      items,
      nextCursor: next < filtered.length ? Buffer.from(String(next)).toString("base64url") : undefined,
    };
  }

  private loadOrCreateKey(): Buffer {
    const path = join(this.directory, KEY_FILE);
    if (existsSync(path)) {
      chmodSync(path, 0o600);
      const key = readFileSync(path);
      if (key.length >= 32) return key;
    }
    const key = randomBytes(32);
    try {
      writeFileSync(path, key, { mode: 0o600, flag: "wx" });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const installed = readFileSync(path);
      if (installed.length < 32) throw new Error("installation telemetry HMAC key is invalid");
      chmodSync(path, 0o600);
      return installed;
    }
    chmodSync(path, 0o600);
    return key;
  }

  private recoverCurrentTail(): void {
    const path = join(this.directory, CURRENT_FILE);
    if (!existsSync(path)) return;
    const content = readFileSync(path, "utf8");
    let validBytes = 0;
    for (const part of content.matchAll(/.*(?:\n|$)/g)) {
      const line = part[0];
      if (!line) continue;
      if (!line.endsWith("\n")) break;
      try {
        if (!normalizeEvent(JSON.parse(line))) break;
        validBytes += Buffer.byteLength(line);
      } catch {
        break;
      }
    }
    if (validBytes !== Buffer.byteLength(content)) truncateSync(path, validBytes);
  }

  private rotateCurrent(): void {
    const current = join(this.directory, CURRENT_FILE);
    if (!existsSync(current) || statSync(current).size === 0) return;
    this.rotationSequence += 1;
    const target = join(this.directory, `${FILE_PREFIX}.${this.now()}.${process.pid}.${this.rotationSequence}.jsonl`);
    renameSync(current, target);
  }

  private enforceRetention(): void {
    const cutoff = this.now() - this.maxAgeMs;
    const agedFiles = eventFiles(this.directory).map((name) => ({ name, path: join(this.directory, name), stat: statSync(join(this.directory, name)) }));
    for (const file of agedFiles) {
      if (file.name !== CURRENT_FILE && file.stat.mtimeMs < cutoff) unlinkSync(file.path);
    }
    const files = eventFiles(this.directory).map((name) => ({ name, path: join(this.directory, name), size: statSync(join(this.directory, name)).size }));
    let total = files.reduce((sum, file) => sum + file.size, 0);
    for (const file of files) {
      if (total <= this.maxBytes || file.name === CURRENT_FILE) continue;
      unlinkSync(file.path);
      total -= file.size;
    }
    if (total > this.maxBytes && existsSync(join(this.directory, CURRENT_FILE))) {
      this.trimCurrentToBytes(Math.max(0, this.maxBytes));
    }
  }

  private trimCurrentToBytes(maxBytes: number): void {
    const path = join(this.directory, CURRENT_FILE);
    const lines = readFileSync(path, "utf8").split("\n").filter(Boolean);
    const kept: string[] = [];
    let size = 0;
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const line = `${lines[index]}\n`;
      const bytes = Buffer.byteLength(line);
      if (size + bytes > maxBytes) break;
      kept.unshift(line);
      size += bytes;
    }
    const temporary = `${path}.${process.pid}.trim.tmp`;
    writeFileSync(temporary, kept.join(""), { mode: 0o600 });
    renameSync(temporary, path);
  }

  private readAll(): PersistedCacheDecision[] {
    if (!this.enabled) return [];
    const events: PersistedCacheDecision[] = [];
    for (const name of eventFiles(this.directory)) {
      const content = readFileSync(join(this.directory, name), "utf8");
      for (const line of content.split("\n")) {
        if (!line) continue;
        try {
          const event = normalizeEvent(JSON.parse(line));
          if (event && event.timestamp >= this.now() - this.maxAgeMs) events.push(event);
        } catch {
          // Rotated segments are immutable; skip malformed records defensively.
        }
      }
    }
    return events.sort((a, b) => a.timestamp - b.timestamp);
  }
}
