import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CacheEventStore, type PersistedCacheDecision } from "./cache-event-store";
import { MetricsCollector } from "./metrics";

const roots: string[] = [];

function directory(): string {
  const path = mkdtempSync(join(tmpdir(), "clap-cache-events-"));
  roots.push(path);
  return path;
}

function event(id: string, overrides: Partial<PersistedCacheDecision> = {}): PersistedCacheDecision {
  return {
    schemaVersion: 2,
    source: "persisted",
    requestId: id,
    timestamp: Date.now(),
    serverLaunchId: "server-launch",
    model: "model-a",
    backend: "mlx",
    status: "ok",
    cache: { hit: true, reusedTokens: 12 },
    ...overrides,
  };
}

afterEach(() => {
  for (const path of roots.splice(0)) rmSync(path, { recursive: true, force: true });
});

describe("CacheEventStore", () => {
  test("survives restart and filters with pagination", () => {
    const root = directory();
    const first = new CacheEventStore({ directory: root });
    first.append(event("one", { model: "a", cache: { hit: true } }));
    first.append(event("two", { model: "b", status: "cancelled", cache: { hit: false, missReason: "absent_anchor" } }));
    first.append(event("three", { model: "a", cache: { hit: false } }));

    const restarted = new CacheEventStore({ directory: root });
    expect(restarted.get("two")?.status).toBe("cancelled");
    expect(restarted.list({ model: "a" }, 1).items.map((item) => item.requestId)).toEqual(["three"]);
    const page = restarted.list({ model: "a" }, 1);
    expect(restarted.list({ model: "a" }, 1, page.nextCursor).items.map((item) => item.requestId)).toEqual(["one"]);
    expect(restarted.list({ hit: false }, 10).items.map((item) => item.requestId)).toEqual(["three", "two"]);
  });

  test("recovers a truncated current tail", () => {
    const root = directory();
    const store = new CacheEventStore({ directory: root });
    store.append(event("complete"));
    const path = join(root, "cache-decisions.current.jsonl");
    writeFileSync(path, `${readFileSync(path, "utf8")}{\"schemaVersion\":2,\"requestId\":`);

    const recovered = new CacheEventStore({ directory: root });
    expect(recovered.list().items.map((item) => item.requestId)).toEqual(["complete"]);
    expect(readFileSync(path, "utf8").endsWith("\n")).toBe(true);
  });

  test("rotates and remains bounded", () => {
    const root = directory();
    const store = new CacheEventStore({ directory: root, segmentBytes: 16 * 1024, maxBytes: 64 * 1024 });
    for (let index = 0; index < 200; index += 1) {
      store.append(event(`request-${index}`, { model: `model-${index}-${"x".repeat(400)}` }));
    }
    const files = readdirSync(root).filter((name) => name.endsWith(".jsonl"));
    const total = files.reduce((sum, name) => sum + Bun.file(join(root, name)).size, 0);
    expect(files.length).toBeGreaterThan(1);
    expect(total).toBeLessThanOrEqual(64 * 1024);
    expect(store.get("request-199")?.requestId).toBe("request-199");
  });

  test("migrates schema v1 records while rejecting unknown versions", () => {
    const root = directory();
    writeFileSync(join(root, "cache-decisions.current.jsonl"), `${JSON.stringify({ ...event("old"), schemaVersion: 1, source: undefined })}\n${JSON.stringify({ ...event("future"), schemaVersion: 99 })}\n`);
    const store = new CacheEventStore({ directory: root });
    expect(store.get("old")?.schemaVersion).toBe(2);
    expect(store.get("future")).toBeUndefined();
  });

  test("uses installation-local fingerprints and secure key permissions", () => {
    const firstRoot = directory();
    const secondRoot = directory();
    const first = new CacheEventStore({ directory: firstRoot });
    const restarted = new CacheEventStore({ directory: firstRoot });
    const second = new CacheEventStore({ directory: secondRoot });
    expect(first.fingerprint("same identity")).toBe(restarted.fingerprint("same identity"));
    expect(first.fingerprint("same identity")).not.toBe(second.fingerprint("same identity"));
    const mode = Bun.file(join(firstRoot, "telemetry.hmac-key"));
    expect(mode.size).toBe(32);
    expect((require("node:fs").statSync(join(firstRoot, "telemetry.hmac-key")).mode & 0o777)).toBe(0o600);
  });

  test("persists stable boundaries only as complete positive triples", () => {
    const store = new CacheEventStore({ directory: directory() });
    store.append(event("positive", {
      stableBoundaryTokenHash: "token-hash",
      stableBoundaryTokenCount: 24,
      stableBoundaryKind: "prompt",
    }));
    store.append(event("unavailable", {
      stableBoundaryTokenHash: "empty-sequence-hash",
      stableBoundaryTokenCount: 0,
    }));
    expect(store.get("positive")).toMatchObject({
      stableBoundaryTokenHash: "token-hash",
      stableBoundaryTokenCount: 24,
      stableBoundaryKind: "prompt",
    });
    expect(store.get("unavailable")?.stableBoundaryTokenHash).toBeUndefined();
    expect(store.get("unavailable")?.stableBoundaryTokenCount).toBeUndefined();
    expect(store.get("unavailable")?.stableBoundaryKind).toBeUndefined();
  });

  test("does not synthesize a stable boundary from request content", () => {
    const store = new CacheEventStore({ directory: directory() });
    const metrics = new MetricsCollector(store);
    const unavailable = metrics.start("model", "/v1/chat/completions", false);
    unavailable.capture({ messages: [{ role: "system", content: "system" }, { role: "user", content: "user" }] });
    unavailable.finish({ status: "ok", stableBoundaryTokenHash: "empty-sequence-hash", stableBoundaryTokenCount: 0 });
    expect(store.get(unavailable.record.id)?.stableBoundaryTokenHash).toBeUndefined();

    const positive = metrics.start("model", "/v1/chat/completions", false);
    positive.capture({ messages: [{ role: "user", content: "user" }] });
    positive.finish({ status: "ok", stableBoundaryTokenHash: "native-token-hash", stableBoundaryTokenCount: 12, stableBoundaryKind: "prompt" });
    expect(store.get(positive.record.id)).toMatchObject({ stableBoundaryTokenCount: 12, stableBoundaryKind: "prompt" });
    expect(store.get(positive.record.id)?.stableBoundaryTokenHash).not.toBe("native-token-hash");
  });

  test("persists arbitrary boundary labels only as telemetry with hashed token identities", () => {
    const store = new CacheEventStore({ directory: directory() });
    const metrics = new MetricsCollector(store);
    const handle = metrics.start("model", "/v1/chat/completions", false);
    handle.capture({ messages: [{ role: "user", content: "PRIVATE SLICE CONTENT" }] });
    handle.finish({
      status: "ok",
      stableBoundaries: [
        { tokenHash: "native-token-hash-a", tokenCount: 12, kind: "messages", label: "checkpoint.alpha", requested: true, status: "resolved", materialized: true },
        { tokenHash: "native-token-hash-b", tokenCount: 18, kind: "prompt", requested: false, status: "resolved", materialized: false },
        { kind: "tools", label: "slice:tools-v2", requested: true, status: "skipped", skipReason: "unsupported_template_boundary" },
      ],
    });
    const boundaries = store.get(handle.record.id)?.stableBoundaries;
    expect(boundaries?.map(({ tokenHash: _tokenHash, ...boundary }) => boundary)).toEqual([
      { tokenCount: 12, kind: "messages", label: "checkpoint.alpha", requested: true, status: "resolved", materialized: true },
      { tokenCount: 18, kind: "prompt", requested: false, status: "resolved", materialized: false },
      { kind: "tools", label: "slice:tools-v2", requested: true, status: "skipped", skipReason: "unsupported_template_boundary" },
    ]);
    expect(boundaries?.[0]?.tokenHash).not.toBe("native-token-hash-a");
    expect(JSON.stringify(store.get(handle.record.id))).not.toContain("PRIVATE SLICE CONTENT");
  });

  test("persists cancellation and candidate diagnostics without request content", () => {
    const root = directory();
    const store = new CacheEventStore({ directory: root });
    const metrics = new MetricsCollector(store);
    const handle = metrics.start("private-model", "/v1/chat/completions", true);
    handle.capture({
      messages: [{ role: "system", content: "TOP SECRET PROMPT" }, { role: "user", content: "PRIVATE USER CONTENT" }],
      tools: [{ function: { name: "private_tool", description: "SECRET TOOL ARGUMENT", parameters: { password: "hunter2" } } }],
      cache: { namespace: "secret-tenant", session: "secret-session", project: "secret-project" },
    });
    handle.finish({
      status: "cancelled",
      finishReason: "cancel",
      cacheHit: false,
      cacheMissReason: "busy_lease",
      cacheCandidates: [{ slot: 3, generation: 7, state: "busy", sharedPrefixTokens: 42, rejection: "busy_lease" }],
    });

    const persisted = store.get(handle.record.id);
    expect(persisted?.status).toBe("cancelled");
    expect(persisted?.cache?.candidates?.[0]).toEqual({ slot: 3, generation: 7, state: "busy", sharedPrefixTokens: 42, rejection: "busy_lease" });
    const disk = readdirSync(root).filter((name) => name.endsWith(".jsonl")).map((name) => readFileSync(join(root, name), "utf8")).join("");
    for (const secret of ["TOP SECRET PROMPT", "PRIVATE USER CONTENT", "SECRET TOOL ARGUMENT", "hunter2", "secret-tenant", "secret-session", "secret-project"]) {
      expect(disk).not.toContain(secret);
    }
  });

  test("persists parser-style error codes", () => {
    const store = new CacheEventStore({ directory: directory() });
    const metrics = new MetricsCollector(store);
    const handle = metrics.start("model", "/v1/chat/completions", false);
    handle.capture({ messages: [{ role: "user", content: "do not persist me" }] });
    handle.finish({ status: "error", errorCode: "unsupported_content_part", error: "parser rejected content" });
    expect(store.get(handle.record.id)?.errorCode).toBe("unsupported_content_part");
  });

  test("persists classified cacheOutcome while keeping raw cache telemetry", () => {
    const store = new CacheEventStore({ directory: directory() });
    const metrics = new MetricsCollector(store, { checkpointMinimumTokens: () => 2048 });
    const handle = metrics.start("model", "/v1/chat/completions", false);
    handle.finish({
      status: "ok",
      cacheHit: false,
      reusedTokens: 0,
      workerLaunchId: "worker-launch",
      cacheCandidates: [],
      promptTokenCount: 4096,
    });
    const persisted = store.get(handle.record.id);
    expect(persisted?.cacheOutcome?.category).toBe("cold");
    expect(persisted?.cache).toMatchObject({ hit: false, reusedTokens: 0, candidates: [] });
    // Stored event keeps both classification and raw fields; neither replaces the other.
    expect(JSON.stringify(persisted)).toContain("\"hit\":false");
    expect(JSON.stringify(persisted)).toContain("\"category\":\"cold\"");
  });

  test("persists session fingerprint and omits raw session content", () => {
    const root = directory();
    const store = new CacheEventStore({ directory: root });
    const metrics = new MetricsCollector(store);
    const handle = metrics.start("model", "/v1/chat/completions", false);
    handle.capture({
      messages: [{ role: "user", content: "hello" }],
      cache: { session: "RAW-SESSION-S1", namespace: "tenant" },
    });
    handle.finish({ status: "ok", cacheHit: true, reusedTokens: 1, reuseKind: "slot" });
    const persisted = store.get(handle.record.id);
    expect(persisted?.sessionFingerprint).toBeDefined();
    expect(persisted?.sessionFingerprint).not.toBe("RAW-SESSION-S1");
    expect(persisted?.sessionIdentitySource).toBe("request.cache.session");
    expect(persisted?.promptPrefixId).toBeUndefined();
    const disk = readdirSync(root).filter((name) => name.endsWith(".jsonl")).map((name) => readFileSync(join(root, name), "utf8")).join("");
    expect(disk).not.toContain("RAW-SESSION-S1");
  });

  test("reads old records without cacheOutcome without rewriting them", () => {
    const root = directory();
    writeFileSync(join(root, "cache-decisions.current.jsonl"), `${JSON.stringify({
      schemaVersion: 2,
      source: "persisted",
      requestId: "legacy",
      timestamp: Date.now(),
      serverLaunchId: "server-launch",
      model: "model-a",
      status: "ok",
      cache: { hit: false, reusedTokens: 0, candidates: [] },
      promptTokenCount: 4096,
    })}\n`);
    const store = new CacheEventStore({ directory: root });
    const legacy = store.get("legacy");
    expect(legacy?.cacheOutcome).toBeUndefined();
    expect(legacy?.cache?.candidates).toEqual([]);
    // File content is not rewritten with a derived outcome.
    const disk = readFileSync(join(root, "cache-decisions.current.jsonl"), "utf8");
    expect(disk).not.toContain("cacheOutcome");
    expect(disk).toContain("\"hit\":false");
  });

  test("clear removes decision segments but preserves the fingerprint key", () => {
    const root = directory();
    const store = new CacheEventStore({ directory: root, segmentBytes: 16 * 1024 });
    const before = store.fingerprint("identity");
    store.append(event("one"));
    store.append(event("two"));
    expect(store.clear()).toEqual({ deletedEvents: 2 });
    expect(store.list().items).toEqual([]);
    expect(store.fingerprint("identity")).toBe(before);
    expect(readdirSync(root)).toContain("telemetry.hmac-key");
  });
});
