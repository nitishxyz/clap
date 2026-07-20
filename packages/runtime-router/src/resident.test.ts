import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { deriveTokenCapabilities, parseWorkerRetention, parseWorkerTokenCapabilities, ResidentWorkerRegistry, validateTokenBudget } from "./resident";

describe("worker retention telemetry", () => {
  test("parses a complete MLX retention snapshot", () => {
    expect(parseWorkerRetention({
      max_active: 4, queued: 3, previous_max_active: 2,
      last_adjustment_reason: "global_headroom_available", last_adjustment_at: "2026-07-20T11:29:00.000Z",
      retained_growth_reserve_bytes: 1_000, global_resident_memory_bytes: 20_000,
      pressure_state: "normal", active_policy: { mode: "auto", selected_max: 4, backend_ceiling: 16,
        hardware_ceiling: 8, model_ceiling: 16, memory_ceiling: 4, reason: "memory_ceiling",
        inputs: { startup_available_bytes: 12_000, hybrid_or_recurrent: false } },
      active: 2, retained_total: 103, retained_sessions: 100, retained_anchors: 3,
      retained_bytes: 10_000, session_bytes: 8_000, anchor_bytes: 2_000, budget_bytes: 100_000,
      high_watermark_bytes: 90_000, low_watermark_bytes: 75_000, under_pressure: false,
      hard_ceiling: 256, eviction_reason: "byte_pressure", eviction_count: 7,
    })).toEqual({
      maxActive: 4, queued: 3, previousMaxActive: 2,
      lastAdjustmentReason: "global_headroom_available", lastAdjustmentAt: "2026-07-20T11:29:00.000Z",
      retainedGrowthReserveBytes: 1_000, globalResidentMemoryBytes: 20_000,
      pressureState: "normal", activePolicy: { mode: "auto", selectedMax: 4, backendCeiling: 16,
        hardwareCeiling: 8, modelCeiling: 16, memoryCeiling: 4, reason: "memory_ceiling",
        inputs: { startup_available_bytes: 12_000, hybrid_or_recurrent: false } },
      active: 2, retainedTotal: 103, retainedSessions: 100, retainedAnchors: 3,
      retainedBytes: 10_000, sessionBytes: 8_000, anchorBytes: 2_000, budgetBytes: 100_000,
      highWatermarkBytes: 90_000, lowWatermarkBytes: 75_000, underPressure: false,
      hardCeiling: 256, evictionReason: "byte_pressure", evictionCount: 7,
    });
  });

  test("preserves absent telemetry for older and non-MLX workers", () => {
    expect(parseWorkerRetention(undefined)).toBeUndefined();
    expect(parseWorkerRetention({ retained_total: 1 })).toBeUndefined();
    expect(parseWorkerRetention({ max_active: 4, active_policy: { mode: "auto" } })).toBeUndefined();
  });
});

describe("worker token capabilities", () => {
  test.each([
    ["small llama", 4096, 1024],
    ["long mlx", 131072, 8192],
  ])("parses the %s model profile", (_name, context, output) => {
    expect(parseWorkerTokenCapabilities({
      model_context_window: context,
      effective_context_window: context,
      max_input_tokens: context - 1,
      max_output_tokens: output,
      backend_allocation_cap: context,
      user_configured_override: null,
    })).toEqual({
      modelContextWindow: context,
      effectiveContextWindow: context,
      maxInputTokens: context - 1,
      maxOutputTokens: output,
      backendAllocationCap: context,
      userConfiguredOverride: null,
    });
  });

  test("preserves unknown metadata instead of fabricating limits", () => {
    expect(parseWorkerTokenCapabilities({
      model_context_window: null,
      effective_context_window: null,
      max_input_tokens: null,
      max_output_tokens: null,
      backend_allocation_cap: null,
      user_configured_override: null,
    })).toEqual({
      modelContextWindow: null,
      effectiveContextWindow: null,
      maxInputTokens: null,
      maxOutputTokens: null,
      backendAllocationCap: null,
      userConfiguredOverride: null,
    });
  });

  test("preserves authoritative worker metadata sources", () => {
    expect(parseWorkerTokenCapabilities({
      model_context_window: 131072,
      effective_context_window: 131072,
      max_input_tokens: 131071,
      max_output_tokens: null,
      model_context_window_source: "config.json:text_config.max_position_embeddings",
      max_output_tokens_source: null,
      backend_allocation_cap: 131072,
      user_configured_override: null,
    })).toMatchObject({
      modelContextWindowSource: "config.json:text_config.max_position_embeddings",
      maxOutputTokensSource: null,
    });
  });

  test("applies the minimum model, allocation, and admin override", () => {
    expect(deriveTokenCapabilities({
      modelContextWindow: 131072,
      backendAllocationCap: 65536,
      userConfiguredOverride: 32768,
      maxOutputTokens: 8192,
    })).toEqual({
      modelContextWindow: 131072,
      effectiveContextWindow: 32768,
      maxInputTokens: 32767,
      maxOutputTokens: 8192,
      backendAllocationCap: 65536,
      userConfiguredOverride: 32768,
    });
  });

  test("rejects prompt, output, and combined context overflows", () => {
    const capabilities = deriveTokenCapabilities({ modelContextWindow: 4096, backendAllocationCap: 4096, maxOutputTokens: 1024 });
    expect(validateTokenBudget(capabilities, 4096, 1)).toMatchObject({ code: "context_length_exceeded" });
    expect(validateTokenBudget(capabilities, 100, 1025)).toMatchObject({ code: "max_output_tokens_exceeded" });
    expect(validateTokenBudget(capabilities, 3500, 700)).toMatchObject({ code: "context_length_exceeded" });
    expect(validateTokenBudget(capabilities, 3500)).toEqual({ maxTokens: 596 });
  });

  test("requires an explicit output budget when every capability is unknown", () => {
    const capabilities = deriveTokenCapabilities({});
    expect(validateTokenBudget(capabilities, 100)).toMatchObject({ code: "token_capability_unknown" });
    expect(validateTokenBudget(capabilities, 100, 32)).toEqual({ maxTokens: 32 });
  });
});

describe("resident worker registry", () => {
  test("loads, reuses one pid for chats, and shuts down", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-registry-test-"));
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeResidentWorker(dir);
      process.env.CLAP_HOME = join(dir, "fresh-home");
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("key", "llama", join(dir, "model.gguf"));
      await writeFile(join(dir, "model.gguf"), "gguf");

      const info = await worker.load();
      expect(info.state).toBe("resident");
      expect(info.pid).toBeNumber();
      expect(info.tokenCapabilities).toMatchObject({ effectiveContextWindow: 32768, maxOutputTokens: 2048 });
      const tokens: string[] = [];
      let dispatches = 0;
      const first = await worker.chat(
        { model: join(dir, "model.gguf"), messages: [{ role: "user", content: "one" }], stream: false },
        (token) => tokens.push(token),
        undefined,
        undefined,
        () => { dispatches += 1; },
      );
      const second = await worker.chat({ model: join(dir, "model.gguf"), messages: [{ role: "user", content: "two" }], stream: false });
      const withIntent = await worker.chat({
        model: join(dir, "model.gguf"),
        messages: [{ role: "user", content: "cache intent" }],
        stream: false,
        cache: {
          namespace: "tenant-acme",
          project: "payments",
          harness: "coding-v4",
          agent: "builder",
          session: "conversation-123",
          priority: "interactive",
          side_request: true,
          boundaries: [{ kind: "messages", through_message: 0, label: "checkpoint.alpha" }],
        },
      });
      expect(first.content).toBe("resident response");
      expect(first.usage).toEqual({ promptTokens: 12, completionTokens: 2 });
      expect(first.finishReason).toBe("stop");
      expect(first.cache).toMatchObject({ hit: true, reusedTokens: 10, reuseKind: "branch", reuseScope: "conversation", sideRequest: false, slot: 1 });
      expect(first.cache).toMatchObject({ stableBoundaryTokenHash: "boundary-one", stableBoundaryTokenCount: 8, stableBoundaryKind: "prompt" });
      expect(first.cache?.stableBoundaries).toEqual([{
        tokenHash: "boundary-one", tokenCount: 8, kind: "messages", label: "prefix-0",
        requested: true, status: "resolved", skipReason: undefined, materialized: true,
      }]);
      expect(first.cache?.workerLaunchId).toMatch(/^[0-9a-f-]{36}$/);
      expect(tokens).toEqual(["resident ", "response"]);
      expect(dispatches).toBe(1);
      expect(second.content).toBe("resident response");
      expect(second.cache?.reuseKind).toBe("anchor");
      expect(second.cache?.stableBoundaryTokenHash).toBeUndefined();
      expect(second.cache?.stableBoundaryTokenCount).toBeUndefined();
      expect(second.cache?.stableBoundaryKind).toBeUndefined();
      expect(withIntent.cache?.reuseScope).toBe("project");
      expect(withIntent.cache?.sideRequest).toBe(true);
      expect(withIntent.cache?.stableBoundaries?.[0]?.label).toBe("checkpoint.alpha");
      expect(withIntent.cache?.stableBoundaries?.[1]).toMatchObject({
        kind: "tools", label: "slice:tools-v2", requested: true,
        status: "skipped", skipReason: "unsupported_template_boundary",
      });
      expect(withIntent.cache?.stableBoundaries?.[1]?.tokenHash).toBeUndefined();
      expect(withIntent.cache?.stableBoundaries?.[1]?.tokenCount).toBeUndefined();
      expect(withIntent.cache?.stableBoundaryTokenHash).toBeUndefined();
      expect(worker.info().pid).toBe(info.pid);
      expect(worker.info().memory).toEqual({ activeBytes: 1024, cacheBytes: 0, peakActiveBytes: 4096 });

      registry.shutdownAll();
      expect(worker.info().state).toBe("not_started");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("rebalances resident workers in place from one coordinated memory snapshot", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-rebalance-test-"));
    try {
      const { command, log } = await fakeRebalanceWorker(dir);
      process.env.CLAP_LLAMA_WORKER = command;
      const registry = new ResidentWorkerRegistry();
      registry.memorySnapshot = async (pids) => ({
        physicalMemoryBytes: 32 * 1024 ** 3,
        availableMemoryBytes: 12 * 1024 ** 3,
        residentBytesByPid: new Map(pids.map((pid) => [pid, 8 * 1024 ** 3])),
      });
      const first = registry.getOrCreate("first", "llama", join(dir, "first.gguf"));
      const second = registry.getOrCreate("second", "llama", join(dir, "second.gguf"));
      await writeFile(join(dir, "first.gguf"), "gguf");
      await writeFile(join(dir, "second.gguf"), "gguf");
      await Promise.all([first.load(), second.load()]);
      await Bun.sleep(30);
      await registry.rebalance("test_topology");

      expect(first.info().retention).toMatchObject({
        maxActive: 8,
        previousMaxActive: 1,
        globalResidentMemoryBytes: 16 * 1024 ** 3,
        pressureState: "normal",
      });
      expect(second.info().retention).toMatchObject({
        maxActive: 8,
        globalResidentMemoryBytes: 16 * 1024 ** 3,
        pressureState: "normal",
      });
      const commands = (await readFile(log, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
      const updates = commands.filter((command) => command.type === "set_max_active");
      expect(updates.length).toBeGreaterThanOrEqual(2);
      expect(updates.every((command) => command.max_active === 8)).toBe(true);
      expect(commands.some((command) => command.type === "cancel")).toBe(false);
      registry.shutdownAll();
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("sends cancel to the worker when the signal aborts", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-cancel-test-"));
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeCancellableWorker(dir);
      const registry = new ResidentWorkerRegistry();
      const worker = registry.getOrCreate("key", "llama", join(dir, "model.gguf"));
      await writeFile(join(dir, "model.gguf"), "gguf");

      const controller = new AbortController();
      const tokens: string[] = [];
      const pending = worker.chat(
        { model: join(dir, "model.gguf"), messages: [{ role: "user", content: "long" }], stream: true },
        (token) => {
          tokens.push(token);
          if (tokens.length === 3) controller.abort();
        },
        controller.signal,
      );
      const result = await pending;
      expect(result.finishReason).toBe("cancel");
      expect(tokens.length).toBeLessThan(50);
      registry.shutdownAll();
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("rotates stderr and attributes each worker launch", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-resident-log-test-"));
    const home = join(dir, "home");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeResidentWorker(dir);
      process.env.CLAP_HOME = home;
      await writeFile(join(dir, "model.gguf"), "gguf");

      const firstRegistry = new ResidentWorkerRegistry();
      const first = firstRegistry.getOrCreate("first-key", "llama", join(dir, "model.gguf"));
      await first.load();
      await Bun.sleep(20);
      firstRegistry.shutdownAll();
      await Bun.sleep(20);

      const secondRegistry = new ResidentWorkerRegistry();
      const second = secondRegistry.getOrCreate("second-key", "llama", join(dir, "model.gguf"));
      await second.load();
      await Bun.sleep(20);
      const current = await readFile(join(home, "llama-worker.err.log"), "utf8");
      const previous = await readFile(join(home, "llama-worker.err.log.previous"), "utf8");
      const launch = JSON.parse(await readFile(join(home, "llama-worker.err.log.launch.json"), "utf8"));
      expect(current).toMatch(/^worker stderr launch \d+\n$/);
      expect(previous).toMatch(/^worker stderr launch \d+\n$/);
      expect(current).not.toBe(previous);
      expect(launch).toMatchObject({ backend: "llama", key: "second-key" });
      expect(launch.startedAt).toBeString();
      secondRegistry.shutdownAll();
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });
});

async function fakeResidentWorker(dir: string): Promise<string> {
  const path = join(dir, "fake-resident-worker");
  await writeFile(path, `#!/usr/bin/env bun
console.error("worker stderr launch " + process.pid);
const decoder = new TextDecoder();
let buffer = "";
let chatCount = 0;
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const request = JSON.parse(line);
    if (request.type === "shutdown") process.exit(0);
    if (request.type === "load") {
      console.log(JSON.stringify({ id: request.id, loaded: true, done: true, token_capabilities: { model_context_window: 65536, effective_context_window: 32768, max_input_tokens: 32767, max_output_tokens: 2048, backend_allocation_cap: 32768, user_configured_override: 32768 } }));
      continue;
    }
    if (request.type === "unload") {
      console.log(JSON.stringify({ id: request.id, unloaded: true, done: true }));
      continue;
    }
    chatCount += 1;
    console.log(JSON.stringify({ id: request.id, started: true }));
    console.log(JSON.stringify({ id: request.id, token: "resident " }));
    console.log(JSON.stringify({ id: request.id, token: "response" }));
    const stableBoundary = chatCount === 1
      ? { stable_boundary_token_hash: "boundary-one", stable_boundary_token_count: 8, stable_boundary_kind: "prompt", stable_boundaries: [{ token_hash: "boundary-one", token_count: 8, kind: "messages", label: "prefix-0", requested: true, status: "resolved", materialized: true }] }
      : chatCount === 2 ? { stable_boundary_token_hash: "empty-sequence-hash", stable_boundary_token_count: 0 } : {};
    if (request.cache?.boundaries?.[0]?.label === "checkpoint.alpha") stableBoundary.stable_boundaries = [{ token_hash: "slice-boundary", token_count: 9, kind: "messages", label: "checkpoint.alpha", requested: true, status: "resolved", materialized: false }, { kind: "tools", label: "slice:tools-v2", requested: true, status: "skipped", skip_reason: "unsupported_template_boundary" }];
    console.log(JSON.stringify({ id: request.id, done: true, finish_reason: "stop", usage: { prompt_tokens: 12, completion_tokens: 2 }, cache: { hit: true, reused_tokens: 10, reuse_kind: chatCount === 1 ? "branch" : "anchor", reuse_scope: request.cache?.project ? "project" : "conversation", side_request: request.cache?.side_request ?? false, slot: 1, ...stableBoundary } }));
    console.log(JSON.stringify({ memory: { active_bytes: 1024, cache_bytes: 0, peak_active_bytes: 4096 } }));
  }
}
`);
  return `/usr/bin/env bun ${path}`;
}

async function fakeCancellableWorker(dir: string): Promise<string> {
  const path = join(dir, "fake-cancellable-worker");
  await writeFile(path, `#!/usr/bin/env bun
const decoder = new TextDecoder();
let buffer = "";
let active = null;
let cancelled = false;
async function generate(id) {
  for (let i = 0; i < 50; i += 1) {
    if (cancelled) {
      console.log(JSON.stringify({ id, done: true, cancelled: true, finish_reason: "cancel" }));
      return;
    }
    console.log(JSON.stringify({ id, token: "t" + i + " " }));
    await Bun.sleep(5);
  }
  console.log(JSON.stringify({ id, done: true, finish_reason: "stop" }));
}
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const request = JSON.parse(line);
    if (request.type === "shutdown") process.exit(0);
    if (request.type === "cancel") {
      cancelled = true;
      continue;
    }
    if (request.type === "load") {
      console.log(JSON.stringify({ id: request.id, loaded: true, done: true }));
      continue;
    }
    if (request.type === "chat") {
      cancelled = false;
      active = generate(request.id);
    }
  }
}
`);
  return `/usr/bin/env bun ${path}`;
}

async function fakeRebalanceWorker(dir: string): Promise<{ command: string; log: string }> {
  const path = join(dir, "fake-rebalance-worker");
  const log = join(dir, "rebalance-commands.jsonl");
  await writeFile(log, "");
  await writeFile(path, `#!/usr/bin/env bun
import { appendFileSync } from "node:fs";
const log = ${JSON.stringify(log)};
const decoder = new TextDecoder();
let buffer = "";
let maxActive = 1;
function retention(request = {}) {
  return {
    max_active: maxActive,
    queued: 0,
    previous_max_active: request.previous_max_active ?? null,
    last_adjustment_reason: request.last_adjustment_reason ?? null,
    last_adjustment_at: request.last_adjustment_at ?? null,
    retained_growth_reserve_bytes: request.retained_growth_reserve_bytes ?? 0,
    global_resident_memory_bytes: request.global_resident_memory_bytes ?? null,
    pressure_state: request.pressure_state ?? null,
    active_policy: { mode: "auto", selected_max: maxActive, backend_ceiling: 16,
      hardware_ceiling: 16, model_ceiling: 16, memory_ceiling: 16,
      reason: "bounded_backend_default", inputs: { per_active_reserve_bytes: 536870912 } },
    active: 0, retained_total: 1, retained_sessions: 1, retained_anchors: 0,
    retained_bytes: 74448896, session_bytes: 74448896, anchor_bytes: 0,
    budget_bytes: 1216348160, high_watermark_bytes: 1094713344,
    low_watermark_bytes: 912261120, under_pressure: false, hard_ceiling: 64,
    eviction_reason: null, eviction_count: 0,
  };
}
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const request = JSON.parse(line);
    appendFileSync(log, JSON.stringify(request) + "\\n");
    if (request.type === "shutdown") process.exit(0);
    if (request.type === "load") {
      console.log(JSON.stringify({ id: request.id, loaded: true, done: true, retention: retention() }));
    } else if (request.type === "set_max_active") {
      maxActive = request.max_active;
      console.log(JSON.stringify({ id: request.id, done: true, retention: retention(request) }));
    }
  }
}
`);
  return { command: `/usr/bin/env bun ${path}`, log };
}

function restoreEnv(name: string, value: string | undefined) {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}
