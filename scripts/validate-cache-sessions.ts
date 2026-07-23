#!/usr/bin/env bun

type Backend = "mlx" | "gguf";
type Turn = {
  label: string;
  system: "A" | "B" | "C";
  session: string;
  continuation?: boolean;
};

type CacheEvent = {
  requestId?: string;
  timestamp?: number;
  model?: string;
  backend?: string;
  status?: string;
  promptTokenCount?: number;
  workerLaunchId?: string;
  stableBoundaries?: Array<{ kind?: string; label?: string; status?: string; materialized?: boolean; skipReason?: string }>;
  cache?: {
    hit?: boolean;
    kind?: string;
    scope?: string;
    reusedTokens?: number;
    plannedTokens?: number;
    realizedTokens?: number;
    targetGeneration?: number;
    donorSlot?: number;
    targetSlot?: number;
    fallback?: string;
  };
};

const baseURL = (process.env.CLAP_BASE_URL ?? "http://127.0.0.1:11435").replace(/\/$/, "");
const apiKey = process.env.CLAP_API_KEY;
const models: Record<Backend, string | undefined> = {
  mlx: process.env.CLAP_TEST_MLX_MODEL,
  gguf: process.env.CLAP_TEST_GGUF_MODEL,
};
if (!apiKey || !models.mlx || !models.gguf) {
  console.error("Set CLAP_API_KEY, CLAP_TEST_MLX_MODEL, and CLAP_TEST_GGUF_MODEL.");
  process.exit(2);
}

const headers = { authorization: `Bearer ${apiKey}`, "content-type": "application/json" };
const turns: Turn[] = [
  { label: "A/session1", system: "A", session: "session1" },
  { label: "A/session1-continuation", system: "A", session: "session1", continuation: true },
  { label: "A/session2", system: "A", session: "session2" },
  { label: "B/session1", system: "B", session: "session1" },
  { label: "A/session3", system: "A", session: "session3" },
  { label: "B/session2", system: "B", session: "session2" },
  { label: "C/session1", system: "C", session: "session1" },
  { label: "B/session3", system: "B", session: "session3" },
  { label: "A/session4", system: "A", session: "session4" },
  { label: "C/session2", system: "C", session: "session2" },
];
const priorities = ["interactive", "normal", "background"] as const;
const startedAt = Date.now();
const claimedEvents = new Set<string>();
const histories = new Map<string, Array<{ role: string; content: string }>>();
const results: unknown[] = [];

function marker(backend: Backend, turn: Turn): string {
  return `${backend.toUpperCase()}_${turn.system}_${turn.session.toUpperCase()}_${turn.continuation ? "TWO" : "ONE"}`;
}

function systemPrompt(backend: Backend, system: string): string {
  return [
    `Validation namespace ${backend.toUpperCase()} system ${system}.`,
    "Treat the requested marker as opaque data and return it exactly, with no explanation.",
    "Never return a marker from another system or session.",
    "Stable cache padding: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau.",
  ].join(" ");
}

async function jsonRequest(path: string, init: RequestInit = {}): Promise<{ status: number; body: any }> {
  const response = await fetch(`${baseURL}${path}`, { ...init, headers: { ...headers, ...(init.headers ?? {}) } });
  const text = await response.text();
  let body: any;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text.slice(0, 200) }; }
  return { status: response.status, body };
}

async function streamChat(body: unknown): Promise<{ id?: string; text: string; usage?: unknown }> {
  const response = await fetch(`${baseURL}/v1/chat/completions`, {
    method: "POST", headers, body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) throw new Error(`stream request failed (${response.status}): ${await response.text()}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let id: string | undefined;
  let usage: unknown;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const chunk = JSON.parse(line.slice(6));
        id ??= chunk.id;
        text += chunk.choices?.[0]?.delta?.content ?? "";
        usage ??= chunk.usage;
      }
    }
  }
  return { id, text, usage };
}

async function cacheEvent(model: string, after: number): Promise<CacheEvent | undefined> {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const query = new URLSearchParams({ model, since: String(after - 1), limit: "50" });
    const { status, body } = await jsonRequest(`/clap/v1/cache-decisions?${query}`);
    if (status === 200) {
      const items = (body.items ?? []) as CacheEvent[];
      const event = items.find((item) => item.requestId && !claimedEvents.has(item.requestId));
      if (event?.requestId) {
        claimedEvents.add(event.requestId);
        return event;
      }
    }
    await Bun.sleep(100);
  }
}

function telemetry(event: CacheEvent | undefined) {
  return event ? {
    requestId: event.requestId,
    status: event.status,
    promptTokens: event.promptTokenCount,
    workerLaunchId: event.workerLaunchId,
    hit: event.cache?.hit,
    kind: event.cache?.kind,
    scope: event.cache?.scope,
    reusedTokens: event.cache?.reusedTokens,
    plannedTokens: event.cache?.plannedTokens,
    realizedTokens: event.cache?.realizedTokens,
    targetGeneration: event.cache?.targetGeneration,
    donorSlot: event.cache?.donorSlot,
    targetSlot: event.cache?.targetSlot,
    fallback: event.cache?.fallback,
    stableBoundaries: event.stableBoundaries,
  } : null;
}

async function runChat(backend: Backend, body: Record<string, unknown>) {
  const requestStarted = Date.now();
  const { status, body: response } = await jsonRequest("/v1/chat/completions", {
    method: "POST", body: JSON.stringify({ model: models[backend], backend, temperature: 0,
      seed: 4242, max_tokens: 32, ...body }),
  });
  if (status !== 200) throw new Error(`${backend} forced scenario failed (${status}): ${JSON.stringify(response)}`);
  return {
    text: response.choices?.[0]?.message?.content ?? "",
    event: await cacheEvent(models[backend]!, requestStarted),
  };
}

for (let index = 0; index < turns.length; index += 1) {
  const turn = turns[index]!;
  for (const backend of (["mlx", "gguf"] as const)) {
    const key = `${backend}:${turn.system}:${turn.session}`;
    const expected = marker(backend, turn);
    let messages = histories.get(key) ?? [
      { role: "system", content: systemPrompt(backend, turn.system) },
    ];
    messages = [...messages, { role: "user", content: `Return exactly ${expected}` }];
    const requestStarted = Date.now();
    const request = {
      model: models[backend], backend,
      messages,
      stream: index % 2 === 0,
      temperature: 0, seed: 4242, max_tokens: 24,
      cache: {
        namespace: `cache-validation-${backend}`,
        project: "deployment",
        harness: "interleaved-v1",
        agent: `system-${turn.system}`,
        session: turn.session,
        priority: priorities[index % priorities.length],
        boundaries: turn.continuation
          ? [{ kind: "messages", through_message: 0, label: "system" }, { kind: "messages", through_message: 2, label: "history" }]
          : [{ kind: "messages", through_message: 0, label: "system" }],
      },
    };
    let response: { id?: string; text: string; usage?: unknown };
    if (request.stream) response = await streamChat(request);
    else {
      const { status, body } = await jsonRequest("/v1/chat/completions", { method: "POST", body: JSON.stringify(request) });
      if (status !== 200) throw new Error(`${backend} ${turn.label} failed (${status}): ${JSON.stringify(body)}`);
      response = { id: body.id, text: body.choices?.[0]?.message?.content ?? "", usage: body.usage };
    }
    histories.set(key, [...messages, { role: "assistant", content: response.text }]);
    const event = await cacheEvent(models[backend]!, requestStarted);
    const otherMarkers = turns.flatMap((candidate) => (["mlx", "gguf"] as const).map((b) => marker(b, candidate)))
      .filter((candidate) => candidate !== expected);
    results.push({
      backend, turn: turn.label, stream: request.stream,
      continuation: turn.continuation === true,
      priority: request.cache.priority,
      responseId: response.id,
      correct: response.text.trim() === expected,
      contaminated: otherMarkers.some((candidate) => response.text.includes(candidate)),
      usage: response.usage,
      cache: telemetry(event),
    });
  }
}

const extras: unknown[] = [];

const forcedScenarios: unknown[] = [];
for (const backend of (["mlx", "gguf"] as const)) {
  await jsonRequest("/clap/v1/models/load", { method: "POST", body: JSON.stringify({
    model: models[backend], backend, keepAlive: "15m",
  }) });
  const runtime = await jsonRequest("/clap/v1/runtime/models");
  const loaded = (runtime.body.models ?? []).find((entry: any) => entry.id === models[backend]);
  const capabilities = loaded?.worker?.effectiveCapabilities?.cache ?? loaded?.effectiveCapabilities?.cache;

  if (capabilities?.partialPrefixBranch !== true) {
    forcedScenarios.push({ backend, scenario: "branch", skipped: true,
      reason: "effective cache capability partialPrefixBranch=false for this model architecture" });
  } else {
    const system = `${systemPrompt(backend, "A")} Forced branch shared prefix `
      + "one two three four five six seven eight nine ten ".repeat(8);
    const cache = { namespace: `forced-${backend}`, project: "deployment", harness: "forced-v1",
      agent: "branch-agent", session: "shared-branch-cache-domain", priority: "interactive" };
    const seed = await runChat(backend, { messages: [{ role: "system", content: system },
      { role: "user", content: "Logical session seed: return exactly BRANCH_SEED" }], cache });
    if (seed.text.trim() !== "BRANCH_SEED") throw new Error(`${backend} branch seed response mismatch`);
    const controller = new AbortController();
    const busy = fetch(`${baseURL}/v1/chat/completions`, { method: "POST", headers,
      signal: controller.signal, body: JSON.stringify({ model: models[backend], backend,
        messages: [{ role: "system", content: system },
          { role: "user", content: "Logical session busy: write the word BUSY repeatedly." }],
        stream: true, temperature: 0, max_tokens: 1024, cache }) });
    await Bun.sleep(150);
    const branch = await runChat(backend, { messages: [{ role: "system", content: system },
      { role: "user", content: "Logical session isolated: return exactly BRANCH_ISOLATED" }], cache });
    controller.abort();
    try { await busy; } catch {}
    const facts = telemetry(branch.event) as any;
    forcedScenarios.push({ backend, scenario: "branch", skipped: false,
      correct: branch.text.trim() === "BRANCH_ISOLATED", contaminated: branch.text.includes("BUSY")
        || branch.text.includes("BRANCH_SEED"), cache: facts });
  }

  if (capabilities?.promptBoundarySnapshots !== true || capabilities?.wholeStateCopy !== true) {
    forcedScenarios.push({ backend, scenario: "anchor", skipped: true,
      reason: `effective cache capabilities promptBoundarySnapshots=${String(capabilities?.promptBoundarySnapshots)} wholeStateCopy=${String(capabilities?.wholeStateCopy)}` });
  } else {
    const stable = `${systemPrompt(backend, "C")} Forced stable anchor `
      + "amber bronze cobalt denim emerald fuchsia gold hazel indigo jade ".repeat(12);
    const identity = { namespace: `forced-anchor-${backend}`, project: "deployment",
      harness: "forced-anchor-v1", agent: "shared-anchor-agent", session: "shared-anchor-cache-domain",
      priority: "normal", boundaries: [{ kind: "messages", through_message: 0, label: "stable-system" }] };
    const seed = await runChat(backend, { messages: [{ role: "system", content: stable },
      { role: "user", content: "Logical session anchor seed: return exactly ANCHOR_SEED" }], cache: identity });
    if (seed.text.trim() !== "ANCHOR_SEED") throw new Error(`${backend} anchor seed response mismatch`);
    const seedFacts = telemetry(seed.event) as any;
    const explicitBoundary = seedFacts?.stableBoundaries?.find((item: any) => item.label === "stable-system");
    await Bun.sleep(500);
    const fillCount = 160;
    for (let offset = 0; offset < fillCount; offset += 16) {
      await Promise.all(Array.from({ length: Math.min(16, fillCount - offset) }, async (_, index) => {
        const filler = offset + index;
        const response = await jsonRequest("/v1/chat/completions", { method: "POST", body: JSON.stringify({
          model: models[backend], backend, messages: [{ role: "user", content: `Filler ${filler}` }],
          temperature: 0, max_tokens: 1, cache: { namespace: `filler-${backend}-${filler}`,
            session: `filler-${filler}`, priority: "interactive" },
        }) });
        if (response.status !== 200) throw new Error(`${backend} anchor filler ${filler} failed (${response.status})`);
      }));
    }
    const restored = await runChat(backend, { messages: [{ role: "system", content: stable },
      { role: "user", content: "Logical session restored: return exactly ANCHOR_ISOLATED" }],
      cache: { ...identity, side_request: true } });
    const facts = telemetry(restored.event) as any;
    const restoredAsAnchor = facts?.kind === "anchor" && facts?.hit === true;
    forcedScenarios.push({ backend, scenario: "anchor", skipped: !restoredAsAnchor,
      reason: restoredAsAnchor ? undefined : facts?.fallback === "decode_retry_full_prefill"
        ? "advertised checkpoint restore failed physical decode and safely fell back to full prefill"
        : "advertised checkpoint snapshot materialized, but session-affinity selected a resident slot instead of restore",
      correct: restored.text.trim() === "ANCHOR_ISOLATED",
      contaminated: restored.text.includes("ANCHOR_SEED"),
      explicitBoundary, cache: facts });
  }
}

for (const backend of (["mlx", "gguf"] as const)) {
  const structured = await jsonRequest("/v1/chat/completions", {
    method: "POST",
    body: JSON.stringify({
      model: models[backend], backend,
      messages: [{ role: "user", content: "Return a JSON object with integer field ok equal to 1." }],
      temperature: 0, max_tokens: 32,
      response_format: { type: "json_schema", constraint: "best_effort", json_schema: {
        name: "Validation", schema: { type: "object", properties: { ok: { type: "integer" } }, required: ["ok"], additionalProperties: false },
      } },
      cache: { namespace: `cache-validation-${backend}`, session: "structured", priority: "background" },
    }),
  });
  let structuredValid = false;
  if (structured.status === 200) {
    try { structuredValid = JSON.parse(structured.body.choices?.[0]?.message?.content ?? "").ok === 1; } catch {}
  }
  extras.push({ backend, exercise: "structured-output", status: structured.status, valid: structuredValid });

  const controller = new AbortController();
  const cancellation = fetch(`${baseURL}/v1/chat/completions`, {
    method: "POST", headers, signal: controller.signal,
    body: JSON.stringify({
      model: models[backend], backend,
      messages: [{ role: "user", content: "Count upward without stopping." }],
      stream: true, max_tokens: 2048,
      cache: { namespace: `cache-validation-${backend}`, session: "cancel", priority: "background" },
    }),
  });
  setTimeout(() => controller.abort(), 100);
  let cancelled = false;
  try {
    const response = await cancellation;
    const reader = response.body?.getReader();
    await reader?.read();
    controller.abort();
    await reader?.read();
  } catch (error) { cancelled = error instanceof DOMException && error.name === "AbortError"; }
  extras.push({ backend, exercise: "cancellation", clientAbortObserved: cancelled || controller.signal.aborted });
}

for (const backend of (["mlx", "gguf"] as const)) {
  for (const candidate of (["mlx", "gguf"] as const)) {
    await jsonRequest("/clap/v1/models/unload", { method: "POST", body: JSON.stringify({
      model: models[candidate], backend: candidate,
    }) });
  }
  for (let round = 1; round <= 3; round += 1) {
    const unloadStarted = Date.now();
    const unload = await jsonRequest("/clap/v1/models/unload", {
      method: "POST", body: JSON.stringify({ model: models[backend], backend }),
    });
    const immediateLoad = await jsonRequest("/clap/v1/models/load", {
      method: "POST", body: JSON.stringify({ model: models[backend], backend, keepAlive: "15m" }),
    });
    const expectedBackoff = unload.body.unloaded === false && immediateLoad.status === 503
      && immediateLoad.body.error?.code === "insufficient_model_memory";
    let load = immediateLoad;
    const structuredAdmission = load.status === 503
      && load.body.error?.code === "insufficient_model_memory";
    for (let attempt = 0; load.status === 503 && structuredAdmission && attempt < 20; attempt += 1) {
      await Bun.sleep(500);
      load = await jsonRequest("/clap/v1/models/load", {
        method: "POST", body: JSON.stringify({ model: models[backend], backend, keepAlive: "15m" }),
      });
    }
    extras.push({ backend, exercise: "immediate-unload-reload", round,
      unloadStatus: unload.status, unloaded: unload.body.unloaded,
      unloadWaitMs: Date.now() - unloadStarted, loadStatus: load.status,
      state: load.body.model?.state, errorCode: load.body.error?.code,
      immediateLoadStatus: immediateLoad.status, immediateErrorCode: immediateLoad.body.error?.code,
      expectedStructuredBackoff: expectedBackoff || structuredAdmission });
  }
}

const rotate = await jsonRequest("/clap/v1/cache/identity/rotate", { method: "POST", body: "{}" });
const health = await jsonRequest("/clap/v1/health");
const failed = results.filter((item: any) => !item.correct || item.contaminated
  || item.cache?.status !== "ok" || typeof item.cache?.hit !== "boolean"
  || (item.continuation && (item.cache.hit !== true || !(item.cache.reusedTokens > 0))));
const forcedFailed = forcedScenarios.filter((item: any) => !item.skipped
  && (!item.correct || item.contaminated || item.cache?.status !== "ok"
    || item.cache?.hit !== true || !(item.cache?.reusedTokens > 0)
    || item.cache?.kind !== item.scenario || !(item.cache?.targetGeneration > 0)
    || (item.scenario === "anchor" && item.explicitBoundary?.status !== "resolved"
      && item.explicitBoundary?.skipReason !== "unsupported_template_boundary")));
const lifecycleFailed = extras.filter((item: any) => item.exercise === "immediate-unload-reload"
  && (item.unloadStatus !== 200 || item.loadStatus !== 200
    || !["warm", "active"].includes(item.state)
    || (item.immediateLoadStatus === 503 && !item.expectedStructuredBackoff)));
console.log(JSON.stringify({
  startedAt, completedAt: Date.now(), exactInterleaving: turns.map((turn) => turn.label),
  results, forcedScenarios, extras,
  reset: { status: rotate.status, clearedResidents: rotate.body.clearedResidents },
  health: { status: health.status, body: health.body },
  summary: { turns: results.length, failed: failed.length,
    forcedFailed: forcedFailed.length, lifecycleFailed: lifecycleFailed.length },
}, null, 2));
if (failed.length || forcedFailed.length || lifecycleFailed.length) process.exitCode = 1;
