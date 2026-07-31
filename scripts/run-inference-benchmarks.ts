#!/usr/bin/env bun
// Inference benchmark runner (docs/inference-runtime-benchmark-follow-up.md).
// Runs against an already-running clap server (local or remote), preserves raw
// samples as JSONL, and derives summaries separately so they can always be
// recomputed. Correctness gates are never replaced by these numbers.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { hostname } from "node:os";
import { join, resolve } from "node:path";
import {
  accountChunk, emptyStreamAccounting, metricsDelta, parseMetricsCounters,
  sanitizeBenchmarkText, summarizeCase, synthesizeWords, timingsFrom,
  type MetricsCounters, type SampleRecord,
} from "./inference-benchmark-lib";

type MatrixConfig = {
  schemaVersion: 1;
  defaults: { warmups: number; samples: number; timeoutMs: number;
    maxTokens: number; temperature: number };
  prompts: { seed: number; coldPromptWords: number; sharedPrefixWords: number;
    divergenceWords: number };
  suites: Record<string, Record<string, unknown> & { enabled?: boolean }>;
};

type RunnerOptions = {
  baseUrl: string;
  models: string[];
  suites: string[] | null;
  outDir: string;
  label: string;
  samplesOverride: number | null;
  warmupsOverride: number | null;
  apiKey: string | undefined;
};

function parseArgs(argv: string[]): RunnerOptions {
  const options: RunnerOptions = {
    baseUrl: process.env.CLAP_BASE_URL ?? "http://localhost:11435",
    models: [],
    suites: null,
    outDir: "bench-results",
    label: "run",
    samplesOverride: null,
    warmupsOverride: null,
    apiKey: process.env.CLAP_API_KEY,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    const next = () => {
      index += 1;
      const value = argv[index];
      if (value === undefined) throw new Error(`missing value for ${arg}`);
      return value;
    };
    if (arg === "--model") options.models.push(next());
    else if (arg === "--base-url") options.baseUrl = next();
    else if (arg === "--suite") (options.suites ??= []).push(next());
    else if (arg === "--out") options.outDir = next();
    else if (arg === "--label") options.label = next();
    else if (arg === "--samples") options.samplesOverride = Number(next());
    else if (arg === "--warmups") options.warmupsOverride = Number(next());
    else if (arg === "--api-key") options.apiKey = next();
    else throw new Error(`unknown argument: ${arg}`);
  }
  if (options.models.length === 0) {
    throw new Error("usage: run-inference-benchmarks.ts --model <id> [--model <id>] " +
      "[--base-url http://host:11435] [--suite name] [--out dir] [--label name] " +
      "[--samples n] [--warmups n]");
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const matrix = JSON.parse(await readFile(
  resolve(import.meta.dir, "../config/inference-benchmark-matrix.json"), "utf8")) as MatrixConfig;
if (matrix.schemaVersion !== 1) throw new Error("unsupported benchmark matrix schema");

const headers: Record<string, string> = { "content-type": "application/json" };
if (options.apiKey) headers.authorization = `Bearer ${options.apiKey}`;

async function api(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${options.baseUrl}${path}`, {
    ...init,
    headers: { ...headers, ...(init?.headers as Record<string, string> | undefined) },
  });
  return response;
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await api(path, init);
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}: ${(await response.text()).slice(0, 300)}`);
  }
  return await response.json() as T;
}

async function scrapeMetrics(): Promise<MetricsCounters | null> {
  try {
    const response = await api("/metrics");
    if (!response.ok) return null;
    return parseMetricsCounters(await response.text());
  } catch {
    return null;
  }
}

type ChatOptions = {
  model: string;
  messages: Array<{ role: string; content: string }>;
  maxTokens: number;
  timeoutMs: number;
  session: string;
  priority?: "interactive" | "normal" | "background";
  boundaries?: Array<{ kind: "messages"; through_message: number; label?: string }>;
};

async function streamChat(chat: ChatOptions): Promise<{ timings: ReturnType<typeof timingsFrom>; content: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), chat.timeoutMs);
  const startedMs = performance.now();
  const accounting = emptyStreamAccounting();
  try {
    const response = await api("/v1/chat/completions", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        model: chat.model,
        messages: chat.messages,
        max_tokens: chat.maxTokens,
        temperature: matrix.defaults.temperature,
        stream: true,
        cache: { session: chat.session, priority: chat.priority ?? "normal",
          ...(chat.boundaries ? { boundaries: chat.boundaries } : {}) },
      }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`chat -> ${response.status}: ${(await response.text()).slice(0, 300)}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of frame.split("\n")) {
          if (line.startsWith("data:")) {
            accountChunk(accounting, line.slice(5).trim(), performance.now());
          }
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    clearTimeout(timer);
  }
  if (accounting.error) throw new Error(`stream error: ${accounting.error}`);
  const finishedMs = performance.now();
  return { timings: timingsFrom(accounting, startedMs, finishedMs), content: accounting.content };
}

const promptSeed = matrix.prompts.seed;
const sharedPrefix = synthesizeWords(matrix.prompts.sharedPrefixWords, promptSeed);
const records: SampleRecord[] = [];
const runId = `${new Date().toISOString().replaceAll(/[:.]/g, "-")}-${options.label}`;
const outDir = join(options.outDir, runId);
await mkdir(outDir, { recursive: true });

function pushRecord(record: SampleRecord): void {
  records.push(record);
  const status = record.error ? `FAILED (${record.error.slice(0, 80)})` : [
    record.timings.ttftMs !== null ? `ttft=${record.timings.ttftMs.toFixed(0)}ms` : null,
    record.timings.decodeTps !== null ? `decode=${record.timings.decodeTps.toFixed(1)}t/s` : null,
    `total=${record.timings.totalMs.toFixed(0)}ms`,
    record.metricsDelta ? `reused=${record.metricsDelta.kvReusedTokens}` : null,
  ].filter(Boolean).join(" ");
  console.log(`[${record.suite}/${record.case}] ${record.model} #${record.sampleIndex}${record.warmup ? " (warmup)" : ""}: ${status}`);
}

async function measuredSample(suite: string, caseName: string, model: string,
  sampleIndex: number, warmup: boolean,
  run: () => Promise<{ timings: ReturnType<typeof timingsFrom>; extra?: Record<string, unknown> }>): Promise<void> {
  const before = await scrapeMetrics();
  const startedAt = new Date().toISOString();
  try {
    const result = await run();
    const after = await scrapeMetrics();
    pushRecord({
      suite, case: caseName, model, sampleIndex, warmup, startedAt,
      timings: result.timings,
      metricsDelta: before && after ? metricsDelta(before, after) : null,
      extra: result.extra,
    });
  } catch (error) {
    pushRecord({
      suite, case: caseName, model, sampleIndex, warmup, startedAt,
      timings: { ttftMs: null, totalMs: 0, decodeTokens: 0, decodeTps: null, finishReason: null },
      metricsDelta: null,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

function suiteConfig(name: string): (Record<string, unknown> & { enabled?: boolean }) | null {
  const config = matrix.suites[name];
  if (!config || config.enabled === false) return null;
  if (options.suites && !options.suites.includes(name)) return null;
  return config;
}

function counts(config: Record<string, unknown>): { warmups: number; samples: number } {
  return {
    warmups: options.warmupsOverride ?? (typeof config.warmups === "number" ? config.warmups : matrix.defaults.warmups),
    samples: options.samplesOverride ?? (typeof config.samples === "number" ? config.samples : matrix.defaults.samples),
  };
}

// --- Preflight -------------------------------------------------------------
const health = await apiJson<{ status: string; version?: string }>("/clap/v1/health");
const runtime = await apiJson<Record<string, unknown>>("/clap/v1/runtime").catch(() => ({}));
console.log(`clap server healthy (${health.version ?? "unknown version"}) at ${options.baseUrl}`);

// --- Suite: load-residency -------------------------------------------------
for (const model of options.models) {
  const config = suiteConfig("load-residency");
  if (!config) break;
  const { samples } = counts(config);
  for (let index = 0; index < samples; index += 1) {
    await measuredSample("load-residency", "load", model, index, false, async () => {
      await api("/clap/v1/models/unload", { method: "POST", body: JSON.stringify({ model }) });
      const startedMs = performance.now();
      await apiJson("/clap/v1/models/load", { method: "POST",
        body: JSON.stringify({ model }) });
      const loadedMs = performance.now();
      const residency = await apiJson<{ models?: Array<Record<string, unknown>> }>(
        "/clap/v1/runtime/models").catch(() => ({}) as { models?: Array<Record<string, unknown>> });
      const entry = residency.models?.find((item) => item.id === model || item.model === model);
      return {
        timings: { ttftMs: null, totalMs: loadedMs - startedMs, decodeTokens: 0,
          decodeTps: null, finishReason: null },
        extra: { residency: entry ?? null },
      };
    });
  }
}

// --- Suite: single-generation ---------------------------------------------
for (const model of options.models) {
  const config = suiteConfig("single-generation");
  if (!config) break;
  const { warmups, samples } = counts(config);
  const cases = Array.isArray(config.cases)
    ? config.cases as string[] : ["cold", "continuation", "branch"];
  const total = warmups + samples;

  if (cases.includes("cold")) {
    for (let index = 0; index < total; index += 1) {
      // A unique leading nonce defeats prefix reuse: this measures full prefill.
      const nonce = synthesizeWords(24, promptSeed + 1_000 + index);
      const body = synthesizeWords(matrix.prompts.coldPromptWords, promptSeed + 2_000 + index);
      await measuredSample("single-generation", "cold", model, index, index < warmups,
        async () => await streamChat({
          model, maxTokens: matrix.defaults.maxTokens, timeoutMs: matrix.defaults.timeoutMs,
          session: `bench-cold-${runId}-${index}`,
          messages: [{ role: "user", content: `${nonce}\n${body}\nSummarize the theme in one sentence.` }],
        }));
    }
  }

  if (cases.includes("continuation")) {
    // One growing conversation: turn N+1 extends turn N. Exact continuation
    // should keep prefill near zero after the first turn.
    const messages: Array<{ role: string; content: string }> = [
      { role: "user", content: `${sharedPrefix}\nAnswer briefly: what is this text about?` },
    ];
    for (let index = 0; index < total; index += 1) {
      await measuredSample("single-generation", "continuation", model, index, index < warmups,
        async () => {
          const result = await streamChat({
            model, maxTokens: matrix.defaults.maxTokens, timeoutMs: matrix.defaults.timeoutMs,
            session: `bench-cont-${runId}`,
            messages,
          });
          messages.push({ role: "assistant", content: result.content || "(empty)" });
          messages.push({ role: "user", content: `Follow-up ${index + 1}: add one more short sentence.` });
          return result;
        });
    }
  }

  if (cases.includes("branch")) {
    // Distinct sessions sharing one long harness prompt; later sessions
    // should branch or anchor-restore instead of re-prefilling the shared
    // tokens. The shared prefix is its own system message with a declared
    // cache boundary so rotating-cache models can plant a prompt-boundary
    // anchor exactly at the reusable slice.
    for (let index = 0; index < total; index += 1) {
      const divergence = synthesizeWords(matrix.prompts.divergenceWords, promptSeed + 3_000 + index);
      await measuredSample("single-generation", "branch", model, index, index < warmups,
        async () => await streamChat({
          model, maxTokens: matrix.defaults.maxTokens, timeoutMs: matrix.defaults.timeoutMs,
          session: `bench-branch-${runId}-${index}`,
          boundaries: [{ kind: "messages", through_message: 0, label: "bench-harness" }],
          messages: [
            { role: "system", content: `${sharedPrefix}\nFollow the task instructions.` },
            { role: "user", content: `Task ${index}: ${divergence}\nAnswer in one sentence.` },
          ],
        }));
    }
  }
}

// --- Suite: concurrency ----------------------------------------------------
for (const model of options.models) {
  const config = suiteConfig("concurrency");
  if (!config) break;
  const { warmups, samples } = counts(config);
  const streams = typeof config.streams === "number" ? config.streams : 4;
  const maxTokens = typeof config.maxTokens === "number" ? config.maxTokens : matrix.defaults.maxTokens;
  const total = warmups + samples;
  for (let round = 0; round < total; round += 1) {
    const warmup = round < warmups;
    // Solo control first: one stream with nothing else running.
    await measuredSample("concurrency", "solo-control", model, round, warmup,
      async () => await streamChat({
        model, maxTokens, timeoutMs: matrix.defaults.timeoutMs,
        session: `bench-conc-solo-${runId}-${round}`,
        boundaries: [{ kind: "messages", through_message: 0, label: "bench-harness" }],
        messages: [
          { role: "system", content: `${sharedPrefix}\nFollow the task instructions.` },
          { role: "user", content: `Control ${round}: answer in one sentence.` },
        ],
      }));
    // Then N concurrent streams sharing the prefix with unique tails.
    const startedMs = performance.now();
    const results = await Promise.all(Array.from({ length: streams }, (_, stream) => {
      const divergence = synthesizeWords(matrix.prompts.divergenceWords,
        promptSeed + 4_000 + round * 100 + stream);
      return streamChat({
        model, maxTokens, timeoutMs: matrix.defaults.timeoutMs,
        session: `bench-conc-${runId}-${round}-${stream}`,
        priority: stream === 0 ? "interactive" : "normal",
        boundaries: [{ kind: "messages", through_message: 0, label: "bench-harness" }],
        messages: [
          { role: "system", content: `${sharedPrefix}\nFollow the task instructions.` },
          { role: "user", content: `Stream ${stream}: ${divergence}\nAnswer in one sentence.` },
        ],
      }).then((result) => ({ result, stream, error: null as string | null }))
        .catch((error) => ({
          result: null, stream,
          error: error instanceof Error ? error.message : String(error),
        }));
    }));
    const wallMs = performance.now() - startedMs;
    for (const item of results) {
      pushRecord({
        suite: "concurrency", case: `stream-${item.stream === 0 ? "interactive" : "normal"}`,
        model, sampleIndex: round, warmup, startedAt: new Date().toISOString(),
        timings: item.result?.timings ?? { ttftMs: null, totalMs: 0, decodeTokens: 0,
          decodeTps: null, finishReason: null },
        metricsDelta: null,
        error: item.error ?? undefined,
        extra: { wallMs, streams },
      });
    }
  }
}

// --- Persist ---------------------------------------------------------------
const grouped = new Map<string, SampleRecord[]>();
for (const record of records) {
  const key = `${record.suite}\u0000${record.case}\u0000${record.model}`;
  grouped.set(key, [...(grouped.get(key) ?? []), record]);
}
const summaries = [...grouped.values()]
  .map((group) => summarizeCase(group))
  .filter((summary) => summary !== null);

const meta = {
  schemaVersion: 1,
  runId,
  label: options.label,
  baseUrl: options.baseUrl,
  startedAt: records[0]?.startedAt ?? new Date().toISOString(),
  finishedAt: new Date().toISOString(),
  hostname: hostname(),
  serverVersion: health.version ?? null,
  serverRuntime: runtime,
  matrix,
  models: options.models,
  note: "Raw samples in samples.jsonl are the source of record; summary.json is derived.",
};

const sanitize = (content: string) => sanitizeBenchmarkText(content, [
  options.apiKey, process.env.CLAP_HF_TOKEN, process.env.HF_TOKEN,
]);
await writeFile(join(outDir, "meta.json"), sanitize(JSON.stringify(meta, null, 2)));
await writeFile(join(outDir, "samples.jsonl"),
  sanitize(records.map((record) => JSON.stringify(record)).join("\n") + "\n"));
await writeFile(join(outDir, "summary.json"), sanitize(JSON.stringify(summaries, null, 2)));

console.log(`\nWrote ${records.length} samples, ${summaries.length} case summaries -> ${outDir}`);
for (const summary of summaries) {
  const ttft = summary!.ttftMs ? `ttft p50=${summary!.ttftMs.p50.toFixed(0)}ms p95=${summary!.ttftMs.p95.toFixed(0)}ms` : "no ttft";
  const tps = summary!.decodeTps ? `decode p50=${summary!.decodeTps.p50.toFixed(1)}t/s` : "no decode";
  const reused = summary!.kvReusedTokens ? `reused p50=${summary!.kvReusedTokens.p50.toFixed(0)}tok` : "";
  console.log(`  ${summary!.suite}/${summary!.case} [${summary!.model}] n=${summary!.samples} fail=${summary!.failures}: ${ttft}, ${tps} ${reused}`);
}
