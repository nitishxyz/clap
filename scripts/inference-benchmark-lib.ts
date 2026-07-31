// Pure helpers for the inference benchmark runner. Kept import-safe (no side
// effects) so statistics, prompt synthesis, SSE accounting, and sanitization
// are unit-testable without a running server.

export type SampleTimings = {
  ttftMs: number | null;
  totalMs: number;
  decodeTokens: number;
  decodeTps: number | null;
  finishReason: string | null;
};

export type MetricsCounters = {
  promptTokens: number;
  completionTokens: number;
  kvReusedTokens: number;
  cacheHits: number;
  cacheMisses: number;
};

export type SampleRecord = {
  suite: string;
  case: string;
  model: string;
  sampleIndex: number;
  warmup: boolean;
  startedAt: string;
  timings: SampleTimings;
  metricsDelta: MetricsCounters | null;
  error?: string;
  extra?: Record<string, unknown>;
};

export type CaseSummary = {
  suite: string;
  case: string;
  model: string;
  samples: number;
  failures: number;
  ttftMs: Distribution | null;
  totalMs: Distribution | null;
  decodeTps: Distribution | null;
  kvReusedTokens: Distribution | null;
  promptTokens: Distribution | null;
};

export type Distribution = {
  count: number;
  mean: number;
  min: number;
  max: number;
  p50: number;
  p95: number;
  p99: number;
  stddev: number;
};

// Nearest-rank percentile on a sorted copy; deterministic and simple. Raw
// samples are preserved in JSONL so summaries can always be recomputed.
export function percentile(values: number[], p: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[rank]!;
}

export function summarize(values: number[]): Distribution | null {
  const usable = values.filter((value) => Number.isFinite(value));
  if (usable.length === 0) return null;
  const mean = usable.reduce((a, b) => a + b, 0) / usable.length;
  const variance = usable.reduce((a, b) => a + (b - mean) ** 2, 0) / usable.length;
  return {
    count: usable.length,
    mean,
    min: Math.min(...usable),
    max: Math.max(...usable),
    p50: percentile(usable, 50),
    p95: percentile(usable, 95),
    p99: percentile(usable, 99),
    stddev: Math.sqrt(variance),
  };
}

export function summarizeCase(records: SampleRecord[]): CaseSummary | null {
  const measured = records.filter((record) => !record.warmup);
  if (measured.length === 0) return null;
  const first = measured[0]!;
  const ok = measured.filter((record) => !record.error);
  return {
    suite: first.suite,
    case: first.case,
    model: first.model,
    samples: measured.length,
    failures: measured.length - ok.length,
    ttftMs: summarize(ok.map((r) => r.timings.ttftMs ?? Number.NaN)),
    totalMs: summarize(ok.map((r) => r.timings.totalMs)),
    decodeTps: summarize(ok.map((r) => r.timings.decodeTps ?? Number.NaN)),
    kvReusedTokens: summarize(ok.map((r) => r.metricsDelta?.kvReusedTokens ?? Number.NaN)),
    promptTokens: summarize(ok.map((r) => r.metricsDelta?.promptTokens ?? Number.NaN)),
  };
}

// Deterministic xorshift32 PRNG: benchmark prompts must be redistributable
// and reproducible across runs and machines.
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0 || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0xffffffff;
  };
}

const WORDS = [
  "system", "policy", "review", "agent", "harness", "token", "budget", "audit",
  "channel", "vector", "matrix", "kernel", "stream", "buffer", "socket", "ledger",
  "schema", "branch", "anchor", "prefix", "session", "project", "quantum", "cache",
  "window", "context", "decode", "sample", "batch", "metric", "report", "trace",
];

// Synthesizes deterministic filler text. Word count only approximates token
// count; real prompt tokens are read from server metrics deltas, never from
// this approximation.
export function synthesizeWords(count: number, seed: number): string {
  const random = seededRandom(seed);
  const parts: string[] = [];
  for (let index = 0; index < count; index += 1) {
    parts.push(WORDS[Math.floor(random() * WORDS.length)]!);
    if (index % 13 === 12) parts.push(`item-${index}`);
  }
  return parts.join(" ");
}

// Parses Prometheus text output for the counters the runner tracks.
export function parseMetricsCounters(text: string): MetricsCounters {
  const read = (pattern: RegExp): number => {
    const match = text.match(pattern);
    return match ? Number(match[1]) : 0;
  };
  return {
    promptTokens: read(/^clap_tokens_total\{kind="prompt"\} ([\d.eE+-]+)$/m),
    completionTokens: read(/^clap_tokens_total\{kind="completion"\} ([\d.eE+-]+)$/m),
    kvReusedTokens: read(/^clap_tokens_total\{kind="kv_reused"\} ([\d.eE+-]+)$/m),
    cacheHits: read(/^clap_kv_cache_total\{outcome="hit"\} ([\d.eE+-]+)$/m),
    cacheMisses: read(/^clap_kv_cache_total\{outcome="miss"\} ([\d.eE+-]+)$/m),
  };
}

export function metricsDelta(before: MetricsCounters, after: MetricsCounters): MetricsCounters {
  return {
    promptTokens: after.promptTokens - before.promptTokens,
    completionTokens: after.completionTokens - before.completionTokens,
    kvReusedTokens: after.kvReusedTokens - before.kvReusedTokens,
    cacheHits: after.cacheHits - before.cacheHits,
    cacheMisses: after.cacheMisses - before.cacheMisses,
  };
}

export type StreamAccounting = {
  firstDeltaAtMs: number | null;
  lastDeltaAtMs: number | null;
  deltaCount: number;
  finishReason: string | null;
  content: string;
  // SSE transports errors inside a 200 stream; a silent zero-token "success"
  // must never be recorded as a measurement.
  error: string | null;
};

export function emptyStreamAccounting(): StreamAccounting {
  return { firstDeltaAtMs: null, lastDeltaAtMs: null, deltaCount: 0,
    finishReason: null, content: "", error: null };
}

// Feeds one OpenAI chat.completion.chunk SSE data payload into the running
// accounting. nowMs is injected for determinism in tests.
export function accountChunk(accounting: StreamAccounting, payload: string,
                             nowMs: number): void {
  if (payload === "[DONE]") return;
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return;
  }
  const envelope = parsed as { error?: { message?: string } | string;
    choices?: Array<{ delta?: { content?: string; reasoning?: string };
    finish_reason?: string | null }> };
  if (envelope.error) {
    accounting.error = typeof envelope.error === "string"
      ? envelope.error : envelope.error.message ?? "stream error";
    return;
  }
  const choice = envelope.choices?.[0];
  if (!choice) return;
  const text = choice.delta?.content ?? choice.delta?.reasoning ?? "";
  if (text.length > 0) {
    if (accounting.firstDeltaAtMs === null) accounting.firstDeltaAtMs = nowMs;
    accounting.lastDeltaAtMs = nowMs;
    accounting.deltaCount += 1;
    if (choice.delta?.content) accounting.content += choice.delta.content;
  }
  if (choice.finish_reason) accounting.finishReason = choice.finish_reason;
}

export function timingsFrom(accounting: StreamAccounting, startedMs: number,
                            finishedMs: number): SampleTimings {
  const ttftMs = accounting.firstDeltaAtMs === null
    ? null : accounting.firstDeltaAtMs - startedMs;
  const decodeSpanMs = accounting.firstDeltaAtMs !== null &&
    accounting.lastDeltaAtMs !== null
    ? accounting.lastDeltaAtMs - accounting.firstDeltaAtMs : null;
  const decodeTps = decodeSpanMs !== null && decodeSpanMs > 0 &&
    accounting.deltaCount > 1
    ? ((accounting.deltaCount - 1) * 1000) / decodeSpanMs : null;
  return {
    ttftMs,
    totalMs: finishedMs - startedMs,
    decodeTokens: accounting.deltaCount,
    decodeTps,
    finishReason: accounting.finishReason,
  };
}

// Sanitizes benchmark artifacts before sharing: strips home/temp paths and
// credential-shaped values while keeping measurements intact.
export function sanitizeBenchmarkText(content: string, extraSecrets: Array<string | undefined> = []): string {
  let sanitized = content;
  for (const secret of extraSecrets) {
    if (secret) sanitized = sanitized.replaceAll(secret, "<redacted>");
  }
  return sanitized
    .replace(/\bhf_[A-Za-z0-9]{8,}\b/g, "<redacted>")
    .replace(/\bclap_sk_[A-Za-z0-9_-]{8,}\b/g, "<redacted>")
    .replace(/(?:\/Users|\/home|\/Volumes|\/private|\/tmp)\/[^\s:"'\\,}\]]+/g, "<path>")
    .replace(/((?:token|secret|password|credential|api[_-]?key)"?\s*[=:]\s*"?)[^\s",}]+/gi, "$1<redacted>");
}
