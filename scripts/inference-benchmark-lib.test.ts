import { describe, expect, test } from "bun:test";
import {
  accountChunk, emptyStreamAccounting, metricsDelta, parseMetricsCounters,
  percentile, sanitizeBenchmarkText, seededRandom, summarize, summarizeCase,
  synthesizeWords, timingsFrom, type SampleRecord,
} from "./inference-benchmark-lib";

describe("benchmark statistics", () => {
  test("percentiles use nearest rank on unsorted input", () => {
    const values = [900, 100, 500, 300, 700];
    expect(percentile(values, 50)).toBe(500);
    expect(percentile(values, 95)).toBe(900);
    expect(percentile([42], 99)).toBe(42);
    expect(Number.isNaN(percentile([], 50))).toBe(true);
  });

  test("summarize reports distribution and drops non-finite values", () => {
    const summary = summarize([10, 20, 30, Number.NaN]);
    expect(summary).not.toBeNull();
    expect(summary!.count).toBe(3);
    expect(summary!.mean).toBe(20);
    expect(summary!.min).toBe(10);
    expect(summary!.max).toBe(30);
    expect(summarize([Number.NaN])).toBeNull();
  });

  test("case summaries keep failures in the denominator", () => {
    const base: Omit<SampleRecord, "sampleIndex" | "error"> = {
      suite: "single-generation", case: "cold", model: "m",
      warmup: false, startedAt: "t",
      timings: { ttftMs: 100, totalMs: 200, decodeTokens: 10, decodeTps: 50, finishReason: "stop" },
      metricsDelta: { promptTokens: 700, completionTokens: 10, kvReusedTokens: 0, cacheHits: 0, cacheMisses: 1 },
    };
    const summary = summarizeCase([
      { ...base, sampleIndex: 0 },
      { ...base, sampleIndex: 1, error: "boom" },
      { ...base, sampleIndex: 2, warmup: true },
    ]);
    expect(summary!.samples).toBe(2);
    expect(summary!.failures).toBe(1);
    expect(summary!.ttftMs!.count).toBe(1);
  });
});

describe("deterministic prompts", () => {
  test("seeded random and prompts are reproducible", () => {
    const a = seededRandom(7);
    const b = seededRandom(7);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
    expect(synthesizeWords(50, 123)).toBe(synthesizeWords(50, 123));
    expect(synthesizeWords(50, 123)).not.toBe(synthesizeWords(50, 124));
    expect(synthesizeWords(50, 123).split(" ").length).toBeGreaterThanOrEqual(50);
  });
});

describe("stream accounting", () => {
  const chunk = (content: string | null, finish: string | null = null) =>
    JSON.stringify({ choices: [{ delta: content !== null ? { content } : {},
      finish_reason: finish }] });

  test("ttft and decode tps come from delta timestamps", () => {
    const accounting = emptyStreamAccounting();
    accountChunk(accounting, chunk("a"), 1_100);
    accountChunk(accounting, chunk("b"), 1_150);
    accountChunk(accounting, chunk("c"), 1_200);
    accountChunk(accounting, chunk(null, "stop"), 1_210);
    accountChunk(accounting, "[DONE]", 1_220);
    const timings = timingsFrom(accounting, 1_000, 1_250);
    expect(timings.ttftMs).toBe(100);
    expect(timings.decodeTokens).toBe(3);
    expect(timings.decodeTps).toBe(20);
    expect(timings.totalMs).toBe(250);
    expect(timings.finishReason).toBe("stop");
  });

  test("SSE error payloads are surfaced, never silent zero-token successes", () => {
    const accounting = emptyStreamAccounting();
    accountChunk(accounting,
      JSON.stringify({ error: { message: "prompt exceeds context window" } }), 1_000);
    expect(accounting.error).toBe("prompt exceeds context window");
    const stringError = emptyStreamAccounting();
    accountChunk(stringError, JSON.stringify({ error: "overloaded" }), 1_000);
    expect(stringError.error).toBe("overloaded");
  });

  test("malformed payloads and empty deltas are ignored", () => {
    const accounting = emptyStreamAccounting();
    accountChunk(accounting, "not json", 1_000);
    accountChunk(accounting, chunk(""), 1_010);
    expect(accounting.deltaCount).toBe(0);
    const timings = timingsFrom(accounting, 900, 1_020);
    expect(timings.ttftMs).toBeNull();
    expect(timings.decodeTps).toBeNull();
  });
});

describe("prometheus parsing", () => {
  const text = [
    'clap_tokens_total{kind="prompt"} 1200',
    'clap_tokens_total{kind="completion"} 340',
    'clap_tokens_total{kind="kv_reused"} 900',
    'clap_kv_cache_total{outcome="hit"} 3',
    'clap_kv_cache_total{outcome="miss"} 2',
  ].join("\n");

  test("reads token and cache counters and computes deltas", () => {
    const before = parseMetricsCounters(text);
    expect(before).toEqual({ promptTokens: 1200, completionTokens: 340,
      kvReusedTokens: 900, cacheHits: 3, cacheMisses: 2 });
    const after = parseMetricsCounters(text.replaceAll("1200", "1500")
      .replaceAll('kv_reused"} 900', 'kv_reused"} 1100'));
    expect(metricsDelta(before, after)).toEqual({ promptTokens: 300,
      completionTokens: 0, kvReusedTokens: 200, cacheHits: 0, cacheMisses: 0 });
  });

  test("missing series read as zero instead of failing", () => {
    expect(parseMetricsCounters("")).toEqual({ promptTokens: 0,
      completionTokens: 0, kvReusedTokens: 0, cacheHits: 0, cacheMisses: 0 });
  });
});

describe("benchmark sanitizer", () => {
  test("strips paths, tokens, keys, and explicit secrets", () => {
    const raw = JSON.stringify({
      path: "/Users/someone/.clap/models/huggingface/x",
      linux: "/home/user/models/y",
      hf: "hf_AbCdEf123456789",
      key: "clap_sk_0123456789abcdef",
      api_key: "plain-secret-value",
      ttftMs: 123.4,
    });
    const sanitized = sanitizeBenchmarkText(raw, ["explicit-secret"]);
    expect(sanitized).not.toContain("/Users/someone");
    expect(sanitized).not.toContain("/home/user");
    expect(sanitized).not.toContain("hf_AbCdEf");
    expect(sanitized).not.toContain("clap_sk_0123456789abcdef");
    expect(sanitized).not.toContain("plain-secret-value");
    expect(sanitized).toContain("123.4");
    expect(sanitizeBenchmarkText("has explicit-secret inside", ["explicit-secret"]))
      .not.toContain("explicit-secret");
  });

  test("keeps JSON parseable when paths precede escaped quotes", () => {
    const record = JSON.stringify({
      error: 'worker not ready. See /Users/x/.clap/logs/w.stderr.log',
      nested: JSON.stringify({ message: "See /tmp/x/launch.log", code: "backend_error" }),
    });
    const sanitized = sanitizeBenchmarkText(record);
    const parsed = JSON.parse(sanitized) as { error: string; nested: string };
    expect(parsed.error).toContain("<path>");
    expect(JSON.parse(parsed.nested).code).toBe("backend_error");
  });
});
