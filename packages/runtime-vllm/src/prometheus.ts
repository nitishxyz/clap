import type { VllmCacheConfiguration } from "./types";

export type PrometheusSample = {
  name: string;
  labels: Record<string, string>;
  value: number;
};

export type ParsedVllmMetrics = {
  cache: VllmCacheConfiguration;
  cacheUsage: number | null;
  prefixQueries: number | null;
  prefixHits: number | null;
  externalPrefixQueries: number | null;
  externalPrefixHits: number | null;
};

const names = {
  usage: ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
  queries: ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries",
    "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries"],
  hits: ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits",
    "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits"],
  externalQueries: ["vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_queries"],
  externalHits: ["vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_hits"],
};

export function parsePrometheus(text: string): PrometheusSample[] {
  const samples: PrometheusSample[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const match = /^([^\s{]+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$/.exec(line);
    if (!match) continue;
    const value = Number(match[3]);
    if (!Number.isFinite(value)) continue;
    samples.push({ name: match[1]!, labels: parseLabels(match[2] ?? ""), value });
  }
  return samples;
}

export function parseVllmMetrics(text: string, model: string): ParsedVllmMetrics {
  const samples = parsePrometheus(text);
  const select = (accepted: string[]): PrometheusSample | undefined => samples.find((sample) =>
    accepted.includes(sample.name) && (!sample.labels.model_name || sample.labels.model_name === model));
  const config = select(["vllm:cache_config_info"]);
  const blockTokens = positiveInteger(config?.labels.block_size);
  const gpuBlocks = nonnegativeInteger(config?.labels.num_gpu_blocks);
  const cpuBlocks = nonnegativeInteger(config?.labels.num_cpu_blocks);
  const explicitCapacity = positiveInteger(config?.labels.kv_cache_size_tokens);
  const derivedCapacity = blockTokens !== null && gpuBlocks !== null
    ? safeProduct(blockTokens, gpuBlocks) : null;
  return {
    cache: {
      enabled: parseBoolean(config?.labels.enable_prefix_caching),
      hashAlgorithm: stringValue(config?.labels.prefix_caching_hash_algo),
      blockTokens,
      gpuBlocks,
      cpuBlocks,
      capacityTokens: explicitCapacity ?? derivedCapacity,
      cacheDtype: stringValue(config?.labels.cache_dtype),
    },
    cacheUsage: boundedFraction(select(names.usage)?.value),
    prefixQueries: counter(select(names.queries)?.value),
    prefixHits: counter(select(names.hits)?.value),
    externalPrefixQueries: counter(select(names.externalQueries)?.value),
    externalPrefixHits: counter(select(names.externalHits)?.value),
  };
}

function parseLabels(value: string): Record<string, string> {
  const labels: Record<string, string> = {};
  let offset = 0;
  while (offset < value.length) {
    const match = /\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"])*)"\s*(?:,|$)/y;
    match.lastIndex = offset;
    const parsed = match.exec(value);
    if (!parsed) break;
    labels[parsed[1]!] = parsed[2]!
      .replace(/\\n/g, "\n").replace(/\\"/g, "\"").replace(/\\\\/g, "\\");
    offset = match.lastIndex;
  }
  return labels;
}

function parseBoolean(value: string | undefined): boolean | null {
  if (value === undefined) return null;
  if (/^(true|1)$/i.test(value)) return true;
  if (/^(false|0)$/i.test(value)) return false;
  return null;
}

function stringValue(value: string | undefined): string | null {
  return value && value !== "None" && value !== "null" ? value : null;
}

function positiveInteger(value: string | undefined): number | null {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}

function nonnegativeInteger(value: string | undefined): number | null {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null;
}

function safeProduct(left: number, right: number): number | null {
  const result = left * right;
  return Number.isSafeInteger(result) && result >= 0 ? result : null;
}

function boundedFraction(value: number | undefined): number | null {
  return value !== undefined && value >= 0 && value <= 1 ? value : null;
}

function counter(value: number | undefined): number | null {
  return value !== undefined && value >= 0 ? value : null;
}
