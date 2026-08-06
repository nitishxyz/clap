import { describe, expect, test } from "bun:test";
import {
  deriveVllmCacheSalt, parsePrometheus, parseVllmMetrics,
  VllmReplicaAdapter, VllmReplicaPool,
} from "./index";

const secret = "0123456789abcdef0123456789abcdef";
const prefix = "ab".repeat(32);

const metrics = (usage: number, queries: number, hits: number, externalHits = 0) => `
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{model_name="model-a",enable_prefix_caching="True",prefix_caching_hash_algo="sha256_cbor",block_size="16",num_gpu_blocks="100",num_cpu_blocks="0",kv_cache_size_tokens="1600",cache_dtype="auto"} 1
vllm:kv_cache_usage_perc{model_name="model-a",engine="0"} ${usage}
vllm:prefix_cache_queries_total{model_name="model-a",engine="0"} ${queries}
vllm:prefix_cache_hits_total{model_name="model-a",engine="0"} ${hits}
vllm:external_prefix_cache_queries_total{model_name="model-a",engine="0"} ${queries}
vllm:external_prefix_cache_hits_total{model_name="model-a",engine="0"} ${externalHits}
`;

const openapi = {
  paths: { "/v1/chat/completions": { post: { requestBody: { content: {
    "application/json": { schema: { $ref: "#/components/schemas/ChatCompletionRequest" } },
  } } } } },
  components: { schemas: { ChatCompletionRequest: {
    type: "object", properties: { model: { type: "string" }, cache_salt: { type: "string" } },
  } } },
};

function replicaFetch(state: {
  versions?: Record<string, string>;
  metricsByReplica?: Record<string, string[]>;
  requests?: Array<{ replica: string; body: Record<string, unknown> }>;
  cacheSalt?: boolean;
}) {
  const scrapes = new Map<string, number>();
  return (async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url);
    const replica = url.hostname.split(".")[0]!;
    if (url.pathname === "/health") return new Response("", { status: 200 });
    if (url.pathname === "/version") return Response.json({ version: state.versions?.[replica] ?? "0.26.0" });
    if (url.pathname === "/v1/models") return Response.json({ data: [{ id: "model-a" }] });
    if (url.pathname === "/openapi.json") return Response.json(state.cacheSalt === false ?
      { components: { schemas: { Request: { properties: { model: {} } } } } } : openapi);
    if (url.pathname === "/metrics") {
      const values = state.metricsByReplica?.[replica] ?? [metrics(0.25, 100, 50)];
      const index = scrapes.get(replica) ?? 0;
      scrapes.set(replica, index + 1);
      return new Response(values[Math.min(index, values.length - 1)]);
    }
    if (url.pathname === "/v1/chat/completions") {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      state.requests?.push({ replica, body });
      return Response.json({ id: "chatcmpl-test", choices: [] });
    }
    return new Response("not found", { status: 404 });
  }) as typeof fetch;
}

describe("vLLM native prefix adapter", () => {
  test("derives opaque stable salts scoped by deployment, model, and share identity", () => {
    const first = deriveVllmCacheSalt({ secret, deployment: "prod", model: "model-a", shareIdentity: "tenant-a" });
    const repeat = deriveVllmCacheSalt({ secret, deployment: "prod", model: "model-a", shareIdentity: "tenant-a" });
    const other = deriveVllmCacheSalt({ secret, deployment: "prod", model: "model-a", shareIdentity: "tenant-b" });
    expect(first).toBe(repeat);
    expect(first).not.toBe(other);
    expect(first).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(first).not.toContain("tenant-a");
  });

  test("parses current and legacy-compatible Prometheus names with exact config", () => {
    const samples = parsePrometheus(metrics(0.75, 200, 150, 25));
    expect(samples.some((sample) => sample.name === "vllm:prefix_cache_hits_total")).toBe(true);
    const parsed = parseVllmMetrics(metrics(0.75, 200, 150, 25), "model-a");
    expect(parsed.cache).toMatchObject({ enabled: true, hashAlgorithm: "sha256_cbor",
      blockTokens: 16, gpuBlocks: 100, capacityTokens: 1600, cacheDtype: "auto" });
    expect(parsed.cacheUsage).toBe(0.75);
    expect(parsed.prefixHits).toBe(150);
    expect(parsed.externalPrefixHits).toBe(25);
  });

  test("discovers a paged descriptor and computes restart-safe counter deltas", async () => {
    const adapter = new VllmReplicaAdapter({ id: "r1", baseUrl: "http://r1.test",
      model: "model-a" }, replicaFetch({ metricsByReplica: { r1: [
        metrics(0.2, 100, 40), metrics(0.2, 100, 40),
        metrics(0.3, 140, 70), metrics(0.1, 5, 2),
      ] } }));
    const capabilities = await adapter.discover();
    expect(capabilities.adapter).toMatchObject({ kind: "paged", operations: ["inspect"],
      format: { block_tokens: 16 }, constraints: { byte_accounting: "unknown" } });
    const baseline = await adapter.scrape();
    expect(baseline.prefixHitDelta).toBeNull();
    const observed = await adapter.scrape();
    expect(observed.prefixHitDelta).toBe(30);
    expect(observed.prefixQueryDelta).toBe(40);
    expect(observed.prefixHitRatio).toBe(0.75);
    const reset = await adapter.scrape();
    expect(reset.prefixHitDelta).toBeNull();
    expect(reset.prefixQueryDelta).toBeNull();
  });

  test("rejects prefix caching that lacks cache_salt isolation", async () => {
    const adapter = new VllmReplicaAdapter({ id: "unsafe", baseUrl: "http://unsafe.test",
      model: "model-a" }, replicaFetch({ cacheSalt: false }));
    await expect(adapter.discover()).rejects.toMatchObject({ code: "unsafe_prefix_cache" });
  });

  test("routes repeated prefixes by locality and overwrites caller cache salts", async () => {
    const requests: Array<{ replica: string; body: Record<string, unknown> }> = [];
    const fetcher = replicaFetch({ requests, metricsByReplica: {
      r1: [metrics(0.2, 100, 50), metrics(0.2, 120, 65)],
      r2: [metrics(0.8, 100, 50), metrics(0.8, 120, 55)],
    } });
    const pool = new VllmReplicaPool({ deployment: "prod", cacheSaltSecret: secret, replicas: [
      { id: "r1", baseUrl: "http://r1.test", model: "model-a" },
      { id: "r2", baseUrl: "http://r2.test", model: "model-a" },
    ] }, fetcher);
    await pool.discover();
    await pool.scrape();
    const route = { shareIdentity: "tenant-a", prefixFingerprint: prefix };
    const first = pool.route(route);
    expect(pool.route(route).replicaId).toBe(first.replicaId);
    const dispatched = await pool.chat({ messages: [], cache_salt: "caller-controlled" }, route);
    expect(dispatched.response.ok).toBe(true);
    expect(requests).toHaveLength(1);
    expect(requests[0]!.replica).toBe(first.replicaId);
    expect(requests[0]!.body.cache_salt).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(requests[0]!.body.cache_salt).not.toBe("caller-controlled");
  });
});
