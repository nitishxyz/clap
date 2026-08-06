import type { PhysicalCacheAdapter } from "@clap/worker-protocol";
import { parseVllmMetrics, type ParsedVllmMetrics } from "./prometheus";
import { physicalFormatGeneration } from "./identity";
import type {
  VllmCacheTelemetry, VllmFetch, VllmReplicaCapabilities, VllmReplicaConfiguration,
} from "./types";

export class VllmAdapterError extends Error {
  constructor(message: string, readonly code: string) {
    super(message);
    this.name = "VllmAdapterError";
  }
}

export class VllmReplicaAdapter {
  readonly configuration: VllmReplicaConfiguration;
  private readonly fetcher: VllmFetch;
  private capabilitiesValue: VllmReplicaCapabilities | null = null;
  private formatFingerprint: string | null = null;
  private previous: ParsedVllmMetrics | null = null;

  constructor(configuration: VllmReplicaConfiguration, fetcher: VllmFetch = fetch) {
    if (!configuration.id || !configuration.model) {
      throw new VllmAdapterError("vLLM replica id and model are required", "invalid_configuration");
    }
    const baseUrl = new URL(configuration.baseUrl);
    if (!/^https?:$/.test(baseUrl.protocol)) {
      throw new VllmAdapterError("vLLM replica URL must use HTTP or HTTPS", "invalid_configuration");
    }
    this.configuration = { ...configuration, baseUrl: baseUrl.toString().replace(/\/$/, "") };
    this.fetcher = fetcher;
  }

  get capabilities(): VllmReplicaCapabilities | null { return this.capabilitiesValue; }

  async discover(): Promise<VllmReplicaCapabilities> {
    const health = await this.request("/health");
    if (!health.ok) throw new VllmAdapterError(
      `vLLM replica ${this.configuration.id} is unhealthy (${health.status})`, "unhealthy");
    const [versionResponse, modelsResponse, openapiResponse, metricsResponse] = await Promise.all([
      this.request("/version"), this.request("/v1/models"),
      this.request("/openapi.json"), this.request("/metrics"),
    ]);
    const versionBody = await optionalJson(versionResponse);
    const modelsBody = await optionalJson(modelsResponse);
    const openapiBody = await optionalJson(openapiResponse);
    const metricsText = metricsResponse.ok ? await metricsResponse.text() : "";
    const metrics = parseVllmMetrics(metricsText, this.configuration.model);
    const servedModels = modelIds(modelsBody);
    if (!modelsResponse.ok || !servedModels.includes(this.configuration.model)) {
      throw new VllmAdapterError(
        `vLLM replica ${this.configuration.id} does not serve ${this.configuration.model}`,
        "model_mismatch");
    }
    const supportsCacheSalt = openapiResponse.ok &&
      chatSchemaContainsProperty(openapiBody, "cache_salt");
    if (metrics.cache.enabled !== false && !supportsCacheSalt) {
      throw new VllmAdapterError(
        `vLLM replica ${this.configuration.id} enables prefix caching without cache_salt isolation`,
        "unsafe_prefix_cache");
    }
    const version = objectString(versionBody, "version");
    const nextFingerprint = physicalFormatGeneration({ version, model: this.configuration.model,
      cache: metrics.cache, supportsCacheSalt });
    const generation = this.formatFingerprint === null ? 1
      : this.formatFingerprint === nextFingerprint ? (this.capabilitiesValue?.generation ?? 1)
      : (this.capabilitiesValue?.generation ?? 1) + 1;
    if (this.formatFingerprint !== null && this.formatFingerprint !== nextFingerprint) this.previous = null;
    this.formatFingerprint = nextFingerprint;
    const adapter = physicalDescriptor(version, metrics);
    this.capabilitiesValue = {
      replicaId: this.configuration.id,
      version,
      model: this.configuration.model,
      supportsCacheSalt,
      supportsPrefixMetrics: metrics.prefixQueries !== null && metrics.prefixHits !== null,
      supportsExternalPrefixMetrics: metrics.externalPrefixQueries !== null &&
        metrics.externalPrefixHits !== null,
      cache: metrics.cache,
      adapter,
      generation,
    };
    return this.capabilitiesValue;
  }

  async scrape(): Promise<VllmCacheTelemetry> {
    const capabilities = this.capabilitiesValue ?? await this.discover();
    const response = await this.request("/metrics");
    if (!response.ok) return unavailableTelemetry(capabilities, false);
    const current = parseVllmMetrics(await response.text(), this.configuration.model);
    const prefixQueryDelta = delta(current.prefixQueries, this.previous?.prefixQueries ?? null);
    const prefixHitDelta = delta(current.prefixHits, this.previous?.prefixHits ?? null);
    const externalQueryDelta = delta(
      current.externalPrefixQueries, this.previous?.externalPrefixQueries ?? null);
    const externalHitDelta = delta(
      current.externalPrefixHits, this.previous?.externalPrefixHits ?? null);
    this.previous = current;
    return {
      replicaId: this.configuration.id,
      observedAt: new Date().toISOString(),
      healthy: true,
      generation: capabilities.generation,
      cacheUsage: current.cacheUsage,
      prefixQueries: current.prefixQueries,
      prefixHits: current.prefixHits,
      prefixQueryDelta,
      prefixHitDelta,
      prefixHitRatio: ratio(prefixHitDelta, prefixQueryDelta),
      externalPrefixQueries: current.externalPrefixQueries,
      externalPrefixHits: current.externalPrefixHits,
      externalPrefixQueryDelta: externalQueryDelta,
      externalPrefixHitDelta: externalHitDelta,
      externalPrefixHitRatio: ratio(externalHitDelta, externalQueryDelta),
      capacityTokens: current.cache.capacityTokens,
      blockTokens: current.cache.blockTokens,
    };
  }

  async dispatch(body: Record<string, unknown>, cacheSalt: string, signal?: AbortSignal): Promise<Response> {
    const capabilities = this.capabilitiesValue ?? await this.discover();
    const request: Record<string, unknown> = { ...body, model: this.configuration.model };
    delete request.cache_salt;
    if (capabilities.supportsCacheSalt) request.cache_salt = cacheSalt;
    return this.request("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(request),
      signal,
      headers: { "content-type": "application/json" },
    });
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.configuration.timeoutMs ?? 10_000);
    const signal = init.signal
      ? AbortSignal.any([init.signal, controller.signal]) : controller.signal;
    try {
      return await this.fetcher(`${this.configuration.baseUrl}${path}`, {
        ...init,
        signal,
        headers: {
          ...(this.configuration.apiKey
            ? { authorization: `Bearer ${this.configuration.apiKey}` } : {}),
          ...this.configuration.headers,
          ...init.headers,
        },
      });
    } catch (error) {
      throw new VllmAdapterError(
        `vLLM replica ${this.configuration.id} request failed: ${String(error)}`, "request_failed");
    } finally {
      clearTimeout(timeout);
    }
  }
}

function physicalDescriptor(version: string | null, metrics: ParsedVllmMetrics): PhysicalCacheAdapter | null {
  const blockTokens = metrics.cache.blockTokens;
  if (metrics.cache.enabled !== true) return null;
  const live = blockTokens !== null;
  return {
    contract_version: 1,
    kind: live ? "paged" : "paged_skeleton",
    operations: ["inspect"],
    format: {
      backend: "vllm",
      engine: version ? `vllm-${version}` : "vllm-unknown",
      cache_format: live ? "vllm-native-prefix-block" : "vllm-prefix-observation",
      cache_format_version: 1,
      kv_data_type: metrics.cache.cacheDtype,
      block_tokens: blockTokens,
    },
    constraints: {
      restore_granularity: "block",
      fork_semantics: "copy_on_write",
      minimum_trim_tokens: null,
      safe_busy_donor: false,
      prompt_boundary_snapshots: false,
      recurrent_or_hybrid: false,
      byte_accounting: "unknown",
      tiers: ["device"],
      transfer_format: null,
    },
  };
}

function unavailableTelemetry(capabilities: VllmReplicaCapabilities, healthy: boolean): VllmCacheTelemetry {
  return {
    replicaId: capabilities.replicaId, observedAt: new Date().toISOString(), healthy,
    generation: capabilities.generation, cacheUsage: null, prefixQueries: null, prefixHits: null,
    prefixQueryDelta: null, prefixHitDelta: null, prefixHitRatio: null,
    externalPrefixQueries: null, externalPrefixHits: null, externalPrefixQueryDelta: null,
    externalPrefixHitDelta: null, externalPrefixHitRatio: null,
    capacityTokens: capabilities.cache.capacityTokens, blockTokens: capabilities.cache.blockTokens,
  };
}

function delta(current: number | null, previous: number | null): number | null {
  return current !== null && previous !== null && current >= previous ? current - previous : null;
}

function ratio(numerator: number | null, denominator: number | null): number | null {
  return numerator !== null && denominator !== null && denominator > 0
    ? Math.min(1, numerator / denominator) : null;
}

async function optionalJson(response: Response): Promise<unknown> {
  if (!response.ok) return null;
  try { return await response.json(); } catch { return null; }
}

function objectString(value: unknown, key: string): string | null {
  if (!value || typeof value !== "object") return null;
  const entry = (value as Record<string, unknown>)[key];
  return typeof entry === "string" && entry ? entry : null;
}

function modelIds(value: unknown): string[] {
  if (!value || typeof value !== "object") return [];
  const data = (value as Record<string, unknown>).data;
  if (!Array.isArray(data)) return [];
  return data.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const id = (entry as Record<string, unknown>).id;
    return typeof id === "string" ? [id] : [];
  });
}

function chatSchemaContainsProperty(document: unknown, property: string): boolean {
  if (!document || typeof document !== "object") return false;
  const root = document as Record<string, unknown>;
  const paths = root.paths;
  if (!paths || typeof paths !== "object") return false;
  const chat = (paths as Record<string, unknown>)["/v1/chat/completions"];
  if (!chat || typeof chat !== "object") return false;
  const post = (chat as Record<string, unknown>).post;
  if (!post || typeof post !== "object") return false;
  const requestBody = (post as Record<string, unknown>).requestBody;
  if (!requestBody || typeof requestBody !== "object") return false;
  const content = (requestBody as Record<string, unknown>).content;
  if (!content || typeof content !== "object") return false;
  const json = (content as Record<string, unknown>)["application/json"];
  if (!json || typeof json !== "object") return false;
  return schemaContainsProperty(root, (json as Record<string, unknown>).schema, property);
}

function schemaContainsProperty(root: Record<string, unknown>, value: unknown, property: string,
                                seen = new Set<object>()): boolean {
  if (!value || typeof value !== "object" || seen.has(value)) return false;
  seen.add(value);
  if (!Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    if (typeof record.$ref === "string" && record.$ref.startsWith("#/")) {
      const resolved = record.$ref.slice(2).split("/").reduce<unknown>((current, segment) => {
        if (!current || typeof current !== "object") return null;
        return (current as Record<string, unknown>)[segment.replace(/~1/g, "/").replace(/~0/g, "~")];
      }, root);
      if (schemaContainsProperty(root, resolved, property, seen)) return true;
    }
    if (record.properties && typeof record.properties === "object" &&
        property in (record.properties as Record<string, unknown>)) return true;
    return Object.values(record).some((entry) => schemaContainsProperty(root, entry, property, seen));
  }
  return value.some((entry) => schemaContainsProperty(root, entry, property, seen));
}
