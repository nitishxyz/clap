import { deriveVllmCacheSalt, rendezvousScore } from "./identity";
import { VllmAdapterError, VllmReplicaAdapter } from "./adapter";
import type {
  VllmCacheTelemetry, VllmChatDispatch, VllmFetch, VllmReplicaConfiguration,
  VllmReplicaCapabilities, VllmRouteDecision, VllmRouteRequest,
} from "./types";

export type VllmReplicaPoolConfiguration = {
  deployment: string;
  cacheSaltSecret: Uint8Array | string;
  replicas: VllmReplicaConfiguration[];
};

export class VllmReplicaPool {
  private readonly adapters: VllmReplicaAdapter[];
  private readonly telemetry = new Map<string, VllmCacheTelemetry>();
  private readonly deployment: string;
  private readonly secret: Uint8Array | string;

  constructor(configuration: VllmReplicaPoolConfiguration, fetcher: VllmFetch = fetch) {
    if (!configuration.deployment || configuration.replicas.length === 0) {
      throw new VllmAdapterError("vLLM deployment and at least one replica are required",
        "invalid_configuration");
    }
    if (new Set(configuration.replicas.map((replica) => replica.id)).size !==
        configuration.replicas.length) {
      throw new VllmAdapterError("vLLM replica ids must be unique", "invalid_configuration");
    }
    this.deployment = configuration.deployment;
    this.secret = configuration.cacheSaltSecret;
    deriveVllmCacheSalt({ secret: this.secret, deployment: this.deployment,
      model: configuration.replicas[0]!.model, shareIdentity: "configuration-check" });
    this.adapters = configuration.replicas.map((replica) => new VllmReplicaAdapter(replica, fetcher));
  }

  async discover(): Promise<void> {
    const results = await Promise.allSettled(this.adapters.map((adapter) => adapter.discover()));
    if (results.every((result) => result.status === "rejected")) {
      const reasons = results.map((result) => result.status === "rejected" ? String(result.reason) : "")
        .filter(Boolean).join("; ");
      throw new VllmAdapterError(`no compatible vLLM replicas: ${reasons}`, "no_compatible_replicas");
    }
  }

  async scrape(): Promise<VllmCacheTelemetry[]> {
    const results = await Promise.allSettled(this.adapters.map((adapter) => adapter.scrape()));
    return results.flatMap((result, index) => {
      if (result.status === "fulfilled") {
        this.telemetry.set(result.value.replicaId, result.value);
        return [result.value];
      }
      const adapter = this.adapters[index]!;
      const capability = adapter.capabilities;
      if (!capability) return [];
      const unavailable: VllmCacheTelemetry = {
        replicaId: capability.replicaId, observedAt: new Date().toISOString(), healthy: false,
        generation: capability.generation, cacheUsage: null, prefixQueries: null, prefixHits: null,
        prefixQueryDelta: null, prefixHitDelta: null, prefixHitRatio: null,
        externalPrefixQueries: null, externalPrefixHits: null, externalPrefixQueryDelta: null,
        externalPrefixHitDelta: null, externalPrefixHitRatio: null,
        capacityTokens: capability.cache.capacityTokens, blockTokens: capability.cache.blockTokens,
      };
      this.telemetry.set(unavailable.replicaId, unavailable);
      return [unavailable];
    });
  }

  route(request: VllmRouteRequest): VllmRouteDecision {
    validateRouteRequest(request);
    const candidates = this.adapters.flatMap((adapter) => {
      const capabilities = adapter.capabilities;
      const telemetry = this.telemetry.get(adapter.configuration.id);
      if (!capabilities || telemetry?.healthy === false) return [];
      const affinity = rendezvousScore(this.secret,
        `${this.deployment}\0${capabilities.model}\0${request.shareIdentity}\0${request.prefixFingerprint}`,
        adapter.configuration.id);
      const pressure = telemetry?.cacheUsage ?? 0.5;
      const localHits = telemetry?.prefixHitRatio ?? 0;
      const externalHits = telemetry?.externalPrefixHitRatio ?? 0;
      const external = capabilities.supportsExternalPrefixMetrics && externalHits > 0;
      const affinityWeight = external ? 250 : 1_000;
      const score = affinity * affinityWeight - pressure * 400 + localHits * 120 + externalHits * 80;
      return [{ adapter, telemetry, score, affinity, external }];
    });
    if (candidates.length === 0) {
      throw new VllmAdapterError("no healthy discovered vLLM replicas", "no_healthy_replicas");
    }
    candidates.sort((left, right) => right.score - left.score ||
      left.adapter.configuration.id.localeCompare(right.adapter.configuration.id));
    const selected = candidates[0]!;
    const reason = candidates.length === 1 ? "only_healthy"
      : selected.external ? "external_cache"
      : selected.affinity >= 0.5 ? "prefix_affinity" : "lowest_pressure";
    return {
      replicaId: selected.adapter.configuration.id,
      reason,
      score: selected.score,
      cacheUsage: selected.telemetry?.cacheUsage ?? null,
      prefixHitRatio: selected.telemetry?.prefixHitRatio ?? null,
      externalPrefixHitRatio: selected.telemetry?.externalPrefixHitRatio ?? null,
    };
  }

  async chat(body: Record<string, unknown>, route: VllmRouteRequest,
             signal?: AbortSignal): Promise<VllmChatDispatch> {
    const decision = this.route(route);
    const adapter = this.adapters.find((candidate) => candidate.configuration.id === decision.replicaId)!;
    const cacheSalt = deriveVllmCacheSalt({
      secret: this.secret,
      deployment: this.deployment,
      model: adapter.configuration.model,
      shareIdentity: route.shareIdentity,
    });
    return { decision, response: await adapter.dispatch(body, cacheSalt, signal) };
  }

  snapshot(): { replicas: Array<{
    capabilities: VllmReplicaCapabilities | null;
    telemetry: VllmCacheTelemetry | null;
  }> } {
    return { replicas: this.adapters.map((adapter) => ({
      capabilities: adapter.capabilities,
      telemetry: this.telemetry.get(adapter.configuration.id) ?? null,
    })) };
  }
}

function validateRouteRequest(request: VllmRouteRequest): void {
  if (!request.shareIdentity || request.shareIdentity.includes("\0")) {
    throw new VllmAdapterError("share identity must be non-empty and contain no NUL bytes",
      "invalid_route_request");
  }
  if (!/^[0-9a-f]{64}$/.test(request.prefixFingerprint)) {
    throw new VllmAdapterError("prefix fingerprint must be 32-byte lowercase hexadecimal",
      "invalid_route_request");
  }
}
