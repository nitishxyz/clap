import type { PhysicalCacheAdapter } from "@clap/worker-protocol";

export type VllmReplicaConfiguration = {
  id: string;
  baseUrl: string;
  model: string;
  apiKey?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
};

export type VllmCacheConfiguration = {
  enabled: boolean | null;
  hashAlgorithm: string | null;
  blockTokens: number | null;
  gpuBlocks: number | null;
  cpuBlocks: number | null;
  capacityTokens: number | null;
  cacheDtype: string | null;
};

export type VllmReplicaCapabilities = {
  replicaId: string;
  version: string | null;
  model: string;
  supportsCacheSalt: boolean;
  supportsPrefixMetrics: boolean;
  supportsExternalPrefixMetrics: boolean;
  cache: VllmCacheConfiguration;
  adapter: PhysicalCacheAdapter | null;
  generation: number;
};

export type VllmCacheTelemetry = {
  replicaId: string;
  observedAt: string;
  healthy: boolean;
  generation: number;
  cacheUsage: number | null;
  prefixQueries: number | null;
  prefixHits: number | null;
  prefixQueryDelta: number | null;
  prefixHitDelta: number | null;
  prefixHitRatio: number | null;
  externalPrefixQueries: number | null;
  externalPrefixHits: number | null;
  externalPrefixQueryDelta: number | null;
  externalPrefixHitDelta: number | null;
  externalPrefixHitRatio: number | null;
  capacityTokens: number | null;
  blockTokens: number | null;
};

export type VllmRouteRequest = {
  shareIdentity: string;
  prefixFingerprint: string;
};

export type VllmRouteDecision = {
  replicaId: string;
  reason: "prefix_affinity" | "external_cache" | "lowest_pressure" | "only_healthy";
  score: number;
  cacheUsage: number | null;
  prefixHitRatio: number | null;
  externalPrefixHitRatio: number | null;
};

export type VllmChatDispatch = {
  decision: VllmRouteDecision;
  response: Response;
};

export type VllmFetch = typeof fetch;
