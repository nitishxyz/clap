import { createOpenApiDocument } from "@clap/api";
import {
  ChatCompletionRequestSchema,
  DownloadsResponseSchema,
  clapVersion,
  defaultBaseURL,
  ErrorResponseSchema,
  LoadedModelsResponseSchema,
  LoadModelRequestSchema,
  LoadModelResponseSchema,
  OllamaChatRequestSchema,
  OllamaGenerateRequestSchema,
  OllamaPullRequestSchema,
  OllamaShowRequestSchema,
  ModelResolveResponseSchema,
  PullModelRequestSchema,
  PullModelResponseSchema,
  ResponseRequestSchema,
  ResponseSchema,
  UnloadModelRequestSchema,
  UnloadModelResponseSchema,
  type ChatCompletionRequest,
  type ChatCompletionResponse,
  type ClapModel,
  type Download,
  type LoadedModel,
  type ResponseRequest,
} from "@clap/api";
import { cachedPullResultForTarget, clapHome, listAliases, listModels, listModelsAsync, pullModel, removeModel, resolveModel, resolveModelOptions, resolvePullTarget, type ResolvedModel } from "@clap/models";
import { assertGgufModelPath, isGgufModel, LlamaWorkerError } from "@clap/runtime-llama";
import { assertMlxModelPath, isMlxModelDirectory, MlxWorkerError } from "@clap/runtime-mlx";
import type { StructuredOutputContract } from "@clap/worker-protocol";
import { InsufficientModelMemoryError, listBackends, ModelLifecycleManager, ResidentWorkerRegistry,
  type CacheIdentity, type ResidentChatResult } from "@clap/runtime-router";
import { Hono, type Context } from "hono";
import { streamSSE } from "hono/streaming";
import { parseStructuredOutput, StructuredOutputError } from "./parsers/structured";
import { join } from "node:path";
import { z } from "zod";
import { ApiKeyVerifier, createApiKey, listApiKeys, resolveRequestIdentity, revokeApiKey, type RequestIdentity } from "./auth";
export { createApiKey, listApiKeys, revokeApiKey, keysFilePath } from "./auth";
import { CacheEventStore, type PersistedCacheDecision } from "./cache-event-store";
import { deriveCacheIdentity, derivePhysicalModelDomain, effectivePhysicalContextAllocation, InstallationSecretProvider, type DerivedCacheIdentity, type PhysicalModelDomain } from "./cache-identity";
import {
  classifyPersistedCacheOutcome,
  firstDecisionIdsForWorkerModelDomain,
  sessionDisplayIdentity,
} from "./dashboard-identity";
import { applyConfigToEnv, loadClapConfig, updateUserConfig, workerEnvForModel } from "./config";
export { configPaths, loadClapConfig } from "./config";
import { limiterFromEnv, QueueFullError } from "./limits";
import { renderPrometheus } from "./prometheus";
import { parseAssistantOutput, prepareChatRequest, profileStreamExtras, remainingDelta, StreamingOutputFilter, type ParserTemplateInfo, type StreamDelta } from "./chat-compat";
import { inferParserFamilies, resolveParserTemplateInfo } from "./parsers/traits";
export { inferParserFamilies } from "./parsers/traits";
import { MetricsCollector, type RequestHandle } from "./metrics";
import { sampleGpuUsage } from "./gpu-usage";
import { cpuCoreCount, processRssBytes, sampleProcessUsage, systemCpuPercent, systemMemoryBytes,
  systemMemorySnapshot, systemMemoryUsedBytes } from "./process-usage";
import { webAsset } from "./web-assets";

const startedAt = Date.now();
const downloads = new Map<string, Download>();
const activeDownloads = new Map<string, { id: string; controller: AbortController }>();
const defaultIdleTimeoutSeconds = 240;
const maxBunIdleTimeoutSeconds = 255;

async function requestCarriesCacheIntent(request: Request): Promise<boolean> {
  if (request.method === "GET" || request.method === "HEAD") return false;
  const contentType = request.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.includes("application/json")) return false;
  try {
    const body = await request.clone().json();
    return typeof body === "object" && body !== null && !Array.isArray(body)
      && "cache" in body && (body as { cache?: unknown }).cache !== undefined;
  } catch {
    // Let endpoint schema handling report malformed JSON.
    return false;
  }
}

export type ServerOptions = {
  port?: number;
  hostname?: string;
  idleTimeout?: number;
};

export function normalizeCacheIntent(request: ChatCompletionRequest): ChatCompletionRequest {
  if (!request.cache) return request;
  const { tenant, ...cache } = request.cache;
  return {
    ...request,
    cache: {
      ...cache,
      namespace: cache.namespace ?? tenant,
    },
  };
}

function workerCacheIdentity(
  identity: DerivedCacheIdentity,
  physical: PhysicalModelDomain,
  sideRequest: boolean,
): CacheIdentity {
  return {
    version: 1,
    generation: identity.generation,
    tenant_root: identity.tenantRoot,
    project_fingerprint: identity.fingerprints.project,
    harness_fingerprint: identity.fingerprints.harness,
    agent_fingerprint: identity.fingerprints.agent,
    session_fingerprint: identity.fingerprints.session,
    scope: identity.scope.kind,
    scope_fingerprint: identity.scope.fingerprint,
    namespace_fingerprint: identity.physical.namespace,
    namespace_id: identity.physical.namespaceId.toString(),
    priority: identity.priority,
    side_request: sideRequest,
    display: identity.display,
    physical: {
      fingerprint: identity.physical.fingerprint,
      backend: physical.backend,
      resolved_revision: physical.resolvedRevision,
      model_artifact_fingerprint: physical.modelArtifactFingerprint,
      tokenizer_fingerprint: physical.tokenizerFingerprint,
      context_allocation: physical.contextAllocation,
      kv_format: physical.kvFormat,
      unified_kv: physical.unifiedKv,
      layout_version: physical.layoutVersion,
    },
  };
}

type ServerEnv = {
  Bindings: {
    requestIP?: (request: Request) => { address?: string } | null;
  };
  Variables: {
    requestIdentity: RequestIdentity;
  };
};

export function createServer(
  residents = new ResidentWorkerRegistry(),
  lifecycle = new ModelLifecycleManager(() => Date.now(), (entry) => residents.shutdownAsync(entry.key)),
) {
  const app = new Hono<ServerEnv>();
  const { config, sources: configSources } = loadClapConfig();
  const cacheEvents = new CacheEventStore({
    directory: join(clapHome(), "telemetry"),
    enabled: config.telemetry.cache_decisions_enabled,
    maxBytes: (config.telemetry.cache_decisions_max_mib ?? 32) * 1024 * 1024,
    maxAgeMs: (config.telemetry.cache_decisions_max_age_days ?? 14) * 24 * 60 * 60 * 1000,
  });
  // Authoritative checkpoint minimum: explicit env wins (applyConfigToEnv
  // seeds it from config when unset), matching what workers receive.
  const checkpointMinimumTokens = () => {
    const fromEnv = Number(process.env.CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS);
    return Number.isInteger(fromEnv) && fromEnv > 0 ? fromEnv : config.cache.checkpoints.minimum_tokens;
  };
  const metrics = new MetricsCollector(cacheEvents, { checkpointMinimumTokens });
  applyConfigToEnv(config);

  // Persisted rows carry launch IDs so the dashboard can tell records from a
  // previous server/worker launch (cache no longer resident) apart from rows
  // belonging to a currently warm worker. Old records without a stored
  // outcome are re-classified reproducibly from their raw persisted fields;
  // the stored events themselves are never rewritten.
  const warmWorkerLaunchIds = (): Set<string> => {
    const warm = new Set<string>();
    for (const entry of lifecycle.list()) {
      const launchId = residents.get(entry.key)?.info().launchId;
      if (typeof launchId === "string" && launchId.length > 0) warm.add(launchId);
    }
    return warm;
  };
  // Cold authority is derived at list/query time from ordered worker+model
  // domain evidence (timestamp, workerLaunchId, model). Stored events are never
  // rewritten; empty candidates without first-decision evidence stay non-cold.
  const firstDecisionAuthority = (events: PersistedCacheDecision[]) =>
    firstDecisionIdsForWorkerModelDomain(events.map((event) => ({
      id: event.requestId,
      timestamp: event.timestamp,
      model: event.model,
      workerLaunchId: event.workerLaunchId,
    })));
  const persistedOutcome = (
    event: PersistedCacheDecision,
    firstIds: Set<string>,
  ) => classifyPersistedCacheOutcome(event, {
    checkpointMinimumTokens: checkpointMinimumTokens(),
    isFirstDecisionForWorkerModelDomain: firstIds.has(event.requestId),
  });
  const persistedIdentity = (event: PersistedCacheDecision) =>
    sessionDisplayIdentity({
      sessionFingerprint: event.sessionFingerprint,
      promptPrefixId: event.promptPrefixId,
    });
  const persistedHistorical = (event: PersistedCacheDecision, warm: Set<string>) =>
    event.serverLaunchId !== metrics.serverLaunchId
    || (event.workerLaunchId !== undefined && !warm.has(event.workerLaunchId));
  residents.memorySnapshot = async (pids) => {
    const [memory, processUsage] = await Promise.all([
      systemMemorySnapshot(),
      sampleProcessUsage(pids),
    ]);
    return {
      physicalMemoryBytes: memory.physicalBytes,
      availableMemoryBytes: memory.availableBytes,
      residentBytesByPid: new Map([...processUsage].map(([pid, usage]) => [pid, usage.rssBytes])),
    };
  };
  residents.rssSampler = processRssBytes;
  residents.configureResidency({
    lifecycle,
    env: process.env,
    osHeadroomBytes: configuredMemoryBytes("CLAP_MODEL_OS_HEADROOM_BYTES", 512 * 1024 ** 2),
    runtimeHeadroomBytes: configuredMemoryBytes("CLAP_MODEL_RUNTIME_HEADROOM_BYTES", 512 * 1024 ** 2),
    policy: { minimumHeadroomBytes: configuredMemoryBytes("CLAP_MODEL_MINIMUM_HEADROOM_BYTES", 1024 ** 3) },
    onDecision: (decision, model) => metrics.event("load",
      `model admission ${decision.reason} requested=${decision.requested.bytes} available=${decision.available.bytes} evicted=${decision.evictedModelKeys.length}`,
      { model: model.modelId }),
    onEvent: (event) => metrics.residencyEvent(event),
  });
  residents.workerEnv = (modelPath) => {
    let modelEnvironment: Record<string, string> = {};
    for (const modelId of Object.keys(config.models)) {
      // HF cache paths embed the repo id as owner--name; match either form.
      if (modelPath === modelId || modelPath.includes(modelId.replace("/", "--"))) {
        modelEnvironment = workerEnvForModel(config, modelId) ?? {};
        break;
      }
    }
    const tokenFingerprintKey = cacheEvents.tokenFingerprintKey();
    return tokenFingerprintKey
      ? { ...modelEnvironment, CLAP_TOKEN_FINGERPRINT_KEY: tokenFingerprintKey }
      : modelEnvironment;
  };
  // Warm-on-boot: models marked pinned (or given keep_alive) in config load
  // shortly after startup so the first user request never pays a cold load.
  // Deferred so server construction stays synchronous and tests unaffected.
  const pinnedModels = Object.entries(config.models)
    .filter(([, section]) => section.pinned || section.keep_alive)
    .map(([modelId, section]) => ({ modelId, keepAlive: section.pinned ? "always" : section.keep_alive }));
  if (pinnedModels.length) {
    setTimeout(async () => {
      for (const { modelId, keepAlive } of pinnedModels) {
        try {
          const resolved = resolveAvailableModel(modelId);
          if ("response" in resolved) {
            metrics.event("error", `warm-on-boot skipped: ${modelId} is not cached locally`, { model: modelId });
            continue;
          }
          await assertResidentModelPath(resolved.model);
          const model = lifecycle.load(resolved.model, { keepAlive });
          const worker = residents.getOrCreate(model.key, resolved.model.backend,
            resolved.model.modelPath ?? resolved.model.input, workerDescriptor(resolved.model));
          try {
            model.worker = await worker.load();
          } catch (error) {
            lifecycle.unload(resolved.model);
            throw error;
          }
          metrics.event("load", `${model.id} warmed on boot (keep-alive ${model.keepAlive})`, { model: model.id });
        } catch (error) {
          metrics.event("error", `warm-on-boot failed for ${modelId}: ${error instanceof Error ? error.message : error}`, { model: modelId });
        }
      }
    }, 50);
  }
  metrics.event("server", `clap server started (v${clapVersion})`);
  lifecycle.removeListener = (entry, reason) => {
    if (reason === "cleanup") return;
    metrics.event(reason === "expire" ? "expire" : "unload", `${entry.id} ${reason === "expire" ? "expired after idle keep-alive" : "unloaded"} (${entry.backend})`, { model: entry.id });
  };
  residents.onCrash = ({ key, backend, exitCode, consecutiveCrashes, launchId, logPath,
    metadataPath, classification }) => {
    // Worker keys are JSON tuples ([model, backend, path]); report the model id.
    let model = key;
    try {
      const parsed = JSON.parse(key);
      if (Array.isArray(parsed) && typeof parsed[0] === "string") model = parsed[0];
    } catch {
      // plain key
    }
    metrics.event("error", `${model} ${backend} worker crashed (exit ${exitCode}, ${consecutiveCrashes} consecutive, ${classification ?? "unknown"}); auto-restarting with backoff`, {
      model, launchId, stderrLogPath: logPath, launchMetadataPath: metadataPath,
      crashClassification: classification,
    });
  };

  app.onError((error, c) => {
    if (error instanceof z.ZodError) {
      const boundaryIssue = error.issues.some((issue) => issue.path[0] === "cache" && issue.path[1] === "boundaries");
      return c.json({
        error: {
          message: error.message,
          type: "invalid_request_error",
          code: boundaryIssue ? "invalid_cache_boundary" : "invalid_json",
        },
      }, 400);
    }
    if (error instanceof InsufficientModelMemoryError) return insufficientMemoryResponse(c, error);
    if (error instanceof StructuredOutputError) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error.message, type: "invalid_request_error", code: error.code },
      }), 422);
    }
    if ((error instanceof LlamaWorkerError || error instanceof MlxWorkerError) &&
        (error.code === "context_length_exceeded" || error.code === "max_output_tokens_exceeded" || error.code === "token_capability_unknown" ||
          error.code === "structured_output_capability_required" || error.code === "invalid_cache_boundary" ||
          error.code === "unsafe_cache_boundary" || error.code === "non_prefix_cache_boundary")) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error.message, type: "invalid_request_error", code: error.code },
      }), 400);
    }
    if (error instanceof LlamaWorkerError) {
      const status = error.code === "model_not_found" ? 404 : 503;
      return c.json(ErrorResponseSchema.parse({
        error: { message: error.message, type: "backend_error", code: error.code },
      }), status);
    }
    if (error instanceof MlxWorkerError) {
      const status = error.code === "model_not_found" ? 404 : 503;
      return c.json(ErrorResponseSchema.parse({
        error: { message: error.message, type: "backend_error", code: error.code },
      }), status);
    }
    if (error instanceof Error && /worker|backend|mlx|llama/i.test(error.message)) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error.message, type: "backend_error", code: "resident_worker_error" },
      }), 503);
    }
    return c.json({ error: { message: error.message, type: "server_error" } }, 500);
  });

  app.get("/openapi.json", (c) => c.json(createOpenApiDocument()));

  app.get("/clap/v1/health", (c) => c.json({
    status: "ok",
    version: clapVersion,
    uptimeMs: Date.now() - startedAt,
  }));

  // API key auth. Loopback clients (CLI, dashboard on the box) stay open
  // unless CLAP_REQUIRE_API_KEY=1 or [auth] require_api_key = true; remote
  // clients must present a valid Bearer key once any active key exists.
  // Health stays open for probes.
  const apiKeys = new ApiKeyVerifier();
  const limiter = limiterFromEnv(config.limits.max_inflight, config.limits.queue_depth);
  const installationSecrets = new InstallationSecretProvider();
  app.use("*", async (c, next) => {
    let address: string | undefined;
    let embedded = true;
    try {
      address = c.env?.requestIP?.(c.req.raw)?.address;
      embedded = address === undefined;
    } catch {
      // No transport info denotes embedded use.
    }
    const identity = resolveRequestIdentity(apiKeys, {
      authorization: c.req.header("authorization"),
      apiKey: c.req.header("x-api-key"),
      address,
      embedded,
    });
    c.set("requestIdentity", identity);
    if (c.req.path === "/clap/v1/health") return next();

    if (identity.credentialPresented && !identity.credentialValid) return invalidApiKey(c);
    if (!identity.cachePrincipal && await requestCarriesCacheIntent(c.req.raw)) {
      return c.json(ErrorResponseSchema.parse({
        error: {
          message: "cache identity requires a valid API key for remote requests",
          type: "authentication_error",
          code: "cache_identity_required",
        },
      }), 401);
    }
    const requireAlways = process.env.CLAP_REQUIRE_API_KEY === "1"
      || (process.env.CLAP_REQUIRE_API_KEY === undefined && config.auth.require_api_key === true);
    const required = requireAlways || (!identity.loopback && apiKeys.hasActiveKeys());
    if (!required || identity.credentialValid) return next();
    return invalidApiKey(c);
  });

  function invalidApiKey(c: Context<ServerEnv>) {
    return c.json(ErrorResponseSchema.parse({
      error: { message: "missing or invalid API key; pass Authorization: Bearer <key>", type: "invalid_request_error", code: "invalid_api_key" },
    }), 401);
  }

  app.post("/clap/v1/cache/identity/rotate", async (c) => {
    const identity = c.get("requestIdentity");
    if (!identity.cachePrincipal) return invalidApiKey(c);
    const result = await installationSecrets.rotate(async (rotation) => ({
      ...rotation,
      clearedResidents: await residents.rotateCacheIdentityGeneration(),
    }));
    return c.json(result);
  });

  app.post("/clap/v1/keys", async (c) => {
    const body = z.object({ name: z.string().min(1) }).parse(await c.req.json());
    const { record, key } = createApiKey(body.name);
    metrics.event("server", `API key created: ${record.name} (${record.id})`);
    return c.json({ ...record, key }, 201);
  });

  app.get("/clap/v1/keys", (c) => c.json({ keys: listApiKeys() }));

  app.get("/clap/v1/config", (c) => c.json({ config, sources: configSources }));

  app.patch("/clap/v1/config", async (c) => {
    const patch = z.record(z.string(), z.unknown()).parse(await c.req.json());
    let updated: ReturnType<typeof updateUserConfig>;
    try {
      updated = updateUserConfig(patch);
    } catch (error) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error instanceof Error ? error.message : String(error), type: "invalid_request_error", code: "invalid_config" },
      }), 400);
    }
    // Live-apply what can apply live: auth requirement and per-model worker
    // env (next worker start). [server] and [limits] need a restart.
    config.auth = updated.config.auth;
    config.models = updated.config.models;
    config.llama = updated.config.llama;
    config.cache = updated.config.cache;
    metrics.event("server", `config updated (${updated.path})`);
    return c.json({
      config: updated.config,
      path: updated.path,
      note: "auth/models/llama apply to new requests; [cache.checkpoints], [server], and [limits] apply on restart",
    });
  });

  app.get("/metrics", (c) => {
    const loaded = lifecycle.list().map((entry) => ({
      ...entry,
      worker: residents.get(entry.key)?.info() ?? entry.worker,
    }));
    return c.text(renderPrometheus({
      totals: metrics.totals,
      activeRequests: metrics.activeRequests().length,
      queue: limiter.stats(),
      loadedModels: loaded.map((entry) => ({
        id: entry.id,
        backend: entry.backend,
        state: entry.worker?.state ?? "not_started",
        crashes: entry.worker?.crashes,
        retention: entry.worker?.retention,
        tokenCapabilities: entry.worker?.tokenCapabilities,
      })),
      uptimeMs: Date.now() - startedAt,
      histograms: metrics.histograms,
      residency: metrics.residency,
      structuredOutputOutcomes: metrics.structuredOutputOutcomes,
      priorityRequestOutcomes: metrics.priorityRequestOutcomes,
      priorityDurationMs: metrics.priorityDurationMs,
    }), 200, { "content-type": "text/plain; version=0.0.4; charset=utf-8" });
  });

  app.delete("/clap/v1/keys/:id", (c) => {
    const revoked = revokeApiKey(c.req.param("id"));
    if (!revoked) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: `no active key with id ${c.req.param("id")}`, type: "invalid_request_error", code: "key_not_found" },
      }), 404);
    }
    metrics.event("server", `API key revoked: ${c.req.param("id")}`);
    return c.json({ revoked: true });
  });

  app.get("/clap/v1/runtime", (c) => c.json({
    platform: process.platform,
    arch: process.arch,
    bunVersion: Bun.version,
    runtime: "bun",
  }));

  app.get("/clap/v1/backends", (c) => c.json({ backends: listBackends() }));

  app.get("/clap/v1/models", async (c) => c.json({ models: await listModelsAsync() }));

  app.get("/clap/v1/aliases", (c) => c.json({ models: listAliases() }));

  app.get("/clap/v1/downloads", (c) => c.json(DownloadsResponseSchema.parse({
    downloads: [...downloads.values()],
  })));

  app.get("/clap/v1/runtime/models", (c) => c.json(LoadedModelsResponseSchema.parse({
    models: lifecycle.list().map((entry) => ({
      ...entry,
      worker: residents.get(entry.key)?.info() ?? entry.worker,
    })),
  })));

  const dashboardPayload = async () => {
    const loaded = lifecycle.list().map((entry) => ({
      ...entry,
      worker: residents.get(entry.key)?.info() ?? entry.worker,
    }));
    const workerPids = loaded
      .map((entry) => entry.worker?.pid)
      .filter((pid): pid is number => typeof pid === "number");
    const [usage, gpus, memoryUsed] = await Promise.all([
      sampleProcessUsage([process.pid, ...workerPids]),
      sampleGpuUsage(),
      systemMemoryUsedBytes(),
    ]);
    const liveRequests = metrics.recent(80).map((request) => ({
      ...request,
      historical: false as const,
    }));
    const liveIds = new Set(liveRequests.map((request) => request.id));
    const warmWorkers = warmWorkerLaunchIds();
    // Pull a wider window so first-decision authority is stable even when the
    // dashboard only returns the newest 80 rows.
    const persistedWindow = cacheEvents.list({}, 200).items;
    const firstIds = firstDecisionAuthority(persistedWindow);
    const persistedRequests = persistedWindow
      .filter((event) => !liveIds.has(event.requestId))
      .slice(0, 80)
      .map((event) => {
        const identity = persistedIdentity(event);
        return {
          source: "persisted" as const,
          id: event.requestId,
          startedAt: event.timestamp - (event.durationMs ?? 0),
          endedAt: event.timestamp,
          durationMs: event.durationMs,
          ttftMs: event.ttftMs,
          priority: event.priority ?? "normal",
          model: event.model,
          endpoint: event.endpoint ?? "/v1/chat/completions",
          stream: false,
          status: event.status,
          phase: "done" as const,
          promptTokens: event.promptTokenCount,
          conversation: identity.promptPrefixId,
          sessionDisplayId: identity.sessionDisplayId,
          sessionIdentityKind: identity.sessionIdentityKind,
          sessionFingerprint: identity.sessionFingerprint,
          cacheHit: event.cache?.hit,
          reusedTokens: event.cache?.reusedTokens,
          reuseKind: event.cache?.kind === "slot" || event.cache?.kind === "branch" || event.cache?.kind === "anchor"
            ? event.cache.kind
            : undefined,
          reuseScope: event.cache?.scope,
          sideRequest: event.side,
          donorSlot: event.cache?.donorSlot,
          targetSlot: event.cache?.targetSlot,
          cacheDecisionUs: event.cache?.decisionUs,
          plannedReuseTokens: event.cache?.plannedTokens,
          realizedReuseTokens: event.cache?.realizedTokens,
          cacheFallback: event.cache?.fallback,
          timing: event.timing,
          finishReason: event.finishReason,
          cacheOutcome: persistedOutcome(event, firstIds),
          structuredOutput: event.structuredOutput,
          historical: persistedHistorical(event, warmWorkers),
        };
      });
    return {
      server: {
        version: clapVersion,
        uptimeMs: Date.now() - startedAt,
        platform: process.platform,
        arch: process.arch,
        bunVersion: Bun.version,
        pid: process.pid,
        rssBytes: usage.get(process.pid)?.rssBytes ?? process.memoryUsage().rss,
        cpuPercent: usage.get(process.pid)?.cpuPercent,
        systemMemoryBytes: systemMemoryBytes(),
        systemMemoryUsedBytes: memoryUsed,
        systemCpuPercent: systemCpuPercent(),
        cpuCount: cpuCoreCount(),
      },
      gpus,
      totals: metrics.totals,
      queue: limiter.stats(),
      active: metrics.activeRequests(),
      requests: [...liveRequests, ...persistedRequests]
        .sort((a, b) => b.startedAt - a.startedAt)
        .slice(0, 80),
      events: metrics.events(50),
      loaded: loaded.map((entry) => ({
        ...entry,
        usage: entry.worker?.pid ? usage.get(entry.worker.pid) : undefined,
        gpuMemoryBytes: entry.worker?.pid ? gpus[0]?.processes?.find((proc) => proc.pid === entry.worker?.pid)?.memoryBytes : undefined,
      })),
      models: await listModelsAsync(),
      downloads: [...downloads.values()],
    };
  };

  app.get("/clap/v1/dashboard", async (c) => c.json(await dashboardPayload()));

  // Live dashboard feed: pushes the full payload every interval so the UI
  // updates without polling. Clients reconnect on drop (standard SSE).
  app.get("/clap/v1/dashboard/stream", (c) => {
    const intervalMs = Math.max(500, Math.min(10000, Number(c.req.query("interval") ?? 2000) || 2000));
    return strictStreamSSE(c, async (stream) => {
      let open = true;
      stream.onAbort(() => {
        open = false;
      });
      while (open) {
        try {
          await stream.writeSSE({ event: "dashboard", data: JSON.stringify(await dashboardPayload()) });
        } catch {
          break;
        }
        await Bun.sleep(intervalMs);
      }
    });
  });

  app.get("/clap/v1/dashboard/requests/:id", (c) => {
    const record = metrics.request(c.req.param("id"));
    if (record) return c.json({ ...record, historical: false });
    const persisted = cacheEvents.get(c.req.param("id"));
    if (persisted) {
      const warmWorkers = warmWorkerLaunchIds();
      // Authority over the same model+worker domain for cold classification.
      const domainPeers = cacheEvents.list({ model: persisted.model }, 200).items
        .filter((event) => event.workerLaunchId === persisted.workerLaunchId);
      const firstIds = firstDecisionAuthority(
        domainPeers.length ? domainPeers : [persisted],
      );
      const identity = persistedIdentity(persisted);
      return c.json({
        source: "persisted",
        id: persisted.requestId,
        startedAt: persisted.timestamp - (persisted.durationMs ?? 0),
        endedAt: persisted.timestamp,
        durationMs: persisted.durationMs,
        ttftMs: persisted.ttftMs,
        priority: persisted.priority ?? "normal",
        model: persisted.model,
        endpoint: persisted.endpoint ?? "/v1/chat/completions",
        stream: false,
        status: persisted.status,
        phase: "done",
        promptTokens: persisted.promptTokenCount,
        conversation: identity.promptPrefixId,
        sessionDisplayId: identity.sessionDisplayId,
        sessionIdentityKind: identity.sessionIdentityKind,
        sessionFingerprint: identity.sessionFingerprint,
        cacheHit: persisted.cache?.hit,
        reusedTokens: persisted.cache?.reusedTokens,
        reuseKind: persisted.cache?.kind === "slot" || persisted.cache?.kind === "branch" || persisted.cache?.kind === "anchor"
          ? persisted.cache.kind
          : undefined,
        reuseScope: persisted.cache?.scope,
        sideRequest: persisted.side,
        donorSlot: persisted.cache?.donorSlot,
        targetSlot: persisted.cache?.targetSlot,
        cacheDecisionUs: persisted.cache?.decisionUs,
        plannedReuseTokens: persisted.cache?.plannedTokens,
        realizedReuseTokens: persisted.cache?.realizedTokens,
        cacheFallback: persisted.cache?.fallback,
        timing: persisted.timing,
        finishReason: persisted.finishReason,
        cacheOutcome: persistedOutcome(persisted, firstIds),
        historical: persistedHistorical(persisted, warmWorkers),
        // Full raw persisted event is preserved for diagnostics; outcome is
        // derived/read-only and never rewrites the stored telemetry.
        cacheDiagnostics: persisted,
      });
    }
    return c.json({ error: { message: "request not found", type: "invalid_request_error" } }, 404);
  });

  app.get("/clap/v1/cache-decisions", (c) => {
    const hit = c.req.query("hit");
    const status = c.req.query("status");
    return c.json(cacheEvents.list({
      requestId: c.req.query("request_id"),
      model: c.req.query("model"),
      backend: c.req.query("backend"),
      status: status === "ok" || status === "error" || status === "cancelled" ? status : undefined,
      hit: hit === "true" ? true : hit === "false" ? false : undefined,
      since: c.req.query("since") ? Number(c.req.query("since")) : undefined,
      until: c.req.query("until") ? Number(c.req.query("until")) : undefined,
    }, Number(c.req.query("limit") ?? 50), c.req.query("cursor")));
  });

  app.get("/clap/v1/cache-decisions/:id", (c) => {
    const event = cacheEvents.get(c.req.param("id"));
    if (!event) return c.json({ error: { message: "cache decision not found", type: "invalid_request_error" } }, 404);
    return c.json(event);
  });

  const serveWeb = (path: string) => {
    const asset = webAsset(path);
    if (!asset) return undefined;
    return new Response(asset.bytes, { headers: { "content-type": asset.type, "cache-control": path.startsWith("assets/") ? "public, max-age=31536000, immutable" : "no-cache" } });
  };
  app.get("/", (c) => serveWeb("index.html") ?? c.text("clap dashboard is not built. Run: bun run build:web", 503));
  app.get("/assets/*", (c) => serveWeb(c.req.path.slice(1)) ?? c.notFound());
  app.get("/dashboard", (c) => c.redirect("/"));

  app.post("/clap/v1/models/load", async (c) => {
    const request = LoadModelRequestSchema.parse(await c.req.json());
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) return resolved.response(c);
    await assertResidentModelPath(resolved.model);
    const model = lifecycle.load(resolved.model, { keepAlive: request.keepAlive });
    const worker = residents.getOrCreate(model.key, resolved.model.backend,
      resolved.model.modelPath ?? resolved.model.input, workerDescriptor(resolved.model));
    let info;
    try {
      info = await worker.load();
    } catch (error) {
      lifecycle.unload(resolved.model);
      if (error instanceof InsufficientModelMemoryError) return insufficientMemoryResponse(c, error);
      return c.json(ErrorResponseSchema.parse({
        error: { message: error instanceof Error ? error.message : String(error), type: "backend_error", code: "resident_worker_error" },
      }), 503);
    }
    model.worker = info;
    metrics.event("load", `${model.id} loaded (keep-alive ${model.keepAlive})`, { model: model.id });
    return c.json(LoadModelResponseSchema.parse({ model }));
  });

  app.post("/clap/v1/models/unload", async (c) => {
    const request = UnloadModelRequestSchema.parse(await c.req.json());
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) return resolved.response(c);
    return c.json(UnloadModelResponseSchema.parse(lifecycle.unload(resolved.model)));
  });

  app.post("/clap/v1/models/remove", async (c) => {
    const request = z.object({ model: z.string() }).parse(await c.req.json());
    for (const entry of lifecycle.list()) {
      if (entry.id === request.model) {
        if (entry.activeRequests > 0) {
          return c.json(ErrorResponseSchema.parse({
            error: { message: `${request.model} is serving ${entry.activeRequests} active request(s); try again when idle`, type: "model_error", code: "model_busy" },
          }), 409);
        }
        lifecycle.unload({ id: entry.id, backend: entry.backend, format: entry.format, input: entry.localPath, modelPath: entry.localPath } as ResolvedModel);
      }
    }
    const removed = await removeModel(request.model);
    if (!removed.length) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: `No cached files found for ${request.model}`, type: "model_error", code: "not_cached" },
      }), 404);
    }
    metrics.event("unload", `removed ${request.model} from disk (${removed.length} path${removed.length > 1 ? "s" : ""})`, { model: request.model });
    return c.json({ removed });
  });

  app.post("/clap/v1/models/resolve", async (c) => {
    const request = PullModelRequestSchema.parse(await c.req.json());
    return c.json(ModelResolveResponseSchema.parse(await resolveModelOptions(request)));
  });

  app.post("/clap/v1/models/pull", async (c) => {
    const request = PullModelRequestSchema.parse(await c.req.json());
    const useResolver = !request.file && request.backend !== "mlx";
    const resolvedOptions = useResolver ? await resolveModelOptions(request) : undefined;
    const selected = resolvedOptions?.selected;
    if (useResolver && !selected) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: `No supported runnable artifacts found for ${request.model}`, type: "model_error", code: "no_supported_artifact" },
      }), 400);
    }
    const target = selected ? resolvePullTarget({ model: selected.repo, backend: selected.backend, file: selected.file, force: request.force }) : resolvePullTarget(request);
    const active = activeDownloads.get(target.key);
    if (active) {
      const download = downloads.get(active.id);
      if (download?.status === "running" || download?.status === "queued") {
        return c.json(PullModelResponseSchema.parse({ download }));
      }
    }

    const id = `pull_${crypto.randomUUID()}`;
    const startedAtIso = new Date().toISOString();
    const cached = request.force ? undefined : cachedPullResultForTarget(target);
    const download: Download = {
      id,
      model: request.model,
      file: selected?.file ?? request.file,
      backend: selected?.backend ?? request.backend,
      selected,
      targetKey: target.key,
      status: cached ? "completed" : "running",
      bytesReceived: 0,
      modelPath: cached?.modelPath,
      startedAt: startedAtIso,
      completedAt: cached ? startedAtIso : undefined,
    };
    downloads.set(id, download);
    if (cached) {
      return c.json(PullModelResponseSchema.parse({ download }));
    }

    const controller = new AbortController();
    activeDownloads.set(target.key, { id, controller });
    metrics.event("download", `pull started: ${download.model}`, { model: download.model });
    void pullModel(selected ? { model: selected.repo, backend: selected.backend, file: selected.file, force: request.force } : request, {
      signal: controller.signal,
      onProgress: (progress) => {
        if (progress.bytesReceived !== undefined) download.bytesReceived = progress.bytesReceived;
        if (progress.totalBytes !== undefined) download.totalBytes = progress.totalBytes;
        if (progress.currentFile !== undefined) download.currentFile = progress.currentFile;
      },
    }).then((result) => {
      if (download.status === "cancelled") return;
      download.status = "completed";
      download.modelPath = result.modelPath;
      if (download.totalBytes === undefined) download.totalBytes = download.bytesReceived;
      download.currentFile = undefined;
      download.completedAt = new Date().toISOString();
      metrics.event("download", `pull completed: ${download.model}`, { model: download.model });
    }).catch((error: unknown) => {
      if (controller.signal.aborted || isAbortError(error)) {
        download.status = "cancelled";
        download.error = undefined;
        download.currentFile = undefined;
        download.completedAt = new Date().toISOString();
        return;
      }
      download.status = "failed";
      download.error = error instanceof Error ? error.message : String(error);
      download.completedAt = new Date().toISOString();
      metrics.event("error", `pull failed: ${download.model} — ${download.error}`, { model: download.model });
    }).finally(() => {
      const active = activeDownloads.get(target.key);
      if (active?.id === id) activeDownloads.delete(target.key);
    });

    return c.json(PullModelResponseSchema.parse({ download }));
  });

  app.post("/clap/v1/downloads/:id/cancel", (c) => {
    const id = c.req.param("id");
    const download = downloads.get(id);
    if (!download) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: `download not found: ${id}`, type: "not_found_error", code: "download_not_found" },
      }), 404);
    }
    if (download.status === "running" || download.status === "queued") {
      const targetKey = download.targetKey;
      const active = targetKey ? activeDownloads.get(targetKey) : undefined;
      if (active?.id === id) active.controller.abort();
      download.status = "cancelled";
      download.currentFile = undefined;
      download.completedAt = new Date().toISOString();
    }
    return c.json(PullModelResponseSchema.parse({ download }));
  });

  app.get("/v1/models", (c) => {
    const includeMetadata = ["1", "true"].includes(c.req.query("metadata")?.toLowerCase() ?? "");
    return listModelsAsync().then((models) => c.json({
      object: "list",
      data: models.map((model) => ({
        ...(includeMetadata ? model : { id: model.id, object: model.object }),
        created: 0,
        owned_by: "clap",
      })),
    }));
  });

  app.post("/v1/chat/completions", async (c) => {
    const request = normalizeCacheIntent(ChatCompletionRequestSchema.parse(await c.req.json()));
    const handle = metrics.start(request.model, "/v1/chat/completions", request.stream);
    handle.capture(request);
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) {
      handle.finish({ status: "error", errorCode: "model_not_found" });
      return resolved.response(c);
    }
    handle.capture({ ...request, backend: resolved.model.backend });
    if (resolved.model.backend === "llama" && resolved.model.modelPath) {
      await assertGgufModelPath(resolved.model.modelPath);
    }

    // Prepare the immutable physical cache descriptor once cache intent is
    // present. Dispatch integration consumes this in the next phase; content
    // hashes are memoized by canonical path and stat signature.
    const physicalModelDomain = await preparePhysicalModelDomain(resolved.model);
    const principal = c.get("requestIdentity").cachePrincipal;
    if (!principal) throw new Error("Authenticated cache principal is unavailable");

    const templateInfo = await resolveParserTemplateInfo(resolved.model);
    const secretLease = await installationSecrets.acquireSecret();
    let derivedIdentity: DerivedCacheIdentity;
    try {
      derivedIdentity = deriveCacheIdentity(secretLease.secret, principal, request.cache ?? {}, {
        backend: physicalModelDomain.backend,
        modelRevision: physicalModelDomain.modelRevision,
        tokenizer: physicalModelDomain.tokenizerFingerprint,
        contextAllocation: physicalModelDomain.contextAllocation,
        kvFormat: physicalModelDomain.kvFormat,
        unifiedKv: physicalModelDomain.unifiedKv,
        layoutVersion: physicalModelDomain.layoutVersion,
      });
    } catch (error) {
      secretLease.release();
      throw error;
    }
    const cacheIdentity = workerCacheIdentity(derivedIdentity, physicalModelDomain, request.cache?.side_request ?? false);
    const localModelPath = resolved.model.modelPath ?? request.model;
    if (isGgufModel(localModelPath)) {
      await assertGgufModelPath(localModelPath);
      return dispatchWithLimit(c, resolved.model, request, templateInfo, handle, cacheIdentity, secretLease.release);
    }
    if (await isMlxModelDirectory(localModelPath)) {
      return dispatchWithLimit(c, resolved.model, request, templateInfo, handle, cacheIdentity, secretLease.release);
    }

    secretLease.release();
    handle.finish({ status: "error", errorCode: "not_cached" });
    return c.json(ErrorResponseSchema.parse({
      error: { message: `Model ${request.model} is not cached as a GGUF file or MLX directory. Run: clap pull ${request.model}, or pass a local .gguf file / MLX directory.`, type: "model_error", code: "not_cached" },
    }), 404);
  });

  // Admission gate shared by both backends: bounded in-flight + fair queue.
  // Saturation answers 429 + Retry-After before any model work happens.
  function forwardedHeaders(c: { req: { header: (name: string) => string | undefined } }): Record<string, string> {
    // Internal delegation (Responses/Ollama -> chat completions) must carry
    // the caller's credentials so auth and per-key fairness see the real
    // client instead of an anonymous embedded request.
    const headers: Record<string, string> = { "content-type": "application/json" };
    const authorization = c.req.header("authorization");
    if (authorization) headers.authorization = authorization;
    const apiKey = c.req.header("x-api-key");
    if (apiKey) headers["x-api-key"] = apiKey;
    return headers;
  }

  async function preparePhysicalModelDomain(model: ResolvedModel): Promise<PhysicalModelDomain> {
    const modelEnvironment = workerEnvForModel(config, model.id) ?? {};
    const prefix = model.backend === "llama" ? "CLAP_LLAMA" : "CLAP_MLX";
    const context = Number(modelEnvironment[`${prefix}_CONTEXT`] ?? process.env[`${prefix}_CONTEXT`] ?? 0);
    const kvFormat = modelEnvironment[`${prefix}_KV_TYPE`] ?? process.env[`${prefix}_KV_TYPE`] ?? "f16";
    const unifiedKv = model.backend === "llama"
      ? (modelEnvironment.CLAP_LLAMA_KV_UNIFIED ?? process.env.CLAP_LLAMA_KV_UNIFIED ?? "1") !== "0"
      : false;
    return derivePhysicalModelDomain(model, {
      contextAllocation: effectivePhysicalContextAllocation(model, context),
      kvFormat,
      unifiedKv,
    });
  }

  async function dispatchWithLimit(
    c: Parameters<typeof streamSSE>[0],
    model: ResolvedModel,
    request: ChatCompletionRequest,
    templateInfo: ParserTemplateInfo | undefined,
    handle: RequestHandle,
    cacheIdentity: CacheIdentity,
    releaseIdentityLease: () => void,
  ) {
    let release: () => void;
    try {
      release = await limiter.acquire(c.get("requestIdentity").clientId,
        request.cache?.priority ?? "normal", c.req.raw.signal);
    } catch (error) {
      if (error instanceof QueueFullError) {
        releaseIdentityLease();
        handle.finish({ status: "error", error: error.message, errorCode: "server_overloaded" });
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json(ErrorResponseSchema.parse({
          error: { message: error.message, type: "rate_limit_error", code: "server_overloaded" },
        }), 429);
      }
      releaseIdentityLease();
      throw error;
    }
    try {
      const worker = residents.getOrCreate(lifecycleKey(model), model.backend, model.modelPath ?? model.input,
        workerDescriptor(model));
      const workerInfo = await worker.load();
      let routedRequest: ChatCompletionRequest;
      try {
        routedRequest = prepareChatRequest(
          { ...request, model: model.modelPath ?? request.model },
          { nativeTools: workerInfo.effectiveCapabilities?.generation.toolTemplateSupport === true
              && templateInfo?.hasToolCalls === true },
        );
      } catch (error) {
        release();
        releaseIdentityLease();
        handle.finish({
          status: "error",
          error: error instanceof Error ? error.message : String(error),
          errorCode: "unsupported_content_part",
        });
        return c.json(ErrorResponseSchema.parse({
          error: { message: error instanceof Error ? error.message : String(error), type: "invalid_request_error", code: "unsupported_content_part" },
        }), 400);
      }
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(model);
        return streamResidentResponse(c, residents, loaded, routedRequest, templateInfo, () => {
          lifecycle.finishUsage(loaded);
          release();
          releaseIdentityLease();
        }, handle, cacheIdentity);
      }
      const response = await lifecycle.withUsage(model, (entry) => jsonResidentResponse(c, residents, entry, routedRequest, templateInfo, handle, cacheIdentity));
      release();
      releaseIdentityLease();
      return response;
    } catch (error) {
      release();
      releaseIdentityLease();
      throw error;
    }
  }

  app.post("/v1/responses", async (c) => {
    const request = ResponseRequestSchema.parse(await c.req.json());
    if (request.previous_response_id) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: "previous_response_id/stateful continuation is not implemented by Clap yet", type: "invalid_request_error", code: "unsupported_stateful_continuation" },
      }), 400);
    }
    let chatRequest: ChatCompletionRequest;
    try {
      chatRequest = chatRequestFromResponse(request);
    } catch (error) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error instanceof Error ? error.message : String(error), type: "invalid_request_error", code: "unsupported_content_part" },
      }), 400);
    }
    const response = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: forwardedHeaders(c),
      body: JSON.stringify({ ...chatRequest, stream: false }),
    });
    if (!response.ok) return response;
    const chat = await response.json() as ChatCompletionResponse;
    const body = responseFromChat(request, chat);
    if (request.stream) return streamResponseBody(body);
    return c.json(ResponseSchema.parse(body));
  });

  app.get("/api/tags", async (c) => c.json({
    models: (await listModelsAsync()).map(ollamaTag),
  }));

  app.post("/api/show", async (c) => {
    const request = OllamaShowRequestSchema.parse(await c.req.json());
    const model = findOllamaModel(request.model);
    if (!model) return ollamaNotFound(c, request.model);
    return c.json({
      license: "",
      modelfile: `FROM ${model.id}`,
      parameters: "",
      template: "",
      details: ollamaDetails(model),
      model_info: model,
      capabilities: model.capabilities,
    });
  });

  app.post("/api/pull", async (c) => {
    const request = OllamaPullRequestSchema.parse(await c.req.json());
    const model = request.model ?? request.name;
    if (!model) return c.json({ error: "model is required" }, 400);
    if (request.stream === false) {
      const result = await pullModel({ model });
      return c.json({ status: "success", digest: result.id, total: result.files.length, completed: result.files.length });
    }
    return new Response(new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        const write = (value: unknown) => controller.enqueue(encoder.encode(`${JSON.stringify(value)}\n`));
        try {
          write({ status: "pulling manifest" });
          const result = await pullModel({ model }, {
            onProgress: (progress) => write({
              status: progress.currentFile ? `downloading ${progress.currentFile}` : "downloading",
              completed: progress.bytesReceived ?? 0,
              total: progress.totalBytes,
            }),
          });
          write({ status: "success", digest: result.id });
          controller.close();
        } catch (error) {
          write({ error: error instanceof Error ? error.message : String(error) });
          controller.close();
        }
      },
    }), { headers: { "content-type": "application/x-ndjson" } });
  });

  app.post("/api/chat", async (c) => {
    const raw = await c.req.json();
    if (hasOllamaImages(raw)) return c.json({ error: "image input is not supported by the selected local text runtime yet" }, 400);
    const request = OllamaChatRequestSchema.parse(raw);
    const stream = request.stream !== false;
    const response = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: forwardedHeaders(c),
      body: JSON.stringify({
        model: request.model,
        messages: request.messages,
        stream,
        tools: request.tools,
        temperature: request.options?.temperature,
        top_p: request.options?.top_p,
        seed: request.options?.seed,
        max_tokens: request.options?.num_predict,
        stop: request.options?.stop,
      }),
    });
    if (!response.ok) return response;
    if (stream) return ollamaStreamFromSse(request.model, response, "chat");
    const body = await response.json() as ChatCompletionResponse;
    return ollamaChatResponse(c, request.model, body, false);
  });

  app.post("/api/generate", async (c) => {
    const raw = await c.req.json();
    if (hasOllamaImages(raw)) return c.json({ error: "image input is not supported by the selected local text runtime yet" }, 400);
    const request = OllamaGenerateRequestSchema.parse(raw);
    const stream = request.stream !== false;
    const messages = [
      ...(request.system ? [{ role: "system" as const, content: request.system }] : []),
      { role: "user" as const, content: request.prompt },
    ];
    const response = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: forwardedHeaders(c),
      body: JSON.stringify({
        model: request.model,
        messages,
        stream,
        response_format: request.format ? request.format === "json" ? { type: "json_object" } : { type: "json_schema", json_schema: { name: "OllamaFormat", schema: request.format } } : undefined,
        temperature: request.options?.temperature,
        top_p: request.options?.top_p,
        seed: request.options?.seed,
        max_tokens: request.options?.num_predict,
        stop: request.options?.stop,
      }),
    });
    if (!response.ok) return response;
    if (stream) return ollamaStreamFromSse(request.model, response, "generate");
    const body = await response.json() as ChatCompletionResponse;
    return ollamaGenerateResponse(c, request.model, body, false);
  });

  const unsupportedOllama = (name: string) => (c: { json: (body: unknown, status?: number) => Response | Promise<Response> }) => c.json({ error: `${name} is not implemented by Clap yet` }, 501);
  app.delete("/api/delete", unsupportedOllama("delete"));
  app.post("/api/delete", unsupportedOllama("delete"));
  app.post("/api/copy", unsupportedOllama("copy"));
  app.post("/api/embeddings", unsupportedOllama("embeddings"));
  app.post("/api/embed", unsupportedOllama("embed"));

  return app;
}

function chatRequestFromResponse(request: ResponseRequest): ChatCompletionRequest {
  const messages = responseInputMessages(request);
  if (request.instructions) messages.unshift({ role: "system", content: request.instructions });
  const cache = request.cache ? {
    ...request.cache,
    boundaries: request.cache.boundaries?.map((boundary) => boundary.kind === "messages" && request.instructions
      ? { ...boundary, through_message: boundary.through_message + 1 }
      : boundary),
  } : undefined;
  return {
    model: request.model,
    messages,
    stream: false,
    tools: request.tools,
    tool_choice: request.tool_choice,
    parallel_tool_calls: request.parallel_tool_calls,
    response_format: request.response_format ?? request.text?.format,
    temperature: request.temperature,
    top_p: request.top_p,
    max_tokens: request.max_output_tokens,
    cache,
  };
}

function responseInputMessages(request: ResponseRequest): ChatCompletionRequest["messages"] {
  if (typeof request.input === "string") return [{ role: "user", content: request.input }];
  return request.input.map((item) => {
    if (Array.isArray(item.content) && item.content.some((part) => part.type === "image_url")) {
      throw new Error("image input is not supported by the selected local text runtime yet");
    }
    return {
      role: item.role ?? "user",
      content: item.content ?? "",
      tool_call_id: item.tool_call_id,
    };
  });
}

function responseFromChat(request: ResponseRequest, chat: ChatCompletionResponse) {
  const choice = chat.choices[0];
  const message = choice?.message;
  type ResponseOutput =
    | { id: string; type: "reasoning"; status: "completed"; summary: Array<{ type: "summary_text"; text: string }>; content: Array<{ type: "reasoning_text"; text: string }> }
    | { id: string; type: "message"; status: "completed"; role: "assistant"; content: Array<{ type: "output_text"; text: string }> }
    | { id: string; type: "function_call"; status: "completed"; call_id: string; name: string; arguments: string };
  const output: ResponseOutput[] = [];
  if (message?.reasoning) {
    output.push({
      id: `rs_${crypto.randomUUID()}`,
      type: "reasoning" as const,
      status: "completed" as const,
      summary: [{ type: "summary_text" as const, text: message.reasoning }],
      content: [{ type: "reasoning_text" as const, text: message.reasoning }],
    });
  }
  if (message?.tool_calls?.length) {
    for (const call of message.tool_calls) {
      output.push({
        id: `fc_${crypto.randomUUID()}`,
        type: "function_call" as const,
        status: "completed" as const,
        call_id: call.id,
        name: call.function.name,
        arguments: call.function.arguments,
      });
    }
  } else {
    output.push({
      id: `msg_${crypto.randomUUID()}`,
      type: "message" as const,
      status: "completed" as const,
      role: "assistant" as const,
      content: [{ type: "output_text" as const, text: typeof message?.content === "string" ? message.content : "" }],
    });
  }
  const outputText = output
    .filter((item): item is Extract<(typeof output)[number], { type: "message" }> => item.type === "message")
    .flatMap((item) => item.content.map((part) => part.text))
    .join("");
  return {
    id: `resp_${crypto.randomUUID()}`,
    object: "response" as const,
    created_at: nowSeconds(),
    status: "completed" as const,
    model: request.model,
    output,
    output_text: outputText,
    usage: chat.usage ? {
      input_tokens: chat.usage.prompt_tokens,
      output_tokens: chat.usage.completion_tokens,
      total_tokens: chat.usage.total_tokens,
    } : undefined,
    error: null,
    incomplete_details: null,
    metadata: request.metadata,
  };
}

function streamResponseBody(body: ReturnType<typeof responseFromChat>): Response {
  const events: Array<{ event: string; data: unknown }> = [
    { event: "response.created", data: { ...body, output: [], output_text: "" } },
  ];
  for (const [index, item] of body.output.entries()) {
    events.push({ event: "response.output_item.added", data: { output_index: index, item } });
    if (item.type === "message") {
      const text = item.content.map((part) => part.text).join("");
      if (text) events.push({ event: "response.output_text.delta", data: { output_index: index, content_index: 0, delta: text } });
    } else if (item.type === "function_call") {
      events.push({ event: "response.function_call_arguments.delta", data: { output_index: index, item_id: item.id, delta: item.arguments } });
    } else {
      const text = item.content?.map((part) => part.text).join("") ?? item.summary?.map((part) => part.text).join("") ?? "";
      if (text) events.push({ event: "response.reasoning_text.delta", data: { output_index: index, item_id: item.id, delta: text } });
    }
  }
  events.push({ event: "response.completed", data: body });
  const content = events.map((entry) => `event: ${entry.event}\ndata: ${JSON.stringify(entry.data)}\n`).join("\n") + "\n";
  return new Response(content, { headers: { "content-type": "text/event-stream" } });
}

function ollamaTag(model: ClapModel) {
  return {
    name: model.id,
    model: model.id,
    modified_at: new Date(0).toISOString(),
    size: 0,
    digest: `sha256:${Bun.hash(model.id).toString(16)}`,
    details: ollamaDetails(model),
  };
}

function ollamaDetails(model: ClapModel) {
  return {
    parent_model: model.source.baseRepo ?? "",
    format: model.format,
    family: model.modelType ?? model.backend,
    families: model.modelType ? [model.modelType] : null,
    parameter_size: "unknown",
    quantization_level: model.quantization ?? "unknown",
  };
}

function findOllamaModel(name: string): ClapModel | undefined {
  return listModels().find((model) => model.id === name || model.name === name || model.displayName === name);
}

function ollamaNotFound(c: { json: (body: unknown, status?: number) => Response | Promise<Response> }, model: string) {
  return c.json({ error: `model '${model}' not found` }, 404);
}

function hasOllamaImages(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some(hasOllamaImages);
  const record = value as Record<string, unknown>;
  if (Array.isArray(record.images) && record.images.length > 0) return true;
  return Object.values(record).some(hasOllamaImages);
}

function ollamaChatResponse(c: { json: (body: unknown) => Response | Promise<Response> }, model: string, body: ChatCompletionResponse, stream: boolean) {
  const choice = body.choices[0];
  const message = {
    role: "assistant",
    content: typeof choice?.message.content === "string" ? choice.message.content : "",
    tool_calls: choice?.message.tool_calls,
  };
  const payload = { model, created_at: new Date().toISOString(), message, done: true, done_reason: choice?.finish_reason ?? "stop" };
  if (!stream) return c.json(payload);
  return ndjsonResponse([payload]);
}

function ollamaGenerateResponse(c: { json: (body: unknown) => Response | Promise<Response> }, model: string, body: ChatCompletionResponse, stream: boolean) {
  const choice = body.choices[0];
  const payload = {
    model,
    created_at: new Date().toISOString(),
    response: typeof choice?.message.content === "string" ? choice.message.content : "",
    done: true,
    done_reason: choice?.finish_reason ?? "stop",
  };
  if (!stream) return c.json(payload);
  return ndjsonResponse([payload]);
}

function ndjsonResponse(values: unknown[]): Response {
  return new Response(values.map((value) => JSON.stringify(value)).join("\n") + "\n", {
    headers: { "content-type": "application/x-ndjson" },
  });
}

type ChatCompletionChunk = {
  error?: { message?: string };
  choices?: Array<{
    delta?: {
      content?: string;
      reasoning?: string;
      tool_calls?: Array<{ index: number; id?: string; type?: string; function?: { name?: string; arguments?: string } }>;
    };
    finish_reason?: string | null;
  }>;
};

async function* sseCompletionChunks(response: Response): AsyncGenerator<ChatCompletionChunk> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let separator: number;
    while ((separator = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        if (data) yield JSON.parse(data) as ChatCompletionChunk;
      }
    }
  }
}

function ollamaStreamFromSse(model: string, sse: Response, kind: "chat" | "generate"): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    async start(controller) {
      const write = (value: unknown) => controller.enqueue(encoder.encode(`${JSON.stringify(value)}\n`));
      let doneReason = "stop";
      try {
        for await (const chunk of sseCompletionChunks(sse)) {
          if (chunk.error) {
            write({ error: chunk.error.message ?? "backend error" });
            controller.close();
            return;
          }
          const choice = chunk.choices?.[0];
          const delta = choice?.delta ?? {};
          if (choice?.finish_reason) doneReason = choice.finish_reason;
          const createdAt = new Date().toISOString();
          if (kind === "chat") {
            if (delta.content) write({ model, created_at: createdAt, message: { role: "assistant", content: delta.content }, done: false });
            if (delta.reasoning) write({ model, created_at: createdAt, message: { role: "assistant", content: "", thinking: delta.reasoning }, done: false });
            if (delta.tool_calls?.length) write({ model, created_at: createdAt, message: { role: "assistant", content: "", tool_calls: delta.tool_calls }, done: false });
          } else {
            if (delta.content) write({ model, created_at: createdAt, response: delta.content, done: false });
            if (delta.reasoning) write({ model, created_at: createdAt, response: "", thinking: delta.reasoning, done: false });
          }
        }
        const finalPayload = kind === "chat"
          ? { model, created_at: new Date().toISOString(), message: { role: "assistant", content: "" }, done: true, done_reason: doneReason }
          : { model, created_at: new Date().toISOString(), response: "", done: true, done_reason: doneReason };
        write(finalPayload);
      } catch (error) {
        write({ error: error instanceof Error ? error.message : String(error) });
      }
      controller.close();
    },
  }), { headers: { "content-type": "application/x-ndjson" } });
}

function resolveAvailableModel(model: string, backend?: "gguf" | "mlx"):
  | { model: ResolvedModel }
  | { response: (c: { json: (body: unknown, status?: number) => Response | Promise<Response> }) => Response | Promise<Response> } {
  const resolved = resolveModel(model, backend);
  if (resolved.status === "available") return { model: resolved };
  return {
    response: (c) => c.json(ErrorResponseSchema.parse({
      error: { message: resolved.message ?? `model is not available: ${model}`, type: "model_error", code: resolved.status },
    }), resolved.status === "unsupported" ? 400 : 404),
  };
}

function lifecycleKey(model: ResolvedModel): string {
  return JSON.stringify([model.id, model.backend, model.modelPath ?? model.input]);
}

async function assertResidentModelPath(model: ResolvedModel): Promise<void> {
  const path = model.modelPath ?? model.input;
  if (model.backend === "llama") await assertGgufModelPath(path);
  else await assertMlxModelPath(path);
}

function workerDescriptor(model: ResolvedModel) {
  return {
    modelId: model.id,
    revision: model.revision,
    artifactBytes: model.artifactBytes,
    architecture: model.architecture,
    modelType: model.modelType,
    quantization: model.quantization,
    context: model.context,
    configuredContext: model.configuredContext ?? configuredContextForBackend(model.backend),
    kv: model.kv,
    cacheBudget: model.cacheBudget,
  };
}

function configuredMemoryBytes(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? Math.ceil(value) : fallback;
}

function configuredContextForBackend(backend: ResolvedModel["backend"]): number | undefined {
  const value = Number(process.env[backend === "mlx" ? "CLAP_MLX_CONTEXT" : "CLAP_LLAMA_CONTEXT"]);
  return Number.isSafeInteger(value) && value > 0 ? value : 4_096;
}

async function jsonResidentResponse(c: { json: (body: ChatCompletionResponse) => Response | Promise<Response>; req: { raw: Request } }, residents: ResidentWorkerRegistry, entry: LoadedModel, request: ChatCompletionRequest, templateInfo: ParserTemplateInfo | undefined, handle: RequestHandle | undefined, cacheIdentity: CacheIdentity) {
  const worker = residents.getOrCreate(entry.key, entry.backend, entry.localPath, { modelId: entry.id });
  let result: ResidentChatResult | undefined;
  try {
    handle?.phase("loading");
    const loadStarted = Date.now();
    entry.worker = await worker.load();
    handle?.loaded(Date.now() - loadStarted);
    const structuredMode = structuredBackendMode(worker.info(), request);
    // The resident worker processes requests serially: mark this request as
    // queued until worker prefill progress or the first token proves it is
    // actually running.
    handle?.phase("queued");
    result = await worker.chat(request, () => handle?.firstToken(), c.req.raw.signal,
      (done, total) => handle?.prefill(done, total), () => handle?.phase("prefill"),
      { cacheIdentity, structuredOutput: structuredOutputContract(request) });
    entry.worker = worker.info();
    const structuredOutput = validateStructuredResult(result, request, structuredMode);
    const body = chatResponse(request, result, templateInfo);
    const message = body.choices[0]?.message;
    handle?.finish({
      status: result.finishReason === "cancel" ? "cancelled" : "ok",
      ...workerResultMetrics(result),
      promptTokens: result.usage?.promptTokens,
      completionTokens: result.usage?.completionTokens,
      cacheHit: result.cache?.hit,
      reusedTokens: result.cache?.reusedTokens,
      reuseKind: result.cache?.reuseKind,
      reuseScope: result.cache?.reuseScope,
      sideRequest: result.cache?.sideRequest,
      slot: result.cache?.slot,
      cacheNamespace: result.cache?.namespace,
      donorSlot: result.cache?.donorSlot,
      targetSlot: result.cache?.targetSlot,
      evictedSlots: result.cache?.evictedSlots,
      cacheDecisionUs: result.cache?.decisionUs,
      plannedReuseTokens: result.cache?.plannedReuseTokens,
      realizedReuseTokens: result.cache?.realizedReuseTokens,
      cacheFallback: result.cache?.fallback,
      finishReason: body.choices[0]?.finish_reason ?? undefined,
      toolCalls: message?.tool_calls?.length,
      response: {
        content: typeof message?.content === "string" ? message.content : undefined,
        reasoning: message?.reasoning ?? undefined,
        toolCalls: message?.tool_calls?.map((call) => ({ name: call.function.name, arguments: call.function.arguments })),
      },
      rawOutput: result.content,
      structuredOutput,
    });
    return c.json(body);
  } catch (error) {
    entry.worker = worker.info();
    handle?.finish({
      status: "error",
      error: error instanceof Error ? error.message : String(error),
      errorCode: error instanceof LlamaWorkerError || error instanceof MlxWorkerError ? error.code : "resident_worker_error",
      ...workerResultMetrics(result),
      workerLaunchId: result?.cache?.workerLaunchId ?? worker.info().launchId,
      rawOutput: result?.content,
      structuredOutput: structuredFailureFacts(error, worker.info(), request),
    });
    throw error;
  }
}

type StrictSseStream = {
  aborted: boolean;
  writeSSE(message: { data: string; event?: string; id?: string; retry?: number }): Promise<void>;
  onAbort(listener: () => void | Promise<void>): void;
};

function strictStreamSSE(c: Parameters<typeof streamSSE>[0], callback: (stream: StrictSseStream) => Promise<void>) {
  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
  const writer = writable.getWriter();
  const reader = readable.getReader();
  const encoder = new TextEncoder();
  const abortSubscribers: Array<() => void | Promise<void>> = [];
  const stream: StrictSseStream = {
    aborted: false,
    async writeSSE(message) {
      const data = message.data.split(/\r\n|\r|\n/).map((line) => `data: ${line}`).join("\n");
      const payload = [
        message.event && `event: ${message.event}`,
        data,
        message.id && `id: ${message.id}`,
        message.retry && `retry: ${message.retry}`,
      ].filter(Boolean).join("\n") + "\n\n";
      await writer.write(encoder.encode(payload));
    },
    onAbort(listener) {
      abortSubscribers.push(listener);
    },
  };
  const abort = () => {
    if (stream.aborted) return;
    stream.aborted = true;
    for (const subscriber of abortSubscribers) void subscriber();
  };
  const responseReadable = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) controller.close();
      else controller.enqueue(value);
    },
    async cancel() {
      abort();
      await reader.cancel().catch(() => undefined);
    },
  });
  c.header("Transfer-Encoding", "chunked");
  c.header("Content-Type", "text/event-stream");
  c.header("Cache-Control", "no-cache");
  c.header("Connection", "keep-alive");
  void (async () => {
    try {
      await callback(stream);
    } finally {
      await writer.close().catch(() => undefined);
    }
  })();
  return c.newResponse(responseReadable);
}

function streamResidentResponse(c: Parameters<typeof streamSSE>[0], residents: ResidentWorkerRegistry, entry: LoadedModel, request: ChatCompletionRequest, templateInfo: ParserTemplateInfo | undefined, onDone: (() => void) | undefined, handle: RequestHandle | undefined, cacheIdentity: CacheIdentity) {
  const worker = residents.getOrCreate(entry.key, entry.backend, entry.localPath, { modelId: entry.id });
  return strictStreamSSE(c, async (stream) => {
    const id = completionId();
    const created = nowSeconds();
    const aborter = new AbortController();
    stream.onAbort(() => aborter.abort());
    if (c.req.raw.signal.aborted) aborter.abort();
    else c.req.raw.signal.addEventListener("abort", () => aborter.abort(), { once: true });
    let wroteRole = false;
    let writeQueue = Promise.resolve();
    const enqueueWrite = (write: () => Promise<void>) => {
      const queued = writeQueue.then(write);
      writeQueue = queued;
      void queued.catch(() => aborter.abort());
    };
    const ensureRole = async () => {
      if (wroteRole) return;
      wroteRole = true;
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { role: "assistant" })) });
    };
    // Send the role chunk immediately so clients get a first byte right away,
    // then heartbeat with empty delta chunks while the worker ingests a long
    // prompt. Real data chunks (not SSE comments) are required: stream parsers
    // ignore comments, so comment-only keepalives still trip client
    // inactivity timeouts during multi-minute prefills.
    await ensureRole();
    let sawOutput = false;
    const heartbeat = setInterval(() => {
      if (aborter.signal.aborted) return;
      enqueueWrite(async () => {
        await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {})) });
      });
    }, 5_000);
    const writeDelta = async (delta: StreamDelta) => {
      await ensureRole();
      const payload = delta.type === "reasoning"
        ? { reasoning: delta.text, reasoning_content: delta.text }
        : { content: delta.text };
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, payload)) });
    };
    let result: ResidentChatResult | undefined;
    try {
      handle?.phase("loading");
      const loadStarted = Date.now();
      entry.worker = await worker.load();
      handle?.loaded(Date.now() - loadStarted);
      const structuredMode = structuredBackendMode(worker.info(), request);
      handle?.phase("queued");
      const streamExtras = profileStreamExtras(request.model, request, templateInfo);
      const filter = new StreamingOutputFilter({
        toolMode: Boolean(request.tools?.length),
        bufferAll: Boolean(request.response_format && request.response_format.type !== "text"),
        stops: typeof request.stop === "string" ? [request.stop] : request.stop ?? [],
        startInReasoning: streamExtras.implicitThink,
        extraMarkers: streamExtras.extraMarkers,
      });
      result = await worker.chat(request, (token) => {
        if (!sawOutput) handle?.firstToken();
        sawOutput = true;
        const deltas = filter.feed(token);
        if (!deltas.length) return;
        enqueueWrite(async () => {
          for (const delta of deltas) await writeDelta(delta);
        });
      }, aborter.signal, (done, total) => handle?.prefill(done, total), () => handle?.phase("prefill"),
      { cacheIdentity, structuredOutput: structuredOutputContract(request) });
      // A disconnected client makes queued SSE writes reject before the
      // worker's cancellation result arrives. Drain that queue without
      // propagating its transport error so the final worker cache decision can
      // still be merged into metrics. worker.chat remains the bounded cleanup:
      // resident workers must answer cancel or their normal watchdog rejects.
      await writeQueue.catch(() => undefined);
      entry.worker = worker.info();
      if (result.finishReason === "cancel" || aborter.signal.aborted) {
        handle?.finish({
          status: "cancelled",
          ...workerResultMetrics(result),
          promptTokens: result.usage?.promptTokens,
          completionTokens: result.usage?.completionTokens,
          cacheHit: result.cache?.hit,
          reusedTokens: result.cache?.reusedTokens,
          reuseKind: result.cache?.reuseKind,
          reuseScope: result.cache?.reuseScope,
          sideRequest: result.cache?.sideRequest,
          slot: result.cache?.slot,
          cacheNamespace: result.cache?.namespace,
          donorSlot: result.cache?.donorSlot,
          targetSlot: result.cache?.targetSlot,
          evictedSlots: result.cache?.evictedSlots,
          cacheDecisionUs: result.cache?.decisionUs,
          plannedReuseTokens: result.cache?.plannedReuseTokens,
          realizedReuseTokens: result.cache?.realizedReuseTokens,
          cacheFallback: result.cache?.fallback,
          finishReason: "cancel",
          structuredOutput: structuredOutputContract(request) ? { backendMode: structuredMode } : undefined,
        });
        return;
      }
      const structuredOutput = validateStructuredResult(result, request, structuredMode);
      const parsed = parseAssistantOutput(result.content, request, templateInfo, { truncated: result.finishReason === "length" });
      await ensureRole();
      const reasoningTail = remainingDelta(parsed.reasoning, filter.emittedReasoning);
      if (reasoningTail) await writeDelta({ type: "reasoning", text: reasoningTail });
      if (parsed.toolCalls?.length) {
        for (const [index, call] of parsed.toolCalls.entries()) {
          await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {
            tool_calls: [{ index, id: call.id, type: "function", function: { name: call.function.name, arguments: call.function.arguments } }],
          })) });
        }
      } else {
        const contentTail = remainingDelta(parsed.content, filter.emittedContent);
        if (contentTail) await writeDelta({ type: "content", text: contentTail });
      }
      const finishReason = parsed.finishReason === "tool_calls" ? "tool_calls" : workerFinishReason(result) ?? parsed.finishReason;
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {}, finishReason, usageFor(request, result))) });
      await stream.writeSSE({ data: "[DONE]" });
      handle?.finish({
        status: "ok",
        ...workerResultMetrics(result),
        promptTokens: result.usage?.promptTokens,
        completionTokens: result.usage?.completionTokens,
        cacheHit: result.cache?.hit,
        reusedTokens: result.cache?.reusedTokens,
        reuseKind: result.cache?.reuseKind,
        reuseScope: result.cache?.reuseScope,
        sideRequest: result.cache?.sideRequest,
        slot: result.cache?.slot,
        cacheNamespace: result.cache?.namespace,
        donorSlot: result.cache?.donorSlot,
        targetSlot: result.cache?.targetSlot,
        evictedSlots: result.cache?.evictedSlots,
        cacheDecisionUs: result.cache?.decisionUs,
        plannedReuseTokens: result.cache?.plannedReuseTokens,
        realizedReuseTokens: result.cache?.realizedReuseTokens,
        cacheFallback: result.cache?.fallback,
        finishReason,
        toolCalls: parsed.toolCalls?.length,
        response: {
          content: parsed.content ?? undefined,
          reasoning: parsed.reasoning,
          toolCalls: parsed.toolCalls?.map((call) => ({ name: call.function.name, arguments: call.function.arguments })),
        },
        rawOutput: result.content,
        structuredOutput,
      });
    } catch (error) {
      entry.worker = worker.info();
      await writeQueue.catch(() => undefined);
      if (aborter.signal.aborted) {
        handle?.finish({ status: "cancelled", finishReason: "cancel", workerLaunchId: worker.info().launchId });
        return;
      }
      handle?.finish({
        status: "error",
        error: error instanceof Error ? error.message : String(error),
        errorCode: error instanceof LlamaWorkerError || error instanceof MlxWorkerError ? error.code : "resident_worker_error",
        ...workerResultMetrics(result),
        workerLaunchId: result?.cache?.workerLaunchId ?? worker.info().launchId,
        rawOutput: result?.content,
        structuredOutput: structuredFailureFacts(error, worker.info(), request),
      });
      await stream.writeSSE({ data: JSON.stringify({ error: backendErrorBody(error).error }) }).catch(() => undefined);
    } finally {
      clearInterval(heartbeat);
      onDone?.();
    }
  });
}

function workerResultMetrics(result?: ResidentChatResult) {
  return {
    promptTokens: result?.usage?.promptTokens,
    completionTokens: result?.usage?.completionTokens,
    cacheHit: result?.cache?.hit,
    reusedTokens: result?.cache?.reusedTokens,
    reuseKind: result?.cache?.reuseKind,
    reuseScope: result?.cache?.reuseScope,
    sideRequest: result?.cache?.sideRequest,
    slot: result?.cache?.slot,
    cacheNamespace: result?.cache?.namespace,
    donorSlot: result?.cache?.donorSlot,
    targetSlot: result?.cache?.targetSlot,
    evictedSlots: result?.cache?.evictedSlots,
    cacheDecisionUs: result?.cache?.decisionUs,
    plannedReuseTokens: result?.cache?.plannedReuseTokens,
    realizedReuseTokens: result?.cache?.realizedReuseTokens,
    cacheFallback: result?.cache?.fallback,
    cacheMissReason: result?.cache?.missReason,
    workerLaunchId: result?.cache?.workerLaunchId,
    donorGeneration: result?.cache?.donorGeneration,
    targetGeneration: result?.cache?.targetGeneration,
    cacheEvictions: result?.cache?.evictions,
    cacheCandidates: result?.cache?.candidates,
    systemTokenHash: result?.cache?.systemTokenHash,
    systemTokenCount: result?.cache?.systemTokenCount,
    toolsTokenHash: result?.cache?.toolsTokenHash,
    toolsTokenCount: result?.cache?.toolsTokenCount,
    stableBoundaryTokenHash: result?.cache?.stableBoundaryTokenHash,
    stableBoundaryTokenCount: result?.cache?.stableBoundaryTokenCount,
    stableBoundaryKind: result?.cache?.stableBoundaryKind,
    stableBoundaries: result?.cache?.stableBoundaries,
    promptTokenHash: result?.cache?.promptTokenHash,
    promptTokenCount: result?.cache?.promptTokenCount,
    prefillMs: result?.timing?.prefillMs ?? result?.cache?.prefillMs,
    timing: result?.timing,
    finishReason: result?.finishReason,
  };
}

function backendErrorBody(error: unknown) {
  if (error instanceof InsufficientModelMemoryError) return insufficientMemoryBody(error);
  const message = error instanceof Error ? error.message : String(error);
  const code = error instanceof StructuredOutputError ? error.code
    : error instanceof LlamaWorkerError || error instanceof MlxWorkerError ? error.code
    : "resident_worker_error";
  return ErrorResponseSchema.parse({
    error: { message, type: "backend_error", code },
  });
}

type StructuredWorkerInfo = ReturnType<ReturnType<ResidentWorkerRegistry["getOrCreate"]>["info"]>;

function structuredBackendMode(worker: StructuredWorkerInfo, request: ChatCompletionRequest): "native" | "post_validate" | undefined {
  const contract = structuredOutputContract(request);
  if (!contract) return undefined;
  const mode = worker.effectiveCapabilities?.generation.structuredOutput[contract.kind];
  return mode === "native" || mode === "post_validate" ? mode : undefined;
}

function structuredFailureFacts(error: unknown, worker: StructuredWorkerInfo, request: ChatCompletionRequest) {
  const contract = structuredOutputContract(request);
  if (!contract) return undefined;
  return {
    backendMode: structuredBackendMode(worker, request),
    outcome: error instanceof LlamaWorkerError || error instanceof MlxWorkerError
      ? error.code === "structured_output_capability_required" ? "capability_rejected" as const : "invalid" as const
      : "invalid" as const,
    repairApplied: false,
    validationMs: error instanceof StructuredOutputError ? error.validationMs : undefined,
  };
}

function validateStructuredResult(result: ResidentChatResult, request: ChatCompletionRequest,
                                  backendMode?: "native" | "post_validate") {
  const format = request.response_format;
  if (!format || format.type === "text" || result.finishReason === "cancel") return undefined;
  const started = performance.now();
  const outcome = parseStructuredOutput(result.content, format);
  const validationMs = Math.round((performance.now() - started) * 100) / 100;
  if (!outcome.ok) {
    throw new StructuredOutputError(
      outcome.error.code === "invalid_schema" ? "schema_unsupported" : "structured_output_invalid",
      outcome.error.message, false, validationMs);
  }
  result.content = outcome.json;
  return {
    backendMode,
    outcome: outcome.repaired ? "repaired_validated" as const
      : backendMode === "native" ? "native_validated" as const : "validated" as const,
    repairApplied: outcome.repaired,
    validationMs,
  };
}

function structuredOutputContract(request: ChatCompletionRequest): StructuredOutputContract | undefined {
  const format = request.response_format;
  if (!format || format.type === "text") return undefined;
  const strength = format.constraint === "required"
    || (format.type === "json_schema" && format.json_schema.strict === true)
    ? "required" : "best_effort";
  return format.type === "json_object"
    ? { kind: "json_object", strength }
    : { kind: "json_schema", strength, schema: format.json_schema.schema };
}

function insufficientMemoryBody(error: InsufficientModelMemoryError) {
  return ErrorResponseSchema.parse({
    error: {
      message: "Insufficient memory to load the requested model safely",
      type: "model_error",
      code: error.code,
      details: error.details,
    },
  });
}

function insufficientMemoryResponse(c: Context, error: InsufficientModelMemoryError) {
  const mapped = mapInsufficientModelMemoryError(error);
  c.header("Retry-After", mapped.retryAfter);
  return c.json(mapped.body, mapped.status);
}

export function mapInsufficientModelMemoryError(error: InsufficientModelMemoryError) {
  return { status: 503 as const, retryAfter: "5", body: insufficientMemoryBody(error) };
}

export function startServer(options: ServerOptions = {}) {
  const { config } = loadClapConfig();
  const port = options.port ?? portFromEnv(config.server.port);
  const hostname = options.hostname ?? hostnameFromEnv(config.server.host);
  const idleTimeout = options.idleTimeout ?? idleTimeoutFromEnv(config.server.idle_timeout_seconds);
  const residents = new ResidentWorkerRegistry();
  const lifecycle = new ModelLifecycleManager(() => Date.now(), (entry) => residents.shutdownAsync(entry.key));
  const server = Bun.serve({
    port,
    hostname,
    idleTimeout,
    fetch: createServer(residents, lifecycle).fetch,
  });
  installShutdownCleanup(server, lifecycle, residents);
  return server;
}

function installShutdownCleanup(server: ReturnType<typeof Bun.serve>, lifecycle: ModelLifecycleManager, residents: ResidentWorkerRegistry): void {
  const cleanup = () => {
    lifecycle.cleanup();
    residents.shutdownAll();
    server.stop(true);
    process.exit(0);
  };
  process.once("SIGINT", cleanup);
  process.once("SIGTERM", cleanup);
}

function portFromEnv(configPort?: number): number {
  if (process.env.PORT === undefined && process.env.CLAP_BASE_URL === undefined && configPort !== undefined) return configPort;
  const fromUrl = new URL(process.env.CLAP_BASE_URL ?? defaultBaseURL).port;
  return Number(process.env.PORT ?? fromUrl ?? 11435);
}

function hostnameFromEnv(configHost?: string): string {
  const fromEnv = process.env.CLAP_HOST?.trim();
  if (fromEnv) return fromEnv;
  if (configHost) return configHost;
  try {
    const fromUrl = new URL(process.env.CLAP_BASE_URL ?? defaultBaseURL).hostname;
    if (fromUrl && fromUrl !== "localhost") return fromUrl;
  } catch {
    // fall through to loopback default
  }
  return "127.0.0.1";
}

export function idleTimeoutFromEnv(configSeconds?: number): number {
  const raw = process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS;
  if (raw === undefined || raw.trim() === "") {
    if (configSeconds !== undefined) return configSeconds === 0 ? 0 : Math.min(configSeconds, maxBunIdleTimeoutSeconds);
    return defaultIdleTimeoutSeconds;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) return defaultIdleTimeoutSeconds;
  if (value === 0) return 0;
  return Math.min(value, maxBunIdleTimeoutSeconds);
}

function completionId(): string {
  return `chatcmpl_${crypto.randomUUID()}`;
}

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function workerFinishReason(result: ResidentChatResult): "stop" | "length" | undefined {
  return result.finishReason === "cancel" ? "stop" : result.finishReason;
}

function chatResponse(request: ChatCompletionRequest, result: ResidentChatResult, templateInfo?: ParserTemplateInfo): ChatCompletionResponse {
  if (process.env.CLAP_DEBUG_RAW) {
    console.error(`[clap] raw model output (${result.content.length} chars): ${JSON.stringify(result.content)}`);
  }
  const parsed = parseAssistantOutput(result.content, request, templateInfo, { truncated: result.finishReason === "length" });
  const message = {
    role: "assistant" as const,
    content: parsed.content,
    reasoning: parsed.reasoning,
    reasoning_content: parsed.reasoning,
    tool_calls: parsed.toolCalls,
  };
  return {
    id: completionId(),
    object: "chat.completion",
    created: nowSeconds(),
    model: request.model,
    choices: [{
      index: 0,
      message,
      finish_reason: parsed.finishReason === "tool_calls" ? "tool_calls" : workerFinishReason(result) ?? parsed.finishReason,
    }],
    usage: usageFor(request, result),
  };
}

function usageFor(request: ChatCompletionRequest, result: ResidentChatResult) {
  const promptTokens = result.usage?.promptTokens ?? estimateTokens(JSON.stringify(request.messages));
  const completionTokens = result.usage?.completionTokens ?? estimateTokens(result.content);
  return { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function chunk(
  id: string,
  created: number,
  model: string,
  delta: { role?: "assistant"; content?: string | null; reasoning?: string; reasoning_content?: string; tool_calls?: Array<{ index: number; id?: string; type?: "function"; function?: { name?: string; arguments?: string } }> },
  finishReason: "stop" | "length" | "tool_calls" | "content_filter" | "function_call" | null = null,
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number },
) {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    usage,
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

if (import.meta.main) {
  const server = startServer();
  console.log(`clap server listening on http://${server.hostname}:${server.port}`);
}
