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
  PullModelRequestSchema,
  PullModelResponseSchema,
  UnloadModelRequestSchema,
  UnloadModelResponseSchema,
  type ChatCompletionRequest,
  type ChatCompletionResponse,
  type Download,
  type LoadedModel,
} from "@clap/api";
import { cachedPullResultForTarget, listAliases, listModels, pullModel, resolveModel, resolvePullTarget, type ResolvedModel } from "@clap/models";
import { completeWithLlama, isGgufModel, LlamaWorkerError, streamWithLlama } from "@clap/runtime-llama";
import { completeWithMlx, isMlxModelDirectory, MlxWorkerError, streamWithMlx } from "@clap/runtime-mlx";
import { listBackends, ModelLifecycleManager } from "@clap/runtime-router";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";

const startedAt = Date.now();
const downloads = new Map<string, Download>();
const activeDownloads = new Map<string, { id: string; controller: AbortController }>();

export type ServerOptions = {
  port?: number;
  hostname?: string;
};

export function createServer(lifecycle = new ModelLifecycleManager()) {
  const app = new Hono();

  app.onError((error, c) => {
    if (error instanceof z.ZodError) {
      return c.json({ error: { message: error.message, type: "invalid_request_error", code: "invalid_json" } }, 400);
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
    return c.json({ error: { message: error.message, type: "server_error" } }, 500);
  });

  app.get("/openapi.json", (c) => c.json(createOpenApiDocument()));

  app.get("/clap/v1/health", (c) => c.json({
    status: "ok",
    version: clapVersion,
    uptimeMs: Date.now() - startedAt,
  }));

  app.get("/clap/v1/runtime", (c) => c.json({
    platform: process.platform,
    arch: process.arch,
    bunVersion: Bun.version,
    runtime: "bun",
  }));

  app.get("/clap/v1/backends", (c) => c.json({ backends: listBackends() }));

  app.get("/clap/v1/models", (c) => c.json({ models: listModels() }));

  app.get("/clap/v1/aliases", (c) => c.json({ models: listAliases() }));

  app.get("/clap/v1/downloads", (c) => c.json(DownloadsResponseSchema.parse({
    downloads: [...downloads.values()],
  })));

  app.get("/clap/v1/runtime/models", (c) => c.json(LoadedModelsResponseSchema.parse({ models: lifecycle.list() })));

  app.post("/clap/v1/models/load", async (c) => {
    const request = LoadModelRequestSchema.parse(await c.req.json());
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) return resolved.response(c);
    const model = lifecycle.load(resolved.model, { keepAlive: request.keepAlive });
    return c.json(LoadModelResponseSchema.parse({ model }));
  });

  app.post("/clap/v1/models/unload", async (c) => {
    const request = UnloadModelRequestSchema.parse(await c.req.json());
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) return resolved.response(c);
    return c.json(UnloadModelResponseSchema.parse(lifecycle.unload(resolved.model)));
  });

  app.post("/clap/v1/models/pull", async (c) => {
    const request = PullModelRequestSchema.parse(await c.req.json());
    const target = resolvePullTarget(request);
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
      file: request.file,
      backend: request.backend,
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
    void pullModel(request, {
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
    return c.json({
      object: "list",
      data: listModels().map((model) => ({
        ...(includeMetadata ? model : { id: model.id, object: model.object }),
        created: 0,
        owned_by: "clap",
      })),
    });
  });

  app.post("/v1/chat/completions", async (c) => {
    const request = ChatCompletionRequestSchema.parse(await c.req.json());
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) return resolved.response(c);

    const routedRequest = { ...request, model: resolved.model.modelPath ?? request.model };
    if (isGgufModel(routedRequest.model)) {
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(resolved.model);
        return streamLlamaResponse(c, routedRequest, () => lifecycle.finishUsage(loaded));
      }
      return lifecycle.withUsage(resolved.model, () => jsonLlamaResponse(c, routedRequest));
    }
    if (await isMlxModelDirectory(routedRequest.model)) {
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(resolved.model);
        return streamMlxResponse(c, routedRequest, () => lifecycle.finishUsage(loaded));
      }
      return lifecycle.withUsage(resolved.model, () => jsonMlxResponse(c, routedRequest));
    }

    return c.json(ErrorResponseSchema.parse({
      error: { message: `Model ${request.model} is not cached as a GGUF file or MLX directory. Run: clap pull ${request.model}, or pass a local .gguf file / MLX directory.`, type: "model_error", code: "not_cached" },
    }), 404);
  });

  return app;
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

async function jsonLlamaResponse(c: { json: (body: ChatCompletionResponse) => Response | Promise<Response> }, request: ChatCompletionRequest) {
  const content = await completeWithLlama({ request, stream: false });
  return c.json({
    id: completionId(),
    object: "chat.completion",
    created: nowSeconds(),
    model: request.model,
    choices: [{
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    }],
  });
}

function streamLlamaResponse(c: Parameters<typeof streamSSE>[0], request: ChatCompletionRequest, onDone?: () => void) {
  return streamSSE(c, async (stream) => {
    try {
      const id = completionId();
      const created = nowSeconds();
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { role: "assistant" })) });
      for await (const token of streamWithLlama({ request, stream: true })) {
        await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { content: token })) });
      }
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {}, "stop")) });
      await stream.writeSSE({ data: "[DONE]" });
    } finally {
      onDone?.();
    }
  });
}

async function jsonMlxResponse(c: { json: (body: ChatCompletionResponse) => Response | Promise<Response> }, request: ChatCompletionRequest) {
  const content = await completeWithMlx({ request, stream: false });
  return c.json({
    id: completionId(),
    object: "chat.completion",
    created: nowSeconds(),
    model: request.model,
    choices: [{
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    }],
  });
}

function streamMlxResponse(c: Parameters<typeof streamSSE>[0], request: ChatCompletionRequest, onDone?: () => void) {
  return streamSSE(c, async (stream) => {
    try {
      const id = completionId();
      const created = nowSeconds();
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { role: "assistant" })) });
      for await (const token of streamWithMlx({ request, stream: true })) {
        await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { content: token })) });
      }
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {}, "stop")) });
      await stream.writeSSE({ data: "[DONE]" });
    } finally {
      onDone?.();
    }
  });
}

export function startServer(options: ServerOptions = {}) {
  const port = options.port ?? portFromEnv();
  const hostname = options.hostname ?? "127.0.0.1";
  const lifecycle = new ModelLifecycleManager();
  const server = Bun.serve({
    port,
    hostname,
    fetch: createServer(lifecycle).fetch,
  });
  installShutdownCleanup(server, lifecycle);
  return server;
}

function installShutdownCleanup(server: ReturnType<typeof Bun.serve>, lifecycle: ModelLifecycleManager): void {
  const cleanup = () => {
    lifecycle.cleanup();
    server.stop(true);
    process.exit(0);
  };
  process.once("SIGINT", cleanup);
  process.once("SIGTERM", cleanup);
}

function portFromEnv(): number {
  const fromUrl = new URL(process.env.CLAP_BASE_URL ?? defaultBaseURL).port;
  return Number(process.env.PORT ?? fromUrl ?? 11435);
}

function completionId(): string {
  return `chatcmpl_${crypto.randomUUID()}`;
}

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function chunk(
  id: string,
  created: number,
  model: string,
  delta: { role?: "assistant"; content?: string },
  finishReason: string | null = null,
) {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

if (import.meta.main) {
  const server = startServer();
  console.log(`clap server listening on http://${server.hostname}:${server.port}`);
}
