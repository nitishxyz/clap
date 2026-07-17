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
import { cachedPullResultForTarget, listAliases, listModels, listModelsAsync, pullModel, removeModel, resolveModel, resolveModelOptions, resolvePullTarget, type ResolvedModel } from "@clap/models";
import { assertGgufModelPath, isGgufModel, LlamaWorkerError } from "@clap/runtime-llama";
import { assertMlxModelPath, isMlxModelDirectory, MlxWorkerError } from "@clap/runtime-mlx";
import { listBackends, ModelLifecycleManager, ResidentWorkerRegistry, type ResidentChatResult } from "@clap/runtime-router";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { readFile, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { z } from "zod";
import { parseAssistantOutput, prepareChatRequest, profileStreamExtras, remainingDelta, StreamingOutputFilter, type ParserTemplateInfo, type StreamDelta } from "./chat-compat";
import { MetricsCollector, type RequestHandle } from "./metrics";
import { sampleProcessUsage, systemMemoryBytes } from "./process-usage";
import { webAsset } from "./web-assets";

const startedAt = Date.now();
const downloads = new Map<string, Download>();
const activeDownloads = new Map<string, { id: string; controller: AbortController }>();
const defaultIdleTimeoutSeconds = 240;
const maxBunIdleTimeoutSeconds = 255;

export type ServerOptions = {
  port?: number;
  hostname?: string;
  idleTimeout?: number;
};

export function createServer(
  residents = new ResidentWorkerRegistry(),
  lifecycle = new ModelLifecycleManager(() => Date.now(), (entry) => residents.shutdown(entry.key)),
) {
  const app = new Hono();
  const metrics = new MetricsCollector();
  metrics.event("server", `clap server started (v${clapVersion})`);
  lifecycle.removeListener = (entry, reason) => {
    if (reason === "cleanup") return;
    metrics.event(reason === "expire" ? "expire" : "unload", `${entry.id} ${reason === "expire" ? "expired after idle keep-alive" : "unloaded"} (${entry.backend})`, { model: entry.id });
  };

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

  app.get("/clap/v1/runtime/models", (c) => c.json(LoadedModelsResponseSchema.parse({ models: lifecycle.list() })));

  app.get("/clap/v1/dashboard", async (c) => {
    const loaded = lifecycle.list();
    const workerPids = loaded
      .map((entry) => entry.worker?.pid)
      .filter((pid): pid is number => typeof pid === "number");
    const usage = await sampleProcessUsage([process.pid, ...workerPids]);
    return c.json({
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
      },
      totals: metrics.totals,
      active: metrics.activeRequests(),
      requests: metrics.recent(80),
      events: metrics.events(50),
      loaded: loaded.map((entry) => ({
        ...entry,
        usage: entry.worker?.pid ? usage.get(entry.worker.pid) : undefined,
      })),
      models: await listModelsAsync(),
      downloads: [...downloads.values()],
    });
  });

  app.get("/clap/v1/dashboard/requests/:id", (c) => {
    const record = metrics.request(c.req.param("id"));
    if (!record) return c.json({ error: { message: "request not found", type: "invalid_request_error" } }, 404);
    return c.json(record);
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
    const worker = residents.getOrCreate(lifecycleKey(resolved.model), resolved.model.backend, resolved.model.modelPath ?? resolved.model.input);
    let info;
    try {
      info = await worker.load();
    } catch (error) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error instanceof Error ? error.message : String(error), type: "backend_error", code: "resident_worker_error" },
      }), 503);
    }
    const model = lifecycle.load(resolved.model, { keepAlive: request.keepAlive, worker: info });
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
    const request = ChatCompletionRequestSchema.parse(await c.req.json());
    const resolved = resolveAvailableModel(request.model, request.backend);
    if ("response" in resolved) return resolved.response(c);

    const templateInfo = await resolveParserTemplateInfo(resolved.model);
    let routedRequest: ChatCompletionRequest;
    try {
      routedRequest = prepareChatRequest(
        { ...request, model: resolved.model.modelPath ?? request.model },
        { nativeTools: resolved.model.backend === "mlx" && Boolean(templateInfo?.hasToolCalls) },
      );
    } catch (error) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error instanceof Error ? error.message : String(error), type: "invalid_request_error", code: "unsupported_content_part" },
      }), 400);
    }
    if (isGgufModel(routedRequest.model)) {
      await assertGgufModelPath(routedRequest.model);
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(resolved.model);
        return streamResidentResponse(c, residents, loaded, routedRequest, templateInfo, () => lifecycle.finishUsage(loaded), metrics.start(request.model, "/v1/chat/completions", true));
      }
      return lifecycle.withUsage(resolved.model, (entry) => jsonResidentResponse(c, residents, entry, routedRequest, templateInfo, metrics.start(request.model, "/v1/chat/completions", false)));
    }
    if (await isMlxModelDirectory(routedRequest.model)) {
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(resolved.model);
        return streamResidentResponse(c, residents, loaded, routedRequest, templateInfo, () => lifecycle.finishUsage(loaded), metrics.start(request.model, "/v1/chat/completions", true));
      }
      return lifecycle.withUsage(resolved.model, (entry) => jsonResidentResponse(c, residents, entry, routedRequest, templateInfo, metrics.start(request.model, "/v1/chat/completions", false)));
    }

    return c.json(ErrorResponseSchema.parse({
      error: { message: `Model ${request.model} is not cached as a GGUF file or MLX directory. Run: clap pull ${request.model}, or pass a local .gguf file / MLX directory.`, type: "model_error", code: "not_cached" },
    }), 404);
  });

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
      headers: { "content-type": "application/json" },
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
      headers: { "content-type": "application/json" },
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
      headers: { "content-type": "application/json" },
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

async function resolveParserTemplateInfo(model: ResolvedModel): Promise<ParserTemplateInfo | undefined> {
  const root = await metadataRoot(model);
  if (!root) return undefined;
  const sourceFiles: string[] = [];
  const chunks: string[] = [];
  const nameChunks: string[] = [];
  for (const file of ["tokenizer_config.json", "tokenizer.json", "config.json", "generation_config.json", "chat_template.jinja"]) {
    const path = join(root, file);
    try {
      const text = await readFile(path, "utf8");
      sourceFiles.push(file);
      chunks.push(text.slice(0, 250_000));
      // tokenizer.json is mostly raw vocabulary; family names like "harmony"
      // or "hermes" appear as ordinary word tokens in any large vocab, so only
      // template/config files may vote on family by name.
      if (file !== "tokenizer.json") nameChunks.push(text.slice(0, 250_000));
    } catch {
      // optional metadata file
    }
  }
  if (!chunks.length) return undefined;
  const haystack = chunks.join("\n").toLowerCase();
  const familyHints = inferParserFamilies(haystack, nameChunks.join("\n").toLowerCase());
  return {
    familyHints,
    hasToolCalls: /tool_call|tool_calls|\[tool_calls\]|python_tag|call:/.test(haystack),
    hasReasoning: /enable_thinking|reasoning_effort|reasoning_content|<think>|analysis|commentary|final/.test(haystack),
    implicitThink: templatePrefillsThink(chunks.join("\n")),
    sourceFiles,
  };
}

// Detects chat templates (Qwen3.6, DeepSeek-R1, ...) that emit a literal
// <think> tag as part of the generation prompt, which means the model's raw
// output begins inside a reasoning block without an opening tag.
function templatePrefillsThink(templateText: string): boolean {
  for (const match of templateText.matchAll(/\{\{-?\s*'([^']*)'\s*-?\}\}/g)) {
    const literal = match[1] ?? "";
    if (literal.includes("<think>") && !literal.includes("</think>")) return true;
  }
  return false;
}

async function metadataRoot(model: ResolvedModel): Promise<string | undefined> {
  const path = model.modelPath ?? model.input;
  try {
    const info = await stat(path);
    return info.isDirectory() ? path : dirname(path);
  } catch {
    return undefined;
  }
}

export function inferParserFamilies(markerText: string, nameText: string): string[] {
  const hints: string[] = [];
  const add = (family: string, markers: RegExp, names?: RegExp) => {
    if ((markers.test(markerText) || (names?.test(nameText) ?? false)) && !hints.includes(family)) hints.push(family);
  };
  // Marker patterns are distinctive protocol tokens safe to match anywhere
  // (including tokenizer vocabulary); family-name patterns only run against
  // template/config metadata where a name is an intentional signal.
  add("harmony", /<\|channel\|>/, /gpt-oss|harmony/);
  // "enable_thinking" is intentionally not a qwen signal: other families
  // (gemma included) adopted the same template toggle name.
  add("qwen", /<\|tool_call_start\|>|<function=/, /qwen/);
  add("deepseek", /<｜tool▁calls▁begin｜>/, /deepseek/);
  add("mistral", /\[tool_calls\]/, /mistral|mixtral/);
  add("llama", /<\|python_tag\|>/, undefined);
  add("gemma", /functiongemma/, /gemma/);
  add("hermes", /<tool_call>/, /hermes|xlam|functionary/);
  return hints;
}

async function jsonResidentResponse(c: { json: (body: ChatCompletionResponse) => Response | Promise<Response>; req: { raw: Request } }, residents: ResidentWorkerRegistry, entry: LoadedModel, request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo, handle?: RequestHandle) {
  try {
    handle?.capture(request);
    const worker = residents.getOrCreate(entry.key, entry.backend, entry.localPath);
    handle?.phase("loading");
    const loadStarted = Date.now();
    entry.worker = await worker.load();
    handle?.loaded(Date.now() - loadStarted);
    // The resident worker processes requests serially: mark this request as
    // queued until worker prefill progress or the first token proves it is
    // actually running.
    handle?.phase("queued");
    const result = await worker.chat(request, () => handle?.firstToken(), c.req.raw.signal, (done, total) => handle?.prefill(done, total));
    entry.worker = worker.info();
    const body = chatResponse(request, result, templateInfo);
    const message = body.choices[0]?.message;
    handle?.finish({
      status: result.finishReason === "cancel" ? "cancelled" : "ok",
      promptTokens: result.usage?.promptTokens,
      completionTokens: result.usage?.completionTokens,
      cacheHit: result.cache?.hit,
      reusedTokens: result.cache?.reusedTokens,
      sideRequest: result.cache?.sideRequest,      slot: result.cache?.slot,
      finishReason: body.choices[0]?.finish_reason ?? undefined,
      toolCalls: message?.tool_calls?.length,
      response: {
        content: typeof message?.content === "string" ? message.content : undefined,
        reasoning: message?.reasoning ?? undefined,
        toolCalls: message?.tool_calls?.map((call) => ({ name: call.function.name, arguments: call.function.arguments })),
      },
      rawOutput: result.content,
    });
    return c.json(body);
  } catch (error) {
    handle?.finish({ status: "error", error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

function streamResidentResponse(c: Parameters<typeof streamSSE>[0], residents: ResidentWorkerRegistry, entry: LoadedModel, request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo, onDone?: () => void, handle?: RequestHandle) {
  return streamSSE(c, async (stream) => {
    const id = completionId();
    const created = nowSeconds();
    const aborter = new AbortController();
    stream.onAbort(() => aborter.abort());
    if (c.req.raw.signal.aborted) aborter.abort();
    else c.req.raw.signal.addEventListener("abort", () => aborter.abort(), { once: true });
    let wroteRole = false;
    let writeQueue = Promise.resolve();
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
      if (sawOutput || aborter.signal.aborted) return;
      writeQueue = writeQueue.then(async () => {
        await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {})) });
      }).catch(() => undefined);
    }, 5_000);
    const writeDelta = async (delta: StreamDelta) => {
      await ensureRole();
      const payload = delta.type === "reasoning"
        ? { reasoning: delta.text, reasoning_content: delta.text }
        : { content: delta.text };
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, payload)) });
    };
    try {
      handle?.capture(request);
      const worker = residents.getOrCreate(entry.key, entry.backend, entry.localPath);
      handle?.phase("loading");
      const loadStarted = Date.now();
      entry.worker = await worker.load();
      handle?.loaded(Date.now() - loadStarted);
      handle?.phase("queued");
      const streamExtras = profileStreamExtras(request.model, request, templateInfo);
      const filter = new StreamingOutputFilter({
        toolMode: Boolean(request.tools?.length),
        bufferAll: Boolean(request.response_format && request.response_format.type !== "text"),
        stops: typeof request.stop === "string" ? [request.stop] : request.stop ?? [],
        startInReasoning: streamExtras.implicitThink,
        extraMarkers: streamExtras.extraMarkers,
      });
      const result = await worker.chat(request, (token) => {
        if (!sawOutput) handle?.firstToken();
        sawOutput = true;
        const deltas = filter.feed(token);
        if (!deltas.length) return;
        writeQueue = writeQueue.then(async () => {
          for (const delta of deltas) await writeDelta(delta);
        });
      }, aborter.signal, (done, total) => handle?.prefill(done, total));
      await writeQueue;
      entry.worker = worker.info();
      if (result.finishReason === "cancel" || aborter.signal.aborted) {
        handle?.finish({
          status: "cancelled",
          promptTokens: result.usage?.promptTokens,
          completionTokens: result.usage?.completionTokens,
          cacheHit: result.cache?.hit,
          reusedTokens: result.cache?.reusedTokens,
          sideRequest: result.cache?.sideRequest,          slot: result.cache?.slot,
          finishReason: "cancel",
        });
        return;
      }
      const parsed = parseAssistantOutput(result.content, request, templateInfo);
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
        promptTokens: result.usage?.promptTokens,
        completionTokens: result.usage?.completionTokens,
        cacheHit: result.cache?.hit,
        reusedTokens: result.cache?.reusedTokens,
        sideRequest: result.cache?.sideRequest,        slot: result.cache?.slot,
        finishReason,
        toolCalls: parsed.toolCalls?.length,
        response: {
          content: parsed.content ?? undefined,
          reasoning: parsed.reasoning,
          toolCalls: parsed.toolCalls?.map((call) => ({ name: call.function.name, arguments: call.function.arguments })),
        },
        rawOutput: result.content,
      });
    } catch (error) {
      await writeQueue.catch(() => undefined);
      handle?.finish({ status: "error", error: error instanceof Error ? error.message : String(error) });
      await stream.writeSSE({ data: JSON.stringify({ error: backendErrorBody(error).error }) });
    } finally {
      clearInterval(heartbeat);
      onDone?.();
    }
  });
}

function backendErrorBody(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return ErrorResponseSchema.parse({
    error: { message, type: "backend_error", code: "resident_worker_error" },
  });
}

export function startServer(options: ServerOptions = {}) {
  const port = options.port ?? portFromEnv();
  const hostname = options.hostname ?? hostnameFromEnv();
  const idleTimeout = options.idleTimeout ?? idleTimeoutFromEnv();
  const residents = new ResidentWorkerRegistry();
  const lifecycle = new ModelLifecycleManager(() => Date.now(), (entry) => residents.shutdown(entry.key));
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

function portFromEnv(): number {
  const fromUrl = new URL(process.env.CLAP_BASE_URL ?? defaultBaseURL).port;
  return Number(process.env.PORT ?? fromUrl ?? 11435);
}

function hostnameFromEnv(): string {
  const fromEnv = process.env.CLAP_HOST?.trim();
  if (fromEnv) return fromEnv;
  try {
    const fromUrl = new URL(process.env.CLAP_BASE_URL ?? defaultBaseURL).hostname;
    if (fromUrl && fromUrl !== "localhost") return fromUrl;
  } catch {
    // fall through to loopback default
  }
  return "127.0.0.1";
}

export function idleTimeoutFromEnv(): number {
  const raw = process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS;
  if (raw === undefined || raw.trim() === "") return defaultIdleTimeoutSeconds;
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
  const parsed = parseAssistantOutput(result.content, request, templateInfo);
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
