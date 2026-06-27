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
import { cachedPullResultForTarget, listAliases, listModels, pullModel, resolveModel, resolvePullTarget, type ResolvedModel } from "@clap/models";
import { assertGgufModelPath, isGgufModel, LlamaWorkerError } from "@clap/runtime-llama";
import { assertMlxModelPath, isMlxModelDirectory, MlxWorkerError } from "@clap/runtime-mlx";
import { listBackends, ModelLifecycleManager, ResidentWorkerRegistry } from "@clap/runtime-router";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import { parseAssistantOutput, prepareChatRequest } from "./chat-compat";

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

    let routedRequest: ChatCompletionRequest;
    try {
      routedRequest = prepareChatRequest({ ...request, model: resolved.model.modelPath ?? request.model });
    } catch (error) {
      return c.json(ErrorResponseSchema.parse({
        error: { message: error instanceof Error ? error.message : String(error), type: "invalid_request_error", code: "unsupported_content_part" },
      }), 400);
    }
    if (isGgufModel(routedRequest.model)) {
      await assertGgufModelPath(routedRequest.model);
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(resolved.model);
        return streamResidentResponse(c, residents, loaded, routedRequest, () => lifecycle.finishUsage(loaded));
      }
      return lifecycle.withUsage(resolved.model, (entry) => jsonResidentResponse(c, residents, entry, routedRequest));
    }
    if (await isMlxModelDirectory(routedRequest.model)) {
      if (routedRequest.stream) {
        const loaded = lifecycle.beginUsage(resolved.model);
        return streamResidentResponse(c, residents, loaded, routedRequest, () => lifecycle.finishUsage(loaded));
      }
      return lifecycle.withUsage(resolved.model, (entry) => jsonResidentResponse(c, residents, entry, routedRequest));
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

  app.get("/api/tags", (c) => c.json({
    models: listModels().map(ollamaTag),
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
    const response = await app.request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: request.model,
        messages: request.messages,
        stream: false,
        tools: request.tools,
        temperature: request.options?.temperature,
        top_p: request.options?.top_p,
        seed: request.options?.seed,
        max_tokens: request.options?.num_predict,
        stop: request.options?.stop,
      }),
    });
    if (!response.ok) return response;
    const body = await response.json() as ChatCompletionResponse;
    return ollamaChatResponse(c, request.model, body, request.stream !== false);
  });

  app.post("/api/generate", async (c) => {
    const raw = await c.req.json();
    if (hasOllamaImages(raw)) return c.json({ error: "image input is not supported by the selected local text runtime yet" }, 400);
    const request = OllamaGenerateRequestSchema.parse(raw);
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
        stream: false,
        response_format: request.format ? request.format === "json" ? { type: "json_object" } : { type: "json_schema", json_schema: { name: "OllamaFormat", schema: request.format } } : undefined,
        temperature: request.options?.temperature,
        top_p: request.options?.top_p,
        seed: request.options?.seed,
        max_tokens: request.options?.num_predict,
        stop: request.options?.stop,
      }),
    });
    if (!response.ok) return response;
    const body = await response.json() as ChatCompletionResponse;
    return ollamaGenerateResponse(c, request.model, body, request.stream !== false);
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
    | { id: string; type: "message"; status: "completed"; role: "assistant"; content: Array<{ type: "output_text"; text: string }> }
    | { id: string; type: "function_call"; status: "completed"; call_id: string; name: string; arguments: string };
  const output: ResponseOutput[] = [];
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
    } else {
      events.push({ event: "response.function_call_arguments.delta", data: { output_index: index, item_id: item.id, delta: item.arguments } });
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

async function jsonResidentResponse(c: { json: (body: ChatCompletionResponse) => Response | Promise<Response> }, residents: ResidentWorkerRegistry, entry: LoadedModel, request: ChatCompletionRequest) {
  const worker = residents.getOrCreate(entry.key, entry.backend, entry.localPath);
  entry.worker = await worker.load();
  const content = await worker.chat(request);
  entry.worker = worker.info();
  return c.json(chatResponse(request, content));
}

function streamResidentResponse(c: Parameters<typeof streamSSE>[0], residents: ResidentWorkerRegistry, entry: LoadedModel, request: ChatCompletionRequest, onDone?: () => void) {
  return streamSSE(c, async (stream) => {
    try {
      const worker = residents.getOrCreate(entry.key, entry.backend, entry.localPath);
      entry.worker = await worker.load();
      const content = await worker.chat(request);
      entry.worker = worker.info();
      await writeParsedStream(stream, request, content);
    } finally {
      onDone?.();
    }
  });
}

export function startServer(options: ServerOptions = {}) {
  const port = options.port ?? portFromEnv();
  const hostname = options.hostname ?? "127.0.0.1";
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

function chatResponse(request: ChatCompletionRequest, rawContent: string): ChatCompletionResponse {
  const parsed = parseAssistantOutput(rawContent, request);
  const message = {
    role: "assistant" as const,
    content: parsed.content,
    reasoning: parsed.reasoning,
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
      finish_reason: parsed.finishReason,
    }],
    usage: usageFor(request, rawContent),
  };
}

async function writeParsedStream(stream: { writeSSE: (event: { data: string }) => Promise<void> }, request: ChatCompletionRequest, rawContent: string): Promise<void> {
  const id = completionId();
  const created = nowSeconds();
  const parsed = parseAssistantOutput(rawContent, request);
  await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { role: "assistant" })) });
  if (parsed.reasoning) {
    await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { reasoning: parsed.reasoning })) });
  }
  if (parsed.toolCalls?.length) {
    for (const [index, call] of parsed.toolCalls.entries()) {
      await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {
        tool_calls: [{ index, id: call.id, type: "function", function: { name: call.function.name, arguments: call.function.arguments } }],
      })) });
    }
  } else if (parsed.content) {
    await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, { content: parsed.content })) });
  }
  await stream.writeSSE({ data: JSON.stringify(chunk(id, created, request.model, {}, parsed.finishReason, usageFor(request, rawContent))) });
  await stream.writeSSE({ data: "[DONE]" });
}

function usageFor(request: ChatCompletionRequest, completion: string) {
  const promptText = JSON.stringify(request.messages);
  const promptTokens = estimateTokens(promptText);
  const completionTokens = estimateTokens(completion);
  return { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function chunk(
  id: string,
  created: number,
  model: string,
  delta: { role?: "assistant"; content?: string | null; reasoning?: string; tool_calls?: Array<{ index: number; id?: string; type?: "function"; function?: { name?: string; arguments?: string } }> },
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
