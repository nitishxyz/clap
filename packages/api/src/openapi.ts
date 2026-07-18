import { OpenAPIRegistry, OpenApiGeneratorV3, extendZodWithOpenApi } from "@asteasolutions/zod-to-openapi";
import { z } from "zod";
import {
  BackendsResponseSchema,
  ChatCompletionRequestSchema,
  ChatCompletionResponseSchema,
  clapVersion,
  ClapModelsResponseSchema,
  DownloadsResponseSchema,
  ErrorResponseSchema,
  HealthResponseSchema,
  LoadedModelsResponseSchema,
  LoadModelRequestSchema,
  LoadModelResponseSchema,
  ModelResolveResponseSchema,
  OllamaChatRequestSchema,
  OllamaGenerateRequestSchema,
  OllamaPullRequestSchema,
  OllamaShowRequestSchema,
  OllamaTagsResponseSchema,
  OpenAIModelsResponseSchema,
  PullModelRequestSchema,
  PullModelResponseSchema,
  ResponseRequestSchema,
  ResponseSchema,
  RuntimeResponseSchema,
  UnloadModelRequestSchema,
  UnloadModelResponseSchema,
} from "./schemas";

extendZodWithOpenApi(z);

export function createOpenApiDocument() {
  const registry = new OpenAPIRegistry();

  registry.register("ErrorResponse", ErrorResponseSchema);
  registry.register("HealthResponse", HealthResponseSchema);
  registry.register("RuntimeResponse", RuntimeResponseSchema);
  registry.register("BackendsResponse", BackendsResponseSchema);
  registry.register("ClapModelsResponse", ClapModelsResponseSchema);
  registry.register("DownloadsResponse", DownloadsResponseSchema);
  registry.register("LoadedModelsResponse", LoadedModelsResponseSchema);
  registry.register("LoadModelRequest", LoadModelRequestSchema);
  registry.register("LoadModelResponse", LoadModelResponseSchema);
  registry.register("UnloadModelRequest", UnloadModelRequestSchema);
  registry.register("UnloadModelResponse", UnloadModelResponseSchema);
  registry.register("PullModelRequest", PullModelRequestSchema);
  registry.register("PullModelResponse", PullModelResponseSchema);
  registry.register("ModelResolveResponse", ModelResolveResponseSchema);
  registry.register("OpenAIModelsResponse", OpenAIModelsResponseSchema);
  registry.register("ChatCompletionRequest", ChatCompletionRequestSchema);
  registry.register("ChatCompletionResponse", ChatCompletionResponseSchema);
  registry.register("ResponseRequest", ResponseRequestSchema);
  registry.register("Response", ResponseSchema);
  registry.register("OllamaTagsResponse", OllamaTagsResponseSchema);
  registry.register("OllamaShowRequest", OllamaShowRequestSchema);
  registry.register("OllamaPullRequest", OllamaPullRequestSchema);
  registry.register("OllamaChatRequest", OllamaChatRequestSchema);
  registry.register("OllamaGenerateRequest", OllamaGenerateRequestSchema);

  registry.registerPath({
    method: "get",
    path: "/clap/v1/health",
    summary: "Health check",
    responses: jsonResponses(HealthResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/clap/v1/runtime",
    summary: "Runtime metadata",
    responses: jsonResponses(RuntimeResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/clap/v1/backends",
    summary: "Bundled backend metadata",
    responses: jsonResponses(BackendsResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/clap/v1/models",
    summary: "Clap model inventory",
    responses: jsonResponses(ClapModelsResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/clap/v1/downloads",
    summary: "Model download status",
    responses: jsonResponses(DownloadsResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/clap/v1/runtime/models",
    summary: "Loaded runtime model registry",
    responses: jsonResponses(LoadedModelsResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/clap/v1/models/load",
    summary: "Load or warm a local model in the runtime lifecycle registry",
    request: {
      body: {
        content: {
          "application/json": { schema: LoadModelRequestSchema },
        },
      },
    },
    responses: jsonResponses(LoadModelResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/clap/v1/models/unload",
    summary: "Unload a model from the runtime lifecycle registry",
    request: {
      body: {
        content: {
          "application/json": { schema: UnloadModelRequestSchema },
        },
      },
    },
    responses: jsonResponses(UnloadModelResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/clap/v1/models/resolve",
    summary: "Resolve runnable model artifacts before pulling",
    request: {
      body: {
        content: {
          "application/json": { schema: PullModelRequestSchema },
        },
      },
    },
    responses: jsonResponses(ModelResolveResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/clap/v1/models/pull",
    summary: "Download a Hugging Face model into the local cache",
    request: {
      body: {
        content: {
          "application/json": { schema: PullModelRequestSchema },
        },
      },
    },
    responses: jsonResponses(PullModelResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/clap/v1/downloads/{id}/cancel",
    summary: "Cancel a running model download",
    request: {
      params: z.object({ id: z.string() }),
    },
    responses: jsonResponses(PullModelResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/v1/models",
    summary: "OpenAI-compatible models list",
    request: {
      query: z.object({ metadata: z.enum(["1", "true"]).optional() }),
    },
    responses: jsonResponses(OpenAIModelsResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/v1/chat/completions",
    summary: "OpenAI-compatible chat completions",
    request: {
      body: {
        content: {
          "application/json": { schema: ChatCompletionRequestSchema },
        },
      },
    },
    responses: jsonResponses(ChatCompletionResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/v1/responses",
    summary: "OpenAI-compatible Responses API",
    request: {
      body: {
        content: {
          "application/json": { schema: ResponseRequestSchema },
        },
      },
    },
    responses: jsonResponses(ResponseSchema),
  });

  registry.registerPath({
    method: "get",
    path: "/api/tags",
    summary: "Ollama-compatible local model tags",
    responses: jsonResponses(OllamaTagsResponseSchema),
  });

  registry.registerPath({
    method: "post",
    path: "/api/show",
    summary: "Ollama-compatible model metadata",
    request: { body: { content: { "application/json": { schema: OllamaShowRequestSchema } } } },
    responses: jsonResponses(z.record(z.unknown())),
  });

  registry.registerPath({
    method: "post",
    path: "/api/pull",
    summary: "Ollama-compatible model pull",
    request: { body: { content: { "application/json": { schema: OllamaPullRequestSchema } } } },
    responses: jsonResponses(z.record(z.unknown())),
  });

  registry.registerPath({
    method: "post",
    path: "/api/chat",
    summary: "Ollama-compatible chat",
    request: { body: { content: { "application/json": { schema: OllamaChatRequestSchema } } } },
    responses: jsonResponses(z.record(z.unknown())),
  });

  registry.registerPath({
    method: "post",
    path: "/api/generate",
    summary: "Ollama-compatible generate",
    request: { body: { content: { "application/json": { schema: OllamaGenerateRequestSchema } } } },
    responses: jsonResponses(z.record(z.unknown())),
  });

  const generator = new OpenApiGeneratorV3(registry.definitions);
  return generator.generateDocument({
    openapi: "3.0.0",
    info: {
      title: "Clap Local Model Server API",
      version: clapVersion,
    },
    servers: [{ url: "http://localhost:11435" }],
  });
}

function jsonResponses(schema: z.ZodTypeAny) {
  return {
    200: {
      description: "OK",
      content: {
        "application/json": { schema },
      },
    },
    400: {
      description: "Bad Request",
      content: {
        "application/json": { schema: ErrorResponseSchema },
      },
    },
  };
}

if (import.meta.main) {
  await Bun.write(new URL("../openapi.json", import.meta.url), `${JSON.stringify(createOpenApiDocument(), null, 2)}\n`);
}
