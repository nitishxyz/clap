import { OpenAPIRegistry, OpenApiGeneratorV3, extendZodWithOpenApi } from "@asteasolutions/zod-to-openapi";
import { z } from "zod";
import {
  BackendsResponseSchema,
  ChatCompletionRequestSchema,
  ChatCompletionResponseSchema,
  ClapModelsResponseSchema,
  DownloadsResponseSchema,
  ErrorResponseSchema,
  HealthResponseSchema,
  LoadedModelsResponseSchema,
  LoadModelRequestSchema,
  LoadModelResponseSchema,
  OpenAIModelsResponseSchema,
  PullModelRequestSchema,
  PullModelResponseSchema,
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
  registry.register("OpenAIModelsResponse", OpenAIModelsResponseSchema);
  registry.register("ChatCompletionRequest", ChatCompletionRequestSchema);
  registry.register("ChatCompletionResponse", ChatCompletionResponseSchema);

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

  const generator = new OpenApiGeneratorV3(registry.definitions);
  return generator.generateDocument({
    openapi: "3.0.0",
    info: {
      title: "Clap Local Model Server API",
      version: "0.1.0",
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
