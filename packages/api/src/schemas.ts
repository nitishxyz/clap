import { z } from "zod";

export const clapVersion = "0.1.2";
export const defaultBaseURL = "http://localhost:11435";

export const ErrorResponseSchema = z.object({
  error: z.object({
    message: z.string(),
    type: z.string().default("invalid_request_error"),
    code: z.string().optional(),
  }),
});

export const HealthResponseSchema = z.object({
  status: z.literal("ok"),
  version: z.string(),
  uptimeMs: z.number(),
});

export const RuntimeResponseSchema = z.object({
  platform: z.string(),
  arch: z.string(),
  bunVersion: z.string(),
  runtime: z.literal("bun"),
});

export const BackendSchema = z.object({
  id: z.enum(["llama", "mlx"]),
  name: z.string(),
  status: z.enum(["available", "unsupported", "not_installed"]),
  formats: z.array(z.string()),
  reason: z.string().optional(),
});

export const BackendsResponseSchema = z.object({
  backends: z.array(BackendSchema),
});

export const ClapModelCapabilitiesSchema = z.object({
  chat: z.boolean(),
  completion: z.boolean(),
  streaming: z.boolean(),
  temperature: z.boolean(),
  system_prompt: z.boolean(),
  attachment: z.boolean(),
  reasoning: z.boolean(),
  tool_call: z.boolean(),
  structured_output: z.boolean(),
});

export const ClapModelLimitSchema = z.object({
  context: z.number().int().positive().nullable(),
  output: z.number().int().positive().nullable(),
});

export const ClapModelModalitiesSchema = z.object({
  input: z.array(z.enum(["text", "image", "audio"])),
  output: z.array(z.enum(["text", "image", "audio"])),
});

export const ClapModelSchema = z.object({
  id: z.string(),
  object: z.literal("model").default("model"),
  name: z.string(),
  displayName: z.string(),
  provider: z.string(),
  source: z.object({
    type: z.enum(["huggingface", "local", "alias"]),
    repo: z.string().optional(),
    baseRepo: z.string().optional(),
  }),
  backend: z.enum(["llama", "mlx"]),
  format: z.string(),
  status: z.enum(["available", "not_downloaded", "unsupported"]),
  modalities: ClapModelModalitiesSchema,
  capabilities: ClapModelCapabilitiesSchema,
  limit: ClapModelLimitSchema,
  upstream: z.object({
    modalities: ClapModelModalitiesSchema.optional(),
    capabilities: ClapModelCapabilitiesSchema.partial().optional(),
    limit: ClapModelLimitSchema.optional(),
  }).optional(),
  architecture: z.string().optional(),
  modelType: z.string().optional(),
  quantization: z.string().optional(),
  sizeBytes: z.number().int().nonnegative().optional(),
  alias: z.string().optional(),
  repo: z.string().optional(),
  file: z.string().optional(),
  localPath: z.string().optional(),
  pull: z.object({ model: z.string(), file: z.string().optional(), backend: z.enum(["gguf", "mlx"]).optional() }).optional(),
  reason: z.string().optional(),
});

export const ClapModelsResponseSchema = z.object({
  models: z.array(ClapModelSchema),
});

export const ClapAliasesResponseSchema = z.object({
  models: z.array(ClapModelSchema),
});

export const PullModelRequestSchema = z.object({
  model: z.string().min(1),
  file: z.string().min(1).optional(),
  backend: z.enum(["gguf", "mlx"]).optional(),
  force: z.boolean().optional().default(false),
});

export const ModelResolveOptionSchema = z.object({
  id: z.string(),
  model: z.string(),
  backend: z.enum(["gguf", "mlx"]),
  format: z.enum(["gguf", "mlx", "safetensors"]),
  repo: z.string(),
  file: z.string().optional(),
  sizeBytes: z.number().int().nonnegative().optional(),
  quantization: z.string().optional(),
  supported: z.boolean(),
  unsupportedReason: z.string().optional(),
  recommended: z.boolean(),
  reason: z.string(),
});

export const ModelResolveResponseSchema = z.object({
  model: z.string(),
  repo: z.string(),
  options: z.array(ModelResolveOptionSchema),
  selected: ModelResolveOptionSchema.optional(),
});

export const DownloadSchema = z.object({
  id: z.string(),
  model: z.string(),
  file: z.string().optional(),
  backend: z.enum(["gguf", "mlx"]).optional(),
  targetKey: z.string().optional(),
  currentFile: z.string().optional(),
  status: z.enum(["queued", "running", "completed", "failed", "cancelled"]),
  bytesReceived: z.number().int().nonnegative().default(0),
  totalBytes: z.number().int().nonnegative().optional(),
  modelPath: z.string().optional(),
  error: z.string().optional(),
  startedAt: z.string(),
  completedAt: z.string().optional(),
  selected: ModelResolveOptionSchema.optional(),
});

export const PullModelResponseSchema = z.object({
  download: DownloadSchema,
});

export const DownloadsResponseSchema = z.object({
  downloads: z.array(DownloadSchema),
});

export const KeepAliveSchema = z.union([z.literal("always"), z.string().regex(/^\d+(ms|s|m|h|d)$/)]);

export const LoadModelRequestSchema = z.object({
  model: z.string().min(1),
  backend: z.enum(["gguf", "mlx"]).optional(),
  keepAlive: KeepAliveSchema.optional(),
});

export const UnloadModelRequestSchema = z.object({
  model: z.string().min(1),
  backend: z.enum(["gguf", "mlx"]).optional(),
});

export const LoadedModelSchema = z.object({
  key: z.string(),
  id: z.string(),
  backend: z.enum(["llama", "mlx"]),
  format: z.enum(["gguf", "mlx"]),
  localPath: z.string(),
  state: z.enum(["warm", "active", "unloading"]),
  activeRequests: z.number().int().nonnegative(),
  loadedAt: z.string(),
  lastUsedAt: z.string(),
  keepAlive: z.string(),
  expiresAt: z.string().nullable(),
  pinned: z.boolean(),
  always: z.boolean(),
  worker: z.object({
    pid: z.number().int().positive().optional(),
    state: z.enum(["not_started", "one_shot", "resident", "exited"]),
    limitation: z.string().optional(),
    crashes: z.number().int().nonnegative().optional(),
    lastCrashAt: z.string().optional(),
    memory: z.object({
      activeBytes: z.number().int().nonnegative(),
      cacheBytes: z.number().int().nonnegative(),
      peakActiveBytes: z.number().int().nonnegative(),
    }).optional(),
  }),
});

export const LoadedModelsResponseSchema = z.object({
  models: z.array(LoadedModelSchema),
});

export const LoadModelResponseSchema = z.object({
  model: LoadedModelSchema,
});

export const UnloadModelResponseSchema = z.object({
  unloaded: z.boolean(),
  model: LoadedModelSchema.optional(),
});

export const OpenAIModelSchema = z.object({
  id: z.string(),
  object: z.literal("model"),
  created: z.number(),
  owned_by: z.string(),
});

export const OpenAIRichModelSchema = ClapModelSchema.extend({
  created: z.number(),
  owned_by: z.string(),
});

export const OpenAIModelsResponseSchema = z.object({
  object: z.literal("list"),
  data: z.array(z.union([OpenAIModelSchema, OpenAIRichModelSchema])),
});

export const ChatContentPartSchema = z.union([
  z.object({ type: z.literal("text"), text: z.string() }),
  z.object({ type: z.literal("image_url"), image_url: z.object({ url: z.string(), detail: z.string().optional() }) }),
]);

export const ChatToolCallFunctionSchema = z.object({
  name: z.string(),
  arguments: z.string(),
});

export const ChatToolCallSchema = z.object({
  id: z.string(),
  type: z.literal("function"),
  function: ChatToolCallFunctionSchema,
});

export const ChatMessageSchema = z.object({
  role: z.enum(["system", "user", "assistant", "tool"]),
  content: z.union([z.string(), z.array(ChatContentPartSchema), z.null()]).default(""),
  name: z.string().optional(),
  tool_call_id: z.string().optional(),
  tool_calls: z.array(ChatToolCallSchema).optional(),
});

export const ChatToolSchema = z.object({
  type: z.literal("function"),
  function: z.object({
    name: z.string(),
    description: z.string().optional(),
    parameters: z.record(z.unknown()).optional(),
  }),
});

export const ResponseFormatSchema = z.union([
  z.object({ type: z.literal("text") }),
  z.object({ type: z.literal("json_object") }),
  z.object({
    type: z.literal("json_schema"),
    json_schema: z.object({
      name: z.string(),
      description: z.string().optional(),
      schema: z.record(z.unknown()).optional(),
      strict: z.boolean().optional(),
    }),
  }),
]);

export const ChatCompletionRequestSchema = z.object({
  model: z.string().min(1),
  messages: z.array(ChatMessageSchema).min(1),
  stream: z.boolean().optional().default(false),
  backend: z.enum(["gguf", "mlx"]).optional(),
  temperature: z.number().min(0).max(2).optional(),
  max_tokens: z.number().int().positive().optional(),
  tools: z.array(ChatToolSchema).optional(),
  tool_choice: z.union([z.literal("none"), z.literal("auto"), z.literal("required"), z.object({ type: z.literal("function"), function: z.object({ name: z.string() }) })]).optional(),
  parallel_tool_calls: z.boolean().optional(),
  response_format: ResponseFormatSchema.optional(),
  stop: z.union([z.string(), z.array(z.string())]).optional(),
  seed: z.number().int().optional(),
  top_p: z.number().min(0).max(1).optional(),
  presence_penalty: z.number().min(-2).max(2).optional(),
  frequency_penalty: z.number().min(-2).max(2).optional(),
});

export const ChatCompletionChoiceSchema = z.object({
  index: z.number(),
  message: ChatMessageSchema.extend({ reasoning: z.string().optional(), reasoning_content: z.string().optional() }),
  finish_reason: z.enum(["stop", "length", "tool_calls", "content_filter", "function_call"]).nullable(),
});

export const ChatCompletionResponseSchema = z.object({
  id: z.string(),
  object: z.literal("chat.completion"),
  created: z.number(),
  model: z.string(),
  choices: z.array(ChatCompletionChoiceSchema),
  usage: z.object({
    prompt_tokens: z.number().int().nonnegative(),
    completion_tokens: z.number().int().nonnegative(),
    total_tokens: z.number().int().nonnegative(),
  }).optional(),
});

export const ChatToolCallDeltaSchema = z.object({
  index: z.number(),
  id: z.string().optional(),
  type: z.literal("function").optional(),
  function: z.object({
    name: z.string().optional(),
    arguments: z.string().optional(),
  }).optional(),
});

export const ChatCompletionChunkSchema = z.object({
  id: z.string(),
  object: z.literal("chat.completion.chunk"),
  created: z.number(),
  model: z.string(),
  choices: z.array(z.object({
    index: z.number(),
    delta: z.object({
      role: z.literal("assistant").optional(),
      content: z.string().nullable().optional(),
      reasoning: z.string().optional(),
      reasoning_content: z.string().optional(),
      tool_calls: z.array(ChatToolCallDeltaSchema).optional(),
    }),
    finish_reason: z.enum(["stop", "length", "tool_calls", "content_filter", "function_call"]).nullable(),
  })),
  usage: z.object({
    prompt_tokens: z.number().int().nonnegative(),
    completion_tokens: z.number().int().nonnegative(),
    total_tokens: z.number().int().nonnegative(),
  }).optional(),
});

export const ResponseInputItemSchema = z.object({
  role: z.enum(["system", "user", "assistant", "tool"]).optional().default("user"),
  content: z.union([z.string(), z.array(ChatContentPartSchema), z.null()]).default(""),
  type: z.string().optional(),
  tool_call_id: z.string().optional(),
});

export const ResponseTextConfigSchema = z.object({
  format: ResponseFormatSchema.optional(),
}).optional();

export const ResponseRequestSchema = z.object({
  model: z.string().min(1),
  input: z.union([z.string(), z.array(ResponseInputItemSchema)]),
  instructions: z.string().optional(),
  tools: z.array(ChatToolSchema).optional(),
  tool_choice: z.union([z.literal("none"), z.literal("auto"), z.literal("required"), z.object({ type: z.literal("function"), function: z.object({ name: z.string() }) })]).optional(),
  parallel_tool_calls: z.boolean().optional(),
  text: ResponseTextConfigSchema,
  response_format: ResponseFormatSchema.optional(),
  stream: z.boolean().optional().default(false),
  temperature: z.number().min(0).max(2).optional(),
  top_p: z.number().min(0).max(1).optional(),
  max_output_tokens: z.number().int().positive().optional(),
  metadata: z.record(z.unknown()).optional(),
  previous_response_id: z.string().optional(),
});

export const ResponseOutputItemSchema = z.union([
  z.object({
    id: z.string(),
    type: z.literal("reasoning"),
    status: z.literal("completed"),
    summary: z.array(z.object({ type: z.literal("summary_text"), text: z.string() })).optional(),
    content: z.array(z.object({ type: z.literal("reasoning_text"), text: z.string() })).optional(),
  }),
  z.object({
    id: z.string(),
    type: z.literal("message"),
    status: z.literal("completed"),
    role: z.literal("assistant"),
    content: z.array(z.object({ type: z.literal("output_text"), text: z.string() })),
  }),
  z.object({
    id: z.string(),
    type: z.literal("function_call"),
    status: z.literal("completed"),
    call_id: z.string(),
    name: z.string(),
    arguments: z.string(),
  }),
]);

export const ResponseSchema = z.object({
  id: z.string(),
  object: z.literal("response"),
  created_at: z.number(),
  status: z.enum(["completed", "failed", "incomplete"]),
  model: z.string(),
  output: z.array(ResponseOutputItemSchema),
  output_text: z.string(),
  usage: z.object({
    input_tokens: z.number().int().nonnegative(),
    output_tokens: z.number().int().nonnegative(),
    total_tokens: z.number().int().nonnegative(),
  }).optional(),
  error: z.object({ code: z.string(), message: z.string() }).nullable().optional(),
  incomplete_details: z.object({ reason: z.string() }).nullable().optional(),
  metadata: z.record(z.unknown()).optional(),
});

export const OllamaTagsResponseSchema = z.object({
  models: z.array(z.object({
    name: z.string(),
    model: z.string(),
    modified_at: z.string(),
    size: z.number().int().nonnegative(),
    digest: z.string(),
    details: z.object({
      parent_model: z.string(),
      format: z.string(),
      family: z.string(),
      families: z.array(z.string()).nullable(),
      parameter_size: z.string(),
      quantization_level: z.string(),
    }),
  })),
});

export const OllamaShowRequestSchema = z.object({
  model: z.string().min(1),
});

export const OllamaPullRequestSchema = z.object({
  name: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  stream: z.boolean().optional().default(true),
});

export const OllamaChatRequestSchema = z.object({
  model: z.string().min(1),
  messages: z.array(ChatMessageSchema).min(1),
  stream: z.boolean().optional().default(true),
  tools: z.array(ChatToolSchema).optional(),
  options: z.object({
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    seed: z.number().int().optional(),
    num_predict: z.number().int().positive().optional(),
    stop: z.union([z.string(), z.array(z.string())]).optional(),
  }).optional(),
});

export const OllamaGenerateRequestSchema = z.object({
  model: z.string().min(1),
  prompt: z.string().default(""),
  system: z.string().optional(),
  stream: z.boolean().optional().default(true),
  format: z.union([z.literal("json"), z.record(z.unknown())]).optional(),
  options: z.object({
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    seed: z.number().int().optional(),
    num_predict: z.number().int().positive().optional(),
    stop: z.union([z.string(), z.array(z.string())]).optional(),
  }).optional(),
});

export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type RuntimeResponse = z.infer<typeof RuntimeResponseSchema>;
export type Backend = z.infer<typeof BackendSchema>;
export type BackendsResponse = z.infer<typeof BackendsResponseSchema>;
export type ClapModel = z.infer<typeof ClapModelSchema>;
export type ClapModelsResponse = z.infer<typeof ClapModelsResponseSchema>;
export type ClapAliasesResponse = z.infer<typeof ClapAliasesResponseSchema>;
export type PullModelRequest = z.infer<typeof PullModelRequestSchema>;
export type ModelResolveOption = z.infer<typeof ModelResolveOptionSchema>;
export type ModelResolveResponse = z.infer<typeof ModelResolveResponseSchema>;
export type Download = z.infer<typeof DownloadSchema>;
export type PullModelResponse = z.infer<typeof PullModelResponseSchema>;
export type DownloadsResponse = z.infer<typeof DownloadsResponseSchema>;
export type LoadModelRequest = z.infer<typeof LoadModelRequestSchema>;
export type UnloadModelRequest = z.infer<typeof UnloadModelRequestSchema>;
export type LoadedModel = z.infer<typeof LoadedModelSchema>;
export type LoadedModelsResponse = z.infer<typeof LoadedModelsResponseSchema>;
export type LoadModelResponse = z.infer<typeof LoadModelResponseSchema>;
export type UnloadModelResponse = z.infer<typeof UnloadModelResponseSchema>;
export type OpenAIModelsResponse = z.infer<typeof OpenAIModelsResponseSchema>;
export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatToolCall = z.infer<typeof ChatToolCallSchema>;
export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequestSchema>;
export type ChatCompletionResponse = z.infer<typeof ChatCompletionResponseSchema>;
export type ChatCompletionChunk = z.infer<typeof ChatCompletionChunkSchema>;
export type ResponseRequest = z.infer<typeof ResponseRequestSchema>;
export type ResponseResponse = z.infer<typeof ResponseSchema>;
export type OllamaChatRequest = z.infer<typeof OllamaChatRequestSchema>;
export type OllamaGenerateRequest = z.infer<typeof OllamaGenerateRequestSchema>;
