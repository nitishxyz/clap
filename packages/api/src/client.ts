import {
  BackendsResponseSchema,
  ClapAliasesResponseSchema,
  ChatCompletionRequestSchema,
  ChatCompletionResponseSchema,
  ClapModelsResponseSchema,
  DownloadsResponseSchema,
  HealthResponseSchema,
  LoadedModelsResponseSchema,
  LoadModelRequestSchema,
  LoadModelResponseSchema,
  ModelResolveResponseSchema,
  OpenAIModelsResponseSchema,
  PullModelRequestSchema,
  PullModelResponseSchema,
  ResponseRequestSchema,
  ResponseSchema,
  RuntimeResponseSchema,
  UnloadModelRequestSchema,
  UnloadModelResponseSchema,
  defaultBaseURL,
  type BackendsResponse,
  type ClapAliasesResponse,
  type ChatCompletionRequest,
  type ChatCompletionResponse,
  type ClapModelsResponse,
  type DownloadsResponse,
  type HealthResponse,
  type LoadedModelsResponse,
  type LoadModelRequest,
  type LoadModelResponse,
  type ModelResolveResponse,
  type OpenAIModelsResponse,
  type PullModelRequest,
  type PullModelResponse,
  type ResponseRequest,
  type ResponseResponse,
  type RuntimeResponse,
  type UnloadModelRequest,
  type UnloadModelResponse,
} from "./schemas";

export type ClapClientOptions = {
  baseURL?: string;
  fetch?: typeof fetch;
};

export class ClapApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(message);
    this.name = "ClapApiError";
  }
}

export class ClapClient {
  private readonly baseURL: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClapClientOptions = {}) {
    this.baseURL = (options.baseURL ?? process.env.CLAP_BASE_URL ?? defaultBaseURL).replace(/\/$/, "");
    this.fetchImpl = options.fetch ?? fetch;
  }

  async health(): Promise<HealthResponse> {
    return HealthResponseSchema.parse(await this.getJson("/clap/v1/health"));
  }

  async runtime(): Promise<RuntimeResponse> {
    return RuntimeResponseSchema.parse(await this.getJson("/clap/v1/runtime"));
  }

  async backends(): Promise<BackendsResponse> {
    return BackendsResponseSchema.parse(await this.getJson("/clap/v1/backends"));
  }

  async clapModels(): Promise<ClapModelsResponse> {
    return ClapModelsResponseSchema.parse(await this.getJson("/clap/v1/models"));
  }

  async clapAliases(): Promise<ClapAliasesResponse> {
    return ClapAliasesResponseSchema.parse(await this.getJson("/clap/v1/aliases"));
  }

  async models(): Promise<OpenAIModelsResponse> {
    return OpenAIModelsResponseSchema.parse(await this.getJson("/v1/models"));
  }

  async downloads(): Promise<DownloadsResponse> {
    return DownloadsResponseSchema.parse(await this.getJson("/clap/v1/downloads"));
  }

  async loadedModels(): Promise<LoadedModelsResponse> {
    return LoadedModelsResponseSchema.parse(await this.getJson("/clap/v1/runtime/models"));
  }

  async loadModel(request: LoadModelRequest): Promise<LoadModelResponse> {
    const body = LoadModelRequestSchema.parse(request);
    return LoadModelResponseSchema.parse(await this.postJson("/clap/v1/models/load", body));
  }

  async unloadModel(request: UnloadModelRequest): Promise<UnloadModelResponse> {
    const body = UnloadModelRequestSchema.parse(request);
    return UnloadModelResponseSchema.parse(await this.postJson("/clap/v1/models/unload", body));
  }

  async pullModel(request: PullModelRequest): Promise<PullModelResponse> {
    const body = PullModelRequestSchema.parse(request);
    return PullModelResponseSchema.parse(await this.postJson("/clap/v1/models/pull", body));
  }

  async resolveModel(request: PullModelRequest): Promise<ModelResolveResponse> {
    const body = PullModelRequestSchema.parse(request);
    return ModelResolveResponseSchema.parse(await this.postJson("/clap/v1/models/resolve", body));
  }

  async cancelDownload(id: string): Promise<PullModelResponse> {
    return PullModelResponseSchema.parse(await this.postJson(`/clap/v1/downloads/${encodeURIComponent(id)}/cancel`, {}));
  }

  async chatCompletions(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    const body = ChatCompletionRequestSchema.parse({ ...request, stream: false });
    return ChatCompletionResponseSchema.parse(await this.postJson("/v1/chat/completions", body));
  }

  async responses(request: ResponseRequest): Promise<ResponseResponse> {
    const body = ResponseRequestSchema.parse({ ...request, stream: false });
    return ResponseSchema.parse(await this.postJson("/v1/responses", body));
  }

  async *streamChatCompletions(request: ChatCompletionRequest): AsyncGenerator<string> {
    const body = ChatCompletionRequestSchema.parse({ ...request, stream: true });
    const response = await this.fetchImpl(this.url("/v1/chat/completions"), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw await this.toError(response);
    }

    const reader = response.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf("\n\n")) >= 0) {
        const event = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") return;
          const parsed = JSON.parse(data) as { choices?: Array<{ delta?: { content?: string } }> };
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) yield content;
        }
      }
    }
  }

  private async getJson(path: string): Promise<unknown> {
    const response = await this.fetchImpl(this.url(path));
    if (!response.ok) throw await this.toError(response);
    return response.json();
  }

  private async postJson(path: string, body: unknown): Promise<unknown> {
    const response = await this.fetchImpl(this.url(path), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw await this.toError(response);
    return response.json();
  }

  private url(path: string): string {
    return `${this.baseURL}${path}`;
  }

  private async toError(response: Response): Promise<ClapApiError> {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = await response.text();
    }
    const message = typeof body === "object" && body && "error" in body
      ? String((body as { error?: { message?: unknown } }).error?.message ?? response.statusText)
      : response.statusText;
    return new ClapApiError(message, response.status, body);
  }
}

export function createClapClient(options?: ClapClientOptions): ClapClient {
  return new ClapClient(options);
}
