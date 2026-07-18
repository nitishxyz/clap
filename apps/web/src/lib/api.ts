export type DashboardServer = {
  version: string;
  uptimeMs: number;
  platform: string;
  arch: string;
  bunVersion: string;
  pid: number;
  rssBytes?: number;
  cpuPercent?: number;
  systemMemoryBytes?: number;
};

export type DashboardTotals = {
  requests: number;
  ok: number;
  errors: number;
  cancelled: number;
  promptTokens: number;
  completionTokens: number;
  cacheHits: number;
  cacheMisses: number;
  reusedTokens: number;
};

export type DetailMessage = {
  role: string;
  content: string;
  truncated?: boolean;
  toolCalls?: Array<{ name: string; arguments: string }>;
};

export type RequestDetail = {
  params: {
    temperature?: number;
    topP?: number;
    maxTokens?: number;
    stop?: string[];
    responseFormat?: string;
  };
  toolNames: string[];
  messages: DetailMessage[];
  droppedMessages: number;
  response?: {
    content?: string;
    reasoning?: string;
    toolCalls?: Array<{ name: string; arguments: string }>;
  };
  rawOutput?: string;
};

export type DashboardRequest = {
  id: string;
  startedAt: number;
  endedAt?: number;
  durationMs?: number;
  ttftMs?: number;
  loadMs?: number;
  queuedMs?: number;
  model: string;
  endpoint: string;
  stream: boolean;
  status: "active" | "ok" | "error" | "cancelled";
  phase: "queued" | "loading" | "prefill" | "decode" | "done";
  prefillDone?: number;
  prefillTotal?: number;
  conversation?: string;
  promptTokens?: number;
  completionTokens?: number;
  tokensPerSecond?: number;
  cacheHit?: boolean;
  reusedTokens?: number;
  sideRequest?: boolean;
  slot?: number;
  finishReason?: string;
  toolCalls?: number;
  messageCount?: number;
  error?: string;
  detail?: RequestDetail;
};

export type ServerEvent = {
  id: string;
  at: number;
  type: "server" | "load" | "unload" | "expire" | "error" | "download";
  message: string;
  model?: string;
  durationMs?: number;
};

export type DashboardLoadedModel = {
  key: string;
  id: string;
  backend: "llama" | "mlx";
  format: "gguf" | "mlx";
  localPath: string;
  state: "warm" | "active" | "unloading";
  activeRequests: number;
  loadedAt: string;
  lastUsedAt: string;
  keepAlive: string;
  expiresAt: string | null;
  pinned: boolean;
  worker: { pid?: number; state: string };
  usage?: { rssBytes: number; cpuPercent: number };
};

export type DashboardModel = {
  id: string;
  displayName: string;
  backend: "llama" | "mlx";
  format: string;
  status: "available" | "not_downloaded" | "unsupported";
  quantization?: string;
  limit?: { context?: number };
  capabilities?: { toolCall?: boolean; reasoning?: boolean };
  modalities?: { input?: string[] };
};

export type DashboardDownload = {
  id: string;
  model: string;
  file?: string;
  backend?: "gguf" | "mlx";
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  bytesReceived: number;
  totalBytes?: number;
  currentFile?: string;
  error?: string;
};

export type ModelResolveOption = {
  id: string;
  model: string;
  backend: "gguf" | "mlx";
  format: "gguf" | "mlx" | "safetensors";
  repo: string;
  file?: string;
  sizeBytes?: number;
  quantization?: string;
  supported: boolean;
  unsupportedReason?: string;
  recommended: boolean;
  reason: string;
};

export type ModelResolveResponse = {
  model: string;
  repo: string;
  options: ModelResolveOption[];
  selected?: ModelResolveOption;
};

export type DashboardData = {
  server: DashboardServer;
  totals: DashboardTotals;
  active: DashboardRequest[];
  requests: DashboardRequest[];
  events: ServerEvent[];
  loaded: DashboardLoadedModel[];
  models: DashboardModel[];
  downloads: DashboardDownload[];
};

export async function fetchDashboard(): Promise<DashboardData> {
  const response = await fetch("/clap/v1/dashboard");
  if (!response.ok) throw new Error(`dashboard fetch failed: ${response.status}`);
  return response.json() as Promise<DashboardData>;
}

export async function fetchRequestDetail(id: string): Promise<DashboardRequest> {
  const response = await fetch(`/clap/v1/dashboard/requests/${id}`);
  if (!response.ok) throw new Error(`request detail fetch failed: ${response.status}`);
  return response.json() as Promise<DashboardRequest>;
}

async function post(path: string, body: Record<string, unknown>): Promise<unknown> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => undefined) as { error?: { message?: string } } | undefined;
  if (!response.ok) {
    throw new Error(payload?.error?.message ?? `${path} failed: ${response.status}`);
  }
  return payload;
}

export function loadModel(model: string, keepAlive?: string): Promise<unknown> {
  return post("/clap/v1/models/load", keepAlive ? { model, keepAlive } : { model });
}

export function unloadModel(model: string): Promise<unknown> {
  return post("/clap/v1/models/unload", { model });
}

export function removeModel(model: string): Promise<unknown> {
  return post("/clap/v1/models/remove", { model });
}

export function pullModel(model: string, opts?: { file?: string; backend?: "gguf" | "mlx" }): Promise<unknown> {
  return post("/clap/v1/models/pull", { model, ...(opts?.file ? { file: opts.file } : {}), ...(opts?.backend ? { backend: opts.backend } : {}) });
}

export function resolveModel(model: string): Promise<ModelResolveResponse> {
  return post("/clap/v1/models/resolve", { model }) as Promise<ModelResolveResponse>;
}

export function cancelDownload(id: string): Promise<unknown> {
  return post(`/clap/v1/downloads/${id}/cancel`, {});
}
