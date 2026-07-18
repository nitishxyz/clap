import { makeRequestHistograms } from "./prometheus";

export type RequestStatus = "active" | "ok" | "error" | "cancelled";

export type RequestPhase = "queued" | "loading" | "prefill" | "decode" | "done";

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

export type RequestRecord = {
  id: string;
  startedAt: number;
  endedAt?: number;
  durationMs?: number;
  ttftMs?: number;
  queuedMs?: number;
  loadMs?: number;
  model: string;
  endpoint: string;
  stream: boolean;
  status: RequestStatus;
  phase: RequestPhase;
  prefillDone?: number;
  prefillTotal?: number;
  conversation?: string;
  promptTokens?: number;
  completionTokens?: number;
  tokensPerSecond?: number;
  cacheHit?: boolean;
  reusedTokens?: number;
  reuseKind?: "slot" | "branch" | "anchor";
  reuseScope?: "system" | "conversation";
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

export type MetricsTotals = {
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

export type RequestFinish = {
  status?: RequestStatus;
  promptTokens?: number;
  completionTokens?: number;
  cacheHit?: boolean;
  reusedTokens?: number;
  reuseKind?: "slot" | "branch" | "anchor";
  reuseScope?: "system" | "conversation";
  sideRequest?: boolean;
  slot?: number;
  finishReason?: string;
  toolCalls?: number;
  error?: string;
  response?: RequestDetail["response"];
  rawOutput?: string;
};

export type RequestHandle = {
  record: RequestRecord;
  capture(request: ChatLikeRequest): void;
  phase(phase: RequestPhase): void;
  prefill(done: number, total: number): void;
  loaded(durationMs: number): void;
  firstToken(): void;
  finish(result: RequestFinish): void;
};

export type ChatLikeRequest = {
  messages?: Array<{ role: string; content?: unknown; tool_calls?: Array<{ function?: { name?: string; arguments?: string } }> }>;
  tools?: Array<{ function?: { name?: string } }>;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  response_format?: { type?: string };
};

const HISTORY_LIMIT = 200;
const EVENT_LIMIT = 100;
const MESSAGE_LIMIT = 60;
const CONTENT_LIMIT = 6000;
const RAW_LIMIT = 12000;

function truncate(text: string, limit: number): { text: string; truncated: boolean } {
  if (text.length <= limit) return { text, truncated: false };
  return { text: `${text.slice(0, limit)}…`, truncated: true };
}

function contentToString(content: unknown): string {
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  try {
    return JSON.stringify(content);
  } catch {
    return String(content);
  }
}

// Short stable fingerprint of the conversation identity (model + system
// prompt + first user message) so the dashboard can group requests belonging
// to the same session: multi-turn agent loops share one tag while unrelated
// side requests (title generation, other tools) get their own.
function conversationFingerprint(model: string, messages: ChatLikeRequest["messages"]): string | undefined {
  const list = messages ?? [];
  const system = list.find((message) => message.role === "system");
  const user = list.find((message) => message.role === "user");
  if (!system && !user) return undefined;
  const seed = `${model}\n${contentToString(system?.content).slice(0, 4000)}\n${contentToString(user?.content).slice(0, 500)}`;
  let hash = 0x811c9dc5;
  for (let i = 0; i < seed.length; i += 1) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0").slice(0, 6);
}

export class MetricsCollector {
  private readonly history: RequestRecord[] = [];
  private readonly active = new Map<string, RequestRecord>();
  private readonly byId = new Map<string, RequestRecord>();
  private readonly eventLog: ServerEvent[] = [];
  private sequence = 0;
  private eventSequence = 0;
  readonly histograms = makeRequestHistograms();

  readonly totals: MetricsTotals = {
    requests: 0,
    ok: 0,
    errors: 0,
    cancelled: 0,
    promptTokens: 0,
    completionTokens: 0,
    cacheHits: 0,
    cacheMisses: 0,
    reusedTokens: 0,
  };

  event(type: ServerEvent["type"], message: string, extra?: { model?: string; durationMs?: number }): void {
    this.eventSequence += 1;
    this.eventLog.push({
      id: `e${this.eventSequence}`,
      at: Date.now(),
      type,
      message,
      model: extra?.model,
      durationMs: extra?.durationMs,
    });
    if (this.eventLog.length > EVENT_LIMIT) this.eventLog.shift();
  }

  events(limit = 50): ServerEvent[] {
    return this.eventLog.slice(-limit).reverse();
  }

  start(model: string, endpoint: string, stream: boolean): RequestHandle {
    this.sequence += 1;
    const record: RequestRecord = {
      id: `r${this.sequence}`,
      startedAt: Date.now(),
      model,
      endpoint,
      stream,
      status: "active",
      phase: "queued",
    };
    this.active.set(record.id, record);
    this.byId.set(record.id, record);
    this.totals.requests += 1;
    let finished = false;
    // Queue accounting: a request may wait behind others on the same serial
    // worker. Time spent in the "queued" phase is tracked separately so TTFT
    // reflects actual model work instead of billing every waiting request for
    // the requests ahead of it.
    let queueStart: number | undefined = Date.now();
    let queuedTotal = 0;
    let dispatchedAt: number | undefined;
    const leaveQueue = () => {
      if (queueStart === undefined) return;
      queuedTotal += Date.now() - queueStart;
      queueStart = undefined;
      dispatchedAt = Date.now();
      if (queuedTotal >= 50) record.queuedMs = queuedTotal;
    };
    return {
      record,
      capture: (request) => {
        const messages = request.messages ?? [];
        const kept = messages.slice(-MESSAGE_LIMIT);
        record.messageCount = messages.length;
        record.conversation = conversationFingerprint(record.model, messages);
        record.detail = {
          params: {
            temperature: request.temperature,
            topP: request.top_p,
            maxTokens: request.max_tokens,
            stop: typeof request.stop === "string" ? [request.stop] : request.stop,
            responseFormat: request.response_format?.type,
          },
          toolNames: (request.tools ?? [])
            .map((tool) => tool.function?.name)
            .filter((name): name is string => Boolean(name)),
          droppedMessages: messages.length - kept.length,
          messages: kept.map((message) => {
            const body = truncate(contentToString(message.content), CONTENT_LIMIT);
            const entry: DetailMessage = { role: message.role, content: body.text };
            if (body.truncated) entry.truncated = true;
            const calls = (message.tool_calls ?? [])
              .map((call) => ({
                name: call.function?.name ?? "?",
                arguments: truncate(call.function?.arguments ?? "", 2000).text,
              }));
            if (calls.length) entry.toolCalls = calls;
            return entry;
          }),
        };
      },
      phase: (phase) => {
        if (phase === "queued") {
          if (queueStart === undefined) queueStart = Date.now();
        } else {
          leaveQueue();
        }
        record.phase = phase;
      },
      prefill: (done, total) => {
        leaveQueue();
        record.phase = "prefill";
        record.prefillDone = done;
        record.prefillTotal = total;
      },
      loaded: (durationMs) => {
        record.loadMs = durationMs;
        if (durationMs > 1500) {
          this.event("load", `${record.model} loaded into memory in ${(durationMs / 1000).toFixed(1)}s`, { model: record.model, durationMs });
        }
      },
      firstToken: () => {
        leaveQueue();
        if (record.ttftMs === undefined) record.ttftMs = Date.now() - (dispatchedAt ?? record.startedAt);
        record.phase = "decode";
      },
      finish: (result) => {
        if (finished) return;
        finished = true;
        leaveQueue();
        this.active.delete(record.id);
        record.endedAt = Date.now();
        record.durationMs = record.endedAt - record.startedAt;
        record.status = result.status ?? "ok";
        record.phase = "done";
        record.promptTokens = result.promptTokens;
        record.completionTokens = result.completionTokens;
        record.cacheHit = result.cacheHit;
        record.reusedTokens = result.reusedTokens;
        record.reuseKind = result.reuseKind;
        record.reuseScope = result.reuseScope;
        record.sideRequest = result.sideRequest;
        record.slot = result.slot;
        record.finishReason = result.finishReason;
        record.toolCalls = result.toolCalls;
        record.error = result.error;
        if (record.detail) {
          if (result.response) {
            record.detail.response = {
              content: result.response.content ? truncate(result.response.content, CONTENT_LIMIT).text : undefined,
              reasoning: result.response.reasoning ? truncate(result.response.reasoning, CONTENT_LIMIT).text : undefined,
              toolCalls: result.response.toolCalls?.map((call) => ({
                name: call.name,
                arguments: truncate(call.arguments, 2000).text,
              })),
            };
          }
          if (result.rawOutput) record.detail.rawOutput = truncate(result.rawOutput, RAW_LIMIT).text;
        }
        if (result.completionTokens && record.durationMs > 0) {
          const working = record.durationMs - (record.queuedMs ?? 0) - (record.loadMs ?? 0);
          const afterFirstToken = record.ttftMs !== undefined ? working - record.ttftMs : 0;
          const decodeMs = afterFirstToken > 50 ? afterFirstToken : Math.max(working, 1);
          record.tokensPerSecond = Math.round((result.completionTokens / decodeMs) * 1000 * 10) / 10;
        }
        if (record.status === "ok") this.totals.ok += 1;
        else if (record.status === "error") this.totals.errors += 1;
        else if (record.status === "cancelled") this.totals.cancelled += 1;
        if (record.ttftMs !== undefined) this.histograms.ttftMs.observe(record.ttftMs);
        if (record.durationMs !== undefined) this.histograms.durationMs.observe(record.durationMs);
        this.histograms.queuedMs.observe(record.queuedMs ?? 0);
        if (record.completionTokens !== undefined) this.histograms.completionTokens.observe(record.completionTokens);
        this.totals.promptTokens += result.promptTokens ?? 0;
        this.totals.completionTokens += result.completionTokens ?? 0;
        if (result.cacheHit === true) {
          this.totals.cacheHits += 1;
          this.totals.reusedTokens += result.reusedTokens ?? 0;
        } else if (result.cacheHit === false) {
          this.totals.cacheMisses += 1;
        }
        this.history.push(record);
        if (this.history.length > HISTORY_LIMIT) {
          const removed = this.history.shift();
          if (removed) this.byId.delete(removed.id);
        }
      },
    };
  }

  activeRequests(): RequestRecord[] {
    return [...this.active.values()].map((record) => ({ ...record, detail: undefined }));
  }

  recent(limit = 50): RequestRecord[] {
    return this.history.slice(-limit).reverse().map((record) => ({ ...record, detail: undefined }));
  }

  request(id: string): RequestRecord | undefined {
    return this.byId.get(id);
  }
}
