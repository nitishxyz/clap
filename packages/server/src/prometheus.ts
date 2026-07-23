// Prometheus text exposition (version 0.0.4) for clap.
// The dashboard and /metrics read the same underlying collector so the two
// views can never disagree; this module only formats.

export class Histogram {
  private readonly counts: number[];
  private sum = 0;
  private count = 0;

  constructor(private readonly buckets: number[]) {
    this.counts = new Array(buckets.length).fill(0);
  }

  observe(value: number): void {
    if (!Number.isFinite(value) || value < 0) return;
    this.sum += value;
    this.count += 1;
    for (let index = 0; index < this.buckets.length; index += 1) {
      const bucket = this.buckets[index] ?? 0;
      if (value <= bucket) this.counts[index] = (this.counts[index] ?? 0) + 1;
    }
  }

  render(name: string, labels = ""): string[] {
    const attach = (extra: string) => {
      const parts = [labels, extra].filter(Boolean).join(",");
      return parts ? `{${parts}}` : "";
    };
    const lines: string[] = [];
    for (let index = 0; index < this.buckets.length; index += 1) {
      lines.push(`${name}_bucket${attach(`le="${this.buckets[index]}"`)} ${this.counts[index]}`);
    }
    lines.push(`${name}_bucket${attach('le="+Inf"')} ${this.count}`);
    lines.push(`${name}_sum${labels ? `{${labels}}` : ""} ${this.sum}`);
    lines.push(`${name}_count${labels ? `{${labels}}` : ""} ${this.count}`);
    return lines;
  }
}

export type PromSnapshot = {
  totals: {
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
  activeRequests: number;
  queue: { inflight: number; queued: number; maxInflight: number; queueDepth: number };
  loadedModels: Array<{
    id: string;
    state: string;
    crashes?: number;
    retention?: {
      maxActive: number;
      active: number;
      retainedTotal: number;
      retainedSessions: number;
      retainedAnchors: number;
      retainedBytes: number;
      sessionBytes: number;
      anchorBytes: number;
      budgetBytes: number;
      highWatermarkBytes: number;
      lowWatermarkBytes: number;
      underPressure: boolean;
      hardCeiling: number;
      evictionReason?: string;
      evictionCount: number;
    };
    tokenCapabilities?: {
      effectiveContextWindow: number | null;
      maxInputTokens: number | null;
      maxOutputTokens: number | null;
    };
  }>;
  uptimeMs: number;
  histograms: {
    ttftMs: Histogram;
    durationMs: Histogram;
    queuedMs: Histogram;
    completionTokens: Histogram;
  };
  structuredOutputOutcomes: Map<string, number>;
  residency: {
    reservedBytes: number;
    activeReservations: number;
    outcomes: Map<string, number>;
    evictions: Map<string, number>;
    estimateObservedRatioSum: number;
    estimateObservedRatioCount: number;
  };
};

export function makeRequestHistograms(): PromSnapshot["histograms"] {
  return {
    ttftMs: new Histogram([50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]),
    durationMs: new Histogram([100, 500, 1000, 5000, 15000, 30000, 60000, 120000, 300000, 600000]),
    queuedMs: new Histogram([10, 50, 100, 500, 1000, 5000, 15000, 60000]),
    completionTokens: new Histogram([16, 64, 256, 1024, 4096, 16384]),
  };
}

function esc(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll('"', '\\"').replaceAll("\n", "\\n");
}

export function renderPrometheus(snapshot: PromSnapshot): string {
  const lines: string[] = [];
  const counter = (name: string, help: string, entries: Array<[string, number]>) => {
    lines.push(`# HELP ${name} ${help}`, `# TYPE ${name} counter`);
    for (const [labels, value] of entries) lines.push(`${name}${labels} ${value}`);
  };
  const gauge = (name: string, help: string, entries: Array<[string, number]>) => {
    lines.push(`# HELP ${name} ${help}`, `# TYPE ${name} gauge`);
    for (const [labels, value] of entries) lines.push(`${name}${labels} ${value}`);
  };

  counter("clap_requests_total", "Completed inference requests by status", [
    ['{status="ok"}', snapshot.totals.ok],
    ['{status="error"}', snapshot.totals.errors],
    ['{status="cancelled"}', snapshot.totals.cancelled],
  ]);
  counter("clap_tokens_total", "Tokens processed", [
    ['{kind="prompt"}', snapshot.totals.promptTokens],
    ['{kind="completion"}', snapshot.totals.completionTokens],
    ['{kind="kv_reused"}', snapshot.totals.reusedTokens],
  ]);
  counter("clap_structured_output_total", "Structured output requests by bounded contract and outcome",
    [...snapshot.structuredOutputOutcomes].map(([key, value]) => {
      const [kind, strength, mode, outcome] = key.split("\0");
      return [`{kind="${esc(kind!)}",strength="${esc(strength!)}",mode="${esc(mode!)}",outcome="${esc(outcome!)}"}`, value];
    }));
  counter("clap_kv_cache_total", "KV cache lookups by outcome", [
    ['{outcome="hit"}', snapshot.totals.cacheHits],
    ['{outcome="miss"}', snapshot.totals.cacheMisses],
  ]);

  gauge("clap_requests_active", "Requests currently executing or streaming", [["", snapshot.activeRequests]]);
  gauge("clap_queue_inflight", "Requests admitted past the fair limiter", [["", snapshot.queue.inflight]]);
  gauge("clap_queue_waiting", "Requests waiting in the fair queue", [["", snapshot.queue.queued]]);
  gauge("clap_queue_inflight_limit", "Configured max_inflight", [["", snapshot.queue.maxInflight]]);
  gauge("clap_queue_depth_limit", "Configured queue_depth", [["", snapshot.queue.queueDepth]]);
  gauge("clap_uptime_seconds", "Server uptime", [["", Math.round(snapshot.uptimeMs / 1000)]]);
  gauge("clap_residency_reserved_bytes", "Bytes held by active model-load reservations", [["", snapshot.residency.reservedBytes]]);
  gauge("clap_residency_active_reservations", "Active model-load reservations", [["", snapshot.residency.activeReservations]]);
  counter("clap_residency_load_outcomes_total", "Model-load admission outcomes by backend and reason",
    [...snapshot.residency.outcomes].map(([key, value]) => {
      const [backend, reason, outcome] = key.split("\0");
      return [`{backend="${esc(backend!)}",reason="${esc(reason!)}",outcome="${esc(outcome!)}"}`, value];
    }));
  counter("clap_residency_evictions_total", "Models evicted for load admission by backend and reason",
    [...snapshot.residency.evictions].map(([key, value]) => {
      const [backend, reason] = key.split("\0");
      return [`{backend="${esc(backend!)}",reason="${esc(reason!)}"}`, value];
    }));
  gauge("clap_residency_estimate_observed_ratio", "Mean estimated bytes divided by observed RSS",
    [["", snapshot.residency.estimateObservedRatioCount > 0
      ? snapshot.residency.estimateObservedRatioSum / snapshot.residency.estimateObservedRatioCount : 0]]);

  gauge("clap_model_loaded", "Loaded models (1 per loaded model)", snapshot.loadedModels.map(
    (model) => [`{model="${esc(model.id)}",state="${esc(model.state)}"}`, 1],
  ));
  for (const [kind, field] of [
    ["context", "effectiveContextWindow"],
    ["input", "maxInputTokens"],
    ["output", "maxOutputTokens"],
  ] as const) {
    gauge(`clap_model_${kind}_token_limit`, `Effective model ${kind} token limit (omitted when unknown)`, snapshot.loadedModels
      .filter((model) => model.tokenCapabilities?.[field] !== null && model.tokenCapabilities?.[field] !== undefined)
      .map((model) => [`{model="${esc(model.id)}"}`, model.tokenCapabilities![field]!]));
  }
  counter("clap_worker_crashes_total", "Worker crashes since server start", snapshot.loadedModels
    .filter((model) => (model.crashes ?? 0) > 0)
    .map((model) => [`{model="${esc(model.id)}"}`, model.crashes ?? 0]));
  const retained = snapshot.loadedModels.filter((model) => model.retention !== undefined);
  for (const [name, help, field] of [
    ["clap_mlx_retention_max_active", "Maximum simultaneously active MLX retained entries", "maxActive"],
    ["clap_mlx_retention_active", "Currently active MLX retained entries", "active"],
    ["clap_mlx_retention_entries", "MLX retained entries", "retainedTotal"],
    ["clap_mlx_retention_session_entries", "MLX retained session entries", "retainedSessions"],
    ["clap_mlx_retention_anchor_entries", "MLX retained anchor entries", "retainedAnchors"],
    ["clap_mlx_retention_bytes", "Physical bytes retained by MLX caches", "retainedBytes"],
    ["clap_mlx_retention_session_bytes", "Physical bytes retained by MLX session caches", "sessionBytes"],
    ["clap_mlx_retention_anchor_bytes", "Physical bytes retained by MLX prefix anchors", "anchorBytes"],
    ["clap_mlx_retention_budget_bytes", "MLX retained cache byte budget", "budgetBytes"],
    ["clap_mlx_retention_high_watermark_bytes", "MLX retained cache high watermark", "highWatermarkBytes"],
    ["clap_mlx_retention_low_watermark_bytes", "MLX retained cache low watermark", "lowWatermarkBytes"],
    ["clap_mlx_retention_hard_ceiling", "MLX retained entry hard ceiling", "hardCeiling"],
  ] as const) {
    gauge(name, help, retained.map((model) => [`{model="${esc(model.id)}"}`, model.retention![field]]));
  }
  gauge("clap_mlx_retention_under_pressure", "Whether MLX retained caches are under byte pressure", retained.map(
    (model) => [`{model="${esc(model.id)}"}`, model.retention!.underPressure ? 1 : 0],
  ));
  counter("clap_mlx_retention_evictions_total", "MLX retained cache evictions", retained.map(
    (model) => [`{model="${esc(model.id)}",reason="${esc(model.retention!.evictionReason ?? "none")}"}`, model.retention!.evictionCount],
  ));

  lines.push("# HELP clap_request_ttft_ms Time to first token (excludes queue wait)", "# TYPE clap_request_ttft_ms histogram");
  lines.push(...snapshot.histograms.ttftMs.render("clap_request_ttft_ms"));
  lines.push("# HELP clap_request_duration_ms End-to-end request duration", "# TYPE clap_request_duration_ms histogram");
  lines.push(...snapshot.histograms.durationMs.render("clap_request_duration_ms"));
  lines.push("# HELP clap_request_queued_ms Time spent waiting before dispatch", "# TYPE clap_request_queued_ms histogram");
  lines.push(...snapshot.histograms.queuedMs.render("clap_request_queued_ms"));
  lines.push("# HELP clap_request_completion_tokens Completion tokens per request", "# TYPE clap_request_completion_tokens histogram");
  lines.push(...snapshot.histograms.completionTokens.render("clap_request_completion_tokens"));

  return `${lines.join("\n")}\n`;
}
