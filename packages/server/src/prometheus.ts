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
  queue: { inflight: number; queued: number; maxInflight: number; queueDepth: number;
    inflightByPriority: Record<"interactive" | "normal" | "background", number>;
    waitingByPriority: Record<"interactive" | "normal" | "background", number>;
    outcomesByPriority: Record<"interactive" | "normal" | "background",
      Record<"admitted" | "rejected" | "aborted", number>> };
  loadedModels: Array<{
    id: string;
    backend: string;
    state: string;
    crashes?: number;
    retention?: {
      maxActive: number;
      active: number;
      retainedTotal: number;
      retainedSessions: number;
      retainedAnchors: number;
      retainedBytes: number | null;
      retainedBytesSource?: string;
      retainedBytesBasis?: string;
      sessionBytes: number | null;
      sessionBytesSource?: string;
      sessionBytesBasis?: string;
      anchorBytes: number | null;
      anchorBytesSource?: string;
      anchorBytesBasis?: string;
      evictedBytes?: number | null;
      evictedBytesSource?: string;
      evictedBytesBasis?: string;
      estimatedRetainedBytes?: number | null;
      estimatedRetainedBytesSource?: string;
      estimatedRetainedBytesBasis?: string;
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
  priorityRequestOutcomes: Map<string, number>;
  priorityDurationMs: Record<"interactive" | "normal" | "background", Histogram>;
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

  counter("clap_requests_total", "Completed inference requests by priority and status",
    [...snapshot.priorityRequestOutcomes].map(([key, value]) => {
      const [priority, status] = key.split("\0");
      return [`{priority="${esc(priority!)}",status="${esc(status!)}"}`, value];
    }));
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
  gauge("clap_queue_inflight", "Requests admitted past the fair limiter by priority",
    Object.entries(snapshot.queue.inflightByPriority).map(([priority, value]) =>
      [`{priority="${priority}"}`, value]));
  counter("clap_queue_outcomes_total", "Fair limiter outcomes by priority",
    Object.entries(snapshot.queue.outcomesByPriority).flatMap(([priority, outcomes]) =>
      Object.entries(outcomes).map(([outcome, value]) =>
        [`{priority="${priority}",outcome="${outcome}"}`, value] as [string, number])));
  gauge("clap_queue_waiting", "Requests waiting in the fair queue by priority",
    Object.entries(snapshot.queue.waitingByPriority).map(([priority, value]) =>
      [`{priority="${priority}"}`, value]));
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

  const grouped = (values: Array<[string, number]>) => [...values.reduce((result, [labels, value]) =>
    result.set(labels, (result.get(labels) ?? 0) + value), new Map<string, number>())];
  const groupedMax = (values: Array<[string, number]>) => [...values.reduce((result, [labels, value]) =>
    result.set(labels, Math.max(result.get(labels) ?? Number.NEGATIVE_INFINITY, value)), new Map<string, number>())];
  gauge("clap_model_loaded", "Loaded models grouped by backend and state", grouped(snapshot.loadedModels.map(
    (model) => [`{backend="${esc(model.backend)}",state="${esc(model.state)}"}`, 1],
  )));
  for (const [kind, field] of [
    ["context", "effectiveContextWindow"],
    ["input", "maxInputTokens"],
    ["output", "maxOutputTokens"],
  ] as const) {
    gauge(`clap_model_${kind}_token_limit`, `Maximum effective model ${kind} token limit by backend (omitted when unknown)`, groupedMax(snapshot.loadedModels
      .filter((model) => model.tokenCapabilities?.[field] !== null && model.tokenCapabilities?.[field] !== undefined)
      .map((model) => [`{backend="${esc(model.backend)}"}`, model.tokenCapabilities![field]!] as [string, number])));
  }
  counter("clap_worker_crashes_total", "Worker crashes since server start", grouped(snapshot.loadedModels
    .filter((model) => (model.crashes ?? 0) > 0)
    .map((model) => [`{backend="${esc(model.backend)}"}`, model.crashes ?? 0])));
  const retained = snapshot.loadedModels.filter((model) => model.retention !== undefined);
  for (const [name, help, field] of [
    ["clap_mlx_retention_max_active", "Maximum simultaneously active MLX retained entries", "maxActive"],
    ["clap_mlx_retention_active", "Currently active MLX retained entries", "active"],
    ["clap_mlx_retention_entries", "MLX retained entries", "retainedTotal"],
    ["clap_mlx_retention_session_entries", "MLX retained session entries", "retainedSessions"],
    ["clap_mlx_retention_anchor_entries", "MLX retained anchor entries", "retainedAnchors"],
    ["clap_mlx_retention_budget_bytes", "MLX retained cache byte budget", "budgetBytes"],
    ["clap_mlx_retention_high_watermark_bytes", "MLX retained cache high watermark", "highWatermarkBytes"],
    ["clap_mlx_retention_low_watermark_bytes", "MLX retained cache low watermark", "lowWatermarkBytes"],
    ["clap_mlx_retention_hard_ceiling", "MLX retained entry hard ceiling", "hardCeiling"],
  ] as const) {
    gauge(name, help, grouped(retained.map((model) => [`{backend="${esc(model.backend)}"}`, model.retention![field]])));
  }
  for (const [name, help, field] of [
    ["clap_retention_bytes", "Retained cache bytes by honest source and basis", "retainedBytes"],
    ["clap_retention_session_bytes", "Retained session cache bytes by honest source and basis", "sessionBytes"],
    ["clap_retention_anchor_bytes", "Retained anchor cache bytes by honest source and basis", "anchorBytes"],
    ["clap_retention_evicted_bytes", "Evicted cache bytes by honest source and basis", "evictedBytes"],
    ["clap_retention_estimated_bytes", "Separately reported retained cache estimate", "estimatedRetainedBytes"],
  ] as const) {
    gauge(name, help, grouped(retained.flatMap((model) => {
      const value = model.retention![field];
      const source = model.retention![`${field}Source`];
      const basis = model.retention![`${field}Basis`];
      return value === null || value === undefined || source === "unavailable" ? [] : [[
        `{backend="${esc(model.backend)}",source="${esc(source ?? "legacy")}",basis="${esc(basis ?? "not_reported")}"}`,
        value,
      ]];
    })));
  }
  gauge("clap_mlx_retention_under_pressure", "Whether any retained cache is under byte pressure", groupedMax(retained.map(
    (model) => [`{backend="${esc(model.backend)}"}`, model.retention!.underPressure ? 1 : 0],
  )));
  counter("clap_mlx_retention_evictions_total", "MLX retained cache evictions", grouped(retained.map(
    (model) => [`{backend="${esc(model.backend)}",reason="${esc(model.retention!.evictionReason ?? "none")}"}`, model.retention!.evictionCount],
  )));

  lines.push("# HELP clap_request_ttft_ms Time to first token (excludes queue wait)", "# TYPE clap_request_ttft_ms histogram");
  lines.push(...snapshot.histograms.ttftMs.render("clap_request_ttft_ms"));
  lines.push("# HELP clap_request_duration_ms End-to-end request duration", "# TYPE clap_request_duration_ms histogram");
  for (const priority of ["interactive", "normal", "background"] as const) {
    lines.push(...snapshot.priorityDurationMs[priority].render("clap_request_duration_ms", `priority="${priority}"`));
  }
  lines.push("# HELP clap_request_queued_ms Time spent waiting before dispatch", "# TYPE clap_request_queued_ms histogram");
  lines.push(...snapshot.histograms.queuedMs.render("clap_request_queued_ms"));
  lines.push("# HELP clap_request_completion_tokens Completion tokens per request", "# TYPE clap_request_completion_tokens histogram");
  lines.push(...snapshot.histograms.completionTokens.render("clap_request_completion_tokens"));

  return `${lines.join("\n")}\n`;
}
