import type { ModelTokenCapabilities } from "@clap/api";
import { parseWorkerRetention, parseWorkerTokenCapabilities, type ResidentCacheInfo,
  type ResidentChatResult, type ResidentMlxMemory, type ResidentMlxRetention,
  type ResidentProgress, type ResidentTiming, type ResidentUsage } from "../resident";

export type PendingWorkerResult = {
  content: string[];
  resolve: (result: ResidentChatResult) => void;
  reject: (error: Error) => void;
  onToken?: (token: string) => void;
  onProgress?: ResidentProgress;
  onDispatch?: () => void;
  usage?: ResidentUsage;
  finishReason?: "stop" | "length" | "cancel";
  cache?: ResidentCacheInfo;
  timing?: ResidentTiming;
  tokenCapabilities?: ModelTokenCapabilities;
};

export type WorkerPayloadContext = {
  pending: Map<string, PendingWorkerResult>;
  retention?: ResidentMlxRetention;
  workerLaunchId?: string;
  setMemory: (memory: ResidentMlxMemory) => void;
  setRetention: (retention: ResidentMlxRetention) => void;
  setTokenCapabilities: (capabilities: ModelTokenCapabilities) => void;
  onRetention: (previous: ResidentMlxRetention | undefined, current: ResidentMlxRetention) => void;
  workerError: (message: string, code?: string) => Error;
};

export function mapWorkerResultPayload(
  result: Record<string, unknown>,
  requestId: string,
  hasStreamedContent: boolean,
): Record<string, unknown> {
  const { content: resultContent, ...resultWithoutContent } = result;
  const content = !hasStreamedContent && typeof resultContent === "string"
    ? { content: resultContent }
    : {};
  return { ...resultWithoutContent, ...content, id: requestId, done: true };
}

export function mapWorkerTelemetryPayload(
  telemetry: Record<string, unknown>,
): Record<string, unknown> {
  return "memory" in telemetry || "retention" in telemetry
    ? telemetry
    : { retention: telemetry };
}

export function applyWorkerPayload(
  message: Record<string, unknown>,
  context: WorkerPayloadContext,
): void {
    const id = typeof message.id === "string" ? message.id : undefined;
    if (message.memory && typeof message.memory === "object") {
      const memory = message.memory as Record<string, unknown>;
      if (typeof memory.active_bytes === "number" && typeof memory.cache_bytes === "number" && typeof memory.peak_active_bytes === "number") {
        context.setMemory({
          activeBytes: memory.active_bytes,
          cacheBytes: memory.cache_bytes,
          peakActiveBytes: memory.peak_active_bytes,
        });
      }
    }
    const parsedRetention = parseWorkerRetention(message.retention);
    if (parsedRetention) {
      const previous = context.retention;
      context.setRetention(parsedRetention);
      context.onRetention(previous, parsedRetention);
    }
    const parsedCapabilities = parseWorkerTokenCapabilities(message.token_capabilities);
    if (parsedCapabilities) context.setTokenCapabilities(parsedCapabilities);
    const pending = id ? context.pending.get(id) : undefined;
    if (!pending) return;
    if (message.started === true) {
      const onDispatch = pending.onDispatch;
      pending.onDispatch = undefined;
      onDispatch?.();
    }
    if (message.error) {
      context.pending.delete(id ?? "");
      const code = typeof message.code === "string" ? message.code : undefined;
      pending.reject(context.workerError(String(message.error), code));
      return;
    }
    if (typeof message.token === "string") {
      pending.content.push(message.token);
      pending.onToken?.(message.token);
    }
    if (message.prefill && typeof message.prefill === "object") {
      const prefill = message.prefill as Record<string, unknown>;
      if (typeof prefill.done === "number" && typeof prefill.total === "number") {
        pending.onProgress?.(prefill.done, prefill.total);
      }
    }
    if (typeof message.content === "string") {
      pending.content.push(message.content);
      pending.onToken?.(message.content);
    }
    if (message.usage && typeof message.usage === "object") {
      const usage = message.usage as Record<string, unknown>;
      pending.usage = {
        promptTokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : undefined,
        completionTokens: typeof usage.completion_tokens === "number" ? usage.completion_tokens : undefined,
      };
    }
    if (message.timing && typeof message.timing === "object") {
      const timing = message.timing as Record<string, unknown>;
      const number = (key: string) => typeof timing[key] === "number" ? timing[key] as number : undefined;
      pending.timing = {
        receivedToAdmittedMs: number("received_to_admitted_ms"),
        templateTokenizeMs: number("template_tokenize_ms"),
        coordinatorWaitMs: number("coordinator_wait_ms"),
        coordinatorPlanMs: number("coordinator_plan_ms"),
        coordinatorApplyMs: number("coordinator_apply_ms"),
        schedulerWaitMs: number("scheduler_wait_ms"),
        cacheMaterializeMs: number("cache_materialize_ms"),
        prefillMs: number("prefill_ms"),
        residualPrefillTokens: number("residual_prefill_tokens"),
        prefillTokens: number("prefill_tokens"),
        prefillChunks: number("prefill_chunks"),
        firstDecodeMs: number("first_decode_ms"),
        firstEmitMs: number("first_emit_ms"),
        normalPrefillQuantum: number("normal_prefill_quantum"),
        contendedPrefillQuantum: number("contended_prefill_quantum"),
      };
    }
    if (message.cache && typeof message.cache === "object") {
      const cache = message.cache as Record<string, unknown>;
      const rejectionReasons = new Set(["namespace", "model_domain", "generation", "capability", "busy_lease", "materialization", "session", "nontrim", "min_prefix", "capacity", "absent_anchor", "lower_rank"]);
      const candidates = Array.isArray(cache.candidates) ? cache.candidates.flatMap((value) => {
        if (!value || typeof value !== "object") return [];
        const candidate = value as Record<string, unknown>;
        if (typeof candidate.slot !== "number" || typeof candidate.shared_prefix_tokens !== "number") return [];
        return [{
          slot: candidate.slot,
          generation: typeof candidate.generation === "number" ? candidate.generation : undefined,
          state: typeof candidate.state === "string" ? candidate.state : undefined,
          sharedPrefixTokens: candidate.shared_prefix_tokens,
          namespaceCompatible: typeof candidate.namespace_compatible === "boolean" ? candidate.namespace_compatible : undefined,
          modelCompatible: typeof candidate.model_compatible === "boolean" ? candidate.model_compatible : undefined,
          sessionCompatible: typeof candidate.session_compatible === "boolean" ? candidate.session_compatible : undefined,
          generationCompatible: typeof candidate.generation_compatible === "boolean" ? candidate.generation_compatible : undefined,
          busyEligible: typeof candidate.busy_eligible === "boolean" ? candidate.busy_eligible : undefined,
          leaseEligible: typeof candidate.lease_eligible === "boolean" ? candidate.lease_eligible : undefined,
          materialized: typeof candidate.materialized === "boolean" ? candidate.materialized : undefined,
          trimEligible: typeof candidate.trim_eligible === "boolean" ? candidate.trim_eligible : undefined,
          copyEligible: typeof candidate.copy_eligible === "boolean" ? candidate.copy_eligible : undefined,
          eligible: typeof candidate.eligible === "boolean" ? candidate.eligible : undefined,
          selected: typeof candidate.selected === "boolean" ? candidate.selected : undefined,
          rejection: typeof candidate.rejection === "string" && rejectionReasons.has(candidate.rejection)
            ? candidate.rejection as NonNullable<ResidentCacheInfo["candidates"]>[number]["rejection"]
            : undefined,
        }];
      }) : undefined;
      pending.cache = {
        hit: typeof cache.hit === "boolean" ? cache.hit : undefined,
        reusedTokens: typeof cache.reused_tokens === "number" ? cache.reused_tokens : undefined,
        reuseKind: cache.reuse_kind === "slot" || cache.reuse_kind === "branch" || cache.reuse_kind === "anchor" ? cache.reuse_kind : undefined,
        reuseScope: cache.reuse_scope === "system" || cache.reuse_scope === "conversation" || cache.reuse_scope === "session" || cache.reuse_scope === "agent" || cache.reuse_scope === "project" || cache.reuse_scope === "harness" || cache.reuse_scope === "tenant" ? cache.reuse_scope : undefined,
        sideRequest: typeof cache.side_request === "boolean" ? cache.side_request : undefined,
        slot: typeof cache.slot === "number" ? cache.slot : undefined,
        namespace: typeof cache.namespace === "string" ? cache.namespace : undefined,
        donorSlot: typeof cache.donor_slot === "number" ? cache.donor_slot : undefined,
        targetSlot: typeof cache.target_slot === "number" ? cache.target_slot : undefined,
        evictedSlots: Array.isArray(cache.evicted_slots) && cache.evicted_slots.every((slot) => typeof slot === "number") ? cache.evicted_slots as number[] : undefined,
        decisionUs: typeof cache.decision_us === "number" ? cache.decision_us : undefined,
        plannedReuseTokens: typeof cache.planned_reuse_tokens === "number" ? cache.planned_reuse_tokens : undefined,
        realizedReuseTokens: typeof cache.realized_reuse_tokens === "number" ? cache.realized_reuse_tokens : undefined,
        fallback: typeof cache.fallback === "string" ? cache.fallback : undefined,
        missReason: typeof cache.miss_reason === "string" ? cache.miss_reason : undefined,
        workerLaunchId: typeof cache.worker_launch_id === "string" ? cache.worker_launch_id : undefined,
        donorGeneration: typeof cache.donor_generation === "number" ? cache.donor_generation : undefined,
        targetGeneration: typeof cache.target_generation === "number" ? cache.target_generation : undefined,
        evictions: Array.isArray(cache.evictions) ? cache.evictions.flatMap((value) => {
          if (!value || typeof value !== "object") return [];
          const eviction = value as Record<string, unknown>;
          return typeof eviction.slot === "number" ? [{ slot: eviction.slot, reason: typeof eviction.reason === "string" ? eviction.reason : undefined }] : [];
        }) : undefined,
        candidates,
        systemTokenHash: typeof cache.system_token_hash === "string" ? cache.system_token_hash : undefined,
        systemTokenCount: typeof cache.system_token_count === "number" ? cache.system_token_count : undefined,
        toolsTokenHash: typeof cache.tools_token_hash === "string" ? cache.tools_token_hash : undefined,
        toolsTokenCount: typeof cache.tools_token_count === "number" ? cache.tools_token_count : undefined,
        ...(typeof cache.stable_boundary_token_hash === "string" && cache.stable_boundary_token_hash.length > 0
          && typeof cache.stable_boundary_token_count === "number" && Number.isInteger(cache.stable_boundary_token_count)
          && cache.stable_boundary_token_count > 0
          && typeof cache.stable_boundary_kind === "string" && cache.stable_boundary_kind.length > 0 ? {
            stableBoundaryTokenHash: cache.stable_boundary_token_hash,
            stableBoundaryTokenCount: cache.stable_boundary_token_count,
            stableBoundaryKind: cache.stable_boundary_kind,
          } : {}),
        stableBoundaries: Array.isArray(cache.stable_boundaries) ? cache.stable_boundaries.flatMap((value) => {
          if (!value || typeof value !== "object") return [];
          const boundary = value as Record<string, unknown>;
          if (typeof boundary.kind !== "string" ||
              (boundary.status !== "resolved" && boundary.status !== "authorized" && boundary.status !== "skipped")) return [];
          const resolved = (boundary.status === "resolved" || boundary.status === "authorized") && typeof boundary.token_hash === "string" &&
              typeof boundary.token_count === "number" && typeof boundary.materialized === "boolean";
          if ((boundary.status === "resolved" || boundary.status === "authorized") && !resolved) return [];
          const label = typeof boundary.label === "string" && boundary.label.length <= 64 &&
              /^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(boundary.label) ? boundary.label : undefined;
          const skipReason = boundary.skip_reason === "unsupported_template_boundary" ||
              boundary.skip_reason === "non_prefix_template_boundary" ? boundary.skip_reason : undefined;
          return [{
            tokenHash: resolved ? boundary.token_hash as string : undefined,
            tokenCount: resolved ? boundary.token_count as number : undefined,
            kind: boundary.kind,
            label,
            requested: boundary.requested === true,
            status: boundary.status as "resolved" | "authorized" | "skipped",
            skipReason,
            materialized: resolved ? boundary.materialized as boolean : undefined,
          }];
        }) : undefined,
        promptTokenHash: typeof cache.prompt_token_hash === "string" ? cache.prompt_token_hash : undefined,
        promptTokenCount: typeof cache.prompt_token_count === "number" ? cache.prompt_token_count : undefined,
        prefillMs: typeof cache.prefill_ms === "number" ? cache.prefill_ms : undefined,
      };
    }
    if (message.finish_reason === "stop" || message.finish_reason === "length" || message.finish_reason === "cancel") {
      pending.finishReason = message.finish_reason;
    }
    if (parsedCapabilities) pending.tokenCapabilities = parsedCapabilities;
    if (message.loaded === true || message.unloaded === true || message.done === true) {
      if (id) context.pending.delete(id);
      if (pending.cache && !pending.cache.workerLaunchId) pending.cache.workerLaunchId = context.workerLaunchId;
      pending.resolve({ content: pending.content.join(""), usage: pending.usage, finishReason: pending.finishReason, cache: pending.cache, timing: pending.timing, tokenCapabilities: pending.tokenCapabilities });
    }
  }
