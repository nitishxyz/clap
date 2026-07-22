import ClapCachePolicy

struct TimingTelemetryFacts {
  let receivedToAdmittedMs: Double
  let templateTokenizeMs: Double
  let coordinatorPlanMs: Double
  let coordinatorApplyMs: Double
  let schedulerWaitMs: Double
  let cacheMaterializeMs: Double
  let prefillMs: Double
  let promptTokens: Int
  let reusedTokens: Int
  let prefillTokens: Int
  let prefillChunks: Int
  let firstDecodeMs: Double
  let firstEmitMs: Double
}

func workerTiming(_ facts: TimingTelemetryFacts) -> WorkerTiming {
  WorkerTiming(
    received_to_admitted_ms: facts.receivedToAdmittedMs,
    template_tokenize_ms: facts.templateTokenizeMs,
    coordinator_wait_ms: facts.receivedToAdmittedMs,
    coordinator_plan_ms: facts.coordinatorPlanMs,
    coordinator_apply_ms: facts.coordinatorApplyMs,
    scheduler_wait_ms: facts.schedulerWaitMs,
    cache_materialize_ms: facts.cacheMaterializeMs,
    prefill_ms: facts.prefillMs,
    residual_prefill_tokens: facts.promptTokens - facts.reusedTokens,
    prefill_tokens: facts.prefillTokens,
    prefill_chunks: facts.prefillChunks,
    first_decode_ms: facts.firstDecodeMs,
    first_emit_ms: facts.firstEmitMs,
    normal_prefill_quantum: LatencyScheduler.normalPrefillQuantum,
    contended_prefill_quantum: LatencyScheduler.contendedPrefillQuantum)
}
