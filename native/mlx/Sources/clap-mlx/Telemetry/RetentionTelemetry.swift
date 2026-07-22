import ClapCacheBridge
import ClapCachePolicy

struct RetentionTelemetryFacts {
  let telemetry: cc_retention_telemetry_t
  let configuration: WorkerConfiguration
  let activePolicy: ActiveConcurrencyDecision
  let maxActive: Int
  let queued: Int
  let previousMaxActive: Int?
  let limitingReason: String?
  let lastAdjustmentReason: String?
  let lastAdjustmentAt: String?
  let coordinatedGrowthReserveBytes: UInt64?
  let globalResidentMemoryBytes: UInt64?
  let pressureState: String?
  let modelActiveBytes: UInt64?
  let hybridOrRecurrent: Bool
  let activeCount: Int
  let lastEvictionReason: String?
}

func workerRetention(_ facts: RetentionTelemetryFacts) -> WorkerRetention {
  let telemetry = facts.telemetry
  let retention = facts.configuration.retention
  let retainedBytes = min(telemetry.total_bytes, retention.physicalByteBudget)
  let remainingBudget = retention.physicalByteBudget - retainedBytes
  let growthReserve = min(remainingBudget, max(facts.configuration.retainedGrowthMinimumBytes,
    remainingBudget / 100 * min(100, facts.configuration.retainedGrowthReservePercent)))
  let policy = WorkerActivePolicy(mode: facts.activePolicy.mode,
    selected_max: facts.maxActive, backend_ceiling: facts.activePolicy.backendCeiling,
    hardware_ceiling: facts.activePolicy.hardwareCeiling,
    model_ceiling: facts.activePolicy.modelCeiling,
    memory_ceiling: facts.activePolicy.memoryCeiling,
    reason: facts.limitingReason ?? facts.activePolicy.reason,
    inputs: WorkerActivePolicyInputs(
      physical_memory_bytes: facts.configuration.physicalMemoryBytes,
      startup_available_bytes: facts.configuration.availableMemoryAtStartup,
      model_active_bytes: facts.modelActiveBytes,
      retained_budget_bytes: retention.physicalByteBudget,
      retained_bytes: retainedBytes,
      retained_growth_reserve_bytes: growthReserve,
      os_reserve_bytes: facts.activePolicy.osReserveBytes,
      usable_runtime_bytes: facts.activePolicy.usableRuntimeBytes,
      per_active_reserve_bytes: facts.activePolicy.perActiveReserveBytes,
      processor_count: facts.configuration.processorCount,
      hybrid_or_recurrent: facts.hybridOrRecurrent))
  return WorkerRetention(max_active: facts.maxActive, queued: facts.queued,
    previous_max_active: facts.previousMaxActive,
    last_adjustment_reason: facts.lastAdjustmentReason,
    last_adjustment_at: facts.lastAdjustmentAt,
    retained_growth_reserve_bytes: facts.coordinatedGrowthReserveBytes ?? growthReserve,
    global_resident_memory_bytes: facts.globalResidentMemoryBytes,
    pressure_state: facts.pressureState, active_policy: policy,
    active: facts.activeCount,
    retained_total: Int(telemetry.total_slots), retained_sessions: Int(telemetry.session_slots),
    retained_anchors: Int(telemetry.anchor_slots), retained_bytes: telemetry.total_bytes,
    session_bytes: telemetry.session_bytes, anchor_bytes: telemetry.anchor_bytes,
    automatic_checkpoint_count: Int(telemetry.automatic_checkpoint_slots),
    automatic_checkpoint_bytes: telemetry.automatic_checkpoint_bytes,
    automatic_checkpoint_budget_bytes: telemetry.automatic_checkpoint_byte_budget,
    automatic_checkpoints_enabled: facts.configuration.checkpoints.enabled,
    automatic_checkpoint_minimum_tokens: facts.configuration.checkpoints.minimumTokens,
    automatic_checkpoint_interval_tokens: facts.configuration.checkpoints.intervalTokens,
    automatic_checkpoint_max: facts.configuration.checkpoints.maximum,
    budget_bytes: telemetry.physical_byte_budget,
    high_watermark_bytes: telemetry.high_watermark_bytes,
    low_watermark_bytes: telemetry.low_watermark_bytes,
    under_pressure: telemetry.under_pressure != 0,
    hard_ceiling: retention.hardCeiling, eviction_reason: facts.lastEvictionReason,
    eviction_count: telemetry.evictions)
}
