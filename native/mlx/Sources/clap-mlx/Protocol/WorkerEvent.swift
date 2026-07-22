import Foundation

struct V1ProtocolError: Encodable {
  let code: String
  let message: String
  let retryable: Bool
  let fatal: Bool
}

struct WorkerUsage: Encodable {
  let prompt_tokens: Int
  let completion_tokens: Int
}

struct WorkerCacheCandidate: Encodable {
  let slot: Int
  let generation: UInt64
  let state: String
  let shared_prefix_tokens: Int
  let namespace_compatible: Bool
  let model_compatible: Bool
  let session_compatible: Bool
  let generation_compatible: Bool
  let busy_eligible: Bool
  let lease_eligible: Bool
  let materialized: Bool
  let trim_eligible: Bool
  let copy_eligible: Bool
  let eligible: Bool
  let selected: Bool
  let rejection: String?
}

struct WorkerCacheBoundary: Encodable {
  let token_hash: String?
  let token_count: Int?
  let kind: String
  let label: String?
  let requested: Bool
  let status: String
  let skip_reason: String?
  let materialized: Bool?
}

struct WorkerCache: Encodable {
  let hit: Bool
  let reused_tokens: Int
  let reuse_kind: String?
  let reuse_scope: String?
  let side_request: Bool
  let namespace: String?
  let donor_slot: Int?
  let target_slot: Int
  let evicted_slots: [Int]
  let decision_us: UInt64
  let planned_reuse_tokens: Int
  let realized_reuse_tokens: Int
  let fallback: String?
  let miss_reason: String?
  let candidates: [WorkerCacheCandidate]
  let prompt_token_hash: String
  let prompt_token_count: Int
  let stable_boundary_token_hash: String?
  let stable_boundary_token_count: Int
  let stable_boundary_kind: String?
  let automatic_checkpoint_proposed: Int
  let automatic_checkpoint_authorized: Int
  let automatic_checkpoint_materialized: Int
  let automatic_checkpoint_deduped: Int
  let automatic_checkpoint_skipped: Int
  let stable_boundaries: [WorkerCacheBoundary]
}

struct WorkerPrefill: Encodable {
  let done: Int
  let total: Int
}

struct WorkerTiming: Encodable {
  let received_to_admitted_ms: Double
  let template_tokenize_ms: Double
  let coordinator_wait_ms: Double
  let coordinator_plan_ms: Double
  let coordinator_apply_ms: Double
  let scheduler_wait_ms: Double
  let cache_materialize_ms: Double
  let prefill_ms: Double
  let residual_prefill_tokens: Int
  let prefill_tokens: Int
  let prefill_chunks: Int
  let first_decode_ms: Double
  let first_emit_ms: Double
  let normal_prefill_quantum: Int
  let contended_prefill_quantum: Int
}

struct WorkerMemory: Encodable {
  let active_bytes: Int
  let cache_bytes: Int
  let peak_active_bytes: Int
}

struct WorkerActivePolicyInputs: Encodable {
  let physical_memory_bytes: UInt64
  let startup_available_bytes: UInt64?
  let model_active_bytes: UInt64?
  let retained_budget_bytes: UInt64
  let retained_bytes: UInt64
  let retained_growth_reserve_bytes: UInt64
  let os_reserve_bytes: UInt64
  let usable_runtime_bytes: UInt64
  let per_active_reserve_bytes: UInt64
  let processor_count: Int
  let hybrid_or_recurrent: Bool
}

struct WorkerActivePolicy: Encodable {
  let mode: String
  let selected_max: Int
  let backend_ceiling: Int
  let hardware_ceiling: Int
  let model_ceiling: Int
  let memory_ceiling: Int
  let reason: String
  let inputs: WorkerActivePolicyInputs
}

struct WorkerRetention: Encodable {
  let max_active: Int
  let queued: Int
  let previous_max_active: Int?
  let last_adjustment_reason: String?
  let last_adjustment_at: String?
  let retained_growth_reserve_bytes: UInt64
  let global_resident_memory_bytes: UInt64?
  let pressure_state: String?
  let active_policy: WorkerActivePolicy
  let active: Int
  let retained_total: Int
  let retained_sessions: Int
  let retained_anchors: Int
  let retained_bytes: UInt64
  let session_bytes: UInt64
  let anchor_bytes: UInt64
  let automatic_checkpoint_count: Int
  let automatic_checkpoint_bytes: UInt64
  let automatic_checkpoint_budget_bytes: UInt64
  let automatic_checkpoints_enabled: Bool
  let automatic_checkpoint_minimum_tokens: Int
  let automatic_checkpoint_interval_tokens: Int
  let automatic_checkpoint_max: Int
  let budget_bytes: UInt64
  let high_watermark_bytes: UInt64
  let low_watermark_bytes: UInt64
  let under_pressure: Bool
  let hard_ceiling: Int
  let eviction_reason: String?
  let eviction_count: UInt64
}

struct WorkerTokenCapabilities: Encodable {
  enum CodingKeys: String, CodingKey {
    case model_context_window
    case effective_context_window
    case max_input_tokens
    case max_output_tokens
    case model_context_window_source
    case max_output_tokens_source
    case backend_allocation_cap
    case user_configured_override
  }

  let model_context_window: Int?
  let effective_context_window: Int?
  let max_input_tokens: Int?
  let max_output_tokens: Int?
  let model_context_window_source: String?
  let max_output_tokens_source: String?
  let backend_allocation_cap: Int?
  let user_configured_override: Int?

  func encode(to encoder: any Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(model_context_window, forKey: .model_context_window)
    try container.encode(effective_context_window, forKey: .effective_context_window)
    try container.encode(max_input_tokens, forKey: .max_input_tokens)
    try container.encode(max_output_tokens, forKey: .max_output_tokens)
    try container.encode(model_context_window_source, forKey: .model_context_window_source)
    try container.encode(max_output_tokens_source, forKey: .max_output_tokens_source)
    try container.encode(backend_allocation_cap, forKey: .backend_allocation_cap)
    try container.encode(user_configured_override, forKey: .user_configured_override)
  }
}

struct WorkerMessage: Encodable {
  let id: String?
  let started: Bool?
  let token: String?
  let content: String?
  let loaded: Bool?
  let unloaded: Bool?
  let done: Bool?
  let error: String?
  let code: String?
  let cancelled: Bool?
  let finish_reason: String?
  let usage: WorkerUsage?
  let cache: WorkerCache?
  let timing: WorkerTiming?
  let prefill: WorkerPrefill?
  let memory: WorkerMemory?
  let retention: WorkerRetention?
  let token_capabilities: WorkerTokenCapabilities?
}
