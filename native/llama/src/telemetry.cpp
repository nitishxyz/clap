#include "clap/llama/telemetry.h"

namespace clap::llama {

nlohmann::json serialize_retention_telemetry(const RetentionTelemetrySnapshot& snapshot) {
  using json = nlohmann::json;
  return json{
    {"max_active", snapshot.max_active},
    {"queued", snapshot.queued},
    {"previous_max_active", snapshot.previous_max_active > 0
        ? json(snapshot.previous_max_active) : json(nullptr)},
    {"last_adjustment_reason", snapshot.last_adjustment_reason.empty()
        ? json(nullptr) : json(snapshot.last_adjustment_reason)},
    {"last_adjustment_at", snapshot.last_adjustment_at.empty()
        ? json(nullptr) : json(snapshot.last_adjustment_at)},
    {"retained_growth_reserve_bytes", snapshot.retained_growth_reserve_bytes},
    {"global_resident_memory_bytes", snapshot.global_resident_memory_bytes > 0
        ? json(snapshot.global_resident_memory_bytes) : json(nullptr)},
    {"pressure_state", snapshot.pressure_state.empty()
        ? json(nullptr) : json(snapshot.pressure_state)},
    {"active_policy", {
      {"mode", snapshot.active_policy.mode},
      {"selected_max", snapshot.active_policy.selected_max},
      {"backend_ceiling", snapshot.active_policy.backend_ceiling},
      {"hardware_ceiling", snapshot.active_policy.hardware_ceiling},
      {"model_ceiling", snapshot.active_policy.model_ceiling},
      {"memory_ceiling", snapshot.active_policy.memory_ceiling},
      {"reason", snapshot.active_policy.reason},
      {"inputs", {
        {"startup_available_bytes", snapshot.active_policy.startup_available_bytes > 0
            ? json(snapshot.active_policy.startup_available_bytes) : json(nullptr)},
        {"model_file_bytes", snapshot.active_policy.model_file_bytes > 0
            ? json(snapshot.active_policy.model_file_bytes) : json(nullptr)},
        {"context_capacity", snapshot.active_policy.context_capacity},
        {"context_ceiling", snapshot.active_policy.context_ceiling},
        {"per_active_reserve_cells", snapshot.active_policy.per_active_reserve_cells},
        {"per_active_reserve_bytes", snapshot.active_policy.per_active_reserve_bytes},
        {"processor_count", snapshot.active_policy.processor_count},
        {"hybrid_or_recurrent", snapshot.active_policy.hybrid_or_recurrent},
      }},
    }},
    {"active", snapshot.active},
    {"retained_total", snapshot.retained_total},
    {"retained_sessions", snapshot.retained_sessions},
    {"retained_anchors", snapshot.retained_anchors},
    // Byte pressure is intentionally disabled: pinned llama.cpp has no public
    // authoritative KV byte or used/free-cell telemetry.
    {"retained_bytes", 0}, {"session_bytes", 0}, {"anchor_bytes", 0},
    {"budget_bytes", 0}, {"high_watermark_bytes", 0}, {"low_watermark_bytes", 0},
    {"under_pressure", false},
    {"hard_ceiling", snapshot.hard_ceiling},
    {"eviction_reason", snapshot.eviction_reason.empty()
        ? json(nullptr) : json(snapshot.eviction_reason)},
    {"eviction_count", snapshot.eviction_count},
    {"physical_cell_capacity", snapshot.physical_cell_capacity},
    {"physical_cells_used", nullptr},
    {"physical_cells_free", nullptr},
  };
}

}  // namespace clap::llama
