#include "clap/llama/telemetry.h"

#include <limits>

namespace clap::llama {

namespace {

uint64_t saturating_multiply(uint64_t left, uint64_t right) {
  if (left == 0 || right == 0) return 0;
  if (left > std::numeric_limits<uint64_t>::max() / right) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left * right;
}

}  // namespace

uint64_t estimate_configured_retained_bytes(const RetentionTelemetrySnapshot& snapshot) {
  if (snapshot.retained_total == 0) return 0;
  const auto& policy = snapshot.active_policy;
  if (snapshot.hard_ceiling <= 0 || policy.context_capacity <= 0 ||
      policy.per_active_reserve_cells <= 0 || policy.per_active_reserve_bytes == 0) return 0;
  // llama.cpp does not expose used KV bytes. This is a configuration estimate:
  // divide configured context cells over retained slots and apply the explicit
  // per-cell scheduling reserve. It must never be presented as a measurement.
  const uint64_t cells_per_slot =
      (static_cast<uint64_t>(policy.context_capacity) + snapshot.hard_ceiling - 1) /
      static_cast<uint64_t>(snapshot.hard_ceiling);
  const uint64_t bytes_per_cell =
      (policy.per_active_reserve_bytes + policy.per_active_reserve_cells - 1) /
      static_cast<uint64_t>(policy.per_active_reserve_cells);
  return saturating_multiply(snapshot.retained_total,
      saturating_multiply(cells_per_slot, bytes_per_cell));
}

nlohmann::json serialize_retention_telemetry(const RetentionTelemetrySnapshot& snapshot) {
  using json = nlohmann::json;
  const uint64_t estimated_retained_bytes = estimate_configured_retained_bytes(snapshot);
  const bool estimate_available = snapshot.hard_ceiling > 0 &&
      snapshot.active_policy.context_capacity > 0 &&
      snapshot.active_policy.per_active_reserve_cells > 0 &&
      snapshot.active_policy.per_active_reserve_bytes > 0;
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
    // authoritative KV byte or used/free-cell telemetry. Preserve the legacy
    // fields, but represent unavailable observations honestly.
    {"retained_bytes", nullptr},
    {"retained_bytes_source", "unavailable"},
    {"retained_bytes_basis", "not_observed"},
    {"session_bytes", nullptr},
    {"session_bytes_source", "unavailable"},
    {"session_bytes_basis", "not_observed"},
    {"anchor_bytes", nullptr},
    {"anchor_bytes_source", "unavailable"},
    {"anchor_bytes_basis", "not_observed"},
    {"evicted_bytes", nullptr},
    {"evicted_bytes_source", "unavailable"},
    {"evicted_bytes_basis", "not_observed"},
    {"estimated_retained_bytes", estimate_available
        ? json(estimated_retained_bytes) : json(nullptr)},
    {"estimated_retained_bytes_source", estimate_available
        ? json("estimated") : json("unavailable")},
    {"estimated_retained_bytes_basis", estimate_available
        ? json("context_configuration") : json("not_observed")},
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
