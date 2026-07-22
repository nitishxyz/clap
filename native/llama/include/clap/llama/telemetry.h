#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace clap::llama {

struct ActivePolicyTelemetrySnapshot {
  const std::string mode;
  const int selected_max;
  const int backend_ceiling;
  const int hardware_ceiling;
  const int model_ceiling;
  const int memory_ceiling;
  const std::string reason;
  const uint64_t startup_available_bytes;
  const uint64_t model_file_bytes;
  const int context_capacity;
  const int context_ceiling;
  const int per_active_reserve_cells;
  const uint64_t per_active_reserve_bytes;
  const unsigned int processor_count;
  const bool hybrid_or_recurrent;
};

struct RetentionTelemetrySnapshot {
  const int max_active;
  const std::size_t queued;
  const int previous_max_active;
  const std::string last_adjustment_reason;
  const std::string last_adjustment_at;
  const uint64_t retained_growth_reserve_bytes;
  const uint64_t global_resident_memory_bytes;
  const std::string pressure_state;
  const ActivePolicyTelemetrySnapshot active_policy;
  const std::size_t active;
  const uint32_t retained_total;
  const uint32_t retained_sessions;
  const uint32_t retained_anchors;
  const int hard_ceiling;
  const std::string eviction_reason;
  const uint64_t eviction_count;
  const int physical_cell_capacity;
};

nlohmann::json serialize_retention_telemetry(const RetentionTelemetrySnapshot& snapshot);

}  // namespace clap::llama
