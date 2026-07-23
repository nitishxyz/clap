#include "clap/llama/telemetry.h"

#include <cassert>
#include <fstream>
#include <set>
#include <string>

namespace {

clap::llama::RetentionTelemetrySnapshot snapshot(bool nullable) {
  return {
    2,
    1,
    nullable ? 0 : 1,
    nullable ? "" : "pressure_warning",
    nullable ? "" : "2026-07-20T11:29:00.000Z",
    67108864,
    nullable ? UINT64_C(0) : UINT64_C(1073741824),
    nullable ? "" : "warning",
    {
      "auto", 2, 32, 8, 32, 2, "memory_ceiling",
      nullable ? UINT64_C(0) : UINT64_C(8589934592),
      nullable ? UINT64_C(0) : UINT64_C(4294967296),
      nullable ? 0 : 32768, 8, 4096, 536870912, 8, false,
    },
    1,
    2,
    1,
    1,
    64,
    nullable ? "" : "high_watermark",
    3,
    32768,
  };
}

std::set<std::string> keys(const nlohmann::json& value) {
  std::set<std::string> result;
  for (auto it = value.begin(); it != value.end(); ++it) result.insert(it.key());
  return result;
}

nlohmann::json fixture_retention() {
  std::ifstream fixture(CLAP_LLAMA_TELEMETRY_FIXTURE);
  assert(fixture);
  std::string line;
  while (std::getline(fixture, line)) {
    const auto event = nlohmann::json::parse(line);
    if (event.contains("retention")) return event["retention"];
  }
  assert(false);
  return {};
}

}  // namespace

int main() {
  const auto normal = clap::llama::serialize_retention_telemetry(snapshot(false));
  assert(normal["max_active"] == 2);
  assert(normal["queued"] == 1);
  assert(normal["previous_max_active"] == 1);
  assert(normal["last_adjustment_reason"] == "pressure_warning");
  assert(normal["global_resident_memory_bytes"] == 1073741824);
  assert(normal["pressure_state"] == "warning");
  assert(normal["active_policy"]["inputs"]["startup_available_bytes"] == 8589934592);
  assert(normal["active_policy"]["inputs"]["model_file_bytes"] == 4294967296);
  assert(normal["active_policy"]["inputs"]["hybrid_or_recurrent"] == false);
  assert(normal["retained_total"] == 2);
  assert(normal["retained_bytes"].is_null());
  assert(normal["retained_bytes_source"] == "unavailable");
  assert(normal["retained_bytes_basis"] == "not_observed");
  assert(normal["session_bytes"].is_null());
  assert(normal["session_bytes_source"] == "unavailable");
  assert(normal["anchor_bytes"].is_null());
  assert(normal["anchor_bytes_source"] == "unavailable");
  assert(normal["evicted_bytes"].is_null());
  assert(normal["evicted_bytes_source"] == "unavailable");
  assert(normal["evicted_bytes_basis"] == "not_observed");
  assert(normal["estimated_retained_bytes"] == 134217728);
  assert(normal["estimated_retained_bytes_source"] == "estimated");
  assert(normal["estimated_retained_bytes_basis"] == "context_configuration");
  assert(normal["budget_bytes"] == 0);
  assert(normal["high_watermark_bytes"] == 0);
  assert(normal["low_watermark_bytes"] == 0);
  assert(normal["under_pressure"] == false);
  assert(normal["eviction_reason"] == "high_watermark");
  assert(normal["physical_cells_used"].is_null());
  assert(normal["physical_cells_free"].is_null());

  const auto nullable = clap::llama::serialize_retention_telemetry(snapshot(true));
  assert(nullable["previous_max_active"].is_null());
  assert(nullable["last_adjustment_reason"].is_null());
  assert(nullable["last_adjustment_at"].is_null());
  assert(nullable["global_resident_memory_bytes"].is_null());
  assert(nullable["pressure_state"].is_null());
  assert(nullable["active_policy"]["inputs"]["startup_available_bytes"].is_null());
  assert(nullable["active_policy"]["inputs"]["model_file_bytes"].is_null());
  assert(nullable["eviction_reason"].is_null());
  assert(nullable["retained_bytes"].is_null());
  assert(nullable["estimated_retained_bytes"].is_null());
  assert(nullable["estimated_retained_bytes_source"] == "unavailable");
  assert(nullable["estimated_retained_bytes_basis"] == "not_observed");

  const auto fixture_keys = keys(fixture_retention());
  const auto normal_keys = keys(normal);
  for (const auto& key : fixture_keys) assert(normal_keys.count(key) == 1);
}
