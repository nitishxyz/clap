#ifndef CLAP_LLAMA_ACTIVE_CONCURRENCY_H
#define CLAP_LLAMA_ACTIVE_CONCURRENCY_H

#include <algorithm>
#include <cstdint>
#include <string>

namespace clap::llama_active {

struct Inputs {
  int explicit_max = 0;
  uint64_t startup_available_bytes = 0;
  uint64_t model_file_bytes = 0;
  int processor_count = 1;
  int context_cells = 0;
  int retained_ceiling = 1;
  bool hybrid_or_recurrent = false;
  bool encoder = false;
  // Estimated bytes of K+V state for one token across all layers (see
  // kv-fit.h). Zero falls back to the model-file-size heuristic.
  uint64_t kv_bytes_per_token = 0;
  // Per-session context cap (max_session_ctx). Zero keeps the conservative
  // fixed per-active reserve instead of promising the full pool.
  int session_context_cells = 0;
  // Admin-declared memory budget for active-slot derivation (e.g. a VRAM
  // budget on a discrete GPU, net of weights). Zero uses measured startup
  // availability minus weights and headroom.
  uint64_t explicit_budget_bytes = 0;
};

struct Decision {
  std::string mode;
  int selected_max;
  int backend_ceiling;
  int hardware_ceiling;
  int model_ceiling;
  int memory_ceiling;
  int context_ceiling;
  std::string reason;
  uint64_t per_active_reserve_bytes;
  int per_active_reserve_cells;
};

inline Decision select(const Inputs& input) {
  constexpr int backend_ceiling = 32;
  constexpr uint64_t mib = UINT64_C(1024) * 1024;
  const int processors = std::max(1, input.processor_count);
  const int hardware_ceiling = processors <= 4 ? 2 : processors <= 8 ? 4
      : processors <= 16 ? 8 : processors <= 24 ? 16 : backend_ceiling;
  const int model_ceiling = input.hybrid_or_recurrent || input.encoder ? 1 : backend_ceiling;
  // The allocated unified KV cell pool is authoritative. Reserve the
  // per-session context cap per active request when the operator declared
  // one; otherwise reserve a conservative 4k cells so auto mode never
  // promises every request the full pool.
  const int per_active_cells = input.session_context_cells > 0
      ? (input.context_cells > 0
          ? std::min(input.session_context_cells, input.context_cells)
          : input.session_context_cells)
      : 4096;
  const int context_ceiling = std::max(1,
      std::min(backend_ceiling, input.context_cells / per_active_cells));
  // Per-active memory reserve. With model KV metadata the reserve is the
  // estimated KV cost of one active session at its context reserve (floored
  // so compute scratch is never assumed free); without it, model file size
  // is only a conservative scheduling estimate and is never reported as KV
  // use.
  const uint64_t per_active_bytes = input.kv_bytes_per_token > 0
      ? std::max(UINT64_C(256) * mib,
          input.kv_bytes_per_token * static_cast<uint64_t>(per_active_cells))
      : std::max(UINT64_C(512) * mib,
          input.model_file_bytes > 0 ? input.model_file_bytes / 8 : UINT64_C(1024) * mib);
  // An explicit budget is already net of weights (e.g. a VRAM budget); the
  // measured-availability path reserves the weights and fixed headroom first
  // when the reserve is KV-derived, matching kv-fit.h.
  const uint64_t measured_budget = input.kv_bytes_per_token > 0
      ? (input.startup_available_bytes >
             input.model_file_bytes + UINT64_C(1024) * mib
             ? input.startup_available_bytes - input.model_file_bytes -
                 UINT64_C(1024) * mib
             : 0)
      : input.startup_available_bytes;
  const uint64_t budget_bytes = input.explicit_budget_bytes > 0
      ? input.explicit_budget_bytes : measured_budget;
  const int memory_ceiling =
      (input.startup_available_bytes == 0 && input.explicit_budget_bytes == 0) ? 2
      : std::max(1, static_cast<int>(std::min<uint64_t>(
          backend_ceiling, budget_bytes / per_active_bytes)));
  const int safe_ceiling = std::max(1, std::min({backend_ceiling, hardware_ceiling,
      model_ceiling, memory_ceiling, context_ceiling, std::max(1, input.retained_ceiling)}));
  const bool fixed = input.explicit_max > 0;
  const int selected = fixed ? std::min(input.explicit_max, safe_ceiling)
                             : std::min(8, safe_ceiling);
  std::string reason = "bounded_backend_default";
  if (fixed) reason = selected == input.explicit_max ? "explicit_override"
      : "explicit_override_clamped_to_safe_ceiling";
  else if (selected == model_ceiling) reason = "model_ceiling";
  else if (selected == context_ceiling) reason = "context_ceiling";
  else if (selected == memory_ceiling) reason = "memory_ceiling";
  else if (selected == hardware_ceiling) reason = "hardware_ceiling";
  else if (selected == input.retained_ceiling) reason = "retained_ceiling";
  return {fixed ? "fixed" : "auto", selected, backend_ceiling,
      hardware_ceiling, model_ceiling, memory_ceiling, context_ceiling, reason,
      per_active_bytes, per_active_cells};
}

}  // namespace clap::llama_active

#endif
