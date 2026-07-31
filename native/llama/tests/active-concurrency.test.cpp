#include "active-concurrency.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
  using clap::llama_active::Inputs;
  using clap::llama_active::select;
  constexpr uint64_t gib = UINT64_C(1024) * 1024 * 1024;
  constexpr uint64_t mib = UINT64_C(1024) * 1024;

  {
    // Legacy inputs are unchanged: fixed 4k-cell reserve and the
    // model-file-size memory heuristic.
    Inputs inputs;
    inputs.startup_available_bytes = 64 * gib;
    inputs.model_file_bytes = 8 * gib;
    inputs.processor_count = 16;
    inputs.context_cells = 16384;
    inputs.retained_ceiling = 32;
    const auto decision = select(inputs);
    assert(decision.mode == "auto");
    assert(decision.per_active_reserve_cells == 4096);
    assert(decision.per_active_reserve_bytes == gib);
    assert(decision.context_ceiling == 4);
    assert(decision.selected_max == 4);
    assert(decision.reason == "context_ceiling");
  }

  {
    // A per-session context cap replaces the fixed per-active cell reserve
    // and therefore widens or narrows the context ceiling honestly.
    Inputs inputs;
    inputs.startup_available_bytes = 64 * gib;
    inputs.model_file_bytes = 8 * gib;
    inputs.processor_count = 16;
    inputs.context_cells = 131072;
    inputs.retained_ceiling = 32;
    inputs.session_context_cells = 16384;
    const auto decision = select(inputs);
    assert(decision.per_active_reserve_cells == 16384);
    assert(decision.context_ceiling == 8);
    assert(decision.selected_max == 8);
  }

  {
    // The session cap never exceeds the allocated pool.
    Inputs inputs;
    inputs.startup_available_bytes = 64 * gib;
    inputs.processor_count = 16;
    inputs.context_cells = 8192;
    inputs.retained_ceiling = 32;
    inputs.session_context_cells = 16384;
    const auto decision = select(inputs);
    assert(decision.per_active_reserve_cells == 8192);
    assert(decision.context_ceiling == 1);
  }

  {
    // KV-derived reserve: per-active bytes are the estimated KV cost of one
    // session at its context reserve; the measured budget reserves weights
    // plus fixed headroom first.
    Inputs inputs;
    inputs.startup_available_bytes = 12 * gib;
    inputs.model_file_bytes = 3 * gib;
    inputs.processor_count = 32;
    inputs.context_cells = 131072;
    inputs.retained_ceiling = 32;
    inputs.kv_bytes_per_token = 131072;  // 128 KiB/token
    inputs.session_context_cells = 16384;
    const auto decision = select(inputs);
    assert(decision.per_active_reserve_bytes == 2 * gib);
    // (12 - 3 - 1) GiB budget / 2 GiB per active session.
    assert(decision.memory_ceiling == 4);
    assert(decision.selected_max == 4);
    assert(decision.reason == "memory_ceiling");
  }

  {
    // An explicit admin budget (e.g. VRAM, net of weights) is authoritative
    // even when startup memory measurement was unavailable.
    Inputs inputs;
    inputs.startup_available_bytes = 0;
    inputs.processor_count = 32;
    inputs.context_cells = 131072;
    inputs.retained_ceiling = 32;
    inputs.kv_bytes_per_token = 131072;
    inputs.session_context_cells = 16384;
    inputs.explicit_budget_bytes = 8 * gib;
    const auto decision = select(inputs);
    assert(decision.memory_ceiling == 4);
    assert(decision.selected_max == 4);
  }

  {
    // Without any budget or measurement, auto mode stays conservative.
    Inputs inputs;
    inputs.processor_count = 32;
    inputs.context_cells = 131072;
    inputs.retained_ceiling = 32;
    inputs.kv_bytes_per_token = 131072;
    const auto decision = select(inputs);
    assert(decision.memory_ceiling == 2);
  }

  {
    // Tiny KV models never assume compute scratch is free: the per-active
    // reserve is floored at 256 MiB.
    Inputs inputs;
    inputs.startup_available_bytes = 64 * gib;
    inputs.processor_count = 32;
    inputs.context_cells = 131072;
    inputs.retained_ceiling = 32;
    inputs.kv_bytes_per_token = 1024;
    inputs.session_context_cells = 4096;
    const auto decision = select(inputs);
    assert(decision.per_active_reserve_bytes == 256 * mib);
  }

  {
    // Explicit operator max_active still clamps to the safe ceiling.
    Inputs inputs;
    inputs.explicit_max = 2;
    inputs.startup_available_bytes = 64 * gib;
    inputs.processor_count = 32;
    inputs.context_cells = 131072;
    inputs.retained_ceiling = 32;
    inputs.kv_bytes_per_token = 131072;
    inputs.session_context_cells = 16384;
    const auto decision = select(inputs);
    assert(decision.mode == "fixed");
    assert(decision.selected_max == 2);
    assert(decision.reason == "explicit_override");
  }

  return 0;
}
