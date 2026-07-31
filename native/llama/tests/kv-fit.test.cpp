#include "clap/llama/kv-fit.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
  using clap::llama_kv::estimate_kv_bytes_per_token;
  using clap::llama_kv::fit_context;
  using clap::llama_kv::FitInputs;

  // Per-token estimate: layers * 2 (K+V) * heads_kv * head_dim * bytes/el.
  // 32 layers, 8 KV heads, 128 head dim, f16 -> 32*2*8*128*2 = 131072.
  assert(estimate_kv_bytes_per_token(32, 8, 128, "f16") == 131072);
  assert(estimate_kv_bytes_per_token(32, 8, 128, "q4_0") == 65536);
  // q8_0 uses a conservative 1.5-byte upper bound.
  assert(estimate_kv_bytes_per_token(32, 8, 128, "q8_0") == 98304);
  // Missing metadata yields no estimate.
  assert(estimate_kv_bytes_per_token(0, 8, 128, "f16") == 0);
  assert(estimate_kv_bytes_per_token(32, 0, 128, "f16") == 0);

  {
    // Explicit operator context is never adjusted.
    FitInputs inputs;
    inputs.requested_context = 16384;
    inputs.train_context = 262144;
    inputs.kv_bytes_per_token = 131072;
    inputs.available_bytes = 1ull << 30;
    const auto decision = fit_context(inputs);
    assert(decision.context_cells == 16384);
    assert(!decision.clamped);
    assert(decision.reason == "explicit_context");
  }

  {
    // Training window fits the budget: keep it.
    FitInputs inputs;
    inputs.train_context = 32768;
    inputs.kv_bytes_per_token = 131072;         // 32k cells -> 4 GiB pool
    inputs.available_bytes = 16ull << 30;       // 16 GiB available
    inputs.model_file_bytes = 4ull << 30;       // 4 GiB weights
    const auto decision = fit_context(inputs);  // budget = 16-4-1 = 11 GiB
    assert(decision.context_cells == 32768);
    assert(!decision.clamped);
    assert(decision.reason == "train_context_fits");
  }

  {
    // Oversized training window halves until the estimated pool fits.
    FitInputs inputs;
    inputs.train_context = 262144;
    inputs.kv_bytes_per_token = 131072;         // 262k cells -> 32 GiB pool
    inputs.available_bytes = 24ull << 30;       // 24 GiB available
    inputs.model_file_bytes = 8ull << 30;       // budget = 24-8-1 = 15 GiB
    const auto decision = fit_context(inputs);
    // 262144 -> 131072 (16 GiB, over) -> 65536 (8 GiB, fits)
    assert(decision.context_cells == 65536);
    assert(decision.clamped);
    assert(decision.reason == "kv_budget_fit");
    assert(decision.estimated_pool_bytes == 65536ull * 131072ull);
  }

  {
    // Unavailable memory fails closed to the floor, never the full window.
    FitInputs inputs;
    inputs.train_context = 262144;
    inputs.kv_bytes_per_token = 131072;
    inputs.available_bytes = 0;
    const auto decision = fit_context(inputs);
    assert(decision.context_cells == clap::llama_kv::kFitFloorCells);
    assert(decision.clamped);
    assert(decision.reason == "memory_unavailable_floor");
  }

  {
    // Small training windows never shrink below themselves.
    FitInputs inputs;
    inputs.train_context = 4096;
    inputs.kv_bytes_per_token = 131072;
    inputs.available_bytes = 0;
    const auto decision = fit_context(inputs);
    assert(decision.context_cells == 4096);
    assert(!decision.clamped);
  }

  {
    // Missing KV estimate skips fitting entirely (provenance-honest: no
    // invented budget without an estimate basis).
    FitInputs inputs;
    inputs.train_context = 262144;
    inputs.kv_bytes_per_token = 0;
    inputs.available_bytes = 1ull << 30;
    const auto decision = fit_context(inputs);
    assert(decision.context_cells == 262144);
    assert(!decision.clamped);
    assert(decision.reason == "kv_estimate_unavailable");
  }

  {
    // Budget smaller than everything floors at min(train, floor).
    FitInputs inputs;
    inputs.train_context = 262144;
    inputs.kv_bytes_per_token = 131072;
    inputs.available_bytes = 2ull << 30;
    inputs.model_file_bytes = 4ull << 30;  // reserved exceeds available
    const auto decision = fit_context(inputs);
    assert(decision.context_cells == clap::llama_kv::kFitFloorCells);
    assert(decision.clamped);
    assert(decision.budget_bytes == 0);
  }
}
