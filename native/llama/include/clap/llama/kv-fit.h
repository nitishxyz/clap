#pragma once

// Pure KV allocation-fit policy (scale plan T2.6). Estimates the unified KV
// pool cost from model metadata and shrinks the automatic context allocation
// until the estimate fits measured startup availability. Estimates are
// scheduling inputs only and are never reported as measured KV usage.

#include <algorithm>
#include <cstdint>
#include <string>

namespace clap::llama_kv {

struct FitInputs {
  // <= 0 when the operator pinned the context explicitly (fit is skipped).
  int32_t requested_context = 0;
  int32_t train_context = 0;
  // Estimated bytes of K+V state for one token across all layers; 0 when
  // model metadata was unavailable (fit is skipped).
  uint64_t kv_bytes_per_token = 0;
  // Measured startup availability; 0 means unavailable (fit fails closed to
  // the conservative floor rather than promising the full training window).
  uint64_t available_bytes = 0;
  // Weights are about to be resident too; both reduce the KV budget.
  uint64_t model_file_bytes = 0;
  uint64_t headroom_bytes = UINT64_C(1) << 30;
};

struct FitDecision {
  int32_t context_cells = 0;
  bool clamped = false;
  std::string reason;
  uint64_t estimated_pool_bytes = 0;
  uint64_t budget_bytes = 0;
};

inline uint64_t kv_type_half_bytes(const std::string& kv_type) {
  // Bytes per element, doubled to stay integral: f16=2.0, q8_0~1.0625->2.125,
  // q4_0~0.5625->1.125. Round up to conservative integer halves.
  if (kv_type == "q8_0") return 3;  // 1.5 bytes/element upper bound
  if (kv_type == "q4_0") return 2;  // 1.0 bytes/element upper bound
  return 4;                         // f16: 2 bytes/element
}

// Estimated per-token bytes for K and V across all layers.
inline uint64_t estimate_kv_bytes_per_token(int32_t n_layer, int32_t n_head_kv,
                                            int32_t head_dim,
                                            const std::string& kv_type) {
  if (n_layer <= 0 || n_head_kv <= 0 || head_dim <= 0) return 0;
  const uint64_t per_layer_elements =
      UINT64_C(2) * static_cast<uint64_t>(n_head_kv) * static_cast<uint64_t>(head_dim);
  return static_cast<uint64_t>(n_layer) * per_layer_elements *
      kv_type_half_bytes(kv_type) / 2;
}

constexpr int32_t kFitFloorCells = 8192;

inline FitDecision fit_context(const FitInputs& input) {
  FitDecision decision;
  decision.context_cells = input.train_context;
  if (input.requested_context > 0) {
    // Operator pinned the context; the caller already applied train clamping.
    decision.context_cells = input.requested_context;
    decision.reason = "explicit_context";
    return decision;
  }
  if (input.train_context <= 0) {
    decision.reason = "unknown_train_context";
    return decision;
  }
  if (input.kv_bytes_per_token == 0) {
    decision.reason = "kv_estimate_unavailable";
    return decision;
  }
  const int32_t floor_cells = std::min(input.train_context, kFitFloorCells);
  if (input.available_bytes == 0) {
    // Fail closed: without a memory observation, do not promise the full
    // training window to the unified pool.
    decision.context_cells = floor_cells;
    decision.clamped = decision.context_cells < input.train_context;
    decision.reason = "memory_unavailable_floor";
    return decision;
  }
  const uint64_t reserved = input.model_file_bytes + input.headroom_bytes;
  decision.budget_bytes = input.available_bytes > reserved
      ? input.available_bytes - reserved : 0;
  int32_t cells = input.train_context;
  while (cells > floor_cells &&
         static_cast<uint64_t>(cells) * input.kv_bytes_per_token > decision.budget_bytes) {
    cells = std::max(floor_cells, cells / 2);
  }
  decision.context_cells = cells;
  decision.clamped = cells < input.train_context;
  decision.estimated_pool_bytes = static_cast<uint64_t>(cells) * input.kv_bytes_per_token;
  decision.reason = decision.clamped ? "kv_budget_fit" : "train_context_fits";
  return decision;
}

}  // namespace clap::llama_kv
