#pragma once

// Pure multi-GPU split policy parsing (scale plan T4.13). The pinned
// llama.cpp exposes split_mode / main_gpu / tensor_split on
// llama_model_params; this header only translates operator configuration
// into those inputs so the policy is testable without GPUs.

#include "llama.h"

#include <cstddef>
#include <string>
#include <vector>

namespace clap::llama_gpu {

struct SplitDecision {
  bool valid = true;
  llama_split_mode mode = LLAMA_SPLIT_MODE_LAYER;
  int32_t main_gpu = 0;
  // Proportions per device; empty means llama.cpp's automatic distribution.
  std::vector<float> tensor_split;
  std::string error;
};

inline bool parse_split_mode(const std::string& raw, llama_split_mode* out) {
  if (raw.empty() || raw == "layer") {
    *out = LLAMA_SPLIT_MODE_LAYER;
    return true;
  }
  if (raw == "none") {
    *out = LLAMA_SPLIT_MODE_NONE;
    return true;
  }
  if (raw == "row") {
    *out = LLAMA_SPLIT_MODE_ROW;
    return true;
  }
  return false;
}

// Parses a comma-separated proportion list ("3,1" offloads 3/4 to device 0).
// Rejects malformed entries, negatives, all-zero lists, and lists longer
// than the device limit so a typo never silently changes placement.
inline bool parse_tensor_split(const std::string& raw, std::size_t max_devices,
                               std::vector<float>* out) {
  out->clear();
  if (raw.empty()) return true;
  std::size_t start = 0;
  bool any_positive = false;
  while (start <= raw.size()) {
    const std::size_t comma = raw.find(',', start);
    const std::string item = raw.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    std::size_t consumed = 0;
    float value = 0.0f;
    try {
      value = std::stof(item, &consumed);
    } catch (...) {
      return false;
    }
    if (consumed != item.size() || value < 0.0f) return false;
    if (value > 0.0f) any_positive = true;
    out->push_back(value);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (!any_positive || out->size() > max_devices) return false;
  return true;
}

inline SplitDecision derive_split(const std::string& mode_raw,
                                  int32_t main_gpu,
                                  const std::string& tensor_split_raw,
                                  std::size_t max_devices) {
  SplitDecision decision;
  if (!parse_split_mode(mode_raw, &decision.mode)) {
    decision.valid = false;
    decision.error = "unknown split mode '" + mode_raw + "' (expected none|layer|row)";
    return decision;
  }
  if (main_gpu < 0 ||
      (max_devices > 0 && static_cast<std::size_t>(main_gpu) >= max_devices)) {
    decision.valid = false;
    decision.error = "main_gpu " + std::to_string(main_gpu) +
        " out of range for " + std::to_string(max_devices) + " device(s)";
    return decision;
  }
  decision.main_gpu = main_gpu;
  if (!parse_tensor_split(tensor_split_raw, max_devices, &decision.tensor_split)) {
    decision.valid = false;
    decision.error = "invalid tensor_split '" + tensor_split_raw +
        "' (expected up to " + std::to_string(max_devices) +
        " comma-separated non-negative proportions with at least one > 0)";
    return decision;
  }
  return decision;
}

}  // namespace clap::llama_gpu
