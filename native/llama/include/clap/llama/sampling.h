#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace clap::llama {

struct SamplingParams {
  // Zero means omitted by the caller; admission derives a safe request-local
  // default from the loaded model's remaining effective context.
  int max_tokens = 0;
  double temperature = 0.7;
  double top_p = 0.95;
  int top_k = 0;
  uint32_t seed = LLAMA_DEFAULT_SEED;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::vector<std::string> stops;
};

SamplingParams sampling_from_request(const nlohmann::json& request);
llama_sampler* make_sampler(const SamplingParams& params);

}  // namespace clap::llama
