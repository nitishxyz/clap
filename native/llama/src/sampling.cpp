#include "clap/llama/sampling.h"

namespace clap::llama {

SamplingParams sampling_from_request(const nlohmann::json& request) {
  SamplingParams params;
  params.max_tokens = request.value("max_tokens", params.max_tokens);
  params.temperature = request.value("temperature", params.temperature);
  params.top_p = request.value("top_p", params.top_p);
  params.top_k = request.value("top_k", params.top_k);
  params.presence_penalty = request.value("presence_penalty", params.presence_penalty);
  params.frequency_penalty = request.value("frequency_penalty", params.frequency_penalty);
  if (request.contains("seed") && request["seed"].is_number_integer()) {
    params.seed = static_cast<uint32_t>(request["seed"].get<int64_t>());
  }
  if (request.contains("stop")) {
    const auto& stop = request["stop"];
    if (stop.is_string()) {
      params.stops.push_back(stop.get<std::string>());
    } else if (stop.is_array()) {
      for (const auto& value : stop) {
        if (value.is_string()) params.stops.push_back(value.get<std::string>());
      }
    }
  }
  return params;
}

llama_sampler* make_sampler(const SamplingParams& params) {
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  llama_sampler* sampler = llama_sampler_chain_init(sparams);
  if (params.presence_penalty != 0.0 || params.frequency_penalty != 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
      64,
      1.0f,
      static_cast<float>(params.frequency_penalty),
      static_cast<float>(params.presence_penalty)));
  }
  if (params.temperature <= 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    if (params.top_k > 0) llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(params.top_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(params.temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));
  }
  return sampler;
}

}  // namespace clap::llama
