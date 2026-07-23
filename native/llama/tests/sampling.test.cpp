#include "clap/llama/sampling.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

std::vector<std::string> sampler_names(llama_sampler* sampler) {
  std::vector<std::string> names;
  const int count = llama_sampler_chain_n(sampler);
  for (int index = 0; index < count; ++index) {
    names.emplace_back(llama_sampler_name(llama_sampler_chain_get(sampler, index)));
  }
  return names;
}

}  // namespace

int main() {
  const auto defaults = clap::llama::sampling_from_request(nlohmann::json::object());
  assert(defaults.max_tokens == 0);
  assert(defaults.temperature == 0.7);
  assert(defaults.top_p == 0.95);
  assert(defaults.top_k == 0);
  assert(defaults.seed == LLAMA_DEFAULT_SEED);
  assert(defaults.presence_penalty == 0.0);
  assert(defaults.frequency_penalty == 0.0);
  assert(defaults.stops.empty());

  const auto configured = clap::llama::sampling_from_request({
      {"max_tokens", 17},
      {"temperature", 0.4},
      {"top_p", 0.8},
      {"top_k", 23},
      {"seed", -1},
      {"presence_penalty", 0.2},
      {"frequency_penalty", -0.3},
      {"stop", nlohmann::json::array({"one", 4, "two"})},
  });
  assert(configured.max_tokens == 17);
  assert(configured.temperature == 0.4);
  assert(configured.top_p == 0.8);
  assert(configured.top_k == 23);
  assert(configured.seed == static_cast<uint32_t>(-1));
  assert(configured.presence_penalty == 0.2);
  assert(configured.frequency_penalty == -0.3);
  assert((configured.stops == std::vector<std::string>{"one", "two"}));

  const auto single_stop = clap::llama::sampling_from_request({{"stop", "done"}});
  assert((single_stop.stops == std::vector<std::string>{"done"}));
  const auto ignored_stop = clap::llama::sampling_from_request({{"stop", 12}});
  assert(ignored_stop.stops.empty());
  const auto ignored_seed = clap::llama::sampling_from_request({{"seed", 1.5}});
  assert(ignored_seed.seed == LLAMA_DEFAULT_SEED);

  llama_sampler* default_sampler = clap::llama::make_sampler(defaults);
  assert((sampler_names(default_sampler) ==
          std::vector<std::string>{"top-p", "temp", "dist"}));
  llama_sampler_free(default_sampler);

  llama_sampler* default_sampler_unconstrained =
      clap::llama::make_sampler(defaults, nullptr, nullptr);
  assert((sampler_names(default_sampler_unconstrained) ==
          std::vector<std::string>{"top-p", "temp", "dist"}));
  llama_sampler_free(default_sampler_unconstrained);

  llama_sampler* configured_sampler = clap::llama::make_sampler(configured);
  assert((sampler_names(configured_sampler) ==
          std::vector<std::string>{"penalties", "top-k", "top-p", "temp", "dist"}));
  llama_sampler_free(configured_sampler);

  auto greedy = defaults;
  greedy.temperature = 0.0;
  greedy.top_k = 50;
  greedy.presence_penalty = 0.1;
  llama_sampler* greedy_sampler = clap::llama::make_sampler(greedy);
  assert((sampler_names(greedy_sampler) ==
          std::vector<std::string>{"penalties", "greedy"}));
  llama_sampler_free(greedy_sampler);
}
