#include "clap/llama/model-runtime.h"
#include "clap/llama/request-state.h"
#include "clap/llama/sampling.h"
#include "clap/llama/structured-output.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
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

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
  const int count = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                    nullptr, 0, false, true);
  assert(count > 0);
  std::vector<llama_token> tokens(static_cast<std::size_t>(count));
  assert(llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                        tokens.data(), static_cast<int32_t>(tokens.size()), false, true) >= 0);
  return tokens;
}

bool permitted(llama_sampler* sampler, llama_token token) {
  llama_token_data data[] = {{token, 0.0f, 0.0f}};
  llama_token_data_array candidates{data, 1, -1, false};
  llama_sampler_apply(sampler, &candidates);
  return !std::isinf(candidates.data[0].logit) || candidates.data[0].logit > 0.0f;
}

void assert_accepts_sequence(llama_sampler* sampler, const std::vector<llama_token>& tokens) {
  for (llama_token token : tokens) {
    assert(permitted(sampler, token));
    llama_sampler_accept(sampler, token);
  }
}

clap::llama::StructuredOutputConstraint json_constraint() {
  const auto request = nlohmann::json::object({
      {"structured_output", nlohmann::json::object({
          {"kind", "json_schema"},
          {"strength", "required"},
          {"schema", nlohmann::json::object({
              {"type", "object"},
              {"additionalProperties", false},
              {"required", nlohmann::json::array({"ok"})},
              {"properties", nlohmann::json::object({
                  {"ok", nlohmann::json::object({{"type", "boolean"}})},
              })},
          })},
      })},
  });
  auto parsed = clap::llama::parse_structured_output(request);
  assert(parsed);
  assert(!parsed->grammar.empty());
  return *parsed;
}

}  // namespace

int main() {
  const char* model = std::getenv("CLAP_TEST_GGUF_MODEL");
  if (!model || !*model) {
    std::cout << "SKIP structured sampling: CLAP_TEST_GGUF_MODEL is not set\n";
    return 0;
  }

  clap::llama::ModelRuntime runtime;
  assert(runtime.load(model));
  const llama_vocab* vocab = runtime.vocab();
  assert(vocab != nullptr);

  auto constraint = json_constraint();
  clap::llama::SamplingParams greedy;
  greedy.temperature = 0.0;
  greedy.presence_penalty = 0.25;
  greedy.frequency_penalty = 0.5;

  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> ordered(
      clap::llama::make_sampler(greedy, &constraint, vocab), llama_sampler_free);
  assert((sampler_names(ordered.get()) ==
          std::vector<std::string>{"grammar", "penalties", "greedy"}));

  clap::llama::SamplingParams stochastic;
  stochastic.temperature = 0.7;
  stochastic.top_k = 4;
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> distributed(
      clap::llama::make_sampler(stochastic, &constraint, vocab), llama_sampler_free);
  assert((sampler_names(distributed.get()) ==
          std::vector<std::string>{"grammar", "top-k", "top-p", "temp", "dist"}));

  const auto invalid = tokenize(vocab, "x");
  const auto valid_json = tokenize(vocab, "{\"ok\":true}");
  assert(!invalid.empty());
  assert(!valid_json.empty());

  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> rejects(
      clap::llama::make_sampler(stochastic, &constraint, vocab), llama_sampler_free);
  assert(!permitted(rejects.get(), invalid.front()));
  assert(permitted(rejects.get(), valid_json.front()));

  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sequence(
      clap::llama::make_sampler(stochastic, &constraint, vocab), llama_sampler_free);
  assert_accepts_sequence(sequence.get(), valid_json);
  assert(!permitted(sequence.get(), invalid.front()));

  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> first(
      clap::llama::make_sampler(stochastic, &constraint, vocab), llama_sampler_free);
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> second(
      clap::llama::make_sampler(stochastic, &constraint, vocab), llama_sampler_free);
  assert(permitted(first.get(), valid_json.front()));
  llama_sampler_accept(first.get(), valid_json.front());
  assert(!permitted(first.get(), valid_json.front()));
  assert(permitted(second.get(), valid_json.front()));

  {
    clap::llama::PreparedRequest prepared;
    prepared.id = "cancelled-grammar";
    prepared.params = stochastic;
    prepared.structured_output = constraint;
    clap::llama::ActiveRequest active(std::move(prepared));
    active.sampler.reset(clap::llama::make_sampler(active.params,
        active.structured_output ? &*active.structured_output : nullptr, vocab));
    active.cancelled = true;
    active.finish_reason = "cancel";
    auto completion = active.complete(false, {});
    assert(completion);
    assert(completion->cancelled);
  }
}
