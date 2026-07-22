#include "clap/llama/cache-executor.h"
#include "clap/llama/generation-stepper.h"
#include "clap/llama/model-runtime.h"
#include "tests/support/cache-model-probe.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using clap::llama::CacheExecutor;
using clap::llama::ModelRuntime;
using clap::llama::test::CacheProbeObservation;
using clap::llama::test::Sha256;
using clap::llama::test::logit_fingerprint;
using clap::llama::test::token_fingerprint;
using clap::llama::test::top_quantized_logits;

void deterministic_helpers() {
  const std::string abc = "abc";
  assert(Sha256::hex(reinterpret_cast<const uint8_t*>(abc.data()), abc.size()) ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert(token_fingerprint({1, 2, -1}) ==
         "0a1ce634879f6a487527c9a185b1a4a3de7f41238ad4139e3c6d9e0da723628c");
  const float logits[] = {1.0f, 2.0f, 2.0f, -1.0f};
  const auto top = top_quantized_logits(logits, 4, 3);
  assert((top == std::vector<clap::llama::test::QuantizedLogit>{
      {1, 2048}, {2, 2048}, {0, 1024}}));
  assert(logit_fingerprint(top) ==
         "d663b2bef022dd313400ebbe1658c8c8f05f83014e5c0c98a0265206e801731c");
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
  const int count = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
  assert(count > 0);
  std::vector<llama_token> tokens(static_cast<std::size_t>(count));
  assert(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(),
                        true, true) >= 0);
  return tokens;
}

CacheProbeObservation run_probe(const std::string& model_path) {
  ModelRuntime runtime;
  assert(runtime.load(model_path));
  const auto prompt = tokenize(runtime.vocab(), "The deterministic next token is");

  clap::llama::CacheExecutorConfig config;
  config.slot_count = 1;
  config.logical_token_capacity = 4096;
  config.max_anchors = 0;
  config.hard_max_retained_entries = 1;
  config.automatic_checkpoints = false;
  CacheExecutor executor(config,
      std::make_unique<clap::llama::LlamaPhysicalCacheBackend>(runtime.context()));
  clap::llama_cache::Identity identity;
  identity.name_space = clap::llama_cache::fingerprint("real-model-probe");
  identity.tenant = clap::llama_cache::hash("test-tenant");
  identity.session = clap::llama_cache::hash("test-session");
  identity.scope = CLAP_CACHE_SCOPE_SESSION;
  const auto admission = executor.admit({prompt, identity, 0, 1,
      CLAP_CACHE_SLOT_SESSION, {}, {}, false, {}});

  llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt.size()), 0, 1);
  batch.n_tokens = static_cast<int32_t>(prompt.size());
  for (int32_t index = 0; index < batch.n_tokens; ++index) {
    batch.token[index] = prompt[static_cast<std::size_t>(index)];
    batch.pos[index] = index;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = static_cast<llama_seq_id>(admission.target_slot);
    batch.logits[index] = index + 1 == batch.n_tokens ? 1 : 0;
  }
  assert(llama_decode(runtime.context(), batch) == 0);
  llama_batch_free(batch);

  const float* logits = llama_get_logits_ith(runtime.context(), -1);
  assert(logits != nullptr);
  const std::size_t vocabulary = static_cast<std::size_t>(llama_vocab_n_tokens(runtime.vocab()));
  const auto top = top_quantized_logits(logits, vocabulary);
  assert(!top.empty());
  const int32_t selected = static_cast<int32_t>(std::distance(logits,
      std::max_element(logits, logits + vocabulary)));
  const uint64_t generation = executor.advance(admission.target_slot,
      admission.target_generation, prompt.data(), prompt.size(),
      CLAP_CACHE_SLOT_SESSION, false);

  std::vector<uint8_t> physical;
  std::string descriptor = runtime.cache_domain();
  descriptor.push_back('\0');
  descriptor += runtime.kv_format();
  physical.insert(physical.end(), descriptor.begin(), descriptor.end());
  for (llama_token token : prompt) clap::llama::test::append_i32(physical, token);
  return {admission.operation, admission.reuse_tokens, generation,
      token_fingerprint(prompt), Sha256::hex(physical), selected,
      logit_fingerprint(top), top};
}

void print(const CacheProbeObservation& observation) {
  std::cout << "operation=" << observation.operation
            << " reused=" << observation.reused
            << " generation=" << observation.generation
            << " logical_token_sha256=" << observation.logical_token_sha256
            << " physical_state_sha256=" << observation.physical_state_sha256
            << " selected_next_token=" << observation.selected_next_token
            << " top16_quantized_logit_sha256="
            << observation.top16_quantized_logit_sha256 << " top16=[";
  for (std::size_t index = 0; index < observation.top16_quantized_logits.size(); ++index) {
    if (index) std::cout << ',';
    const auto& value = observation.top16_quantized_logits[index];
    std::cout << value.token << ':' << value.value;
  }
  std::cout << "]\n";
}

}  // namespace

int main() {
  deterministic_helpers();
  const char* model = std::getenv("CLAP_TEST_GGUF_MODEL");
  if (!model || !*model) {
    std::cout << "SKIP real-model probe: CLAP_TEST_GGUF_MODEL is not set\n";
    return 0;
  }
  print(run_probe(model));
}
