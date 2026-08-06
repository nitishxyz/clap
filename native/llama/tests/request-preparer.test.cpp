#include "clap/llama/request-preparer.h"

#include "clap/llama/protocol.h"

#include <cassert>
#include <string>

int main() {
  // Hybrid models cannot trim, so boundary snapshots are their only route to
  // prompt reuse across turns; they must be advertised, and they must never
  // come bundled with the trim/partial-branch capabilities.
  assert((clap::llama::cache_capabilities(true, true) & CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT) != 0);
  assert((clap::llama::cache_capabilities(true, true) & CLAP_CACHE_CAP_RECURRENT_OR_HYBRID) != 0);
  assert((clap::llama::cache_capabilities(true, true) & CLAP_CACHE_CAP_WHOLE_STATE_COPY) != 0);
  assert((clap::llama::cache_capabilities(true, true) & CLAP_CACHE_CAP_PARTIAL_SUFFIX_TRIM) == 0);
  assert((clap::llama::cache_capabilities(true, true) & CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH) == 0);
  assert((clap::llama::cache_capabilities(true, false) & CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT) == 0);
  assert((clap::llama::cache_capabilities(false, false) & CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT) == 0);
  assert((clap::llama::cache_capabilities(false, true) & CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT) != 0);
  const auto sequence_adapter = clap::llama::physical_cache_adapter_descriptor(false, true, "f16");
  assert(sequence_adapter["contract_version"] == 1);
#ifdef LLAMA_MEMORY_KV_CACHE_VIEW_API
  assert(sequence_adapter["kind"] == "paged");
  assert(sequence_adapter["format"]["cache_format"] == "llama-kv-cell");
  assert(sequence_adapter["format"]["block_tokens"] == 1);
  assert(sequence_adapter["constraints"]["byte_accounting"] == "exact");
#else
  assert(sequence_adapter["kind"] == "sequence");
  assert(sequence_adapter["format"]["cache_format"] == "llama-sequence");
  assert(sequence_adapter["format"]["block_tokens"].is_null());
  assert(sequence_adapter["constraints"]["byte_accounting"] == "unknown");
#endif
  assert(sequence_adapter["constraints"]["fork_semantics"] == "copy_on_write");
  assert(sequence_adapter["constraints"]["minimum_trim_tokens"] == 1);
  assert(sequence_adapter["operations"] == nlohmann::json({
      "inspect", "continue", "restore", "fork", "release", "trim", "snapshot"}));
  const auto hybrid_adapter = clap::llama::physical_cache_adapter_descriptor(true, false, "");
  assert(hybrid_adapter["format"]["kv_data_type"].is_null());
  assert(hybrid_adapter["constraints"]["minimum_trim_tokens"].is_null());
  assert(hybrid_adapter["constraints"]["fork_semantics"] == "whole_state_copy");
  assert(hybrid_adapter["operations"] == nlohmann::json({
      "inspect", "continue", "restore", "fork", "release"}));
  auto budget = clap::llama::validate_request_budget(10, 100, 0, 20, 0);
  assert(budget.max_tokens == 20);
  assert(budget.output_reserve == 20);

  budget = clap::llama::validate_request_budget(90, 100, 0, 0, 0);
  assert(budget.max_tokens == 10);

  try {
    clap::llama::validate_request_budget(100, 100, 0, 0, 0);
    assert(false);
  } catch (const clap::llama::RequestError& error) {
    assert(error.code == "context_length_exceeded");
    assert(std::string(error.what()) ==
        "prompt is too long for the loaded model; prompt_tokens=100, "
        "max_input_tokens=99, effective_context_window=100.");
  }

  try {
    clap::llama::validate_request_budget(10, 100, 21, 20, 0);
    assert(false);
  } catch (const clap::llama::RequestError& error) {
    assert(error.code == "max_output_tokens_exceeded");
  }

  try {
    clap::llama::validate_request_budget(90, 100, 11, 0, 0);
    assert(false);
  } catch (const clap::llama::RequestError& error) {
    assert(error.code == "context_length_exceeded");
    assert(std::string(error.what()) ==
        "prompt plus requested output exceeds the loaded model context; "
        "prompt_tokens=90, requested_output_tokens=11, effective_context_window=100.");
  }

  try {
    clap::llama::validate_request_budget(20, 100, 20, 0, 30);
    assert(false);
  } catch (const clap::llama::RequestError& error) {
    assert(error.code == "context_length_exceeded");
    assert(std::string(error.what()).find("max_session_ctx=30") != std::string::npos);
  }

  const auto sampling = clap::llama::sampling_from_request({
      {"max_tokens", 7}, {"temperature", 0.2}, {"stop", "done"}});
  assert(sampling.max_tokens == 7);
  assert(sampling.temperature == 0.2);
  assert(sampling.stops.size() == 1 && sampling.stops[0] == "done");
}
