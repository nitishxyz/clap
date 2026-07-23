#pragma once

#include "clap/llama/model-runtime.h"
#include "clap/llama/request-state.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace clap::llama {

class CacheBackpressure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct RequestSlotState {
  std::vector<int32_t> tokens;
  uint64_t generation = 0;
  bool busy = false;
  bool anchor = false;
};

struct RequestBudget {
  int max_tokens;
  int output_reserve;
};

RequestBudget validate_request_budget(int prompt_count, int context_size,
                                      int requested_max_tokens,
                                      int configured_max_output,
                                      int session_cap);
uint64_t cache_capabilities(bool hybrid);

class RequestPreparer {
 public:
  using Fingerprint = std::function<std::string(const std::vector<llama_token>&, std::size_t)>;

  RequestPreparer(ModelRuntime& runtime, CacheExecutor* cache_executor,
                  std::vector<RequestSlotState> slots, Fingerprint fingerprint);

  PreparedRequest prepare(const std::string& id, const nlohmann::json& request);

 private:
  uint64_t capabilities() const;

  ModelRuntime& runtime_;
  CacheExecutor* cache_executor_;
  std::vector<RequestSlotState> slots_;
  Fingerprint fingerprint_;
};

}  // namespace clap::llama
