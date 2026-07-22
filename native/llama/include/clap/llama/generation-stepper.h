#pragma once

#include "clap/llama/cache-executor.h"
#include "clap/llama/model-runtime.h"
#include "clap/llama/request-state.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace clap::llama {

struct DecodeContribution {
  llama_token token = 0;
  llama_pos position = 0;
  llama_seq_id sequence = 0;
  bool logits = false;
};

class GenerationBackend {
 public:
  virtual ~GenerationBackend() = default;
  virtual int decode(const std::vector<DecodeContribution>& contribution) = 0;
  virtual llama_token sample(llama_sampler* sampler, int32_t logits_index) = 0;
  virtual std::string token_piece(llama_token token) = 0;
  virtual bool is_eog(llama_token token) = 0;
  virtual int32_t context_size() const = 0;
};

struct GenerationEvent {
  enum class Type {
    Token,
    Prefill,
    Complete,
    Failure,
    CacheAppend,
    CacheResetSlot,
    CacheResetAll,
    CacheAnchor,
  };

  Type type = Type::Token;
  ActiveRequest* request = nullptr;
  std::string text;
  std::string code;
  int32_t done = 0;
  int32_t total = 0;
  uint32_t slot = 0;
  uint64_t generation = 0;
  bool anchor = false;
  bool clear = true;
  std::vector<llama_token> tokens;
  std::vector<uint32_t> eviction_slots;
};

class GenerationStepper {
 public:
  GenerationStepper(GenerationBackend& backend, CacheExecutor* cache_executor = nullptr);
  GenerationStepper(ModelRuntime& runtime, CacheExecutor* cache_executor = nullptr);
  ~GenerationStepper();

  GenerationStepper(const GenerationStepper&) = delete;
  GenerationStepper& operator=(const GenerationStepper&) = delete;

  std::vector<GenerationEvent> step(const std::vector<ActiveRequest*>& ordered,
                                    int32_t batch_budget, bool sole_active);

 private:
  void add_contribution(std::vector<DecodeContribution>& batch, ActiveRequest& request,
                        int32_t budget);
  void post_decode(ActiveRequest& request, std::vector<GenerationEvent>& events);
  void process_sampled(ActiveRequest& request, llama_token token,
                       std::vector<GenerationEvent>& events);
  void decode_failure(ActiveRequest& request, bool sole_active,
                      std::vector<GenerationEvent>& events);
  void isolated(ActiveRequest& request, bool sole_active,
                std::vector<GenerationEvent>& events);
  void maybe_create_anchor(ActiveRequest& request, std::vector<GenerationEvent>& events);

  std::unique_ptr<GenerationBackend> owned_backend_;
  GenerationBackend* backend_ = nullptr;
  CacheExecutor* cache_executor_ = nullptr;
};

}  // namespace clap::llama
