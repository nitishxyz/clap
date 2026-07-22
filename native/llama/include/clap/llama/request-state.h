#pragma once

#include "clap/llama/cache-executor.h"
#include "clap/llama/prompt.h"
#include "clap/llama/sampling.h"
#include "clap/llama/stop-buffer.h"
#include "stable-boundary.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace clap::llama {

struct Usage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
};

struct StableBoundaryCompletionFact {
  std::optional<std::string> token_hash;
  std::optional<std::size_t> token_count;
  std::string kind;
  std::optional<std::string> label;
  bool requested = false;
  std::string status;
  std::optional<std::string> skip_reason;
  std::optional<bool> materialized;
};

struct CacheCompletionFacts {
  bool hit = false;
  int reused_tokens = 0;
  std::optional<std::string> reuse_kind;
  std::optional<std::string> reuse_scope;
  std::optional<std::string> name_space;
  std::optional<int> donor_slot;
  std::optional<uint64_t> donor_generation;
  int target_slot = 0;
  uint64_t target_generation = 0;
  std::optional<std::string> miss_reason;
  nlohmann::json candidates = nlohmann::json::array();
  std::string prompt_token_hash;
  int prompt_token_count = 0;
  std::vector<uint32_t> evicted_slots;
  uint64_t decision_us = 0;
  uint64_t planned_reuse_tokens = 0;
  uint64_t realized_reuse_tokens = 0;
  bool side_request = false;
  std::optional<std::string> fallback;
  int slot = 0;
  std::optional<std::string> stable_boundary_token_hash;
  std::optional<std::size_t> stable_boundary_token_count;
  std::optional<std::string> stable_boundary_kind;
  std::vector<StableBoundaryCompletionFact> stable_boundaries;
};

struct RequestCompletion {
  std::string id;
  std::string finish_reason;
  bool cancelled = false;
  std::string visible_tail;
  Usage usage;
  CacheCompletionFacts cache;
  int released_slot = -1;
};

struct RequestFailure {
  std::string id;
  std::string message;
  std::string code;
  int invalidated_slot = -1;
  uint64_t generation = 0;
  std::string invalidation_error;
};

class SamplerOwner {
 public:
  using Deleter = void (*)(llama_sampler*);

  SamplerOwner() = default;
  explicit SamplerOwner(llama_sampler* sampler, Deleter deleter = llama_sampler_free)
      : sampler_(sampler), deleter_(deleter) {}
  SamplerOwner(const SamplerOwner&) = delete;
  SamplerOwner& operator=(const SamplerOwner&) = delete;
  SamplerOwner(SamplerOwner&& other) noexcept;
  SamplerOwner& operator=(SamplerOwner&& other) noexcept;
  ~SamplerOwner();

  llama_sampler* get() const noexcept { return sampler_; }
  explicit operator bool() const noexcept { return sampler_ != nullptr; }
  void reset(llama_sampler* sampler = nullptr, Deleter deleter = llama_sampler_free);

 private:
  llama_sampler* sampler_ = nullptr;
  Deleter deleter_ = llama_sampler_free;
};

struct PreparedRequest {
  PreparedRequest() = default;
  PreparedRequest(const PreparedRequest&) = delete;
  PreparedRequest& operator=(const PreparedRequest&) = delete;
  PreparedRequest(PreparedRequest&&) noexcept = default;
  PreparedRequest& operator=(PreparedRequest&&) noexcept = default;

  std::string id;
  SamplingParams params;
  CacheLease cache_lease;
  clap::llama_cache::Identity cache_identity;
  std::vector<llama_token> prompt_tokens;
  std::vector<llama_token> full_prompt_tokens;
  int prompt_token_count = 0;
  int cached_prompt_tokens = 0;
  std::string cache_reuse_kind;
  std::string cache_reuse_scope;
  std::string cache_namespace;
  int cache_donor_slot = -1;
  uint64_t cache_donor_generation = 0;
  uint64_t cache_target_generation = 0;
  std::vector<uint32_t> cache_evicted_slots;
  uint64_t cache_planned_reuse_tokens = 0;
  uint64_t cache_realized_reuse_tokens = 0;
  uint64_t cache_decision_us = 0;
  std::string cache_fallback;
  std::string prompt_token_hash;
  clap::llama_boundary::StableBoundary stable_boundary;
  std::vector<std::size_t> anchor_boundaries;
  std::vector<std::size_t> structural_boundaries;
  std::vector<ResolvedPromptBoundary> resolved_boundaries;
  nlohmann::json cache_candidates = nlohmann::json::array();
  bool cache_side_request = false;
  llama_seq_id sequence = 0;
  int32_t initial_position = 0;
  int32_t initial_anchor_at = -1;
};

struct ActiveRequest : PreparedRequest {
  enum class Phase { Prefill, Decode };
  enum class TerminalState { Active, Completed, Failed };

  ActiveRequest() = default;
  explicit ActiveRequest(PreparedRequest&& prepared);
  ActiveRequest(const ActiveRequest&) = delete;
  ActiveRequest& operator=(const ActiveRequest&) = delete;
  ActiveRequest(ActiveRequest&&) noexcept = default;
  ActiveRequest& operator=(ActiveRequest&&) noexcept = default;

  bool mark_terminal(TerminalState state) noexcept;
  bool terminal() const noexcept { return terminal_state != TerminalState::Active; }
  std::optional<RequestCompletion> complete(
      bool flush_tail,
      const std::function<std::string(const std::vector<llama_token>&, std::size_t)>& fingerprint);
  std::optional<RequestFailure> fail(std::string message, std::string code = {});

  SamplerOwner sampler;
  llama_seq_id seq = 0;
  std::size_t ingested = 0;
  int32_t n_pos = 0;
  Phase phase = Phase::Prefill;
  llama_token pending_token = 0;
  StopBuffer stop_buffer;
  int completion_tokens = 0;
  std::string finish_reason = "stop";
  bool cancelled = false;
  bool retried = false;
  bool done = false;
  TerminalState terminal_state = TerminalState::Active;
  int32_t anchor_at = -1;
  bool anchor_planted = false;
  std::vector<std::size_t> materialized_boundaries;
  int32_t logits_index = -1;
  int32_t step_tokens = 0;
};

}  // namespace clap::llama
