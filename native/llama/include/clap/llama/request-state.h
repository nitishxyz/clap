#pragma once

#include "clap/llama/cache-executor.h"
#include "clap/llama/prompt.h"
#include "clap/llama/sampling.h"
#include "clap/llama/stop-buffer.h"
#include "stable-boundary.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace clap::llama {

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
