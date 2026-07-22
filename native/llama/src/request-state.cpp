#include "clap/llama/request-state.h"

#include <algorithm>
#include <exception>
#include <utility>

namespace clap::llama {

ActiveRequest::ActiveRequest(PreparedRequest&& prepared)
    : PreparedRequest(std::move(prepared)), seq(sequence), n_pos(initial_position),
      anchor_at(initial_anchor_at) {
  stop_buffer.reset(params.stops);
}

SamplerOwner::SamplerOwner(SamplerOwner&& other) noexcept
    : sampler_(other.sampler_), deleter_(other.deleter_) {
  other.sampler_ = nullptr;
}

SamplerOwner& SamplerOwner::operator=(SamplerOwner&& other) noexcept {
  if (this != &other) {
    reset();
    sampler_ = other.sampler_;
    deleter_ = other.deleter_;
    other.sampler_ = nullptr;
  }
  return *this;
}

SamplerOwner::~SamplerOwner() {
  reset();
}

void SamplerOwner::reset(llama_sampler* sampler, Deleter deleter) {
  if (sampler_ && deleter_) deleter_(sampler_);
  sampler_ = sampler;
  deleter_ = deleter;
}

bool ActiveRequest::mark_terminal(TerminalState state) noexcept {
  if (terminal() || state == TerminalState::Active) return false;
  terminal_state = state;
  done = true;
  return true;
}

std::optional<RequestCompletion> ActiveRequest::complete(
    bool flush_tail,
    const std::function<std::string(const std::vector<llama_token>&, std::size_t)>& fingerprint) {
  if (!mark_terminal(TerminalState::Completed)) return std::nullopt;
  sampler.reset();
  RequestCompletion result;
  result.id = id;
  result.finish_reason = finish_reason;
  result.cancelled = cancelled;
  if (flush_tail) result.visible_tail = stop_buffer.finish();
  result.usage = {prompt_token_count, completion_tokens};
  result.released_slot = cache_lease ? static_cast<int>(seq) : -1;
  if (cache_lease) cache_lease.release();

  auto& cache = result.cache;
  cache.hit = cached_prompt_tokens > 0;
  cache.reused_tokens = cached_prompt_tokens;
  if (!cache_reuse_kind.empty()) cache.reuse_kind = cache_reuse_kind;
  if (!cache_reuse_scope.empty()) cache.reuse_scope = cache_reuse_scope;
  if (!cache_namespace.empty()) cache.name_space = cache_namespace;
  if (cache_donor_slot >= 0) {
    cache.donor_slot = cache_donor_slot;
    cache.donor_generation = cache_donor_generation;
  }
  cache.target_slot = static_cast<int>(seq);
  cache.target_generation = cache_target_generation;
  if (!cache.hit) cache.miss_reason = "no_shared_prefix";
  cache.candidates = cache_candidates;
  cache.prompt_token_hash = prompt_token_hash;
  cache.prompt_token_count = prompt_token_count;
  cache.evicted_slots = cache_evicted_slots;
  cache.decision_us = cache_decision_us;
  cache.planned_reuse_tokens = cache_planned_reuse_tokens;
  cache.realized_reuse_tokens = cache_realized_reuse_tokens;
  cache.side_request = cache_side_request;
  if (!cache_fallback.empty()) cache.fallback = cache_fallback;
  cache.slot = static_cast<int>(seq);
  if (stable_boundary.available()) {
    cache.stable_boundary_token_hash = stable_boundary.token_hash;
    cache.stable_boundary_token_count = stable_boundary.token_count;
    cache.stable_boundary_kind = stable_boundary.kind;
  }
  for (const auto& resolved : resolved_boundaries) {
    const std::size_t boundary = resolved.token_count;
    const bool available = (resolved.status == "resolved" || resolved.status == "authorized") &&
        boundary > 0;
    StableBoundaryCompletionFact fact;
    if (available) {
      fact.token_hash = fingerprint(full_prompt_tokens, boundary);
      fact.token_count = boundary;
      fact.materialized = std::find(materialized_boundaries.begin(),
          materialized_boundaries.end(), boundary) != materialized_boundaries.end();
    }
    fact.kind = resolved.kind;
    if (!resolved.label.empty()) fact.label = resolved.label;
    fact.requested = resolved.requested;
    fact.status = resolved.status;
    if (!resolved.skip_reason.empty()) fact.skip_reason = resolved.skip_reason;
    cache.stable_boundaries.push_back(std::move(fact));
  }
  return result;
}

std::optional<RequestFailure> ActiveRequest::fail(std::string message, std::string code) {
  if (!mark_terminal(TerminalState::Failed)) return std::nullopt;
  sampler.reset();
  RequestFailure result;
  result.id = id;
  result.message = std::move(message);
  result.code = std::move(code);
  if (cache_lease) {
    result.invalidated_slot = static_cast<int>(seq);
    result.generation = cache_lease.generation();
    if (cache_lease.generation() != 0) {
      try {
        result.generation = cache_lease.invalidate_and_clear();
      } catch (const std::exception& error) {
        result.invalidation_error = error.what();
      }
    }
    cache_lease.release();
  }
  return result;
}

}  // namespace clap::llama
