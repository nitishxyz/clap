#include "clap/llama/request-state.h"

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

}  // namespace clap::llama
