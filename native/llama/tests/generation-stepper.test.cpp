#include "clap/llama/generation-stepper.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <deque>
#include <string>
#include <vector>

namespace {

class FakeBackend final : public clap::llama::GenerationBackend {
 public:
  int decode(const std::vector<clap::llama::DecodeContribution>& contribution) override {
    calls.push_back(contribution);
    if (decode_results.empty()) return 0;
    const int result = decode_results.front();
    decode_results.pop_front();
    return result;
  }

  llama_token sample(llama_sampler*, int32_t) override {
    assert(!sampled.empty());
    const llama_token token = sampled.front();
    sampled.pop_front();
    return token;
  }

  std::string token_piece(llama_token token) override { return pieces[token]; }
  bool is_eog(llama_token token) override { return token == eog; }
  int32_t context_size() const override { return context; }

  std::vector<std::vector<clap::llama::DecodeContribution>> calls;
  std::deque<int> decode_results;
  std::deque<llama_token> sampled;
  std::string pieces[64];
  llama_token eog = 63;
  int32_t context = 128;
};

clap::llama::ActiveRequest request(std::vector<llama_token> prompt, llama_seq_id sequence) {
  clap::llama::PreparedRequest prepared;
  prepared.id = "request-" + std::to_string(sequence);
  prepared.sequence = sequence;
  prepared.prompt_tokens = prompt;
  prepared.full_prompt_tokens = prompt;
  prepared.prompt_token_count = static_cast<int>(prompt.size());
  prepared.params.max_tokens = 8;
  return clap::llama::ActiveRequest(std::move(prepared));
}

}  // namespace

int main() {
  {
    FakeBackend backend;
    backend.sampled = {10};
    backend.pieces[10] = "x";
    auto decode = request({}, 0);
    decode.phase = clap::llama::ActiveRequest::Phase::Decode;
    decode.pending_token = 7;
    auto prefill = request({1, 2, 3, 4}, 1);
    clap::llama::GenerationStepper stepper(backend);
    const auto events = stepper.step({&decode, &prefill}, 3, false);
    assert(backend.calls.size() == 1);
    assert(backend.calls[0].size() == 3);
    assert(backend.calls[0][0].token == 7);
    assert(backend.calls[0][0].logits);
    assert(backend.calls[0][1].token == 1);
    assert(backend.calls[0][2].token == 2);
    assert(prefill.ingested == 2);
    assert(decode.pending_token == 10);
    assert(events.size() == 1);
    assert(events[0].type == clap::llama::GenerationEvent::Type::Token);
    assert(events[0].text == "x");
  }

  {
    FakeBackend backend;
    auto prefill = request({1, 2, 3, 4, 5}, 0);
    prefill.anchor_at = 3;
    clap::llama::GenerationStepper stepper(backend);
    stepper.step({&prefill}, 5, true);
    assert(backend.calls[0].size() == 3);
    assert(prefill.ingested == 3);
  }

  {
    FakeBackend backend;
    backend.decode_results = {1, 0, 0};
    backend.sampled = {11, 63};
    backend.pieces[11] = "z";
    auto first = request({}, 0);
    first.phase = clap::llama::ActiveRequest::Phase::Decode;
    first.pending_token = 8;
    auto second = request({2}, 1);
    clap::llama::GenerationStepper stepper(backend);
    stepper.step({&first, &second}, 4, false);
    assert(backend.calls.size() == 3);
    assert(backend.calls[1].size() == 1);
    assert(backend.calls[2].size() == 1);
    assert(first.pending_token == 11);
    assert(second.ingested == 1);
  }

  {
    FakeBackend backend;
    backend.decode_results = {1};
    auto prefill = request({1, 2, 3}, 0);
    prefill.ingested = 2;
    prefill.n_pos = 2;
    prefill.cached_prompt_tokens = 4;
    prefill.cache_reuse_kind = "slot";
    clap::llama::GenerationStepper stepper(backend);
    const auto events = stepper.step({&prefill}, 2, true);
    assert(events.empty());
    assert(prefill.retried);
    assert(prefill.ingested == 0);
    assert(prefill.n_pos == 0);
    assert(prefill.cached_prompt_tokens == 0);
    assert(prefill.cache_fallback == "decode_retry_full_prefill");
    assert(prefill.prompt_tokens == prefill.full_prompt_tokens);
  }

  {
    FakeBackend backend;
    backend.sampled = {12};
    backend.pieces[12] = "hello<stop>tail";
    auto prefill = request({1}, 0);
    prefill.params.stops = {"<stop>"};
    prefill.stop_buffer.reset(prefill.params.stops);
    clap::llama::GenerationStepper stepper(backend);
    const auto events = stepper.step({&prefill}, 1, true);
    assert(events.size() == 2);
    assert(events[0].type == clap::llama::GenerationEvent::Type::Token);
    assert(events[0].text == "hello");
    assert(events[1].type == clap::llama::GenerationEvent::Type::Complete);
    assert(events[1].completion);
    assert(events[1].completion->usage.completion_tokens == 1);
    assert(events[1].completion->visible_tail.empty());
    assert(prefill.finish_reason == "stop");
    assert(prefill.done);
  }

  {
    FakeBackend backend;
    backend.sampled = {13};
    backend.pieces[13] = "hel";
    auto prefill = request({1}, 0);
    prefill.params.max_tokens = 1;
    prefill.params.stops = {"hello"};
    prefill.stop_buffer.reset(prefill.params.stops);
    clap::llama::GenerationStepper stepper(backend);
    const auto events = stepper.step({&prefill}, 1, true);
    assert(events.size() == 1);
    assert(events[0].completion);
    assert(events[0].completion->visible_tail == "hel");
    assert(events[0].completion->finish_reason == "length");
  }
}
