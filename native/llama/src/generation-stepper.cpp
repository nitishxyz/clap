#include "clap/llama/generation-stepper.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace clap::llama {
namespace {

class LlamaGenerationBackend final : public GenerationBackend {
 public:
  explicit LlamaGenerationBackend(ModelRuntime& runtime) : runtime_(runtime) {}

  int decode(const std::vector<DecodeContribution>& contribution) override {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(contribution.size()), 0, 1);
    batch.n_tokens = 0;
    for (const auto& item : contribution) {
      const int32_t index = batch.n_tokens;
      batch.token[index] = item.token;
      batch.pos[index] = item.position;
      batch.n_seq_id[index] = 1;
      batch.seq_id[index][0] = item.sequence;
      batch.logits[index] = item.logits ? 1 : 0;
      batch.n_tokens += 1;
    }
    const int result = llama_decode(runtime_.context(), batch);
    llama_batch_free(batch);
    return result;
  }

  llama_token sample(llama_sampler* sampler, int32_t logits_index) override {
    return llama_sampler_sample(sampler, runtime_.context(), logits_index);
  }

  std::string token_piece(llama_token token) override {
    char buffer[256];
    int count = llama_token_to_piece(runtime_.vocab(), token, buffer, sizeof(buffer), 0, true);
    if (count >= 0) return std::string(buffer, count);
    std::vector<char> large(static_cast<std::size_t>(-count));
    count = llama_token_to_piece(runtime_.vocab(), token, large.data(),
                                 static_cast<int32_t>(large.size()), 0, true);
    return count < 0 ? std::string() : std::string(large.data(), count);
  }

  bool is_eog(llama_token token) override { return llama_vocab_is_eog(runtime_.vocab(), token); }
  int32_t context_size() const override {
    return static_cast<int32_t>(llama_n_ctx(runtime_.context()));
  }

 private:
  ModelRuntime& runtime_;
};

GenerationEvent event(GenerationEvent::Type type, ActiveRequest& request) {
  GenerationEvent value;
  value.type = type;
  value.request = &request;
  return value;
}

}  // namespace

GenerationStepper::GenerationStepper(GenerationBackend& backend, CacheExecutor* cache_executor,
                                     Fingerprint fingerprint)
    : backend_(&backend), cache_executor_(cache_executor),
      fingerprint_(std::move(fingerprint)) {}

GenerationStepper::GenerationStepper(ModelRuntime& runtime, CacheExecutor* cache_executor,
                                     Fingerprint fingerprint)
    : owned_backend_(std::make_unique<LlamaGenerationBackend>(runtime)),
      backend_(owned_backend_.get()), cache_executor_(cache_executor),
      fingerprint_(std::move(fingerprint)) {}

GenerationStepper::~GenerationStepper() = default;

void GenerationStepper::add_contribution(std::vector<DecodeContribution>& batch,
                                         ActiveRequest& request, int32_t budget) {
  if (request.phase == ActiveRequest::Phase::Decode) {
    request.logits_index = static_cast<int32_t>(batch.size());
    request.step_tokens = 1;
    batch.push_back({request.pending_token, request.n_pos, request.seq, true});
    return;
  }
  std::size_t remaining = request.prompt_tokens.size() - request.ingested;
  const std::size_t absolute = static_cast<std::size_t>(request.cached_prompt_tokens) +
      request.ingested;
  if (request.anchor_at >= 0 && static_cast<std::size_t>(request.anchor_at) > absolute) {
    remaining = std::min(remaining, static_cast<std::size_t>(request.anchor_at) - absolute);
  }
  const int32_t chunk = static_cast<int32_t>(
      std::min<std::size_t>(remaining, static_cast<std::size_t>(budget)));
  request.logits_index = -1;
  request.step_tokens = chunk;
  for (int32_t index = 0; index < chunk; ++index) {
    const bool last = request.ingested + static_cast<std::size_t>(index) + 1 ==
        request.prompt_tokens.size();
    if (last) request.logits_index = static_cast<int32_t>(batch.size());
    batch.push_back({request.prompt_tokens[request.ingested + static_cast<std::size_t>(index)],
                     request.n_pos + index, request.seq, last});
  }
}

void GenerationStepper::process_sampled(ActiveRequest& request, llama_token token,
                                        std::vector<GenerationEvent>& events) {
  if (backend_->is_eog(token)) {
    auto value = event(GenerationEvent::Type::Complete, request);
    value.completion = request.complete(true, fingerprint_);
    if (value.completion) {
      events.push_back(std::move(value));
    }
    return;
  }
  request.completion_tokens += 1;
  const std::string piece = backend_->token_piece(token);
  if (!piece.empty()) {
    const auto result = request.stop_buffer.append(piece);
    if (!result.visible.empty()) {
      auto value = event(GenerationEvent::Type::Token, request);
      value.text = result.visible;
      events.push_back(std::move(value));
    }
    if (result.stop_complete) {
      request.finish_reason = "stop";
      auto value = event(GenerationEvent::Type::Complete, request);
      value.completion = request.complete(true, fingerprint_);
      if (value.completion) events.push_back(std::move(value));
      return;
    }
  }
  if (request.completion_tokens >= request.params.max_tokens ||
      request.n_pos + 1 >= backend_->context_size()) {
    request.finish_reason = "length";
    auto value = event(GenerationEvent::Type::Complete, request);
    value.completion = request.complete(true, fingerprint_);
    if (value.completion) events.push_back(std::move(value));
    return;
  }
  request.pending_token = token;
  request.phase = ActiveRequest::Phase::Decode;
}

void GenerationStepper::maybe_create_anchor(ActiveRequest& request,
                                             std::vector<GenerationEvent>& events) {
  if (request.anchor_at < 0 || !request.cache_lease || !cache_executor_) return;
  const std::size_t count = static_cast<std::size_t>(request.anchor_at);
  if (count < 16 || static_cast<std::size_t>(request.cached_prompt_tokens) + request.ingested != count) return;
  const auto next = std::upper_bound(request.anchor_boundaries.begin(),
                                     request.anchor_boundaries.end(), count);
  request.anchor_at = next == request.anchor_boundaries.end()
      ? -1 : static_cast<int32_t>(*next);
  std::vector<llama_token> boundary(request.full_prompt_tokens.begin(),
      request.full_prompt_tokens.begin() + static_cast<std::ptrdiff_t>(count));
  try {
    auto identity = request.cache_identity;
    const bool structural = std::find(request.structural_boundaries.begin(),
        request.structural_boundaries.end(), count) != request.structural_boundaries.end();
    identity.scope = structural ? CLAP_CACHE_SCOPE_HARNESS : CLAP_CACHE_SCOPE_PROJECT;
    const auto result = cache_executor_->create_anchor(
        boundary, identity, static_cast<uint32_t>(request.seq), structural);
    if (!result.materialized) return;
    auto value = event(GenerationEvent::Type::CacheAnchor, request);
    value.slot = result.target_slot;
    value.generation = result.target_generation;
    value.anchor = !result.no_op;
    value.tokens = boundary;
    value.eviction_slots = result.eviction_slots;
    events.push_back(std::move(value));
    request.materialized_boundaries.push_back(count);
  } catch (const std::exception& error) {
    fprintf(stderr, "clap-llama: coordinated anchor skipped: %s\n", error.what());
  }
}

void GenerationStepper::post_decode(ActiveRequest& request,
                                    std::vector<GenerationEvent>& events) {
  if (request.phase == ActiveRequest::Phase::Prefill) {
    const std::size_t chunk = static_cast<std::size_t>(request.step_tokens);
    const std::size_t start = request.ingested;
    if (request.cache_lease) {
      uint64_t generation = request.cache_lease.generation();
      auto append = event(GenerationEvent::Type::CacheAppend, request);
      append.slot = static_cast<uint32_t>(request.seq);
      append.tokens.assign(request.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(start),
                           request.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(start + chunk));
      if (generation != 0) {
        try {
          generation = request.cache_lease.advance(request.prompt_tokens.data() + start,
              chunk, CLAP_CACHE_SLOT_SESSION, true);
        } catch (const std::exception& error) {
          request.cache_fallback = "coordinator_advance_failed";
          events.push_back(std::move(append));
          auto failure = event(GenerationEvent::Type::Failure, request);
          failure.failure = request.fail(
              "cache coordinator advance failed closed", "cache_coordinator_error");
          if (failure.failure) events.push_back(std::move(failure));
          fprintf(stderr, "clap-llama: cache metadata advance failed closed: %s\n", error.what());
          return;
        }
      }
      append.generation = generation;
      events.push_back(std::move(append));
    }
    request.n_pos += request.step_tokens;
    request.ingested += chunk;
    if (request.anchor_at >= 0 && static_cast<std::size_t>(request.cached_prompt_tokens) +
        request.ingested == static_cast<std::size_t>(request.anchor_at)) {
      maybe_create_anchor(request, events);
    }
    if (request.prompt_tokens.size() > 1024 && request.ingested < request.prompt_tokens.size()) {
      auto progress = event(GenerationEvent::Type::Prefill, request);
      progress.done = request.cached_prompt_tokens + static_cast<int32_t>(request.ingested);
      progress.total = request.prompt_token_count;
      events.push_back(std::move(progress));
    }
    if (request.logits_index >= 0) {
      process_sampled(request, backend_->sample(request.sampler.get(), request.logits_index), events);
    }
    return;
  }

  request.n_pos += 1;
  if (request.cache_lease) {
    uint64_t generation = request.cache_lease.generation();
    auto append = event(GenerationEvent::Type::CacheAppend, request);
    append.slot = static_cast<uint32_t>(request.seq);
    append.tokens = {request.pending_token};
    if (generation != 0) {
      try {
        generation = request.cache_lease.advance(
            &request.pending_token, 1, CLAP_CACHE_SLOT_SESSION, true);
      } catch (const std::exception& error) {
        request.cache_fallback = "coordinator_advance_failed";
        events.push_back(std::move(append));
        auto failure = event(GenerationEvent::Type::Failure, request);
        failure.failure = request.fail(
            "cache coordinator advance failed closed", "cache_coordinator_error");
        if (failure.failure) events.push_back(std::move(failure));
        fprintf(stderr, "clap-llama: cache metadata advance failed closed: %s\n", error.what());
        return;
      }
    }
    append.generation = generation;
    events.push_back(std::move(append));
  }
  process_sampled(request, backend_->sample(request.sampler.get(), request.logits_index), events);
}

void GenerationStepper::decode_failure(ActiveRequest& request, bool sole_active,
                                       std::vector<GenerationEvent>& events) {
  if (request.phase == ActiveRequest::Phase::Prefill && !request.retried) {
    request.retried = true;
    if (request.cache_lease) {
      if (sole_active) {
        request.cache_lease.reset_for_retry();
        events.push_back(event(GenerationEvent::Type::CacheResetAll, request));
      } else {
        if (request.cache_lease.generation() != 0) request.cache_lease.invalidate_and_clear(true);
        auto reset = event(GenerationEvent::Type::CacheResetSlot, request);
        reset.slot = static_cast<uint32_t>(request.seq);
        reset.generation = request.cache_lease.generation();
        events.push_back(std::move(reset));
      }
    }
    request.prompt_tokens = request.full_prompt_tokens;
    request.ingested = 0;
    request.n_pos = 0;
    request.cached_prompt_tokens = 0;
    request.materialized_boundaries.clear();
    request.anchor_at = request.anchor_boundaries.empty()
        ? -1 : static_cast<int32_t>(request.anchor_boundaries.front());
    request.cache_realized_reuse_tokens = 0;
    request.cache_reuse_kind.clear();
    request.cache_fallback = "decode_retry_full_prefill";
    fprintf(stderr, "clap-llama: ingest failed for request %s; retrying from scratch\n",
            request.id.c_str());
    return;
  }
  auto failure = event(GenerationEvent::Type::Failure, request);
  failure.failure = request.fail(
      "llama_decode failed; this often indicates llama.cpp GPU memory pressure. "
      "Check the llama worker log. Try a smaller GGUF quant such as Q4_K_M, reduce "
      "CLAP_LLAMA_CONTEXT/CLAP_LLAMA_BATCH/CLAP_LLAMA_UBATCH, set CLAP_LLAMA_GPU_LAYERS "
      "to a lower value, or use CPU fallback with CLAP_LLAMA_GPU_LAYERS=0.");
  if (failure.failure) events.push_back(std::move(failure));
}

void GenerationStepper::isolated(ActiveRequest& request, bool sole_active,
                                 std::vector<GenerationEvent>& events) {
  std::vector<DecodeContribution> batch;
  add_contribution(batch, request, std::max<int32_t>(request.step_tokens, 1));
  if (backend_->decode(batch) == 0) post_decode(request, events);
  else decode_failure(request, sole_active, events);
}

std::vector<GenerationEvent> GenerationStepper::step(
    const std::vector<ActiveRequest*>& ordered, int32_t batch_budget, bool sole_active) {
  std::vector<GenerationEvent> events;
  std::vector<DecodeContribution> batch;
  std::vector<ActiveRequest*> contributors;
  int32_t budget = batch_budget;
  std::size_t runnable_left = std::count_if(ordered.begin(), ordered.end(),
      [](const ActiveRequest* request) { return request && !request->done; });
  for (ActiveRequest* request : ordered) {
    if (!request || request->done || budget <= 0) continue;
    const int32_t reserved_for_others = static_cast<int32_t>(runnable_left > 0 ? runnable_left - 1 : 0);
    const int32_t quantum = sole_active ? batch_budget
        : request->priority == CLAP_CACHE_PRIORITY_INTERACTIVE ? 192
        : request->priority == CLAP_CACHE_PRIORITY_BACKGROUND ? 48 : 96;
    const int32_t request_budget = std::max(1, std::min(quantum, budget - reserved_for_others));
    const std::size_t before = batch.size();
    add_contribution(batch, *request, request_budget);
    const int32_t added = static_cast<int32_t>(batch.size() - before);
    if (added == 0) continue;
    budget -= added;
    contributors.push_back(request);
    runnable_left -= 1;
  }
  if (contributors.empty()) return events;
  if (backend_->decode(batch) == 0) {
    for (ActiveRequest* request : contributors) post_decode(*request, events);
    return events;
  }
  if (contributors.size() == 1) {
    decode_failure(*contributors.front(), sole_active, events);
    return events;
  }
  fprintf(stderr, "clap-llama: mixed batch decode failed; isolating %zu sequences\n",
          contributors.size());
  for (ActiveRequest* request : contributors) isolated(*request, false, events);
  return events;
}

}  // namespace clap::llama
