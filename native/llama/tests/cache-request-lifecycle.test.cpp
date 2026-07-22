#include "clap/llama/cache-executor.h"
#include "clap/llama/generation-stepper.h"
#include "clap/llama/request-state.h"
#include "tests/support/cache-state-backend.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace {

using clap::llama::ActiveRequest;
using clap::llama::CacheAdmissionResult;
using clap::llama::CacheExecutor;
using clap::llama::DecodeContribution;
using clap::llama::GenerationBackend;
using clap::llama::GenerationEvent;
using clap::llama::GenerationStepper;
using clap::llama::PreparedRequest;
using clap::llama::test::CacheStateBackend;

class DeterministicBackend final : public GenerationBackend {
 public:
  int decode(const std::vector<DecodeContribution>& contribution) override {
    decoded.push_back(contribution);
    return decode_result;
  }
  llama_token sample(llama_sampler*, int32_t) override {
    assert(!samples.empty());
    const llama_token value = samples.front();
    samples.pop_front();
    return value;
  }
  std::string token_piece(llama_token token) override {
    return token == 70 ? "hel" : std::to_string(token);
  }
  bool is_eog(llama_token token) override { return token == 99; }
  int32_t context_size() const override { return 256; }

  int decode_result = 0;
  std::deque<llama_token> samples;
  std::vector<std::vector<DecodeContribution>> decoded;
};

clap::llama::CacheExecutorConfig config(uint32_t slots = 1) {
  clap::llama::CacheExecutorConfig value;
  value.slot_count = slots;
  value.min_reuse_tokens = 4;
  value.logical_token_capacity = 64;
  value.max_anchors = slots > 2 ? 1 : 0;
  value.hard_max_retained_entries = slots;
  value.automatic_checkpoints = false;
  return value;
}

clap::llama_cache::Identity identity(const std::string& name) {
  clap::llama_cache::Identity value;
  value.name_space = clap::llama_cache::fingerprint(name);
  value.tenant = clap::llama_cache::hash("tenant");
  value.session = clap::llama_cache::hash(name);
  value.scope = CLAP_CACHE_SCOPE_SESSION;
  return value;
}

std::vector<int32_t> prompt(int32_t start = 1, std::size_t count = 12) {
  std::vector<int32_t> value(count);
  for (std::size_t index = 0; index < count; ++index) {
    value[index] = start + static_cast<int32_t>(index);
  }
  return value;
}

CacheAdmissionResult admit(CacheExecutor& executor, const std::vector<int32_t>& tokens,
                           const clap::llama_cache::Identity& cache_identity) {
  return executor.admit({tokens, cache_identity,
      CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH | CLAP_CACHE_CAP_WHOLE_STATE_COPY,
      1, CLAP_CACHE_SLOT_SESSION, {}, {}, false, {}});
}

ActiveRequest active(CacheExecutor& executor, const CacheAdmissionResult& admission,
                     const std::vector<int32_t>& full_prompt) {
  PreparedRequest prepared;
  prepared.id = "lifecycle";
  prepared.sequence = static_cast<llama_seq_id>(admission.target_slot);
  prepared.cache_lease = executor.lease_admitted(
      admission.target_slot, admission.target_generation);
  prepared.prompt_tokens.assign(
      full_prompt.begin() + static_cast<std::ptrdiff_t>(admission.reuse_tokens),
      full_prompt.end());
  prepared.full_prompt_tokens = full_prompt;
  prepared.prompt_token_count = static_cast<int>(full_prompt.size());
  prepared.cached_prompt_tokens = static_cast<int>(admission.reuse_tokens);
  prepared.cache_target_generation = admission.target_generation;
  prepared.params.max_tokens = 8;
  prepared.params.stops = {"hello"};
  return ActiveRequest(std::move(prepared));
}

void apply_physical(CacheStateBackend& physical,
                    const std::vector<GenerationEvent>& events) {
  for (const auto& event : events) {
    if (event.type == GenerationEvent::Type::CacheAppend) {
      physical.append(event.slot, event.tokens);
    }
  }
}

void cancel(ActiveRequest& request, int& terminal_count) {
  request.cancelled = true;
  request.finish_reason = "cancel";
  const auto completion = request.complete(false, {});
  assert(completion);
  ++terminal_count;
  assert(completion->cancelled);
  assert(completion->visible_tail.empty());
  assert(!request.complete(false, {}));
}

void cancellation_boundaries_are_cold_equivalent() {
  for (int cancellation_point = 0; cancellation_point < 3; ++cancellation_point) {
    auto backend = std::make_unique<CacheStateBackend>(1);
    auto* physical = backend.get();
    CacheExecutor executor(config(), std::move(backend));
    const auto full_prompt = prompt();
    const auto admission = admit(executor, full_prompt,
                                 identity("cancel-" + std::to_string(cancellation_point)));
    auto request = active(executor, admission, full_prompt);
    DeterministicBackend generation;
    generation.samples = {70, 71};
    GenerationStepper stepper(generation, &executor);
    int terminal_count = 0;

    if (cancellation_point >= 1) {
      auto events = stepper.step({&request}, 4, true);
      apply_physical(*physical, events);
      assert(request.ingested == 4);
    }
    if (cancellation_point == 2) {
      while (request.phase == ActiveRequest::Phase::Prefill) {
        auto events = stepper.step({&request}, 64, true);
        apply_physical(*physical, events);
      }
      assert(request.pending_token == 70);
    }

    cancel(request, terminal_count);
    assert(terminal_count == 1);
    assert(!executor.slot(0).busy);
    const auto logical = executor.coordinator().slot(0);
    assert(logical.resident_len == physical->slot(0).size());
    if (cancellation_point == 0) {
      assert(logical.resident_len == 0);
    } else if (cancellation_point == 1) {
      assert(physical->slot(0) == std::vector<int32_t>(full_prompt.begin(),
                                                       full_prompt.begin() + 4));
    } else {
      assert(physical->slot(0) == full_prompt);
      assert(std::find(physical->slot(0).begin(), physical->slot(0).end(), 70) ==
             physical->slot(0).end());
    }
  }
}

void eviction_and_stale_leases_are_generation_guarded() {
  auto pressure = config(2);
  pressure.logical_token_capacity = 24;
  auto backend = std::make_unique<CacheStateBackend>(2);
  auto* physical = backend.get();
  CacheExecutor executor(pressure, std::move(backend));

  std::vector<uint64_t> old_generations(2);
  for (uint32_t index = 0; index < 2; ++index) {
    const auto tokens = prompt(100 + static_cast<int32_t>(index * 20));
    const auto result = admit(executor, tokens, identity("resident-" + std::to_string(index)));
    physical->append(result.target_slot, tokens);
    const uint64_t generation = executor.advance(result.target_slot,
        result.target_generation, tokens.data(), tokens.size(), CLAP_CACHE_SLOT_SESSION, false);
    old_generations[result.target_slot] = generation;
    executor.release(result.target_slot, generation);
  }

  const auto replacement_tokens = prompt(200);
  const auto replacement = admit(executor, replacement_tokens, identity("replacement"));
  assert(!replacement.eviction_slots.empty());
  const uint32_t reused = replacement.target_slot;
  assert(replacement.target_generation != old_generations[reused]);
  executor.release(reused, old_generations[reused]);
  assert(executor.slot(reused).busy);
  executor.release(reused, replacement.target_generation);
  executor.release(reused, replacement.target_generation);
  assert(!executor.slot(reused).busy);
}

void advance_failure_invalidates_physical_and_logical_state() {
  auto backend = std::make_unique<CacheStateBackend>(1);
  auto* physical = backend.get();
  CacheExecutor executor(config(), std::move(backend));
  const auto tokens = prompt(300);
  const auto result = admit(executor, tokens, identity("advance-failure"));
  physical->append(0, tokens);
  executor.release(0, result.target_generation);
  const int32_t extra = 999;
  try {
    executor.advance(0, result.target_generation + 100, &extra, 1,
                     CLAP_CACHE_SLOT_SESSION, true);
    assert(false);
  } catch (const clap::llama_cache::Error&) {
  }
  assert(physical->slot(0).empty());
  assert(executor.slot(0).resident_tokens == 0);
  assert(!executor.slot(0).busy);
}

void reset_invalidates_leases_anchors_and_physical_state() {
  auto backend = std::make_unique<CacheStateBackend>(3);
  auto* physical = backend.get();
  CacheExecutor executor(config(3), std::move(backend));
  const auto tokens = prompt(400, 16);
  const auto admitted = admit(executor, tokens, identity("reset"));
  physical->append(admitted.target_slot, tokens);
  const uint64_t generation = executor.advance(admitted.target_slot,
      admitted.target_generation, tokens.data(), tokens.size(), CLAP_CACHE_SLOT_SESSION, false);
  executor.release(admitted.target_slot, generation);
  const auto anchor = executor.create_anchor(tokens, identity("reset"),
                                              admitted.target_slot, false);
  assert(anchor.materialized && !anchor.no_op);
  auto pre_reset_lease = executor.acquire(admitted.target_slot);
  const uint64_t pre_reset_generation = pre_reset_lease.generation();
  const uint64_t epoch = executor.reset();
  assert(epoch > 0);
  const auto telemetry = executor.telemetry();
  assert(telemetry.resets > 0);
  assert(telemetry.active_slots == 0);
  assert(telemetry.anchors == 0);
  assert(telemetry.read_leases == 0);
  assert(telemetry.write_leases == 0);
  assert(telemetry.physical_bytes == 0);
  for (uint32_t slot = 0; slot < executor.slot_count(); ++slot) {
    assert(physical->slot(slot).empty());
    assert(!executor.slot(slot).busy);
    assert(!executor.slot(slot).anchor);
    assert(executor.slot(slot).resident_tokens == 0);
  }
  pre_reset_lease.release();
  assert(!executor.slot(admitted.target_slot).busy);
  assert(executor.slot(admitted.target_slot).generation != pre_reset_generation);
  const auto fresh = admit(executor, tokens, identity("reset"));
  assert(fresh.operation == CLAP_CACHE_OPERATION_FRESH);
  assert(fresh.reuse_tokens == 0);
}

}  // namespace

int main() {
  cancellation_boundaries_are_cold_equivalent();
  eviction_and_stale_leases_are_generation_guarded();
  advance_failure_invalidates_physical_and_logical_state();
  reset_invalidates_leases_anchors_and_physical_state();
}
