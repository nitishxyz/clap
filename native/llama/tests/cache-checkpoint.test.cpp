#include "clap/llama/cache-executor.h"
#include "tests/support/cache-state-backend.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using clap::llama::CacheExecutor;
using clap::llama::test::CacheStateBackend;

clap::llama_cache::Identity identity() {
  clap::llama_cache::Identity value;
  value.name_space = clap::llama_cache::fingerprint("checkpoint-namespace");
  value.tenant = clap::llama_cache::hash("tenant");
  value.project = clap::llama_cache::hash("project");
  value.scope = CLAP_CACHE_SCOPE_PROJECT;
  return value;
}

clap::llama::CacheExecutorConfig config() {
  clap::llama::CacheExecutorConfig value;
  value.slot_count = 6;
  value.min_reuse_tokens = 2;
  value.logical_token_capacity = 512;
  value.max_anchors = 3;
  value.hard_max_retained_entries = 6;
  value.automatic_checkpoints = true;
  value.checkpoint_minimum_tokens = 8;
  value.checkpoint_interval_tokens = 4;
  value.checkpoint_max = 3;
  value.checkpoint_budget_basis_points = 2'500;
  value.checkpoint_budget_bytes = 384;
  return value;
}

std::vector<int32_t> tokens(std::size_t count) {
  std::vector<int32_t> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = static_cast<int32_t>(index + 1);
  }
  return result;
}

void proposals_honor_interval_max_budget_and_explicit_deduplication() {
  auto backend = std::make_unique<CacheStateBackend>(6);
  CacheExecutor executor(config(), std::move(backend));
  const auto prompt = tokens(22);
  const auto result = executor.preview({prompt, identity(),
      CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT, 1, CLAP_CACHE_SLOT_SESSION,
      {}, {8, 9, 12}, false, {}});
  assert((result.anchor_boundaries == std::vector<uint64_t>{8, 9, 12}));
  assert(std::count(result.anchor_boundaries.begin(),
                    result.anchor_boundaries.end(), 8) == 1);
  assert(std::count(result.anchor_boundaries.begin(),
                    result.anchor_boundaries.end(), 12) == 1);
  const auto retention = executor.retention_telemetry();
  assert(retention.automatic_checkpoint_byte_budget == 384);

  auto unbudgeted = config();
  unbudgeted.checkpoint_budget_bytes = 0;
  auto second_backend = std::make_unique<CacheStateBackend>(6);
  CacheExecutor second(unbudgeted, std::move(second_backend));
  const auto intervals = second.preview({prompt, identity(),
      CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT, 1, CLAP_CACHE_SLOT_SESSION,
      {}, {}, false, {}});
  assert((intervals.anchor_boundaries == std::vector<uint64_t>{8, 16}));
  assert(intervals.anchor_boundaries.size() <= unbudgeted.checkpoint_max);
}

void exact_boundaries_materialize_and_duplicate_is_a_noop() {
  auto backend = std::make_unique<CacheStateBackend>(6);
  auto* physical = backend.get();
  CacheExecutor executor(config(), std::move(backend));
  const auto prompt = tokens(12);
  const auto admitted = executor.admit({prompt, identity(),
      CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT, 1, CLAP_CACHE_SLOT_SESSION,
      {}, {}, false, {}});
  physical->append(admitted.target_slot, prompt);
  const auto generation = executor.advance(admitted.target_slot,
      admitted.target_generation, prompt.data(), prompt.size(),
      CLAP_CACHE_SLOT_SESSION, false);
  executor.release(admitted.target_slot, generation);

  const auto checkpoint = executor.create_anchor(
      prompt, identity(), admitted.target_slot, false);
  assert(checkpoint.materialized && !checkpoint.no_op);
  assert(physical->slot(checkpoint.target_slot) == prompt);
  const auto duplicate = executor.create_anchor(
      prompt, identity(), admitted.target_slot, false);
  assert(duplicate.materialized && duplicate.no_op);
}

void failed_copy_publishes_no_anchor_and_reset_clears_every_checkpoint() {
  auto backend = std::make_unique<CacheStateBackend>(6);
  auto* physical = backend.get();
  CacheExecutor executor(config(), std::move(backend));
  const auto prompt = tokens(16);
  const auto admitted = executor.admit({prompt, identity(), 0, 1,
      CLAP_CACHE_SLOT_SESSION, {}, {}, false, {}});
  physical->append(admitted.target_slot, prompt);
  const auto generation = executor.advance(admitted.target_slot,
      admitted.target_generation, prompt.data(), prompt.size(),
      CLAP_CACHE_SLOT_SESSION, false);
  executor.release(admitted.target_slot, generation);

  physical->fail_next_copy();
  try {
    executor.create_anchor(prompt, identity(), admitted.target_slot, false);
    assert(false);
  } catch (const std::runtime_error&) {
  }
  assert(executor.telemetry().anchors == 0);
  for (uint32_t slot = 0; slot < executor.slot_count(); ++slot) {
    assert(!executor.slot(slot).anchor);
  }

  const auto checkpoint = executor.create_anchor(
      prompt, identity(), admitted.target_slot, false);
  assert(checkpoint.materialized);
  assert(executor.telemetry().anchors == 1);
  executor.reset();
  assert(executor.telemetry().anchors == 0);
  assert(executor.retention_telemetry().automatic_checkpoint_slots == 0);
  for (uint32_t slot = 0; slot < executor.slot_count(); ++slot) {
    assert(!executor.slot(slot).anchor);
    assert(physical->slot(slot).empty());
  }
}

}  // namespace

int main() {
  proposals_honor_interval_max_budget_and_explicit_deduplication();
  exact_boundaries_materialize_and_duplicate_is_a_noop();
  failed_copy_publishes_no_anchor_and_reset_clears_every_checkpoint();
}
