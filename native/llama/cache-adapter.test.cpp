#include "cache-adapter.h"
#include "active-concurrency.h"
#include "stable-boundary.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

using clap::llama_cache::Coordinator;
using clap::llama_cache::Identity;

int main() {
  Coordinator cache(3, 2, 64, 2);
  Identity first;
  first.name_space = clap::llama_cache::fingerprint("llama-test-domain|tenant=a");
  first.tenant = clap::llama_cache::hash("a");
  first.session = clap::llama_cache::hash("session-a");
  first.scope = CLAP_CACHE_SCOPE_SESSION;

  const std::vector<int32_t> original{1, 2, 3, 4};
  auto fresh = cache.plan(original, first, CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 1);
  assert(fresh.view().operation == CLAP_CACHE_OPERATION_FRESH);
  const uint32_t first_slot = fresh.view().target.slot;
  fresh.commit(0, CLAP_CACHE_SLOT_SESSION);
  auto first_info = cache.slot(first_slot);
  clap_cache_slot_ref_t first_ref{first_slot, 0, first_info.generation};
  first_ref.generation = cache.advance(first_ref, original.data(), original.size(),
                                       CLAP_CACHE_SLOT_SESSION, true);
  cache.set_busy(first_ref, false);

  Identity second = first;
  second.session = clap::llama_cache::hash("session-b");
  const std::vector<int32_t> related{1, 2, 3, 9};
  auto branch = cache.plan(related, second,
      CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH | CLAP_CACHE_CAP_SAFE_BUSY_DONOR, 1);
  assert(branch.view().operation == CLAP_CACHE_OPERATION_BRANCH);
  assert(branch.view().reuse_tokens == 3);
  assert(branch.view().donor.slot == first_slot);
  const uint32_t branch_slot = branch.view().target.slot;
  const auto decision = branch.commit(3, CLAP_CACHE_SLOT_SESSION);
  assert(decision.hit == 1);
  assert(decision.realized_reuse_tokens == 3);

  auto branch_info = cache.slot(branch_slot);
  const uint64_t invalidated = cache.invalidate(
      {branch_slot, 0, branch_info.generation});
  assert(invalidated > branch_info.generation);
  assert(cache.slot(branch_slot).state == CLAP_CACHE_SLOT_EMPTY);

  Identity isolated = second;
  isolated.name_space = clap::llama_cache::fingerprint("llama-test-domain|tenant=b");
  isolated.tenant = clap::llama_cache::hash("b");
  auto miss = cache.plan(original, isolated, CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 0);
  assert(miss.view().operation == CLAP_CACHE_OPERATION_FRESH);
  assert(miss.view().reuse_tokens == 0);
  miss.abort();

  bool stale_rejected = false;
  try {
    cache.set_busy({first_slot, 0, first_info.generation}, true);
  } catch (const clap::llama_cache::Error& error) {
    stale_rejected = error.status == CLAP_CACHE_STALE_PLAN;
  }
  assert(stale_rejected);

  Coordinator independent_capacity(1, 2, 1, 1);
  auto oversized = independent_capacity.plan(original, first, 0, 1);
  assert(oversized.view().operation == CLAP_CACHE_OPERATION_FRESH);
  oversized.abort();

  Coordinator saturated(2, 2, 128, 2);
  std::vector<int32_t> harness(32);
  for (int32_t i = 0; i < 32; ++i) harness[static_cast<std::size_t>(i)] = i;
  auto harness_seed = saturated.plan(harness, first, CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 1);
  const uint32_t harness_slot = harness_seed.view().target.slot;
  harness_seed.commit(0, CLAP_CACHE_SLOT_SESSION);
  auto harness_info = saturated.slot(harness_slot);
  clap_cache_slot_ref_t harness_ref{harness_slot, 0, harness_info.generation};
  harness_ref.generation = saturated.advance(harness_ref, harness.data(), harness.size(),
                                             CLAP_CACHE_SLOT_SESSION, true);
  saturated.set_busy(harness_ref, false);
  auto other_seed = saturated.plan(original, isolated, 0, 1);
  const uint32_t other_slot = other_seed.view().target.slot;
  other_seed.commit(0, CLAP_CACHE_SLOT_SESSION);
  auto other_info = saturated.slot(other_slot);
  clap_cache_slot_ref_t other_ref{other_slot, 0, other_info.generation};
  other_ref.generation = saturated.advance(other_ref, original.data(), original.size(),
                                           CLAP_CACHE_SLOT_SESSION, true);
  saturated.set_busy(other_ref, false);
  std::vector<int32_t> continued = harness;
  continued.insert(continued.end(), {40, 41, 42, 43});
  Identity continuation = first;
  continuation.session = clap::llama_cache::hash("session-continuation");
  std::vector<uint8_t> materialization(2, 0);
  materialization[harness_slot] = CLAP_CACHE_SLOT_MATERIALIZED |
      CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM | CLAP_CACHE_SLOT_COPY;
  materialization[other_slot] = CLAP_CACHE_SLOT_MATERIALIZED |
      CLAP_CACHE_SLOT_WRITABLE | CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM |
      CLAP_CACHE_SLOT_COPY;
  auto reuse = saturated.plan(continued, continuation,
      CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 128 - continued.size(),
      CLAP_CACHE_SLOT_SESSION, materialization);
  assert(reuse.view().operation == CLAP_CACHE_OPERATION_BRANCH);
  assert(reuse.view().has_donor == 1);
  assert(reuse.view().reuse_tokens == harness.size());
  reuse.abort();

  materialization[harness_slot] = 0;
  materialization[other_slot] = CLAP_CACHE_SLOT_WRITABLE;
  auto no_ghost = saturated.plan(continued, continuation,
      CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 128 - continued.size(),
      CLAP_CACHE_SLOT_SESSION, materialization);
  assert(no_ghost.view().operation == CLAP_CACHE_OPERATION_FRESH);
  assert(no_ghost.view().has_donor == 0);
  assert(no_ghost.view().reuse_tokens == 0);
  no_ghost.abort();
  assert(saturated.telemetry().read_leases == 0);
  assert(saturated.telemetry().write_leases == 0);

  Coordinator retained(1, 2, 4096, 4, 24);
  for (uint32_t expected = 1; expected < 24; ++expected) {
    const auto registered = retained.register_slot();
    assert(registered.slot == expected);
    assert(registered.generation == 1);
  }
  assert(retained.retention_telemetry().total_slots == 24);
  assert(clap::llama_cache::can_admit(0, 3));
  assert(clap::llama_cache::can_admit(2, 3));
  assert(!clap::llama_cache::can_admit(3, 3));

  std::vector<clap_cache_slot_ref_t> retained_refs;
  for (int32_t index = 0; index < 20; ++index) {
    Identity identity = first;
    identity.name_space = clap::llama_cache::fingerprint(
        "retained-domain-" + std::to_string(index));
    identity.session = clap::llama_cache::hash("retained-session-" + std::to_string(index));
    const std::vector<int32_t> tokens{1000 + index, 2, 3, 4};
    auto plan = retained.plan(tokens, identity, CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 1);
    assert(plan.evictions().empty());
    const uint32_t slot = plan.view().target.slot;
    plan.commit(0, CLAP_CACHE_SLOT_SESSION);
    auto info = retained.slot(slot);
    clap_cache_slot_ref_t ref{slot, 0, info.generation};
    ref.generation = retained.advance(ref, tokens.data(), tokens.size(),
                                      CLAP_CACHE_SLOT_SESSION, true);
    retained.set_busy(ref, false);
    retained_refs.push_back(ref);
  }
  const auto below_pressure = retained.retention_telemetry();
  assert(below_pressure.active_slots == 20);
  assert(below_pressure.session_slots == 20);
  assert(below_pressure.total_bytes == 0);
  assert(below_pressure.under_pressure == 0);
  assert(retained.telemetry().evictions == 0);

  Identity shared = first;
  shared.name_space = clap::llama_cache::fingerprint("retained-domain-0");
  shared.session = clap::llama_cache::hash("shared-branch");
  const std::vector<int32_t> shared_tokens{1000, 2, 3, 9};
  auto shared_branch = retained.plan(shared_tokens, shared,
      CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH | CLAP_CACHE_CAP_UNIFIED_STORAGE, 1);
  assert(shared_branch.view().operation == CLAP_CACHE_OPERATION_BRANCH);
  assert(shared_branch.view().reuse_tokens == 3);
  assert(shared_branch.evictions().empty());
  shared_branch.abort();
  assert(retained.retention_telemetry().total_bytes == 0);

  for (const auto ref : retained_refs) retained.invalidate(ref);
  assert(retained.retention_telemetry().active_slots == 0);
  assert(retained.telemetry().read_leases == 0);
  assert(retained.telemetry().write_leases == 0);
  const uint64_t slot_zero_generation = retained.slot(0).generation;
  retained.reset();
  assert(retained.slot(0).generation > slot_zero_generation);
  assert(retained.retention_telemetry().total_slots == 24);

  bool manager_unavailable = false;
  try {
    Coordinator invalid(0, 1, 1, 0);
    (void) invalid;
  } catch (const clap::llama_cache::Error& error) {
    manager_unavailable = error.status == CLAP_CACHE_INVALID_ARGUMENT;
  }
  assert(manager_unavailable);

  const auto before_reset = cache.telemetry();
  assert(before_reset.commits == 2);
  assert(before_reset.read_leases == 0);
  assert(before_reset.write_leases == 0);
  const uint64_t epoch = cache.reset();
  assert(epoch > 1);
  assert(cache.telemetry().active_slots == 0);

  using clap::llama_active::Inputs;
  const auto small = clap::llama_active::select(Inputs{
      0, UINT64_C(2) << 30, UINT64_C(8) << 30, 12, 32768, 64, false, false});
  assert(small.mode == "auto");
  assert(small.selected_max == 2);
  const auto large = clap::llama_active::select(Inputs{
      0, UINT64_C(64) << 30, UINT64_C(4) << 30, 32, 131072, 64, false, false});
  assert(large.selected_max == 8);
  const auto fixed = clap::llama_active::select(Inputs{
      6, UINT64_C(64) << 30, UINT64_C(4) << 30, 32, 131072, 64, false, false});
  assert(fixed.mode == "fixed" && fixed.selected_max == 6);
  const auto hybrid = clap::llama_active::select(Inputs{
      16, UINT64_C(64) << 30, UINT64_C(4) << 30, 32, 131072, 64, true, false});
  assert(hybrid.selected_max == 1);
  const auto unknown = clap::llama_active::select(Inputs{
      0, 0, 0, 32, 8192, 64, false, false});
  assert(unknown.selected_max >= 1 && unknown.selected_max <= 2);

  const std::vector<int32_t> boundary_tokens{11, 22, 33};
  bool empty_hashed = false;
  const auto exact_boundary = clap::llama_boundary::exact(
      boundary_tokens, 2, "prompt", [&empty_hashed](const auto&, std::size_t count) {
        if (count == 0) empty_hashed = true;
        return std::string("boundary-hash");
      });
  assert(exact_boundary.available());
  assert(exact_boundary.token_hash == "boundary-hash");
  assert(exact_boundary.token_count == 2);
  assert(exact_boundary.kind == "prompt");
  const auto unavailable_boundary = clap::llama_boundary::exact(
      boundary_tokens, 0, "prompt", [&empty_hashed](const auto&, std::size_t count) {
        if (count == 0) empty_hashed = true;
        return std::string("empty-boundary-hash");
      });
  assert(!unavailable_boundary.available());
  assert(unavailable_boundary.token_hash.empty());
  assert(unavailable_boundary.token_count == 0);
  assert(unavailable_boundary.kind.empty());
  assert(!empty_hashed);
  return 0;
}
