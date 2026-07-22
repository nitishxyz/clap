#include "clap/llama/cache-executor.h"
#include "tests/support/cache-state-backend.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clap::llama::CacheAdmissionRequest;
using clap::llama::CacheAdmissionResult;
using clap::llama::CacheExecutor;
using clap::llama::test::CacheMutation;
using clap::llama::test::CacheStateBackend;

clap::llama::CacheExecutorConfig config(uint32_t slots = 4) {
  clap::llama::CacheExecutorConfig value;
  value.slot_count = slots;
  value.min_reuse_tokens = 8;
  value.logical_token_capacity = 256;
  value.max_anchors = slots > 2 ? 2 : 0;
  value.hard_max_retained_entries = slots;
  value.automatic_checkpoints = false;
  return value;
}

clap::llama_cache::Identity identity(const std::string& name_space,
                                     const std::string& session) {
  clap::llama_cache::Identity value;
  value.name_space = clap::llama_cache::fingerprint(name_space);
  value.tenant = clap::llama_cache::hash("authenticated-tenant");
  value.project = clap::llama_cache::hash("authenticated-project");
  value.session = clap::llama_cache::hash(session);
  value.scope = CLAP_CACHE_SCOPE_SESSION;
  return value;
}

std::vector<int32_t> tokens(int32_t first = 1) {
  std::vector<int32_t> value(32);
  for (std::size_t index = 0; index < value.size(); ++index) {
    value[index] = first + static_cast<int32_t>(index);
  }
  return value;
}

CacheAdmissionResult admit(CacheExecutor& executor, const std::vector<int32_t>& prompt,
                           const clap::llama_cache::Identity& request_identity,
                           std::function<void()> commit_hook = {}) {
  return executor.admit(CacheAdmissionRequest{prompt, request_identity,
      CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH | CLAP_CACHE_CAP_WHOLE_STATE_COPY |
          CLAP_CACHE_CAP_SAFE_BUSY_DONOR | CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT,
      1, CLAP_CACHE_SLOT_SESSION, {}, {}, false, std::move(commit_hook)});
}

uint64_t publish(CacheExecutor& executor, CacheStateBackend& physical,
                 const CacheAdmissionResult& admission,
                 const std::vector<int32_t>& prompt) {
  const auto suffix_begin = prompt.begin() + static_cast<std::ptrdiff_t>(admission.reuse_tokens);
  const std::vector<int32_t> suffix(suffix_begin, prompt.end());
  physical.append(admission.target_slot, suffix);
  const uint64_t generation = executor.advance(admission.target_slot,
      admission.target_generation, suffix.data(), suffix.size(), CLAP_CACHE_SLOT_SESSION, false);
  executor.release(admission.target_slot, generation);
  return generation;
}

void cold_continuation_and_branch_are_physically_consistent() {
  auto backend = std::make_unique<CacheStateBackend>(4);
  auto* physical = backend.get();
  CacheExecutor executor(config(), std::move(backend));
  const auto base = tokens();
  const auto shared_identity = identity("namespace-a", "conversation-a");

  const auto cold = admit(executor, base, shared_identity);
  assert(cold.operation == CLAP_CACHE_OPERATION_FRESH);
  assert(cold.reuse_tokens == 0);
  assert(physical->slot(cold.target_slot).empty());
  publish(executor, *physical, cold, base);
  assert(physical->slot(cold.target_slot) == base);

  auto continued_prompt = base;
  continued_prompt.back() = 900;
  const auto continued = admit(executor, continued_prompt, shared_identity);
  assert(continued.operation == CLAP_CACHE_OPERATION_CONTINUE);
  assert(continued.target_slot == cold.target_slot);
  assert(continued.reuse_tokens > 0 && continued.reuse_tokens < continued_prompt.size());
  assert(physical->slot(continued.target_slot) ==
         std::vector<int32_t>(continued_prompt.begin(),
                              continued_prompt.begin() + continued.reuse_tokens));
  publish(executor, *physical, continued, continued_prompt);

  const auto donor_before = physical->slot(continued.target_slot);
  const auto donor_generation = executor.slot(continued.target_slot).generation;
  executor.coordinator().set_busy(
      {continued.target_slot, 0, donor_generation}, true);
  executor.slots()[continued.target_slot].busy = true;
  auto branch_prompt = continued_prompt;
  branch_prompt.back() = 901;
  const auto branched = admit(executor, branch_prompt,
                              identity("namespace-a", "conversation-b"));
  assert(branched.operation == CLAP_CACHE_OPERATION_BRANCH);
  assert(branched.has_donor && branched.donor_slot == continued.target_slot);
  assert(branched.target_slot != branched.donor_slot);
  assert(physical->slot(branched.donor_slot) == donor_before);
  assert(physical->slot(branched.target_slot) ==
         std::vector<int32_t>(branch_prompt.begin(),
                              branch_prompt.begin() + branched.reuse_tokens));
  executor.release(branched.target_slot, branched.target_generation);
  executor.coordinator().set_busy(
      {continued.target_slot, 0, donor_generation}, false);
  executor.slots()[continued.target_slot].busy = false;
}

void anchors_restore_and_duplicate_materialization_is_a_no_op() {
  auto backend = std::make_unique<CacheStateBackend>(4);
  auto* physical = backend.get();
  CacheExecutor executor(config(), std::move(backend));
  const auto prompt = tokens(100);
  const auto request_identity = identity("namespace-anchor", "source");
  const auto source = admit(executor, prompt, request_identity);
  publish(executor, *physical, source, prompt);

  const auto anchor = executor.create_anchor(prompt, request_identity,
                                              source.target_slot, false);
  assert(anchor.materialized && !anchor.no_op);
  assert(anchor.target_slot != source.target_slot);
  assert(physical->slot(anchor.target_slot) == prompt);
  const std::size_t mutation_count = physical->mutations.size();
  const auto duplicate = executor.create_anchor(prompt, request_identity,
                                                 source.target_slot, false);
  assert(duplicate.materialized && duplicate.no_op);
  assert(physical->mutations.size() == mutation_count);

  const auto source_generation = executor.slot(source.target_slot).generation;
  executor.coordinator().set_busy({source.target_slot, 0, source_generation}, true);
  executor.slots()[source.target_slot].busy = true;
  auto restored_prompt = prompt;
  restored_prompt.push_back(999);
  const auto restored = admit(executor, restored_prompt,
                               identity("namespace-anchor", "restored"));
  assert(restored.operation == CLAP_CACHE_OPERATION_RESTORE);
  assert(restored.has_donor && restored.donor_slot == anchor.target_slot);
  assert(physical->slot(restored.target_slot) == prompt);
}

void authenticated_namespaces_never_share_physical_state() {
  auto backend = std::make_unique<CacheStateBackend>(2);
  auto* physical = backend.get();
  CacheExecutor executor(config(2), std::move(backend));
  const auto prompt = tokens(200);
  const auto first = admit(executor, prompt, identity("authenticated-a", "same-session"));
  publish(executor, *physical, first, prompt);
  const auto first_physical = physical->slot(first.target_slot);

  const auto isolated = admit(executor, prompt,
                               identity("authenticated-b", "same-session"));
  assert(isolated.operation == CLAP_CACHE_OPERATION_FRESH);
  assert(!isolated.has_donor && isolated.reuse_tokens == 0);
  assert(physical->slot(first.target_slot) == first_physical);
}

void physical_preparation_precedes_logical_publication() {
  auto backend = std::make_unique<CacheStateBackend>(1);
  auto* physical = backend.get();
  CacheExecutor executor(config(1), std::move(backend));
  const auto original = tokens(300);
  const auto seeded = admit(executor, original, identity("old-namespace", "old"));
  publish(executor, *physical, seeded, original);
  const auto old_logical = executor.coordinator().slot(seeded.target_slot);
  bool observed_old_state = false;
  physical->before_mutation = [&](const CacheMutation& mutation) {
    if (mutation.operation != "remove") return;
    const auto during = executor.coordinator().slot(seeded.target_slot);
    observed_old_state = during.generation == old_logical.generation &&
                         during.resident_len == old_logical.resident_len;
  };
  const auto replacement = admit(executor, tokens(400),
                                 identity("new-namespace", "new"));
  assert(observed_old_state);
  assert(replacement.operation == CLAP_CACHE_OPERATION_FRESH);
}

void failures_do_not_publish_or_leave_a_busy_lease() {
  {
    auto backend = std::make_unique<CacheStateBackend>(1);
    auto* physical = backend.get();
    CacheExecutor executor(config(1), std::move(backend));
    const auto before = executor.coordinator().slot(0);
    physical->fail_next_remove();
    try {
      admit(executor, tokens(500), identity("physical-failure", "one"));
      assert(false);
    } catch (const std::runtime_error& error) {
      assert(std::string(error.what()) == "test physical remove failed");
    }
    const auto after = executor.coordinator().slot(0);
    assert(after.resident_len == before.resident_len);
    assert(!executor.slot(0).busy && executor.slot(0).resident_tokens == 0);
  }
  {
    auto backend = std::make_unique<CacheStateBackend>(1);
    auto* physical = backend.get();
    CacheExecutor executor(config(1), std::move(backend));
    try {
      admit(executor, tokens(600), identity("commit-failure", "one"), [] {
        throw std::runtime_error("test coordinator commit failed");
      });
      assert(false);
    } catch (const std::runtime_error& error) {
      assert(std::string(error.what()) == "test coordinator commit failed");
    }
    assert(physical->slot(0).empty());
    assert(executor.coordinator().slot(0).resident_len == 0);
    assert(!executor.slot(0).busy && executor.slot(0).resident_tokens == 0);
    auto lease = executor.acquire(0);
    assert(lease);
  }
}

void victims_are_reconciled_only_after_commit() {
  auto pressure_config = config(2);
  pressure_config.logical_token_capacity = 64;
  auto backend = std::make_unique<CacheStateBackend>(2);
  auto* physical = backend.get();
  CacheExecutor executor(pressure_config, std::move(backend));
  for (const auto& name : {"first", "second"}) {
    const auto prompt = tokens(name[0] == 'f' ? 700 : 800);
    const auto result = admit(executor, prompt, identity(name, name));
    publish(executor, *physical, result, prompt);
  }

  int32_t prepared_target = -1;
  bool saw_victim_after_commit = false;
  physical->before_mutation = [&](const CacheMutation& mutation) {
    if (mutation.operation != "remove" || mutation.begin != -1) return;
    if (prepared_target < 0) {
      prepared_target = mutation.target;
      return;
    }
    if (mutation.target != prepared_target) {
      saw_victim_after_commit = executor.slot(static_cast<uint32_t>(prepared_target)).busy;
    }
  };
  const auto result = admit(executor, tokens(900), identity("third", "third"));
  assert(!result.eviction_slots.empty());
  assert(saw_victim_after_commit);
}

}  // namespace

int main() {
  cold_continuation_and_branch_are_physically_consistent();
  anchors_restore_and_duplicate_materialization_is_a_no_op();
  authenticated_namespaces_never_share_physical_state();
  physical_preparation_precedes_logical_publication();
  failures_do_not_publish_or_leave_a_busy_lease();
  victims_are_reconciled_only_after_commit();
}
