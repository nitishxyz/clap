#include "clap/llama/cache-executor.h"

#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Operation {
  std::string name;
  std::vector<int32_t> arguments;
};

class RecordingBackend final : public clap::llama::PhysicalCacheBackend {
 public:
  bool remove(int32_t sequence, int32_t begin, int32_t end) override {
    operations.push_back({"remove", {sequence, begin, end}});
    if (fail_remove) throw std::runtime_error("remove failed");
    return remove_result;
  }

  void copy(int32_t source, int32_t target, int32_t begin, int32_t end) override {
    operations.push_back({"copy", {source, target, begin, end}});
  }

  void clear(bool data) override {
    operations.push_back({"clear", {data ? 1 : 0}});
  }

  std::vector<Operation> operations;
  bool remove_result = true;
  bool fail_remove = false;
};

clap::llama::CacheExecutorConfig config() {
  clap::llama::CacheExecutorConfig value;
  value.slot_count = 3;
  value.logical_token_capacity = 64;
  value.max_anchors = 2;
  value.hard_max_retained_entries = 3;
  return value;
}

}  // namespace

int main() {
  auto backend = std::make_unique<RecordingBackend>();
  RecordingBackend* recording = backend.get();
  clap::llama::CacheExecutor executor(config(), std::move(backend));
  assert(executor.slot_count() == 3);
  assert(!executor.slot(0).busy);
  assert(executor.slot(0).generation > 0);

  {
    auto lease = executor.acquire(1);
    assert(lease);
    assert(lease.slot() == 1);
    assert(executor.slot(1).busy);
    try {
      executor.acquire(1);
      assert(false);
    } catch (const std::runtime_error&) {
    }
  }
  assert(!executor.slot(1).busy);

  assert(executor.remove_sequence(2, 3, 4));
  executor.copy_sequence(1, 2, 5, 6);
  executor.clear_physical(false);
  assert(recording->operations.size() == 3);
  assert(recording->operations[0].name == "remove");
  assert((recording->operations[0].arguments == std::vector<int32_t>{2, 3, 4}));
  assert(recording->operations[1].name == "copy");
  assert(recording->operations[2].name == "clear");

  recording->fail_remove = true;
  try {
    executor.remove_sequence(0, -1, -1);
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "remove failed");
  }
  recording->fail_remove = false;

  clap::llama_cache::Identity identity;
  identity.name_space = clap::llama_cache::fingerprint("executor-test");
  identity.tenant = clap::llama_cache::hash("tenant");
  identity.scope = CLAP_CACHE_SCOPE_TENANT;
  const clap::llama::CacheAdmissionRequest admission{
      {1, 2, 3, 4}, identity, CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH,
      1, CLAP_CACHE_SLOT_SESSION, {}, {2}};
  const auto result = executor.preview(admission);
  assert(result.operation == CLAP_CACHE_OPERATION_FRESH);
  assert(result.target_slot < executor.slot_count());

  const uint64_t epoch = executor.reset();
  assert(epoch > 0);
  assert(recording->operations.back().name == "clear");
  assert(recording->operations.back().arguments[0] == 1);
  for (uint32_t slot = 0; slot < executor.slot_count(); ++slot) {
    assert(!executor.slot(slot).busy);
    assert(executor.slot(slot).resident_tokens == 0);
  }

  auto transactional_backend = std::make_unique<RecordingBackend>();
  RecordingBackend* transactional_recording = transactional_backend.get();
  auto transactional_config = config();
  transactional_config.slot_count = 2;
  transactional_config.hard_max_retained_entries = 2;
  transactional_config.max_anchors = 0;
  clap::llama::CacheExecutor transactional(
      transactional_config, std::move(transactional_backend));
  const auto admit_fresh = [&](const std::string& name, std::function<void()> hook = {}) {
    clap::llama_cache::Identity request_identity;
    request_identity.name_space = clap::llama_cache::fingerprint(name);
    request_identity.tenant = clap::llama_cache::hash(name);
    request_identity.scope = CLAP_CACHE_SCOPE_TENANT;
    return transactional.admit({
        std::vector<int32_t>(20, static_cast<int32_t>(name.size())), request_identity,
        CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 1, CLAP_CACHE_SLOT_SESSION, {}, {},
        false, std::move(hook)});
  };
  for (const std::string name : {"first", "second"}) {
    const auto admitted = admit_fresh(name);
    transactional.coordinator().set_busy(
        {admitted.target_slot, 0, admitted.target_generation}, false);
    transactional.slots()[admitted.target_slot].busy = false;
  }
  transactional_recording->operations.clear();
  try {
    clap::llama_cache::Identity invalid_identity;
    invalid_identity.name_space = clap::llama_cache::fingerprint("invalid");
    transactional.admit({std::vector<int32_t>(20, 7), invalid_identity,
        CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH, 1, CLAP_CACHE_SLOT_SESSION,
        {CLAP_CACHE_SLOT_WRITABLE}, {}, false, {}});
    assert(false);
  } catch (const clap::llama_cache::Error&) {
  }
  assert(transactional_recording->operations.empty());

  transactional_recording->fail_remove = true;
  try {
    admit_fresh("materialization-failure");
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "remove failed");
  }
  transactional_recording->fail_remove = false;
  assert(transactional_recording->operations.size() == 2);
  assert(transactional_recording->operations[0].arguments[0] ==
         transactional_recording->operations[1].arguments[0]);

  transactional_recording->operations.clear();
  try {
    admit_fresh("commit-failure", [] { throw std::runtime_error("commit failed"); });
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "commit failed");
  }
  // Before commit, only the selected target is prepared and cleaned up. No
  // planned victim may be physically reconciled.
  assert(transactional_recording->operations.size() == 2);
  assert(transactional_recording->operations[0].name == "remove");
  assert(transactional_recording->operations[1].name == "remove");
  assert(transactional_recording->operations[0].arguments[0] ==
         transactional_recording->operations[1].arguments[0]);

  transactional_recording->operations.clear();
  const auto successful = admit_fresh("successful-eviction");
  assert(!successful.eviction_slots.empty());
  bool reconciled_victim = false;
  for (const uint32_t victim : successful.eviction_slots) {
    if (victim == successful.target_slot) continue;
    reconciled_victim = true;
    int clears = 0;
    for (const auto& operation : transactional_recording->operations) {
      if (operation.name == "remove" && operation.arguments ==
          std::vector<int32_t>{static_cast<int32_t>(victim), -1, -1}) ++clears;
    }
    assert(clears == 1);
  }
  assert(reconciled_victim);
}
