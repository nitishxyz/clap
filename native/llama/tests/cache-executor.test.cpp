#include "clap/llama/cache-executor.h"

#include <cassert>
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
}
