#include "clap/llama/request-state.h"

#include <cassert>
#include <memory>
#include <type_traits>

namespace {

int deleted = 0;
void count_delete(llama_sampler*) { ++deleted; }

class Backend final : public clap::llama::PhysicalCacheBackend {
 public:
  bool remove(int32_t, int32_t, int32_t) override {
    ++removes;
    return true;
  }
  void copy(int32_t, int32_t, int32_t, int32_t) override {}
  void clear(bool) override {}
  int removes = 0;
};

}  // namespace

static_assert(!std::is_copy_constructible_v<clap::llama::PreparedRequest>);
static_assert(std::is_move_constructible_v<clap::llama::PreparedRequest>);
static_assert(!std::is_copy_constructible_v<clap::llama::ActiveRequest>);
static_assert(std::is_move_constructible_v<clap::llama::ActiveRequest>);

int main() {
  {
    clap::llama::SamplerOwner sampler(
        reinterpret_cast<llama_sampler*>(1), count_delete);
    clap::llama::SamplerOwner moved(std::move(sampler));
    assert(moved);
  }
  assert(deleted == 1);

  clap::llama::CacheExecutorConfig config;
  config.slot_count = 1;
  config.logical_token_capacity = 64;
  clap::llama::CacheExecutor executor(config, std::make_unique<Backend>());
  auto lease = executor.acquire(0);
  clap::llama::PreparedRequest prepared;
  prepared.id = "move-only";
  prepared.cache_lease = std::move(lease);
  clap::llama::PreparedRequest moved(std::move(prepared));
  assert(moved.cache_lease);
  assert(executor.slot(0).busy);
  moved.cache_lease.release();
  assert(!executor.slot(0).busy);

  clap::llama::ActiveRequest active;
  assert(active.phase == clap::llama::ActiveRequest::Phase::Prefill);
  assert(active.ingested == 0);
  assert(active.completion_tokens == 0);
  assert(!active.terminal());
  assert(active.mark_terminal(clap::llama::ActiveRequest::TerminalState::Completed));
  assert(active.done);
  assert(!active.mark_terminal(clap::llama::ActiveRequest::TerminalState::Failed));
  assert(active.terminal_state == clap::llama::ActiveRequest::TerminalState::Completed);

  {
    clap::llama::ActiveRequest completed;
    completed.id = "complete";
    completed.seq = 4;
    completed.prompt_token_count = 7;
    completed.completion_tokens = 3;
    completed.cached_prompt_tokens = 0;
    completed.stop_buffer.reset({"hello"});
    assert(completed.stop_buffer.append("hel").visible.empty());
    auto facts = completed.complete(true, [](const auto&, std::size_t count) {
      return "hash-" + std::to_string(count);
    });
    assert(facts);
    assert(facts->visible_tail == "hel");
    assert(facts->usage.prompt_tokens == 7);
    assert(facts->usage.completion_tokens == 3);
    assert(!facts->cache.hit);
    assert(facts->cache.miss_reason == "no_shared_prefix");
    assert(!completed.complete(true, {}));
  }

  {
    clap::llama::ActiveRequest cancelled;
    cancelled.cancelled = true;
    cancelled.finish_reason = "cancel";
    cancelled.stop_buffer.reset({"hello"});
    cancelled.stop_buffer.append("hel");
    auto facts = cancelled.complete(false, {});
    assert(facts && facts->visible_tail.empty());
    assert(facts->cancelled);
  }

  {
    auto backend = std::make_unique<Backend>();
    Backend* observed = backend.get();
    clap::llama::CacheExecutor failure_executor(config, std::move(backend));
    clap::llama::ActiveRequest failed;
    failed.id = "failed";
    failed.seq = 0;
    failed.cache_lease = failure_executor.acquire(0);
    auto facts = failed.fail("decode failed", "decode_error");
    assert(facts);
    assert(facts->message == "decode failed");
    assert(facts->code == "decode_error");
    assert(facts->invalidated_slot == 0);
    assert(observed->removes == 1);
    assert(!failure_executor.slot(0).busy);
    assert(!failed.fail("again"));
  }
}
