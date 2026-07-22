#include "clap/llama/request-state.h"

#include <cassert>
#include <memory>
#include <type_traits>

namespace {

int deleted = 0;
void count_delete(llama_sampler*) { ++deleted; }

class Backend final : public clap::llama::PhysicalCacheBackend {
 public:
  bool remove(int32_t, int32_t, int32_t) override { return true; }
  void copy(int32_t, int32_t, int32_t, int32_t) override {}
  void clear(bool) override {}
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
}
