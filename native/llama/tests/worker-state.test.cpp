#include "clap/llama/worker-state.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<clap::llama::WorkerState>);
static_assert(!std::is_copy_assignable_v<clap::llama::WorkerState>);

int main() {
  clap::llama::WorkerState worker;
  assert(!worker.loaded());
  assert(worker.model_path().empty());
  assert(!worker.has_encoder());
  assert(worker.max_active() == 0);
  assert(worker.batch_capacity() == 0);

  const auto unloaded = worker.retention(2, 3);
  assert(unloaded["active"] == 2);
  assert(unloaded["queued"] == 3);
  assert(unloaded["retained_total"] == 0);
  assert(unloaded["retained_sessions"] == 0);
  assert(unloaded["retained_anchors"] == 0);

  worker.unload();
  worker.unload();
  assert(!worker.loaded());

  assert(worker.set_max_active({4}) == 1);
  assert(worker.max_active() == 1);
  worker.unload();
  assert(worker.max_active() == 0);

  try {
    worker.set_max_active({0});
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "set_max_active.max_active must be positive");
  }

  try {
    worker.load("/definitely/missing/clap-worker-state.gguf");
    assert(false);
  } catch (const std::runtime_error&) {
    assert(!worker.loaded());
    assert(worker.model_path().empty());
    assert(worker.max_active() == 0);
    assert(worker.retained_capacity() == 0);
  }

  worker.unload();
}
