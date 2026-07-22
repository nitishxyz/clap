#include "clap/llama/model-runtime.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<clap::llama::ModelRuntime>);
static_assert(!std::is_copy_assignable_v<clap::llama::ModelRuntime>);
static_assert(!std::is_move_constructible_v<clap::llama::ModelRuntime>);
static_assert(!std::is_move_assignable_v<clap::llama::ModelRuntime>);

int main() {
  clap::llama::ModelRuntime runtime;
  assert(!runtime.loaded());
  assert(runtime.model() == nullptr);
  assert(runtime.context() == nullptr);
  assert(runtime.vocab() == nullptr);
  assert(runtime.model_path().empty());
  assert(runtime.cache_domain().empty());
  assert(runtime.backend_allocation_cap() == 0);

  runtime.reset();
  runtime.reset();
  assert(!runtime.loaded());

  const std::string missing = "/definitely/missing/clap-model-runtime-test.gguf";
  try {
    runtime.load(missing);
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "GGUF model not found: " + missing);
  }
  assert(!runtime.loaded());
  assert(runtime.model() == nullptr);
  assert(runtime.context() == nullptr);
  assert(runtime.model_path().empty());
}
