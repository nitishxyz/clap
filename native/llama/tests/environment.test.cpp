#include "clap/llama/environment.h"

#include <cassert>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

constexpr const char* kVariable = "CLAP_LLAMA_ENVIRONMENT_TEST_VALUE";

void set_value(const char* value) {
  assert(setenv(kVariable, value, 1) == 0);
}

void clear_value() {
  assert(unsetenv(kVariable) == 0);
}

}  // namespace

int main() {
  clear_value();
  assert(!clap::llama::env_enabled(kVariable));
  assert(clap::llama::env_u64(kVariable, 73) == 73);
  assert(clap::llama::env_int(kVariable, 37) == 37);

  set_value("");
  assert(!clap::llama::env_enabled(kVariable));
  assert(clap::llama::env_u64(kVariable, 73) == 73);
  assert(clap::llama::env_int(kVariable, 37) == 37);

  set_value("0");
  assert(!clap::llama::env_enabled(kVariable));
  assert(clap::llama::env_u64(kVariable, 73) == 0);
  assert(clap::llama::env_int(kVariable, 37) == 0);

  set_value("00");
  assert(clap::llama::env_enabled(kVariable));
  set_value("false");
  assert(clap::llama::env_enabled(kVariable));

  set_value("18446744073709551615");
  assert(clap::llama::env_u64(kVariable, 73) == std::numeric_limits<uint64_t>::max());
  set_value("42trailing");
  assert(clap::llama::env_u64(kVariable, 73) == 42);
  assert(clap::llama::env_int(kVariable, 37) == 42);

  set_value("invalid");
  assert(clap::llama::env_u64(kVariable, 73) == 73);
  assert(clap::llama::env_int(kVariable, 37) == 37);
  set_value("18446744073709551616");
  assert(clap::llama::env_u64(kVariable, 73) == 73);
  set_value("2147483648");
  assert(clap::llama::env_int(kVariable, 37) == 37);

  set_value("-17");
  assert(clap::llama::env_int(kVariable, 37) == -17);

  (void)clap::llama::available_memory_bytes();
  clear_value();
}
