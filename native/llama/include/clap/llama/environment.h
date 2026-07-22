#pragma once

#include <cstdint>

namespace clap::llama {

bool env_enabled(const char* name);
uint64_t available_memory_bytes();
uint64_t env_u64(const char* name, uint64_t fallback);
int env_int(const char* name, int fallback);

}  // namespace clap::llama
