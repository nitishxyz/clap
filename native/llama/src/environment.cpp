#include "clap/llama/environment.h"

#include <cstdlib>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace clap::llama {

bool env_enabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw && *raw && std::string(raw) != "0";
}

uint64_t available_memory_bytes() {
#if defined(__APPLE__)
  vm_statistics64_data_t statistics{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_size_t page_size = 0;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&statistics), &count) != KERN_SUCCESS ||
      host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS) return 0;
  const uint64_t pages = statistics.free_count + statistics.inactive_count +
      statistics.purgeable_count;
  return pages > UINT64_MAX / page_size ? UINT64_MAX : pages * page_size;
#elif defined(__linux__)
  struct sysinfo info {};
  if (sysinfo(&info) != 0) return 0;
  return static_cast<uint64_t>(info.freeram) * info.mem_unit;
#else
  return 0;
#endif
}

uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  try {
    return std::stoull(raw);
  } catch (...) {
    return fallback;
  }
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  try {
    return std::stoi(raw);
  } catch (...) {
    return fallback;
  }
}

}  // namespace clap::llama
