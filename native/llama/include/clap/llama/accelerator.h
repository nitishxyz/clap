#pragma once

// What this worker binary can actually offload to, and what it found at
// runtime. A CPU-only worker on a GPU box is a silent 5-10x slowdown, so the
// worker reports this at startup and the server surfaces it: an operator
// should never have to read llama.cpp's stderr to learn which build is live.

#include "ggml-backend.h"

#include <string>
#include <vector>

namespace clap::llama {

struct AcceleratorDevice {
  std::string name;
  std::string description;
  uint64_t total_memory_bytes = 0;
};

struct AcceleratorReport {
  // "cuda", "metal", "hip", "vulkan", "sycl", or "cpu" when this build has no
  // GPU backend compiled in at all.
  std::string compiled;
  std::vector<AcceleratorDevice> devices;
};

// The GPU backend this binary was compiled with, independent of whether any
// device is present. Compile-time so a driverless GPU build still reports
// honestly rather than looking like a CPU build.
inline const char* compiled_accelerator() noexcept {
#if defined(GGML_USE_CUDA)
  return "cuda";
#elif defined(GGML_USE_METAL)
  return "metal";
#elif defined(GGML_USE_HIP)
  return "hip";
#elif defined(GGML_USE_VULKAN)
  return "vulkan";
#elif defined(GGML_USE_SYCL)
  return "sycl";
#elif defined(GGML_USE_MUSA)
  return "musa";
#else
  return "cpu";
#endif
}

inline AcceleratorReport detect_accelerator() {
  AcceleratorReport report;
  report.compiled = compiled_accelerator();
  const std::size_t count = ggml_backend_dev_count();
  for (std::size_t index = 0; index < count; ++index) {
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    if (device == nullptr) continue;
    if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    const char* name = ggml_backend_dev_name(device);
    const char* description = ggml_backend_dev_description(device);
    report.devices.push_back({name ? name : "", description ? description : "",
        static_cast<uint64_t>(total_bytes)});
  }
  return report;
}

}  // namespace clap::llama
