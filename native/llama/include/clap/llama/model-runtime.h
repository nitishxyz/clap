#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

namespace clap::llama {

class ModelRuntime {
 public:
  ModelRuntime() = default;
  ~ModelRuntime();

  ModelRuntime(const ModelRuntime&) = delete;
  ModelRuntime& operator=(const ModelRuntime&) = delete;
  ModelRuntime(ModelRuntime&&) = delete;
  ModelRuntime& operator=(ModelRuntime&&) = delete;

  bool load(const std::string& model_path);
  void reset() noexcept;

  bool loaded() const noexcept { return model_ != nullptr && context_ != nullptr; }
  bool same_path(const std::string& model_path) const noexcept;
  llama_model* model() const noexcept { return model_; }
  llama_context* context() const noexcept { return context_; }
  const llama_vocab* vocab() const noexcept;
  const std::string& model_path() const noexcept { return model_path_; }
  int32_t model_context_window() const noexcept { return model_context_window_; }
  int32_t backend_allocation_cap() const noexcept { return backend_allocation_cap_; }
  int32_t context_override() const noexcept { return context_override_; }
  int32_t max_output_tokens() const noexcept { return max_output_tokens_; }
  int32_t retained_max() const noexcept { return retained_max_; }
  uint64_t startup_available_bytes() const noexcept { return startup_available_bytes_; }
  uint64_t model_file_bytes() const noexcept { return model_file_bytes_; }
  bool hybrid() const noexcept { return hybrid_; }
  bool prompt_boundary_snapshots() const noexcept { return prompt_boundary_snapshots_; }
  bool has_encoder() const noexcept { return has_encoder_; }
  const std::string& cache_domain() const noexcept { return cache_domain_; }
  const std::string& kv_format() const noexcept { return kv_format_; }
  bool unified_kv() const noexcept { return unified_kv_; }

 private:
  llama_model* model_ = nullptr;
  llama_context* context_ = nullptr;
  std::string model_path_;
  int32_t model_context_window_ = 0;
  int32_t backend_allocation_cap_ = 0;
  int32_t context_override_ = 0;
  int32_t max_output_tokens_ = 0;
  int32_t retained_max_ = 0;
  uint64_t startup_available_bytes_ = 0;
  uint64_t model_file_bytes_ = 0;
  bool hybrid_ = false;
  bool prompt_boundary_snapshots_ = false;
  bool has_encoder_ = false;
  std::string cache_domain_;
  std::string kv_format_;
  bool unified_kv_ = true;
};

}  // namespace clap::llama
