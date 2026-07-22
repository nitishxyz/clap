#pragma once

#include "active-concurrency.h"
#include "clap/llama/cache-executor.h"
#include "clap/llama/generation-stepper.h"
#include "clap/llama/model-runtime.h"
#include "clap/llama/request-state.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace clap::llama {

struct MaxActiveUpdate {
  int requested = 0;
  int previous_max_active = 0;
  std::string limiting_reason;
  std::string adjustment_reason;
  std::string adjustment_at;
  uint64_t retained_growth_reserve_bytes = 0;
  uint64_t global_resident_memory_bytes = 0;
  std::string pressure_state;
};

class WorkerState {
 public:
  WorkerState() = default;
  WorkerState(const WorkerState&) = delete;
  WorkerState& operator=(const WorkerState&) = delete;
  ~WorkerState();

  bool loaded() const noexcept { return runtime_.loaded(); }
  bool same_path(const std::string& path) const { return runtime_.same_path(path); }
  const std::string& model_path() const noexcept { return runtime_.model_path(); }
  bool has_encoder() const noexcept { return runtime_.has_encoder(); }
  int32_t max_active() const noexcept { return max_active_; }
  int32_t batch_capacity() const;
  int32_t effective_context_window() const noexcept { return runtime_.backend_allocation_cap(); }
  int32_t model_context_window() const noexcept { return runtime_.model_context_window(); }
  int32_t max_output_tokens() const noexcept { return runtime_.max_output_tokens(); }
  int32_t context_override() const noexcept { return runtime_.context_override(); }
  int32_t retained_capacity() const noexcept { return runtime_.retained_max(); }

  void load(const std::string& model_path);
  void unload();
  int32_t set_max_active(const MaxActiveUpdate& update);

  std::unique_ptr<ActiveRequest> prepare(const std::string& id,
                                         const nlohmann::json& request);
  std::vector<GenerationEvent> step(const std::vector<ActiveRequest*>& ordered,
                                    bool sole_active);
  std::optional<RequestCompletion> complete(ActiveRequest& request, bool flush_tail);
  nlohmann::json retention(std::size_t active, std::size_t queued = 0) const;

 private:
  struct CacheSlot {
    std::vector<llama_token> tokens;
    uint64_t last_used = 0;
    uint64_t coordinator_generation = 0;
    bool busy = false;
    bool is_anchor = false;
  };

  static const std::string& telemetry_key();
  static std::string fingerprint(const std::vector<llama_token>& tokens,
                                 std::size_t count);
  void reconcile(const GenerationEvent& event);
  void reconcile(const RequestCompletion& completion);
  void reconcile(const RequestFailure& failure);

  ModelRuntime runtime_;
  int32_t max_active_ = 0;
  clap::llama_active::Decision active_policy_{};
  std::string last_eviction_reason_;
  int32_t previous_max_active_ = 0;
  std::string last_adjustment_reason_;
  std::string last_adjustment_at_;
  uint64_t retained_growth_reserve_bytes_ = 0;
  uint64_t global_resident_memory_bytes_ = 0;
  std::string pressure_state_;
  std::vector<CacheSlot> slots_;
  uint64_t use_counter_ = 0;
  std::unique_ptr<CacheExecutor> cache_executor_;
};

}  // namespace clap::llama
