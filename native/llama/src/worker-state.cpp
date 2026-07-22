#include "clap/llama/worker-state.h"

#include "clap/llama/environment.h"
#include "clap/llama/request-preparer.h"
#include "clap/llama/sampling.h"
#include "clap/llama/telemetry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace clap::llama {

WorkerState::~WorkerState() {
  try {
    unload();
  } catch (...) {
  }
}

int32_t WorkerState::batch_capacity() const {
  return runtime_.loaded() ? static_cast<int32_t>(llama_n_batch(runtime_.context())) : 0;
}

void WorkerState::unload() {
  if (cache_executor_) cache_executor_->reset();
  cache_executor_.reset();
  slots_.clear();
  runtime_.reset();
  max_active_ = 0;
  active_policy_ = {};
  last_eviction_reason_.clear();
}

void WorkerState::load(const std::string& model_path) {
  if (runtime_.same_path(model_path)) return;
  unload();
  runtime_.load(model_path);
  const int32_t context = runtime_.backend_allocation_cap();
  const int32_t retained = runtime_.retained_max();
  slots_.assign(static_cast<std::size_t>(retained), {});
  use_counter_ = 0;
  active_policy_ = clap::llama_active::select({
      env_int("CLAP_MAX_ACTIVE", 0), runtime_.startup_available_bytes(),
      runtime_.model_file_bytes(),
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency())), context,
      retained, runtime_.hybrid(), runtime_.has_encoder()});
  max_active_ = active_policy_.selected_max;
  try {
    CacheExecutorConfig config;
    config.slot_count = static_cast<uint32_t>(retained);
    config.logical_token_capacity = static_cast<uint64_t>(context);
    config.max_anchors = static_cast<uint32_t>(retained);
    config.hard_max_retained_entries = static_cast<uint32_t>(retained);
    config.automatic_checkpoints = env_int("CLAP_CACHE_CHECKPOINTS_ENABLED", 1) != 0;
    config.checkpoint_minimum_tokens = static_cast<uint64_t>(
        env_int("CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS", 2048));
    config.checkpoint_interval_tokens = static_cast<uint64_t>(
        env_int("CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS", 2048));
    config.checkpoint_max = static_cast<uint32_t>(env_int("CLAP_CACHE_CHECKPOINT_MAX", 8));
    config.checkpoint_budget_basis_points = static_cast<uint32_t>(
        env_int("CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS", 2500));
    config.checkpoint_budget_bytes = env_u64("CLAP_CACHE_CHECKPOINT_BUDGET_BYTES", 0);
    cache_executor_ = std::make_unique<CacheExecutor>(config,
        std::make_unique<LlamaPhysicalCacheBackend>(runtime_.context()));
  } catch (const std::exception& error) {
    cache_executor_.reset();
    fprintf(stderr,
        "clap-llama: cache coordinator unavailable; using no-cache fresh mode: %s\n",
        error.what());
  }
}

int32_t WorkerState::set_max_active(const MaxActiveUpdate& update) {
  if (update.requested <= 0) {
    throw std::runtime_error("set_max_active.max_active must be positive");
  }
  max_active_ = std::max(1, std::min({update.requested,
      active_policy_.backend_ceiling, active_policy_.hardware_ceiling,
      active_policy_.model_ceiling, active_policy_.context_ceiling}));
  active_policy_.selected_max = max_active_;
  previous_max_active_ = update.previous_max_active;
  if (!update.limiting_reason.empty()) active_policy_.reason = update.limiting_reason;
  last_adjustment_reason_ = update.adjustment_reason;
  last_adjustment_at_ = update.adjustment_at;
  retained_growth_reserve_bytes_ = update.retained_growth_reserve_bytes;
  global_resident_memory_bytes_ = update.global_resident_memory_bytes;
  pressure_state_ = update.pressure_state;
  return max_active_;
}

const std::string& WorkerState::telemetry_key() {
  static const std::string key = [] {
    if (const char* installed = std::getenv("CLAP_TELEMETRY_HMAC_KEY"); installed && *installed) {
      return std::string(installed);
    }
    std::random_device random;
    std::ostringstream out;
    for (int index = 0; index < 8; ++index) out << std::hex << random();
    return out.str();
  }();
  return key;
}

std::string WorkerState::fingerprint(const std::vector<llama_token>& tokens,
                                     std::size_t count) {
  count = std::min(count, tokens.size());
  std::ostringstream encoded;
  encoded << telemetry_key() << "|tokens-v1|" << count << '|';
  for (std::size_t index = 0; index < count; ++index) {
    const uint32_t token = static_cast<uint32_t>(tokens[index]);
    encoded.write(reinterpret_cast<const char*>(&token), sizeof(token));
  }
  const std::string material = encoded.str();
  std::ostringstream result;
  for (int domain = 0; domain < 4; ++domain) {
    result << std::hex << clap::llama_cache::hash(std::to_string(domain) + material);
  }
  return result.str();
}

std::unique_ptr<ActiveRequest> WorkerState::prepare(
    const std::string& id, const nlohmann::json& request) {
  std::vector<RequestSlotState> slots;
  slots.reserve(slots_.size());
  for (const auto& slot : slots_) {
    slots.push_back({slot.tokens, slot.coordinator_generation, slot.busy, slot.is_anchor});
  }
  RequestPreparer preparer(runtime_, cache_executor_.get(), std::move(slots),
      telemetry_key(), fingerprint);
  auto prepared = preparer.prepare(id, request);
  const std::size_t target = static_cast<std::size_t>(prepared.sequence);
  if (runtime_.has_encoder()) {
    for (auto& slot : slots_) slot = {};
  }
  for (const uint32_t victim : prepared.cache_evicted_slots) {
    if (victim != target && victim < slots_.size()) slots_[victim] = {};
    last_eviction_reason_ = "hard_ceiling";
  }
  if (target < slots_.size()) {
    auto& slot = slots_[target];
    slot.tokens.assign(prepared.full_prompt_tokens.begin(),
        prepared.full_prompt_tokens.begin() + prepared.initial_position);
    slot.coordinator_generation = prepared.cache_target_generation;
    slot.last_used = ++use_counter_;
    slot.busy = true;
    slot.is_anchor = false;
  }
  auto active = std::make_unique<ActiveRequest>(std::move(prepared));
  active->sampler.reset(make_sampler(active->params));
  return active;
}

void WorkerState::reconcile(const RequestCompletion& completion) {
  if (completion.released_slot >= 0) {
    slots_[static_cast<std::size_t>(completion.released_slot)].busy = false;
  }
}

void WorkerState::reconcile(const RequestFailure& failure) {
  if (failure.invalidated_slot < 0) return;
  auto& slot = slots_[static_cast<std::size_t>(failure.invalidated_slot)];
  slot.busy = false;
  slot.tokens.clear();
  slot.coordinator_generation = failure.generation;
}

void WorkerState::reconcile(const GenerationEvent& event) {
  switch (event.type) {
    case GenerationEvent::Type::Complete:
      if (event.completion) reconcile(*event.completion);
      break;
    case GenerationEvent::Type::Failure:
      if (event.failure) reconcile(*event.failure);
      break;
    case GenerationEvent::Type::CacheAppend: {
      auto& slot = slots_[event.slot];
      slot.tokens.insert(slot.tokens.end(), event.tokens.begin(), event.tokens.end());
      slot.coordinator_generation = event.generation;
      break;
    }
    case GenerationEvent::Type::CacheResetSlot: {
      auto& slot = slots_[event.slot];
      if (event.clear) slot.tokens.clear();
      slot.coordinator_generation = event.generation;
      break;
    }
    case GenerationEvent::Type::CacheResetAll:
      for (auto& slot : slots_) {
        slot.tokens.clear();
        slot.is_anchor = false;
      }
      slots_[static_cast<std::size_t>(event.request->seq)].coordinator_generation =
          event.request->cache_lease.generation();
      break;
    case GenerationEvent::Type::CacheAnchor:
      if (event.anchor && event.slot < slots_.size()) {
        auto& anchor = slots_[event.slot];
        anchor.tokens = event.tokens;
        anchor.is_anchor = true;
        anchor.last_used = ++use_counter_;
        anchor.coordinator_generation = event.generation;
      }
      for (const uint32_t victim : event.eviction_slots) {
        if (victim != event.slot && victim < slots_.size()) slots_[victim] = {};
        last_eviction_reason_ = "hard_ceiling";
      }
      break;
    default:
      break;
  }
}

std::vector<GenerationEvent> WorkerState::step(
    const std::vector<ActiveRequest*>& ordered, bool sole_active) {
  GenerationStepper stepper(runtime_, cache_executor_.get(), fingerprint);
  auto events = stepper.step(ordered, batch_capacity(), sole_active);
  for (const auto& event : events) reconcile(event);
  return events;
}

std::optional<RequestCompletion> WorkerState::complete(
    ActiveRequest& request, bool flush_tail) {
  auto completion = request.complete(flush_tail, fingerprint);
  if (completion) reconcile(*completion);
  return completion;
}

nlohmann::json WorkerState::retention(std::size_t active, std::size_t queued) const {
  uint32_t retained_total = 0;
  uint32_t retained_sessions = 0;
  uint32_t retained_anchors = 0;
  uint64_t evictions = 0;
  if (cache_executor_) {
    const auto retention = cache_executor_->retention_telemetry();
    const auto telemetry = cache_executor_->telemetry();
    retained_total = retention.active_slots;
    retained_sessions = retention.session_slots;
    retained_anchors = retention.anchor_slots;
    evictions = telemetry.evictions;
  }
  return serialize_retention_telemetry({
    max_active_, queued, previous_max_active_, last_adjustment_reason_,
    last_adjustment_at_, retained_growth_reserve_bytes_, global_resident_memory_bytes_,
    pressure_state_, {active_policy_.mode, active_policy_.selected_max,
      active_policy_.backend_ceiling, active_policy_.hardware_ceiling,
      active_policy_.model_ceiling, active_policy_.memory_ceiling, active_policy_.reason,
      runtime_.startup_available_bytes(), runtime_.model_file_bytes(),
      runtime_.backend_allocation_cap(), active_policy_.context_ceiling,
      active_policy_.per_active_reserve_cells, active_policy_.per_active_reserve_bytes,
      std::max(1u, std::thread::hardware_concurrency()), runtime_.hybrid()},
    active, retained_total, retained_sessions, retained_anchors, runtime_.retained_max(),
    last_eviction_reason_, evictions, runtime_.backend_allocation_cap()});
}

}  // namespace clap::llama
