#include "llama.h"
#include "active-concurrency.h"
#include "cache-adapter.h"
#include "clap/llama/cache-executor.h"
#include "clap/llama/environment.h"
#include "clap/llama/generation-stepper.h"
#include "clap/llama/model-runtime.h"
#include "clap/llama/prompt.h"
#include "clap/llama/protocol.h"
#include "clap/llama/request-preparer.h"
#include "clap/llama/request-state.h"
#include "clap/llama/sampling.h"
#include "clap/llama/stop-buffer.h"
#include "clap/llama/telemetry.h"
#include "native-characterization.h"
#include "stable-boundary.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

using clap::llama::env_int;
using clap::llama::env_u64;
using clap::llama::emit;
using clap::llama::emit_error;
using clap::llama::make_sampler;
using clap::llama::RequestError;
using clap::llama::StdinReader;
using clap::llama::ActiveRequest;
using clap::llama::CacheBackpressure;

struct LoadedLlama {
  clap::llama::ModelRuntime runtime;
  int32_t max_active = 0;
  clap::llama_active::Decision active_policy{};
  std::string last_eviction_reason;
  int32_t previous_max_active = 0;
  std::string last_adjustment_reason;
  std::string last_adjustment_at;
  uint64_t retained_growth_reserve_bytes = 0;
  uint64_t global_resident_memory_bytes = 0;
  std::string pressure_state;
  // KV cache slots: one llama sequence per slot so multiple concurrent
  // sessions (agent loops, chats, side requests) each keep a warm prefix.
  struct CacheSlot {
    std::vector<llama_token> tokens;
    uint64_t last_used = 0;
    uint64_t coordinator_generation = 0;
    bool busy = false;  // an active request is generating on this slot
    bool is_anchor = false;  // holds a shared prefix snapshot, never generates
  };
  std::vector<CacheSlot> slots;
  uint64_t use_counter = 0;
  std::unique_ptr<clap::llama::CacheExecutor> cache_executor;
  clap::llama_cache::Coordinator* coordinator = nullptr;
};

void unload(LoadedLlama& loaded) {
  if (loaded.cache_executor) loaded.cache_executor->reset();
  loaded.coordinator = nullptr;
  loaded.cache_executor.reset();
  loaded.slots.clear();
  loaded.runtime.reset();
  loaded.max_active = 0;
  loaded.active_policy = {};
  loaded.last_eviction_reason.clear();
}

void load_model(LoadedLlama& loaded, const std::string& model_path) {
  if (loaded.runtime.same_path(model_path)) return;
  unload(loaded);
  loaded.runtime.load(model_path);
  const int32_t n_ctx = loaded.runtime.backend_allocation_cap();
  const int32_t retained_max = loaded.runtime.retained_max();
  loaded.slots.assign(static_cast<std::size_t>(retained_max), {});
  loaded.use_counter = 0;
  loaded.active_policy = clap::llama_active::select({
      env_int("CLAP_MAX_ACTIVE", 0), loaded.runtime.startup_available_bytes(),
      loaded.runtime.model_file_bytes(),
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency())), n_ctx,
      retained_max, loaded.runtime.hybrid(), loaded.runtime.has_encoder()});
  loaded.max_active = loaded.active_policy.selected_max;
  try {
    clap::llama::CacheExecutorConfig config;
    config.slot_count = static_cast<uint32_t>(retained_max);
    config.logical_token_capacity = static_cast<uint64_t>(n_ctx);
    config.max_anchors = static_cast<uint32_t>(retained_max);
    config.hard_max_retained_entries = static_cast<uint32_t>(retained_max);
    config.automatic_checkpoints = env_int("CLAP_CACHE_CHECKPOINTS_ENABLED", 1) != 0;
    config.checkpoint_minimum_tokens = static_cast<uint64_t>(
        env_int("CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS", 2048));
    config.checkpoint_interval_tokens = static_cast<uint64_t>(
        env_int("CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS", 2048));
    config.checkpoint_max = static_cast<uint32_t>(env_int("CLAP_CACHE_CHECKPOINT_MAX", 8));
    config.checkpoint_budget_basis_points = static_cast<uint32_t>(
        env_int("CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS", 2500));
    config.checkpoint_budget_bytes = env_u64("CLAP_CACHE_CHECKPOINT_BUDGET_BYTES", 0);
    loaded.cache_executor = std::make_unique<clap::llama::CacheExecutor>(
        config, std::make_unique<clap::llama::LlamaPhysicalCacheBackend>(
            loaded.runtime.context()));
    loaded.coordinator = &loaded.cache_executor->coordinator();
  } catch (const std::exception& error) {
    loaded.coordinator = nullptr;
    loaded.cache_executor.reset();
    fprintf(stderr, "clap-llama: cache coordinator unavailable; using no-cache fresh mode: %s\n", error.what());
  }
}

json retention_telemetry(const LoadedLlama& loaded, std::size_t active, std::size_t queued = 0) {
  uint32_t retained_total = 0;
  uint32_t retained_sessions = 0;
  uint32_t retained_anchors = 0;
  uint64_t evictions = 0;
  if (loaded.cache_executor) {
    const auto retention = loaded.cache_executor->retention_telemetry();
    const auto telemetry = loaded.cache_executor->telemetry();
    retained_total = retention.active_slots;
    retained_sessions = retention.session_slots;
    retained_anchors = retention.anchor_slots;
    evictions = telemetry.evictions;
  }
  const clap::llama::RetentionTelemetrySnapshot snapshot{
    loaded.max_active,
    queued,
    loaded.previous_max_active,
    loaded.last_adjustment_reason,
    loaded.last_adjustment_at,
    loaded.retained_growth_reserve_bytes,
    loaded.global_resident_memory_bytes,
    loaded.pressure_state,
    {
      loaded.active_policy.mode,
      loaded.active_policy.selected_max,
      loaded.active_policy.backend_ceiling,
      loaded.active_policy.hardware_ceiling,
      loaded.active_policy.model_ceiling,
      loaded.active_policy.memory_ceiling,
      loaded.active_policy.reason,
      loaded.runtime.startup_available_bytes(),
      loaded.runtime.model_file_bytes(),
      loaded.runtime.backend_allocation_cap(),
      loaded.active_policy.context_ceiling,
      loaded.active_policy.per_active_reserve_cells,
      loaded.active_policy.per_active_reserve_bytes,
      std::max(1u, std::thread::hardware_concurrency()),
      loaded.runtime.hybrid(),
    },
    active,
    retained_total,
    retained_sessions,
    retained_anchors,
    loaded.runtime.retained_max(),
    loaded.last_eviction_reason,
    evictions,
    loaded.runtime.backend_allocation_cap(),
  };
  return clap::llama::serialize_retention_telemetry(snapshot);
}

const std::string& telemetry_key() {
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

template <typename Token>
std::string token_fingerprint(const std::vector<Token>& tokens, std::size_t count) {
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


void finalize(LoadedLlama& loaded, ActiveRequest& req) {
  req.sampler.reset();
  if (req.cache_lease) {
    loaded.slots[static_cast<std::size_t>(req.seq)].busy = false;
    req.cache_lease.release();
  }
  if (!req.mark_terminal(ActiveRequest::TerminalState::Completed)) return;
  json cache{
    {"hit", req.cached_prompt_tokens > 0},
    {"reused_tokens", req.cached_prompt_tokens},
    {"reuse_kind", req.cache_reuse_kind.empty() ? json(nullptr) : json(req.cache_reuse_kind)},
    {"reuse_scope", req.cache_reuse_scope.empty() ? json(nullptr) : json(req.cache_reuse_scope)},
    {"namespace", req.cache_namespace.empty() ? json(nullptr) : json(req.cache_namespace)},
    {"donor_slot", req.cache_donor_slot < 0 ? json(nullptr) : json(req.cache_donor_slot)},
    {"donor_generation", req.cache_donor_slot < 0 ? json(nullptr) : json(req.cache_donor_generation)},
    {"target_slot", static_cast<int>(req.seq)},
    {"target_generation", req.cache_target_generation},
    {"miss_reason", req.cached_prompt_tokens > 0 ? json(nullptr) : json("no_shared_prefix")},
    {"candidates", req.cache_candidates},
    {"prompt_token_hash", req.prompt_token_hash},
    {"prompt_token_count", req.prompt_token_count},
    {"evicted_slots", req.cache_evicted_slots},
    {"decision_us", req.cache_decision_us},
    {"planned_reuse_tokens", req.cache_planned_reuse_tokens},
    {"realized_reuse_tokens", req.cache_realized_reuse_tokens},
    {"side_request", req.cache_side_request},
    {"fallback", req.cache_fallback.empty() ? json(nullptr) : json(req.cache_fallback)},
    {"slot", static_cast<int>(req.seq)},
  };
  if (req.stable_boundary.available()) {
    cache["stable_boundary_token_hash"] = req.stable_boundary.token_hash;
    cache["stable_boundary_token_count"] = req.stable_boundary.token_count;
    cache["stable_boundary_kind"] = req.stable_boundary.kind;
  }
  cache["stable_boundaries"] = json::array();
  for (const auto& resolved : req.resolved_boundaries) {
    const auto boundary = resolved.token_count;
    const bool available = (resolved.status == "resolved" || resolved.status == "authorized") &&
        boundary > 0;
    cache["stable_boundaries"].push_back(json{
      {"token_hash", available ? json(token_fingerprint(req.full_prompt_tokens, boundary)) : json(nullptr)},
      {"token_count", available ? json(boundary) : json(nullptr)},
      {"kind", resolved.kind},
      {"label", !resolved.label.empty() ? json(resolved.label) : json(nullptr)},
      {"requested", resolved.requested},
      {"status", resolved.status},
      {"skip_reason", !resolved.skip_reason.empty() ? json(resolved.skip_reason) : json(nullptr)},
      {"materialized", available ? json(std::find(req.materialized_boundaries.begin(),
          req.materialized_boundaries.end(), boundary) != req.materialized_boundaries.end()) : json(nullptr)},
    });
  }
  emit(req.id, json{
    {"done", true},
    {"finish_reason", req.finish_reason},
    {"cancelled", req.cancelled},
    {"usage", json{
      {"prompt_tokens", req.prompt_token_count},
      {"completion_tokens", req.completion_tokens},
    }},
    {"cache", std::move(cache)},
  });
}

void fail_request(LoadedLlama& loaded, ActiveRequest& req, const std::string& message) {
  req.sampler.reset();
  if (req.cache_lease) {
    auto& slot = loaded.slots[static_cast<std::size_t>(req.seq)];
    slot.busy = false;
    slot.tokens.clear();
    if (req.cache_lease.generation() != 0) {
      try {
        slot.coordinator_generation = req.cache_lease.invalidate_and_clear();
      } catch (const std::exception& error) {
        fprintf(stderr, "clap-llama: cache failure invalidation failed: %s\n", error.what());
      }
    }
    req.cache_lease.release();
  }
  req.mark_terminal(ActiveRequest::TerminalState::Failed);
  emit_error(req.id, message);
}

// Handles one sampled token: EOS, stop sequences, budget checks, streaming
// emission. Finalizes the request when generation ends.
void step(LoadedLlama& loaded, std::vector<std::unique_ptr<ActiveRequest>>& active) {
  for (auto& req : active) {
    if (!req->done && req->cancelled) {
      req->finish_reason = "cancel";
      finalize(loaded, *req);
    }
  }

  std::vector<clap::llama_native::ScheduleRequest> schedule_requests;
  schedule_requests.reserve(active.size());
  for (const auto& req : active) {
    const bool has_work = req->phase == ActiveRequest::Phase::Decode ||
        req->prompt_tokens.size() != req->ingested;
    schedule_requests.push_back({
        req->phase == ActiveRequest::Phase::Decode
            ? clap::llama_native::SchedulePhase::Decode
            : clap::llama_native::SchedulePhase::Prefill,
        !req->done && has_work,
    });
  }
  std::vector<ActiveRequest*> ordered;
  for (const std::size_t index : clap::llama_native::decode_first_order(schedule_requests)) {
    ordered.push_back(active[index].get());
  }

  clap::llama::GenerationStepper stepper(loaded.runtime, loaded.cache_executor.get());
  const auto events = stepper.step(ordered,
      static_cast<int32_t>(llama_n_batch(loaded.runtime.context())), active.size() == 1);
  for (const auto& event : events) {
    auto& req = *event.request;
    switch (event.type) {
      case clap::llama::GenerationEvent::Type::Token:
        emit(req.id, json{{"token", event.text}});
        break;
      case clap::llama::GenerationEvent::Type::Prefill:
        emit(req.id, json{{"prefill", {{"done", event.done}, {"total", event.total}}}});
        break;
      case clap::llama::GenerationEvent::Type::Complete:
        finalize(loaded, req);
        break;
      case clap::llama::GenerationEvent::Type::Failure:
        if (event.code.empty()) fail_request(loaded, req, event.text);
        else emit(req.id, json{{"error", event.text}, {"code", event.code}});
        break;
      case clap::llama::GenerationEvent::Type::CacheAppend: {
        auto& slot = loaded.slots[event.slot];
        slot.tokens.insert(slot.tokens.end(), event.tokens.begin(), event.tokens.end());
        slot.coordinator_generation = event.generation;
        break;
      }
      case clap::llama::GenerationEvent::Type::CacheResetSlot: {
        auto& slot = loaded.slots[event.slot];
        if (event.clear) slot.tokens.clear();
        slot.coordinator_generation = event.generation;
        break;
      }
      case clap::llama::GenerationEvent::Type::CacheResetAll:
        for (auto& slot : loaded.slots) {
          slot.tokens.clear();
          slot.is_anchor = false;
        }
        loaded.slots[static_cast<std::size_t>(req.seq)].coordinator_generation =
            req.cache_lease.generation();
        break;
      case clap::llama::GenerationEvent::Type::CacheAnchor: {
        if (event.anchor && event.slot < loaded.slots.size()) {
          auto& anchor = loaded.slots[event.slot];
          anchor.tokens = event.tokens;
          anchor.is_anchor = true;
          anchor.last_used = ++loaded.use_counter;
          anchor.coordinator_generation = event.generation;
        }
        for (const uint32_t victim : event.eviction_slots) {
          if (victim != event.slot && victim < loaded.slots.size()) loaded.slots[victim] = {};
          loaded.last_eviction_reason = "hard_ceiling";
        }
        break;
      }
    }
  }
}

std::unique_ptr<ActiveRequest> prepare_request(LoadedLlama& loaded, const std::string& id, const json& request) {
  std::vector<clap::llama::RequestSlotState> slots;
  slots.reserve(loaded.slots.size());
  for (const auto& slot : loaded.slots) {
    slots.push_back({slot.tokens, slot.coordinator_generation, slot.busy, slot.is_anchor});
  }
  clap::llama::RequestPreparer preparer(
      loaded.runtime, loaded.cache_executor.get(), std::move(slots), telemetry_key(),
      [](const auto& tokens, std::size_t count) { return token_fingerprint(tokens, count); });
  auto prepared = preparer.prepare(id, request);
  const std::size_t target = static_cast<std::size_t>(prepared.sequence);
  if (loaded.runtime.has_encoder()) {
    for (auto& slot : loaded.slots) slot = {};
  }
  for (const uint32_t victim : prepared.cache_evicted_slots) {
    if (victim != target && victim < loaded.slots.size()) loaded.slots[victim] = {};
    loaded.last_eviction_reason = "hard_ceiling";
  }
  if (target < loaded.slots.size()) {
    auto& slot = loaded.slots[target];
    slot.tokens.assign(prepared.full_prompt_tokens.begin(),
        prepared.full_prompt_tokens.begin() + prepared.initial_position);
    slot.coordinator_generation = prepared.cache_target_generation;
    slot.last_used = ++loaded.use_counter;
    slot.busy = true;
    slot.is_anchor = false;
  }
  auto active = std::make_unique<ActiveRequest>(std::move(prepared));
  active->sampler.reset(make_sampler(active->params));
  return active;
}

}  // namespace

int main() {
  LoadedLlama loaded;
  std::vector<std::unique_ptr<ActiveRequest>> active;
  std::deque<std::pair<std::string, json>> waiting;

  try {
    ggml_backend_load_all();

    StdinReader reader;
    bool running = true;

    // Returns false on shutdown.
    auto handle_message = [&](const std::string& line) -> bool {
      if (line.empty()) return true;
      std::string id;
      try {
        const json message = json::parse(line);
        id = message.value("id", "");
        const std::string type = message.value("type", "");

        if (type == "shutdown") {
          emit(id, json{{"done", true}});
          return false;
        }
        if (type == "cancel") {
          const std::string target = message.value("id", "");
          for (auto& req : active) {
            if (!req->done && clap::llama_native::active_cancel_matches(target, req->id)) {
              req->cancelled = true;
            }
          }
          for (auto it = waiting.begin(); it != waiting.end(); ++it) {
            if (clap::llama_native::queued_cancel_matches(target, it->first)) {
              emit(target, json{{"done", true}, {"finish_reason", "cancel"}, {"cancelled", true}});
              waiting.erase(it);
              break;
            }
          }
          return true;
        }
        if (type == "set_max_active") {
          const int requested = message.value("max_active", 0);
          if (requested <= 0) throw std::runtime_error("set_max_active.max_active must be positive");
          const int previous = loaded.max_active;
          loaded.max_active = std::max(1, std::min({requested,
              loaded.active_policy.backend_ceiling, loaded.active_policy.hardware_ceiling,
              loaded.active_policy.model_ceiling, loaded.active_policy.context_ceiling}));
          loaded.active_policy.selected_max = loaded.max_active;
          loaded.previous_max_active = message.value("previous_max_active", previous);
          loaded.active_policy.reason = message.value("limiting_reason", loaded.active_policy.reason);
          loaded.last_adjustment_reason = message.value("last_adjustment_reason", "");
          loaded.last_adjustment_at = message.value("last_adjustment_at", "");
          loaded.retained_growth_reserve_bytes = message.value("retained_growth_reserve_bytes", UINT64_C(0));
          loaded.global_resident_memory_bytes = message.value("global_resident_memory_bytes", UINT64_C(0));
          loaded.pressure_state = message.value("pressure_state", "");
          emit(id, json{{"done", true}, {"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
          return true;
        }
        if (type == "unload") {
          if (!active.empty()) throw std::runtime_error("cannot unload while requests are active");
          unload(loaded);
          emit(id, json{{"unloaded", true}, {"done", true}});
          return true;
        }
        if (type == "load") {
          const std::string model = message.value("model", "");
          if (model.empty()) throw std::runtime_error("load.model is required");
          if (!active.empty() && loaded.runtime.loaded() && loaded.runtime.model_path() != model) {
            throw std::runtime_error("cannot switch models while requests are active");
          }
          load_model(loaded, model);
          const int32_t effective = loaded.runtime.backend_allocation_cap();
          emit(id, json{{"loaded", true}, {"done", true}, {"token_capabilities", {
            {"model_context_window", loaded.runtime.model_context_window() > 0 ? json(loaded.runtime.model_context_window()) : json(nullptr)},
            {"effective_context_window", effective},
            {"max_input_tokens", std::max(0, effective - 1)},
            {"max_output_tokens", loaded.runtime.max_output_tokens() > 0 ? json(loaded.runtime.max_output_tokens()) : json(nullptr)},
            {"backend_allocation_cap", loaded.runtime.backend_allocation_cap()},
            {"user_configured_override", loaded.runtime.context_override() > 0 ? json(loaded.runtime.context_override()) : json(nullptr)},
          }}, {"retention", retention_telemetry(loaded, active.size())}});
          return true;
        }
        // Anything else is a chat request; it queues until a slot frees up.
        waiting.emplace_back(id, message);
        emit("", json{{"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
        return true;
      } catch (const RequestError& error) {
        emit_error(id, error.what(), error.code);
        return true;
      } catch (const std::exception& error) {
        emit_error(id, error.what());
        return true;
      }
    };

    while (running) {
      std::string line;
      if (active.empty() && waiting.empty()) {
        if (!reader.next(line)) break;  // idle: block until work or EOF
        running = handle_message(line);
        if (!running) break;
      }
      while (reader.poll(line)) {
        running = handle_message(line);
        if (!running) break;
      }
      if (!running) break;

      // Admit queued requests to free slots.
      while (!waiting.empty()) {
        const std::string req_model = waiting.front().second.value("model", "");
        if (req_model.empty()) {
          emit_error(waiting.front().first, "chat.model is required");
          waiting.pop_front();
          continue;
        }
        // Drain in-flight requests before switching models.
        if (!active.empty() && loaded.runtime.loaded() && loaded.runtime.model_path() != req_model) break;
        try {
          load_model(loaded, req_model);
        } catch (const std::exception& error) {
          emit_error(waiting.front().first, error.what());
          waiting.pop_front();
          continue;
        }
        if (loaded.runtime.has_encoder() && !active.empty()) break;
        if (!clap::llama_cache::can_admit(active.size(),
                                          static_cast<uint32_t>(loaded.max_active))) break;
        auto [wid, wreq] = std::move(waiting.front());
        waiting.pop_front();
        try {
          auto prepared = prepare_request(loaded, wid, wreq);
          active.push_back(std::move(prepared));
          emit(wid, json{{"started", true},
                         {"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
        } catch (const CacheBackpressure&) {
          waiting.emplace_front(std::move(wid), std::move(wreq));
          break;
        } catch (const RequestError& error) {
          emit_error(wid, error.what(), error.code);
        } catch (const std::exception& error) {
          emit_error(wid, error.what());
        }
      }

      if (!active.empty()) {
        step(loaded, active);
        const std::size_t before_cleanup = active.size();
        active.erase(
          std::remove_if(active.begin(), active.end(), [](const std::unique_ptr<ActiveRequest>& r) { return r->done; }),
          active.end());
        if (active.size() != before_cleanup) {
          emit("", json{{"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
        }
      }
    }

    for (auto& req : active) {
      if (!req->done) {
        req->finish_reason = "cancel";
        req->cancelled = true;
        finalize(loaded, *req);
      }
    }
    unload(loaded);
    return 0;
  } catch (const std::exception& error) {
    emit_error("", error.what());
    unload(loaded);
    return 1;
  }
}
