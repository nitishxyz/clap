#include "llama.h"
#include "active-concurrency.h"
#include "cache-adapter.h"
#include "clap/llama/protocol.h"
#include "clap/llama/request-preparer.h"
#include "clap/llama/request-state.h"
#include "clap/llama/worker-state.h"
#include "native-characterization.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

using clap::llama::emit;
using clap::llama::emit_error;
using clap::llama::make_sampler;
using clap::llama::RequestError;
using clap::llama::StdinReader;
using clap::llama::ActiveRequest;
using clap::llama::CacheBackpressure;

void emit_completion(const clap::llama::RequestCompletion& completion) {
  const auto& facts = completion.cache;
  json cache{
    {"hit", facts.hit},
    {"reused_tokens", facts.reused_tokens},
    {"reuse_kind", facts.reuse_kind ? json(*facts.reuse_kind) : json(nullptr)},
    {"reuse_scope", facts.reuse_scope ? json(*facts.reuse_scope) : json(nullptr)},
    {"namespace", facts.name_space ? json(*facts.name_space) : json(nullptr)},
    {"donor_slot", facts.donor_slot ? json(*facts.donor_slot) : json(nullptr)},
    {"donor_generation", facts.donor_generation ? json(*facts.donor_generation) : json(nullptr)},
    {"target_slot", facts.target_slot},
    {"target_generation", facts.target_generation},
    {"miss_reason", facts.miss_reason ? json(*facts.miss_reason) : json(nullptr)},
    {"candidates", facts.candidates},
    {"prompt_token_hash", facts.prompt_token_hash},
    {"prompt_token_count", facts.prompt_token_count},
    {"evicted_slots", facts.evicted_slots},
    {"decision_us", facts.decision_us},
    {"planned_reuse_tokens", facts.planned_reuse_tokens},
    {"realized_reuse_tokens", facts.realized_reuse_tokens},
    {"side_request", facts.side_request},
    {"fallback", facts.fallback ? json(*facts.fallback) : json(nullptr)},
    {"slot", facts.slot},
  };
  if (facts.stable_boundary_token_hash) {
    cache["stable_boundary_token_hash"] = *facts.stable_boundary_token_hash;
    cache["stable_boundary_token_count"] = *facts.stable_boundary_token_count;
    cache["stable_boundary_kind"] = *facts.stable_boundary_kind;
  }
  cache["stable_boundaries"] = json::array();
  for (const auto& boundary : facts.stable_boundaries) {
    cache["stable_boundaries"].push_back(json{
      {"token_hash", boundary.token_hash ? json(*boundary.token_hash) : json(nullptr)},
      {"token_count", boundary.token_count ? json(*boundary.token_count) : json(nullptr)},
      {"kind", boundary.kind},
      {"label", boundary.label ? json(*boundary.label) : json(nullptr)},
      {"requested", boundary.requested},
      {"status", boundary.status},
      {"skip_reason", boundary.skip_reason ? json(*boundary.skip_reason) : json(nullptr)},
      {"materialized", boundary.materialized ? json(*boundary.materialized) : json(nullptr)},
    });
  }
  if (!completion.visible_tail.empty()) emit(completion.id, json{{"token", completion.visible_tail}});
  emit(completion.id, json{
    {"done", true}, {"finish_reason", completion.finish_reason},
    {"cancelled", completion.cancelled},
    {"usage", {{"prompt_tokens", completion.usage.prompt_tokens},
               {"completion_tokens", completion.usage.completion_tokens}}},
    {"cache", std::move(cache)},
  });
}

void emit_failure(const clap::llama::RequestFailure& failure) {
  if (!failure.invalidation_error.empty()) {
    fprintf(stderr, "clap-llama: cache failure invalidation failed: %s\n",
            failure.invalidation_error.c_str());
  }
  if (failure.code.empty()) emit_error(failure.id, failure.message);
  else emit(failure.id, json{{"error", failure.message}, {"code", failure.code}});
}

void step(clap::llama::WorkerState& worker,
          std::vector<std::unique_ptr<ActiveRequest>>& active) {
  for (auto& req : active) {
    if (!req->done && req->cancelled) {
      req->finish_reason = "cancel";
      auto completion = worker.complete(*req, false);
      if (completion) emit_completion(*completion);
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

  for (const auto& event : worker.step(ordered, active.size() == 1)) {
    auto& req = *event.request;
    switch (event.type) {
      case clap::llama::GenerationEvent::Type::Token:
        emit(req.id, json{{"token", event.text}});
        break;
      case clap::llama::GenerationEvent::Type::Prefill:
        emit(req.id, json{{"prefill", {{"done", event.done}, {"total", event.total}}}});
        break;
      case clap::llama::GenerationEvent::Type::Complete:
        if (event.completion) emit_completion(*event.completion);
        break;
      case clap::llama::GenerationEvent::Type::Failure:
        if (event.failure) emit_failure(*event.failure);
        break;
      default:
        break;
    }
  }
}

}  // namespace

int main() {
  clap::llama::WorkerState worker;
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
          worker.set_max_active({requested,
              message.value("previous_max_active", worker.max_active()),
              message.value("limiting_reason", ""),
              message.value("last_adjustment_reason", ""),
              message.value("last_adjustment_at", ""),
              message.value("retained_growth_reserve_bytes", UINT64_C(0)),
              message.value("global_resident_memory_bytes", UINT64_C(0)),
              message.value("pressure_state", "")});
          emit(id, json{{"done", true}, {"retention", worker.retention(active.size(), waiting.size())}});
          return true;
        }
        if (type == "unload") {
          if (!active.empty()) throw std::runtime_error("cannot unload while requests are active");
          worker.unload();
          emit(id, json{{"unloaded", true}, {"done", true}});
          return true;
        }
        if (type == "load") {
          const std::string model = message.value("model", "");
          if (model.empty()) throw std::runtime_error("load.model is required");
          if (!active.empty() && worker.loaded() && !worker.same_path(model)) {
            throw std::runtime_error("cannot switch models while requests are active");
          }
          worker.load(model);
          const int32_t effective = worker.effective_context_window();
          emit(id, json{{"loaded", true}, {"done", true}, {"token_capabilities", {
            {"model_context_window", worker.model_context_window() > 0 ? json(worker.model_context_window()) : json(nullptr)},
            {"effective_context_window", effective},
            {"max_input_tokens", std::max(0, effective - 1)},
            {"max_output_tokens", worker.max_output_tokens() > 0 ? json(worker.max_output_tokens()) : json(nullptr)},
            {"backend_allocation_cap", worker.effective_context_window()},
            {"user_configured_override", worker.context_override() > 0 ? json(worker.context_override()) : json(nullptr)},
          }}, {"retention", worker.retention(active.size())}});
          return true;
        }
        // Anything else is a chat request; it queues until a slot frees up.
        waiting.emplace_back(id, message);
        emit("", json{{"retention", worker.retention(active.size(), waiting.size())}});
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
        if (!active.empty() && worker.loaded() && !worker.same_path(req_model)) break;
        try {
          worker.load(req_model);
        } catch (const std::exception& error) {
          emit_error(waiting.front().first, error.what());
          waiting.pop_front();
          continue;
        }
        if (worker.has_encoder() && !active.empty()) break;
        if (!clap::llama_cache::can_admit(active.size(),
                                          static_cast<uint32_t>(worker.max_active()))) break;
        auto [wid, wreq] = std::move(waiting.front());
        waiting.pop_front();
        try {
          auto prepared = worker.prepare(wid, wreq);
          active.push_back(std::move(prepared));
          emit(wid, json{{"started", true},
                         {"retention", worker.retention(active.size(), waiting.size())}});
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
        step(worker, active);
        const std::size_t before_cleanup = active.size();
        active.erase(
          std::remove_if(active.begin(), active.end(), [](const std::unique_ptr<ActiveRequest>& r) { return r->done; }),
          active.end());
        if (active.size() != before_cleanup) {
          emit("", json{{"retention", worker.retention(active.size(), waiting.size())}});
        }
      }
    }

    for (auto& req : active) {
      if (!req->done) {
        req->finish_reason = "cancel";
        req->cancelled = true;
        auto completion = worker.complete(*req, false);
        if (completion) emit_completion(*completion);
      }
    }
    worker.unload();
    return 0;
  } catch (const std::exception& error) {
    emit_error("", error.what());
    worker.unload();
    return 1;
  }
}
