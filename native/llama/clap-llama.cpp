#include "llama.h"
#include "clap/llama/protocol.h"
#include "clap/llama/request-state.h"
#include "clap/llama/scheduler.h"
#include "clap/llama/worker-state.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

using clap::llama::emit;
using clap::llama::emit_error;
using clap::llama::RequestError;
using clap::llama::StdinReader;
using clap::llama::ActiveRequest;

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

void emit_scheduler_events(clap::llama::WorkerState& worker,
                           const std::vector<clap::llama::SchedulerEvent>& events) {
  for (const auto& event : events) {
    switch (event.type) {
      case clap::llama::SchedulerEvent::Type::Started:
        emit(event.id, json{{"started", true},
            {"retention", worker.retention(event.active, event.queued)}});
        break;
      case clap::llama::SchedulerEvent::Type::QueuedCancelled:
        emit(event.id, json{{"done", true}, {"finish_reason", "cancel"},
                            {"cancelled", true}});
        break;
      case clap::llama::SchedulerEvent::Type::Error:
        if (event.code.empty()) emit_error(event.id, event.message);
        else emit_error(event.id, event.message, event.code);
        break;
      case clap::llama::SchedulerEvent::Type::Completion:
        if (event.completion) emit_completion(*event.completion);
        break;
      case clap::llama::SchedulerEvent::Type::Topology:
        emit("", json{{"retention", worker.retention(event.active, event.queued)}});
        break;
      case clap::llama::SchedulerEvent::Type::Generation: {
        if (!event.generation) break;
        const auto& generation = *event.generation;
        switch (generation.type) {
          case clap::llama::GenerationEvent::Type::Token:
            emit(event.id, json{{"token", generation.text}});
            break;
          case clap::llama::GenerationEvent::Type::Prefill:
            emit(event.id, json{{"prefill", {{"done", generation.done},
                                               {"total", generation.total}}}});
            break;
          case clap::llama::GenerationEvent::Type::Complete:
            if (generation.completion) emit_completion(*generation.completion);
            break;
          case clap::llama::GenerationEvent::Type::Failure:
            if (generation.failure) emit_failure(*generation.failure);
            break;
          default:
            break;
        }
        break;
      }
    }
  }
}

}  // namespace

int main() {
  clap::llama::WorkerState worker;
  clap::llama::Scheduler scheduler({
    [&] { return worker.loaded(); },
    [&](const std::string& path) { return worker.same_path(path); },
    [&](const std::string& path) { worker.load(path); },
    [&] { return worker.has_encoder(); },
    [&] { return worker.max_active(); },
    [&](const std::string& id, const json& request) { return worker.prepare(id, request); },
    [&](const std::vector<ActiveRequest*>& ordered, bool sole) {
      return worker.step(ordered, sole);
    },
    [&](ActiveRequest& request, bool flush) { return worker.complete(request, flush); },
  });

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
          emit_scheduler_events(worker, scheduler.cancel(target));
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
          emit(id, json{{"done", true}, {"retention", worker.retention(
              scheduler.active_count(), scheduler.queued_count())}});
          return true;
        }
        if (type == "unload") {
          if (scheduler.has_active()) throw std::runtime_error("cannot unload while requests are active");
          worker.unload();
          emit(id, json{{"unloaded", true}, {"done", true}});
          return true;
        }
        if (type == "load") {
          const std::string model = message.value("model", "");
          if (model.empty()) throw std::runtime_error("load.model is required");
          if (scheduler.has_active() && worker.loaded() && !worker.same_path(model)) {
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
          }}, {"retention", worker.retention(scheduler.active_count())}});
          return true;
        }
        // Anything else is a chat request; it queues until a slot frees up.
        scheduler.enqueue(id, message);
        emit("", json{{"retention", worker.retention(
            scheduler.active_count(), scheduler.queued_count())}});
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
      if (scheduler.idle()) {
        if (!reader.next(line)) break;  // idle: block until work or EOF
        running = handle_message(line);
        if (!running) break;
      }
      while (reader.poll(line)) {
        running = handle_message(line);
        if (!running) break;
      }
      if (!running) break;

      emit_scheduler_events(worker, scheduler.tick());
    }

    emit_scheduler_events(worker, scheduler.cancel_all());
    worker.unload();
    return 0;
  } catch (const std::exception& error) {
    emit_error("", error.what());
    worker.unload();
    return 1;
  }
}
