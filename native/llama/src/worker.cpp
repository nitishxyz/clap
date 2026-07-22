#include "clap/llama/worker.h"

#include "clap/llama/request-state.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace clap::llama {
namespace {

SchedulerState scheduler_state(WorkerState& state) {
  return {
    [&] { return state.loaded(); },
    [&](const std::string& path) { return state.same_path(path); },
    [&](const std::string& path) { state.load(path); },
    [&] { return state.has_encoder(); },
    [&] { return state.max_active(); },
    [&](const std::string& id, const nlohmann::json& request) {
      return state.prepare(id, request);
    },
    [&](const std::vector<ActiveRequest*>& ordered, bool sole) {
      return state.step(ordered, sole);
    },
    [&](ActiveRequest& request, bool flush) { return state.complete(request, flush); },
  };
}

}  // namespace

Worker::Worker() : Worker(std::cin, std::cout) {}

Worker::Worker(std::istream& input, std::ostream& output)
    : output_(output), state_(), scheduler_(scheduler_state(state_)), reader_(input) {}

void Worker::send(const std::string& id, nlohmann::json fields) {
  emit(id, std::move(fields), output_);
}

void Worker::send_error(const std::string& id, const std::string& message,
                        const std::string& code) {
  emit_error(id, message, code, output_);
}

void Worker::send_completion(const RequestCompletion& completion) {
  const auto& facts = completion.cache;
  nlohmann::json cache{
    {"hit", facts.hit}, {"reused_tokens", facts.reused_tokens},
    {"reuse_kind", facts.reuse_kind ? nlohmann::json(*facts.reuse_kind) : nlohmann::json(nullptr)},
    {"reuse_scope", facts.reuse_scope ? nlohmann::json(*facts.reuse_scope) : nlohmann::json(nullptr)},
    {"namespace", facts.name_space ? nlohmann::json(*facts.name_space) : nlohmann::json(nullptr)},
    {"donor_slot", facts.donor_slot ? nlohmann::json(*facts.donor_slot) : nlohmann::json(nullptr)},
    {"donor_generation", facts.donor_generation ? nlohmann::json(*facts.donor_generation) : nlohmann::json(nullptr)},
    {"target_slot", facts.target_slot}, {"target_generation", facts.target_generation},
    {"miss_reason", facts.miss_reason ? nlohmann::json(*facts.miss_reason) : nlohmann::json(nullptr)},
    {"candidates", facts.candidates}, {"prompt_token_hash", facts.prompt_token_hash},
    {"prompt_token_count", facts.prompt_token_count}, {"evicted_slots", facts.evicted_slots},
    {"decision_us", facts.decision_us}, {"planned_reuse_tokens", facts.planned_reuse_tokens},
    {"realized_reuse_tokens", facts.realized_reuse_tokens}, {"side_request", facts.side_request},
    {"fallback", facts.fallback ? nlohmann::json(*facts.fallback) : nlohmann::json(nullptr)},
    {"slot", facts.slot},
  };
  if (facts.stable_boundary_token_hash) {
    cache["stable_boundary_token_hash"] = *facts.stable_boundary_token_hash;
    cache["stable_boundary_token_count"] = *facts.stable_boundary_token_count;
    cache["stable_boundary_kind"] = *facts.stable_boundary_kind;
  }
  cache["stable_boundaries"] = nlohmann::json::array();
  for (const auto& boundary : facts.stable_boundaries) {
    cache["stable_boundaries"].push_back({
      {"token_hash", boundary.token_hash ? nlohmann::json(*boundary.token_hash) : nlohmann::json(nullptr)},
      {"token_count", boundary.token_count ? nlohmann::json(*boundary.token_count) : nlohmann::json(nullptr)},
      {"kind", boundary.kind},
      {"label", boundary.label ? nlohmann::json(*boundary.label) : nlohmann::json(nullptr)},
      {"requested", boundary.requested}, {"status", boundary.status},
      {"skip_reason", boundary.skip_reason ? nlohmann::json(*boundary.skip_reason) : nlohmann::json(nullptr)},
      {"materialized", boundary.materialized ? nlohmann::json(*boundary.materialized) : nlohmann::json(nullptr)},
    });
  }
  if (!completion.visible_tail.empty()) send(completion.id, {{"token", completion.visible_tail}});
  send(completion.id, {
    {"done", true}, {"finish_reason", completion.finish_reason},
    {"cancelled", completion.cancelled},
    {"usage", {{"prompt_tokens", completion.usage.prompt_tokens},
               {"completion_tokens", completion.usage.completion_tokens}}},
    {"cache", std::move(cache)},
  });
}

void Worker::send_failure(const RequestFailure& failure) {
  if (!failure.invalidation_error.empty()) {
    fprintf(stderr, "clap-llama: cache failure invalidation failed: %s\n",
            failure.invalidation_error.c_str());
  }
  send_error(failure.id, failure.message, failure.code);
}

void Worker::send_scheduler_events(const std::vector<SchedulerEvent>& events) {
  for (const auto& event : events) {
    switch (event.type) {
      case SchedulerEvent::Type::Started:
        send(event.id, {{"started", true},
            {"retention", state_.retention(event.active, event.queued)}});
        break;
      case SchedulerEvent::Type::QueuedCancelled:
        send(event.id, {{"done", true}, {"finish_reason", "cancel"}, {"cancelled", true}});
        break;
      case SchedulerEvent::Type::Error:
        send_error(event.id, event.message, event.code);
        break;
      case SchedulerEvent::Type::Completion:
        if (event.completion) send_completion(*event.completion);
        break;
      case SchedulerEvent::Type::Topology:
        send("", {{"retention", state_.retention(event.active, event.queued)}});
        break;
      case SchedulerEvent::Type::Generation:
        if (!event.generation) break;
        switch (event.generation->type) {
          case GenerationEvent::Type::Token:
            send(event.id, {{"token", event.generation->text}});
            break;
          case GenerationEvent::Type::Prefill:
            send(event.id, {{"prefill", {{"done", event.generation->done},
                                          {"total", event.generation->total}}}});
            break;
          case GenerationEvent::Type::Complete:
            if (event.generation->completion) send_completion(*event.generation->completion);
            break;
          case GenerationEvent::Type::Failure:
            if (event.generation->failure) send_failure(*event.generation->failure);
            break;
          default:
            break;
        }
        break;
    }
  }
}

bool Worker::dispatch(const std::string& line) {
  if (line.empty()) return true;
  std::string id;
  try {
    const nlohmann::json message = nlohmann::json::parse(line);
    id = message.value("id", "");
    const std::string type = message.value("type", "");
    if (type == "shutdown") {
      send(id, {{"done", true}});
      return false;
    }
    if (type == "cancel") {
      send_scheduler_events(scheduler_.cancel(message.value("id", "")));
      return true;
    }
    if (type == "set_max_active") {
      state_.set_max_active({message.value("max_active", 0),
          message.value("previous_max_active", state_.max_active()),
          message.value("limiting_reason", ""), message.value("last_adjustment_reason", ""),
          message.value("last_adjustment_at", ""),
          message.value("retained_growth_reserve_bytes", UINT64_C(0)),
          message.value("global_resident_memory_bytes", UINT64_C(0)),
          message.value("pressure_state", "")});
      send(id, {{"done", true}, {"retention", state_.retention(
          scheduler_.active_count(), scheduler_.queued_count())}});
      return true;
    }
    if (type == "unload") {
      if (scheduler_.has_active()) throw std::runtime_error("cannot unload while requests are active");
      state_.unload();
      send(id, {{"unloaded", true}, {"done", true}});
      return true;
    }
    if (type == "load") {
      const std::string model = message.value("model", "");
      if (model.empty()) throw std::runtime_error("load.model is required");
      if (scheduler_.has_active() && state_.loaded() && !state_.same_path(model)) {
        throw std::runtime_error("cannot switch models while requests are active");
      }
      state_.load(model);
      const int32_t effective = state_.effective_context_window();
      send(id, {{"loaded", true}, {"done", true}, {"token_capabilities", {
        {"model_context_window", state_.model_context_window() > 0
            ? nlohmann::json(state_.model_context_window()) : nlohmann::json(nullptr)},
        {"effective_context_window", effective}, {"max_input_tokens", std::max(0, effective - 1)},
        {"max_output_tokens", state_.max_output_tokens() > 0
            ? nlohmann::json(state_.max_output_tokens()) : nlohmann::json(nullptr)},
        {"backend_allocation_cap", state_.effective_context_window()},
        {"user_configured_override", state_.context_override() > 0
            ? nlohmann::json(state_.context_override()) : nlohmann::json(nullptr)},
      }}, {"retention", state_.retention(scheduler_.active_count())}});
      return true;
    }
    scheduler_.enqueue(id, message);
    send("", {{"retention", state_.retention(
        scheduler_.active_count(), scheduler_.queued_count())}});
    return true;
  } catch (const RequestError& error) {
    send_error(id, error.what(), error.code);
  } catch (const std::exception& error) {
    send_error(id, error.what());
  }
  return true;
}

int Worker::run() {
  try {
    bool running = true;
    while (running) {
      std::string line;
      if (scheduler_.idle()) {
        if (!reader_.next(line)) break;
        running = dispatch(line);
        if (!running) break;
      }
      while (reader_.poll(line)) {
        running = dispatch(line);
        if (!running) break;
      }
      if (!running) break;
      send_scheduler_events(scheduler_.tick());
    }
    send_scheduler_events(scheduler_.cancel_all());
    state_.unload();
    return 0;
  } catch (const std::exception& error) {
    send_error("", error.what());
    try {
      state_.unload();
    } catch (...) {
    }
    return 1;
  }
}

}  // namespace clap::llama
