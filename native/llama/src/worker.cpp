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
    : output_(output), v1_(std::make_unique<ProtocolWriter>(output)),
      state_(), scheduler_(scheduler_state(state_)), reader_(input) {
  v1_->ready({{"backend", "llama"}, {"streaming", true}, {"scheduling", {
      {"fused_multi_sequence_batching", true}, {"interleaved", true}}}}, nullptr);
}

void Worker::send_scheduler_events(const std::vector<SchedulerEvent>& events) {
  send_v1_scheduler_events(events);
}

void Worker::send_v1_completion(const RequestCompletion& completion) {
  nlohmann::json result{{"kind", completion.cancelled ? "cancelled" : "generated"}};
  if (!completion.cancelled) {
    if (!completion.visible_tail.empty()) {
      v1_->token(completion.id, completion.visible_tail);
      generated_content_[completion.id] += completion.visible_tail;
    }
    result["content"] = generated_content_[completion.id];
    result["finish_reason"] = completion.finish_reason;
    result["usage"] = {{"prompt_tokens", completion.usage.prompt_tokens},
                       {"completion_tokens", completion.usage.completion_tokens}};
  }
  v1_->completed(completion.id, std::move(result));
  generated_content_.erase(completion.id);
}

void Worker::send_v1_scheduler_events(const std::vector<SchedulerEvent>& events) {
  for (const auto& event : events) {
    switch (event.type) {
      case SchedulerEvent::Type::Started:
        v1_->started(event.id);
        break;
      case SchedulerEvent::Type::QueuedCancelled:
        v1_->completed(event.id, {{"kind", "cancelled"}});
        generated_content_.erase(event.id);
        break;
      case SchedulerEvent::Type::Error:
        v1_->failed(event.id, event.code.empty() ? "worker_error" : event.code,
                    event.message, false, false);
        break;
      case SchedulerEvent::Type::Completion:
        if (event.completion) send_v1_completion(*event.completion);
        break;
      case SchedulerEvent::Type::Topology:
        v1_->telemetry(state_.retention(event.active, event.queued));
        break;
      case SchedulerEvent::Type::Generation:
        if (!event.generation) break;
        switch (event.generation->type) {
          case GenerationEvent::Type::Token:
            generated_content_[event.id] += event.generation->text;
            v1_->token(event.id, event.generation->text);
            break;
          case GenerationEvent::Type::Prefill:
            v1_->prefill_progress(event.id, event.generation->done, event.generation->total);
            break;
          case GenerationEvent::Type::Complete:
            if (event.generation->completion) send_v1_completion(*event.generation->completion);
            break;
          case GenerationEvent::Type::Failure:
            if (event.generation->failure) {
              const auto& failure = *event.generation->failure;
              v1_->failed(failure.id, failure.code.empty() ? "generation_failed" : failure.code,
                          failure.message, true, false);
              generated_content_.erase(failure.id);
            }
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
  V1Request request;
  try {
    request = decode_v1_request(line);
  } catch (const V1DecodeError& error) {
    if (!error.request_id.empty()) {
      v1_->accepted(error.request_id);
      v1_->failed(error.request_id, error.code, error.what(), false, false);
    } else {
      emit("", {{"protocol", 1}, {"type", "diagnostic"}, {"level", "error"},
                 {"message", error.what()}}, output_);
    }
    return true;
  }

  if (!v1_->accepted(request.request_id)) return true;
  try {
    if (request.type == "shutdown") {
      v1_->completed(request.request_id, {{"kind", "shutdown"}});
      return false;
    }
    if (request.type == "cancel") {
      send_v1_scheduler_events(scheduler_.cancel(request.target_request_id));
      v1_->completed(request.request_id, {{"kind", "cancelled"}});
      return true;
    }
    if (request.type == "set_max_active") {
      const int selected = state_.set_max_active({request.body["max_active"].get<int>()});
      v1_->completed(request.request_id,
          {{"kind", "max_active_updated"}, {"max_active", selected}});
      return true;
    }
    if (request.type == "unload") {
      if (scheduler_.has_active()) throw std::runtime_error("cannot unload while requests are active");
      state_.unload();
      v1_->completed(request.request_id, {{"kind", "unloaded"}});
      return true;
    }
    if (request.type == "load") {
      const std::string model = request.body["model"].get<std::string>();
      if (scheduler_.has_active() && state_.loaded() && !state_.same_path(model)) {
        throw std::runtime_error("cannot switch models while requests are active");
      }
      state_.load(model);
      const int32_t effective = state_.effective_context_window();
      v1_->completed(request.request_id, {{"kind", "loaded"}, {"model", model},
        {"effective_model_capabilities", {
          {"cache", {{"partial_suffix_trim", true}, {"partial_prefix_branch", true},
            {"whole_state_copy", true}, {"prompt_boundary_snapshots", true},
            {"quantized_kv", state_.kv_format().find('q') != std::string::npos}}},
          {"generation", {{"structured_output", {{"json_object", "native"},
            {"json_schema", "native"}, {"post_validation", true},
            {"max_schema_bytes", 64 * 1024}}}, {"tool_templates", false}}},
          {"modalities", {{"input", {"text"}}, {"output", {"text"}}}}
        }},
        {"token_capabilities", {{"model_context_window", state_.model_context_window() > 0
              ? nlohmann::json(state_.model_context_window()) : nlohmann::json(nullptr)},
          {"effective_context_window", effective}, {"max_input_tokens", std::max(0, effective - 1)},
          {"max_output_tokens", state_.max_output_tokens() > 0
              ? nlohmann::json(state_.max_output_tokens()) : nlohmann::json(nullptr)},
          {"backend_allocation_cap", effective},
          {"user_configured_override", state_.context_override() > 0
              ? nlohmann::json(state_.context_override()) : nlohmann::json(nullptr)}}}});
      return true;
    }

    nlohmann::json body = request.body.contains("request") && request.body["request"].is_object()
        ? request.body["request"] : request.body;
    body.erase("protocol");
    body.erase("request_id");
    body.erase("request");
    body["cache_identity"] = request.body["cache_identity"];
    body.erase("structured_output");
    if (request.body.contains("structured_output")) {
      body["structured_output"] = request.body["structured_output"];
    }
    body["id"] = request.request_id;
    body["type"] = "generate";
    scheduler_.enqueue(request.request_id, std::move(body));
    v1_->telemetry(state_.retention(scheduler_.active_count(), scheduler_.queued_count()));
    return true;
  } catch (const RequestError& error) {
    v1_->failed(request.request_id, error.code, error.what(), false, false);
  } catch (const std::exception& error) {
    v1_->failed(request.request_id, "worker_error", error.what(), false, false);
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
    send_scheduler_events(scheduler_.cancel_all(true));
    state_.unload();
    return 0;
  } catch (const std::exception& error) {
    emit("", {{"protocol", 1}, {"type", "diagnostic"}, {"level", "error"},
               {"message", error.what()}}, output_);
    try {
      state_.unload();
    } catch (...) {
    }
    return 1;
  }
}

}  // namespace clap::llama
