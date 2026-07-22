#include "clap/llama/scheduler.h"

#include "clap/llama/protocol.h"
#include "clap/llama/request-preparer.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace clap::llama {
namespace {

SchedulerEvent error_event(const std::string& id, const std::string& message,
                           std::string code = {}) {
  SchedulerEvent event;
  event.type = SchedulerEvent::Type::Error;
  event.id = id;
  event.message = message;
  event.code = std::move(code);
  return event;
}

}  // namespace

Scheduler::Scheduler(SchedulerState state) : state_(std::move(state)) {
  if (!state_.loaded || !state_.same_path || !state_.load || !state_.has_encoder ||
      !state_.max_active || !state_.prepare || !state_.step || !state_.complete) {
    throw std::invalid_argument("scheduler state callbacks are required");
  }
}

void Scheduler::enqueue(std::string id, nlohmann::json request) {
  waiting_.emplace_back(std::move(id), std::move(request));
}

SchedulerEvent Scheduler::topology() const {
  SchedulerEvent event;
  event.type = SchedulerEvent::Type::Topology;
  event.active = active_.size();
  event.queued = waiting_.size();
  return event;
}

std::vector<SchedulerEvent> Scheduler::cancel(const std::string& target) {
  std::vector<SchedulerEvent> events;
  for (auto& request : active_) {
    if (!request->done && (target.empty() || request->id == target)) request->cancelled = true;
  }
  if (!target.empty()) {
    const auto found = std::find_if(waiting_.begin(), waiting_.end(),
        [&target](const auto& queued) { return queued.first == target; });
    if (found != waiting_.end()) {
      SchedulerEvent event;
      event.type = SchedulerEvent::Type::QueuedCancelled;
      event.id = target;
      events.push_back(std::move(event));
      waiting_.erase(found);
    }
  }
  return events;
}

void Scheduler::admit(std::vector<SchedulerEvent>& events) {
  while (!waiting_.empty()) {
    const std::string model = waiting_.front().second.value("model", "");
    if (model.empty()) {
      events.push_back(error_event(waiting_.front().first, "chat.model is required"));
      waiting_.pop_front();
      continue;
    }
    if (!active_.empty() && state_.loaded() && !state_.same_path(model)) break;
    try {
      state_.load(model);
    } catch (const std::exception& error) {
      events.push_back(error_event(waiting_.front().first, error.what()));
      waiting_.pop_front();
      continue;
    }
    if (state_.has_encoder() && !active_.empty()) break;
    if (active_.size() >= static_cast<std::size_t>(std::max(0, state_.max_active()))) break;

    auto queued = std::move(waiting_.front());
    waiting_.pop_front();
    try {
      active_.push_back(state_.prepare(queued.first, queued.second));
      SchedulerEvent started;
      started.type = SchedulerEvent::Type::Started;
      started.id = queued.first;
      started.active = active_.size();
      started.queued = waiting_.size();
      events.push_back(std::move(started));
    } catch (const CacheBackpressure&) {
      waiting_.emplace_front(std::move(queued));
      break;
    } catch (const RequestError& error) {
      events.push_back(error_event(queued.first, error.what(), error.code));
    } catch (const std::exception& error) {
      events.push_back(error_event(queued.first, error.what()));
    }
  }
}

std::vector<ActiveRequest*> Scheduler::decode_first() const {
  std::vector<ActiveRequest*> ordered;
  ordered.reserve(active_.size());
  for (const auto& request : active_) {
    const bool has_work = request->phase == ActiveRequest::Phase::Decode ||
        request->prompt_tokens.size() != request->ingested;
    if (!request->done && has_work && request->phase == ActiveRequest::Phase::Decode) {
      ordered.push_back(request.get());
    }
  }
  for (const auto& request : active_) {
    const bool has_work = request->phase == ActiveRequest::Phase::Decode ||
        request->prompt_tokens.size() != request->ingested;
    if (!request->done && has_work && request->phase == ActiveRequest::Phase::Prefill) {
      ordered.push_back(request.get());
    }
  }
  return ordered;
}

void Scheduler::generate(std::vector<SchedulerEvent>& events) {
  for (auto& request : active_) {
    if (!request->done && request->cancelled) {
      request->finish_reason = "cancel";
      auto completion = state_.complete(*request, false);
      if (completion) {
        SchedulerEvent event;
        event.type = SchedulerEvent::Type::Completion;
        event.id = completion->id;
        event.completion = std::move(completion);
        events.push_back(std::move(event));
      }
    }
  }

  const auto ordered = decode_first();
  if (!ordered.empty()) {
    for (auto& generation : state_.step(ordered, active_.size() == 1)) {
      SchedulerEvent event;
      event.type = SchedulerEvent::Type::Generation;
      event.id = generation.request ? generation.request->id : "";
      generation.request = nullptr;
      event.generation = std::move(generation);
      events.push_back(std::move(event));
    }
  }

  const std::size_t before = active_.size();
  active_.erase(std::remove_if(active_.begin(), active_.end(),
      [](const auto& request) { return request->done; }), active_.end());
  if (active_.size() != before) events.push_back(topology());
}

std::vector<SchedulerEvent> Scheduler::tick() {
  std::vector<SchedulerEvent> events;
  admit(events);
  if (!active_.empty()) generate(events);
  return events;
}

std::vector<SchedulerEvent> Scheduler::cancel_all(bool include_waiting) {
  std::vector<SchedulerEvent> events;
  for (auto& request : active_) {
    if (request->done) continue;
    request->cancelled = true;
    request->finish_reason = "cancel";
    auto completion = state_.complete(*request, false);
    if (completion) {
      SchedulerEvent event;
      event.type = SchedulerEvent::Type::Completion;
      event.id = completion->id;
      event.completion = std::move(completion);
      events.push_back(std::move(event));
    }
  }
  active_.clear();
  if (include_waiting) {
    while (!waiting_.empty()) {
      SchedulerEvent event;
      event.type = SchedulerEvent::Type::QueuedCancelled;
      event.id = waiting_.front().first;
      events.push_back(std::move(event));
      waiting_.pop_front();
    }
  }
  return events;
}

}  // namespace clap::llama
