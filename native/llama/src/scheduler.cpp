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
  uint32_t priority = CLAP_CACHE_PRIORITY_NORMAL;
  if (request.contains("cache_identity") && request["cache_identity"].is_object()) {
    const auto value = request["cache_identity"].value("priority", "normal");
    if (value == "interactive") priority = CLAP_CACHE_PRIORITY_INTERACTIVE;
    else if (value == "background") priority = CLAP_CACHE_PRIORITY_BACKGROUND;
  }
  waiting_.push_back({std::move(id), std::move(request), priority});
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
        [&target](const auto& queued) { return queued.id == target; });
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
  static constexpr uint32_t schedule[] = {
    CLAP_CACHE_PRIORITY_INTERACTIVE, CLAP_CACHE_PRIORITY_INTERACTIVE,
    CLAP_CACHE_PRIORITY_INTERACTIVE, CLAP_CACHE_PRIORITY_INTERACTIVE,
    CLAP_CACHE_PRIORITY_NORMAL, CLAP_CACHE_PRIORITY_NORMAL,
    CLAP_CACHE_PRIORITY_BACKGROUND};
  while (!waiting_.empty()) {
    auto selected = waiting_.begin();
    for (std::size_t offset = 0; offset < std::size(schedule); ++offset) {
      const uint32_t priority = schedule[(priority_cursor_ + offset) % std::size(schedule)];
      const auto found = std::find_if(waiting_.begin(), waiting_.end(),
          [priority](const auto& queued) { return queued.priority == priority; });
      if (found != waiting_.end()) {
        selected = found;
        priority_cursor_ = (priority_cursor_ + offset + 1) % std::size(schedule);
        break;
      }
    }
    const std::string model = selected->request.value("model", "");
    if (model.empty()) {
      events.push_back(error_event(selected->id, "chat.model is required"));
      waiting_.erase(selected);
      continue;
    }
    if (!active_.empty() && state_.loaded() && !state_.same_path(model)) break;
    try {
      state_.load(model);
    } catch (const std::exception& error) {
      events.push_back(error_event(selected->id, error.what()));
      waiting_.erase(selected);
      continue;
    }
    if (state_.has_encoder() && !active_.empty()) break;
    if (active_.size() >= static_cast<std::size_t>(std::max(0, state_.max_active()))) break;

    auto queued = std::move(*selected);
    waiting_.erase(selected);
    try {
      active_.push_back(state_.prepare(queued.id, queued.request));
      SchedulerEvent started;
      started.type = SchedulerEvent::Type::Started;
      started.id = queued.id;
      started.active = active_.size();
      started.queued = waiting_.size();
      events.push_back(std::move(started));
    } catch (const CacheBackpressure&) {
      waiting_.push_front(std::move(queued));
      break;
    } catch (const RequestError& error) {
      events.push_back(error_event(queued.id, error.what(), error.code));
    } catch (const std::exception& error) {
      events.push_back(error_event(queued.id, error.what()));
    }
  }
}

std::vector<ActiveRequest*> Scheduler::decode_first() const {
  std::vector<ActiveRequest*> ordered;
  ordered.reserve(active_.size());
  for (uint32_t priority : {CLAP_CACHE_PRIORITY_INTERACTIVE, CLAP_CACHE_PRIORITY_NORMAL,
                            CLAP_CACHE_PRIORITY_BACKGROUND}) {
    for (const auto& request : active_) {
      const bool has_work = request->phase == ActiveRequest::Phase::Decode ||
          request->prompt_tokens.size() != request->ingested;
      if (!request->done && has_work && request->priority == priority &&
          request->phase == ActiveRequest::Phase::Decode) ordered.push_back(request.get());
    }
    for (const auto& request : active_) {
      const bool has_work = request->phase == ActiveRequest::Phase::Decode ||
          request->prompt_tokens.size() != request->ingested;
      if (!request->done && has_work && request->priority == priority &&
          request->phase == ActiveRequest::Phase::Prefill) ordered.push_back(request.get());
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
      event.id = waiting_.front().id;
      events.push_back(std::move(event));
      waiting_.pop_front();
    }
  }
  return events;
}

}  // namespace clap::llama
