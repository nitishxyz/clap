#include "clap/llama/scheduler.h"

#include "clap/llama/request-preparer.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace {

struct FakeState {
  bool is_loaded = false;
  bool encoder = false;
  int32_t capacity = 2;
  std::string path;
  bool backpressure = false;
  bool finish_on_step = false;
  std::vector<std::string> loaded_paths;
  std::vector<std::string> prepared;
  std::vector<std::string> stepped;

  clap::llama::SchedulerState callbacks() {
    return {
      [&] { return is_loaded; },
      [&](const std::string& candidate) { return is_loaded && path == candidate; },
      [&](const std::string& candidate) {
        is_loaded = true;
        path = candidate;
        loaded_paths.push_back(candidate);
      },
      [&] { return encoder; },
      [&] { return capacity; },
      [&](const std::string& id, const nlohmann::json& input) {
        if (backpressure) throw clap::llama::CacheBackpressure("busy");
        clap::llama::PreparedRequest facts;
        facts.id = id;
        facts.params.max_tokens = 4;
        facts.prompt_tokens = {1};
        facts.full_prompt_tokens = {1};
        const std::string priority = input.value("cache_identity", nlohmann::json::object())
            .value("priority", "normal");
        facts.priority = priority == "interactive" ? CLAP_CACHE_PRIORITY_INTERACTIVE
            : priority == "background" ? CLAP_CACHE_PRIORITY_BACKGROUND : CLAP_CACHE_PRIORITY_NORMAL;
        prepared.push_back(id);
        auto active = std::make_unique<clap::llama::ActiveRequest>(std::move(facts));
        if (id.find("decode") != std::string::npos) {
          active->phase = clap::llama::ActiveRequest::Phase::Decode;
          active->pending_token = 2;
        }
        return active;
      },
      [&](const std::vector<clap::llama::ActiveRequest*>& ordered, bool) {
        stepped.clear();
        for (auto* request : ordered) stepped.push_back(request->id);
        if (finish_on_step && !ordered.empty()) {
          ordered.front()->mark_terminal(clap::llama::ActiveRequest::TerminalState::Completed);
        }
        return std::vector<clap::llama::GenerationEvent>{};
      },
      [&](clap::llama::ActiveRequest& request, bool flush) {
        return request.complete(flush, {});
      },
    };
  }
};

nlohmann::json request(const std::string& model) { return {{"model", model}}; }
nlohmann::json request(const std::string& model, const std::string& priority) {
  return {{"model", model}, {"cache_identity", {{"priority", priority}}}};
}

std::size_t count(const std::vector<clap::llama::SchedulerEvent>& events,
                  clap::llama::SchedulerEvent::Type type) {
  return static_cast<std::size_t>(std::count_if(events.begin(), events.end(),
      [type](const auto& event) { return event.type == type; }));
}

}  // namespace

int main() {
  {
    FakeState state;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("a", request("m"));
    scheduler.enqueue("b", request("m"));
    scheduler.enqueue("c", request("m"));
    const auto events = scheduler.tick();
    assert(state.prepared == std::vector<std::string>({"a", "b"}));
    assert(count(events, clap::llama::SchedulerEvent::Type::Started) == 2);
    assert(scheduler.active_count() == 2 && scheduler.queued_count() == 1);
  }

  {
    FakeState state;
    state.capacity = 3;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("prefill", request("m"));
    scheduler.enqueue("decode", request("m"));
    scheduler.tick();
    assert(state.stepped == std::vector<std::string>({"decode", "prefill"}));
    // The fake requests are mutable only through scheduler-owned state; cancel
    // proves empty target addresses every active request while leaving queued work.
    scheduler.enqueue("queued", request("m"));
    auto cancel_events = scheduler.cancel("");
    assert(cancel_events.empty());
    const auto events = scheduler.tick();
    assert(count(events, clap::llama::SchedulerEvent::Type::Completion) == 2);
    assert(scheduler.active_count() == 1);
  }

  {
    FakeState state;
    state.capacity = 1;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("active", request("one"));
    scheduler.tick();
    scheduler.enqueue("other", request("two"));
    scheduler.tick();
    assert(state.prepared == std::vector<std::string>({"active"}));
    assert(scheduler.queued_count() == 1);
  }

  {
    FakeState state;
    state.encoder = true;
    state.capacity = 4;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("one", request("m"));
    scheduler.enqueue("two", request("m"));
    scheduler.tick();
    assert(scheduler.active_count() == 1 && scheduler.queued_count() == 1);
  }

  {
    FakeState state;
    state.backpressure = true;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("first", request("m"));
    scheduler.enqueue("second", request("m"));
    scheduler.tick();
    assert(scheduler.active_count() == 0 && scheduler.queued_count() == 2);
    state.backpressure = false;
    scheduler.tick();
    assert(state.prepared == std::vector<std::string>({"first", "second"}));
  }

  {
    FakeState state;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("missing", nlohmann::json::object());
    const auto events = scheduler.tick();
    assert(events.front().type == clap::llama::SchedulerEvent::Type::Error);
    assert(events.front().message == "chat.model is required");
    assert(scheduler.idle());
  }

  {
    FakeState state;
    state.capacity = 1;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("active", request("m"));
    scheduler.tick();
    scheduler.enqueue("queued", request("m"));
    const auto ignored = scheduler.cancel("");
    assert(ignored.empty());
    assert(scheduler.queued_count() == 1);
    const auto removed = scheduler.cancel("queued");
    assert(count(removed, clap::llama::SchedulerEvent::Type::QueuedCancelled) == 1);
    assert(scheduler.queued_count() == 0);
  }

  {
    FakeState state;
    state.finish_on_step = true;
    clap::llama::Scheduler scheduler(state.callbacks());
    scheduler.enqueue("done", request("m"));
    const auto events = scheduler.tick();
    assert(count(events, clap::llama::SchedulerEvent::Type::Topology) == 1);
    assert(scheduler.idle());
  }

  {
    FakeState state;
    state.capacity = 7;
    clap::llama::Scheduler scheduler(state.callbacks());
    for (int index = 0; index < 8; ++index) {
      scheduler.enqueue("i" + std::to_string(index), request("m", "interactive"));
    }
    scheduler.enqueue("n", request("m", "normal"));
    scheduler.enqueue("b", request("m", "background"));
    scheduler.tick();
    assert(state.prepared == std::vector<std::string>({"i0", "i1", "i2", "i3", "n", "b", "i4"}));
    assert(scheduler.queued_count() == 3);
    assert(state.stepped == std::vector<std::string>({"i0", "i1", "i2", "i3", "i4", "n", "b"}));
    assert(scheduler.active_count() == 7);  // Priority never destructively preempts active work.
  }
}
