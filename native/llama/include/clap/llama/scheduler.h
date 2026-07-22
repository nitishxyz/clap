#pragma once

#include "clap/llama/generation-stepper.h"
#include "clap/llama/request-state.h"

#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace clap::llama {

struct SchedulerState {
  std::function<bool()> loaded;
  std::function<bool(const std::string&)> same_path;
  std::function<void(const std::string&)> load;
  std::function<bool()> has_encoder;
  std::function<int32_t()> max_active;
  std::function<std::unique_ptr<ActiveRequest>(const std::string&, const nlohmann::json&)> prepare;
  std::function<std::vector<GenerationEvent>(const std::vector<ActiveRequest*>&, bool)> step;
  std::function<std::optional<RequestCompletion>(ActiveRequest&, bool)> complete;
};

struct SchedulerEvent {
  enum class Type { Started, QueuedCancelled, Error, Generation, Completion, Topology };

  Type type = Type::Topology;
  std::string id;
  std::string message;
  std::string code;
  std::optional<GenerationEvent> generation;
  std::optional<RequestCompletion> completion;
  std::size_t active = 0;
  std::size_t queued = 0;
};

class Scheduler {
 public:
  explicit Scheduler(SchedulerState state);
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  void enqueue(std::string id, nlohmann::json request);
  std::vector<SchedulerEvent> cancel(const std::string& target);
  std::vector<SchedulerEvent> tick();
  std::vector<SchedulerEvent> cancel_all();

  bool idle() const noexcept { return active_.empty() && waiting_.empty(); }
  bool has_active() const noexcept { return !active_.empty(); }
  std::size_t active_count() const noexcept { return active_.size(); }
  std::size_t queued_count() const noexcept { return waiting_.size(); }

 private:
  SchedulerEvent topology() const;
  std::vector<ActiveRequest*> decode_first() const;
  void admit(std::vector<SchedulerEvent>& events);
  void generate(std::vector<SchedulerEvent>& events);

  SchedulerState state_;
  std::vector<std::unique_ptr<ActiveRequest>> active_;
  std::deque<std::pair<std::string, nlohmann::json>> waiting_;
};

}  // namespace clap::llama
