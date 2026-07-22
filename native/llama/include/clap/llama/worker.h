#pragma once

#include "clap/llama/protocol.h"
#include "clap/llama/scheduler.h"
#include "clap/llama/worker-state.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>

namespace clap::llama {

class Worker {
 public:
  Worker();
  Worker(std::istream& input, std::ostream& output);
  Worker(std::istream& input, std::ostream& output, ProtocolMode mode);
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  int run();
  bool dispatch(const std::string& line);

 private:
  void send(const std::string& id, nlohmann::json fields);
  void send_error(const std::string& id, const std::string& message,
                  const std::string& code = {});
  void send_completion(const RequestCompletion& completion);
  void send_failure(const RequestFailure& failure);
  void send_scheduler_events(const std::vector<SchedulerEvent>& events);
  bool dispatch_v1(const std::string& line);
  void send_v1_completion(const RequestCompletion& completion);
  void send_v1_scheduler_events(const std::vector<SchedulerEvent>& events);

  std::ostream& output_;
  ProtocolMode mode_;
  std::unique_ptr<ProtocolWriter> v1_;
  std::unordered_map<std::string, std::string> generated_content_;
  WorkerState state_;
  Scheduler scheduler_;
  StdinReader reader_;
};

}  // namespace clap::llama
