#pragma once

#include "clap/llama/protocol.h"
#include "clap/llama/scheduler.h"
#include "clap/llama/worker-state.h"

#include <iosfwd>
#include <string>

namespace clap::llama {

class Worker {
 public:
  Worker();
  Worker(std::istream& input, std::ostream& output);
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

  std::ostream& output_;
  WorkerState state_;
  Scheduler scheduler_;
  StdinReader reader_;
};

}  // namespace clap::llama
