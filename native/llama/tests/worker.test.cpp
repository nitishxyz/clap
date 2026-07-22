#include "clap/llama/worker.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

std::vector<nlohmann::json> lines(const std::string& output) {
  std::vector<nlohmann::json> result;
  std::istringstream input(output);
  std::string line;
  while (std::getline(input, line)) result.push_back(nlohmann::json::parse(line));
  return result;
}

}  // namespace

int main() {
  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    assert(!worker.dispatch(R"({"type":"shutdown","id":"bye"})"));
    const auto events = lines(output.str());
    assert(events.size() == 1);
    assert(events[0] == nlohmann::json({{"done", true}, {"id", "bye"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    assert(worker.dispatch("{"));
    const auto events = lines(output.str());
    assert(events.size() == 1);
    assert(events[0]["error"].is_string());
    assert(!events[0].contains("id"));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    worker.dispatch(R"({"type":"load","id":"load"})");
    worker.dispatch(R"({"type":"unload","id":"unload"})");
    const auto events = lines(output.str());
    assert(events[0] == nlohmann::json({{"error", "load.model is required"}, {"id", "load"}}));
    assert(events[1] == nlohmann::json({{"done", true}, {"unloaded", true}, {"id", "unload"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    worker.dispatch(R"({"type":"set_max_active","id":"limit","max_active":0})");
    const auto events = lines(output.str());
    assert(events.size() == 1);
    assert(events[0] == nlohmann::json({
        {"error", "set_max_active.max_active must be positive"}, {"id", "limit"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    worker.dispatch(R"({"id":"queued","model":"missing.gguf"})");
    worker.dispatch(R"({"type":"cancel","id":"queued"})");
    const auto events = lines(output.str());
    assert(events.size() == 2);
    assert(events[0].contains("retention"));
    assert(!events[0].contains("id"));
    assert(events[1] == nlohmann::json({
        {"done", true}, {"finish_reason", "cancel"},
        {"cancelled", true}, {"id", "queued"}}));
  }

  {
    std::istringstream input(
        R"({"type":"unload","id":"u"})" "\n"
        R"({"type":"shutdown","id":"s"})" "\n");
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    assert(worker.run() == 0);
    const auto events = lines(output.str());
    assert(events.size() == 2);
    assert(events[0]["unloaded"] == true);
    assert(events[1] == nlohmann::json({{"done", true}, {"id", "s"}}));
  }
}
