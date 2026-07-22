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
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::Legacy);
    assert(!worker.dispatch(R"({"type":"shutdown","id":"bye"})"));
    const auto events = lines(output.str());
    assert(events.size() == 1);
    assert(events[0] == nlohmann::json({{"done", true}, {"id", "bye"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::Legacy);
    assert(worker.dispatch("{"));
    const auto events = lines(output.str());
    assert(events.size() == 1);
    assert(events[0]["error"].is_string());
    assert(!events[0].contains("id"));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::Legacy);
    worker.dispatch(R"({"type":"load","id":"load"})");
    worker.dispatch(R"({"type":"unload","id":"unload"})");
    const auto events = lines(output.str());
    assert(events[0] == nlohmann::json({{"error", "load.model is required"}, {"id", "load"}}));
    assert(events[1] == nlohmann::json({{"done", true}, {"unloaded", true}, {"id", "unload"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::Legacy);
    worker.dispatch(R"({"type":"set_max_active","id":"limit","max_active":0})");
    const auto events = lines(output.str());
    assert(events.size() == 1);
    assert(events[0] == nlohmann::json({
        {"error", "set_max_active.max_active must be positive"}, {"id", "limit"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::Legacy);
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
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::Legacy);
    assert(worker.run() == 0);
    const auto events = lines(output.str());
    assert(events.size() == 2);
    assert(events[0]["unloaded"] == true);
    assert(events[1] == nlohmann::json({{"done", true}, {"id", "s"}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::V1);
    assert(!worker.dispatch(
        R"({"protocol":1,"type":"shutdown","request_id":"req_shutdown"})"));
    const auto events = lines(output.str());
    assert(events.size() == 3);
    assert(events[0]["type"] == "ready");
    assert(!events[0].contains("request_id"));
    assert(events[1] == nlohmann::json({{"protocol", 1}, {"type", "accepted"},
        {"request_id", "req_shutdown"}, {"sequence", 0}}));
    assert(events[2] == nlohmann::json({{"protocol", 1}, {"type", "completed"},
        {"request_id", "req_shutdown"}, {"sequence", 1},
        {"result", {{"kind", "shutdown"}}}}));
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::V1);
    worker.dispatch(R"({"protocol":2,"type":"load","request_id":"bad","model":"m"})");
    const auto events = lines(output.str());
    assert(events.size() == 3);
    assert(events[1]["type"] == "accepted" && events[1]["sequence"] == 0);
    assert(events[2]["type"] == "failed" && events[2]["sequence"] == 1);
    assert(events[2]["error"]["code"] == "unsupported_protocol_version");
    assert(events[2]["error"]["message"].get<std::string>().find("expected 1") !=
        std::string::npos);
  }

  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::V1);
    worker.dispatch(
        R"({"protocol":1,"type":"generate","request_id":"target","prompt":"Hello","model":"missing.gguf"})");
    worker.dispatch(
        R"({"protocol":1,"type":"cancel","request_id":"cancel","target_request_id":"target"})");
    const auto events = lines(output.str());
    assert(events.size() == 6);
    assert(events[0]["type"] == "ready");
    assert(events[1]["type"] == "accepted" && events[1]["request_id"] == "target");
    assert(events[2]["type"] == "telemetry");
    assert(events[3]["type"] == "accepted" && events[3]["request_id"] == "cancel");
    assert(events[4]["type"] == "completed" && events[4]["request_id"] == "target");
    assert(events[4]["result"]["kind"] == "cancelled");
    assert(events[5]["type"] == "completed" && events[5]["request_id"] == "cancel");
    assert(events[5]["result"]["kind"] == "cancelled");
    // No late target event may pass the terminal guard.
    assert(events[4]["sequence"] == 1);
  }

  {
    std::istringstream input(
        R"({"protocol":1,"type":"generate","request_id":"waiting","prompt":"x","model":"missing.gguf"})" "\n"
        R"({"protocol":1,"type":"shutdown","request_id":"shutdown"})" "\n");
    std::ostringstream output;
    clap::llama::Worker worker(input, output, clap::llama::ProtocolMode::V1);
    assert(worker.run() == 0);
    const auto events = lines(output.str());
    assert(events.size() == 6);
    assert(events[0]["type"] == "ready");
    assert(events[1]["type"] == "accepted" && events[1]["request_id"] == "waiting");
    assert(events[3]["type"] == "accepted" && events[3]["request_id"] == "shutdown");
    assert(events[4]["type"] == "completed" && events[4]["request_id"] == "shutdown");
    assert(events[5]["type"] == "completed" && events[5]["request_id"] == "waiting");
    assert(events[5]["result"]["kind"] == "cancelled");
  }
}
