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

nlohmann::json identity() {
  return {{"version", 1}, {"generation", "sec_fixture"},
    {"tenant_root", std::string(64, 'a')}, {"scope", "tenant"},
    {"scope_fingerprint", std::string(64, 'a')},
    {"namespace_fingerprint", std::string(64, 'b')},
    {"namespace_id", "13527612320720337851"}, {"priority", "interactive"},
    {"side_request", false}, {"display", nlohmann::json::object()},
    {"physical", {{"fingerprint", std::string(64, 'c')}, {"backend", "llama"},
      {"resolved_revision", "local:fixture"},
      {"model_artifact_fingerprint", std::string(64, 'd')},
      {"tokenizer_fingerprint", std::string(64, 'd')}, {"context_allocation", 4096},
      {"kv_format", "f16"}, {"unified_kv", true}, {"layout_version", 1}}}};
}

}  // namespace

int main() {
  {
    std::istringstream input;
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
    assert(!worker.dispatch(
        R"({"protocol":1,"type":"shutdown","request_id":"req_shutdown"})"));
    const auto events = lines(output.str());
    assert(events.size() == 3);
    assert(events[0]["type"] == "ready");
    assert(events[0]["worker_capabilities"]["backend"] == "llama");
    assert(events[0]["worker_capabilities"]["scheduling"] == nlohmann::json({
        {"fused_multi_sequence_batching", true}, {"interleaved", true},
        {"priority_aware", true}}));
    assert(events[0]["model_capabilities"].is_null());
    assert(!events[0].contains("structured_output"));
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
    clap::llama::Worker worker(input, output);
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
    clap::llama::Worker worker(input, output);
    worker.dispatch(nlohmann::json{{"protocol", 1}, {"type", "generate"},
        {"request_id", "target"}, {"prompt", "Hello"}, {"model", "missing.gguf"},
        {"cache_identity", identity()}}.dump());
    worker.dispatch(
        R"({"protocol":1,"type":"cancel","request_id":"cancel","target_request_id":"target"})");
    const auto events = lines(output.str());
    assert(events.size() == 6);
    assert(events[0]["type"] == "ready");
    assert(events[1]["type"] == "accepted" && events[1]["request_id"] == "target");
    assert(events[2]["type"] == "telemetry");
    const auto& retention = events[2]["telemetry"]["retention"];
    assert(retention["retained_bytes"].is_null());
    assert(retention["retained_bytes_source"] == "unavailable");
    assert(retention["retained_bytes_basis"] == "not_observed");
    assert(retention["evicted_bytes"].is_null());
    assert(retention["evicted_bytes_source"] == "unavailable");
    assert(retention["estimated_retained_bytes"].is_null());
    assert(retention["estimated_retained_bytes_source"] == "unavailable");
    assert(events[3]["type"] == "accepted" && events[3]["request_id"] == "cancel");
    assert(events[4]["type"] == "completed" && events[4]["request_id"] == "target");
    assert(events[4]["result"]["kind"] == "cancelled");
    assert(events[5]["type"] == "completed" && events[5]["request_id"] == "cancel");
    assert(events[5]["result"]["kind"] == "cancelled");
    // No late target event may pass the terminal guard.
    assert(events[4]["sequence"] == 1);
  }

  {
    const std::string commands = nlohmann::json{{"protocol", 1}, {"type", "generate"},
        {"request_id", "waiting"}, {"prompt", "x"}, {"model", "missing.gguf"},
        {"cache_identity", identity()}}.dump() + "\n" +
        R"({"protocol":1,"type":"shutdown","request_id":"shutdown"})" + "\n";
    std::istringstream input(commands);
    std::ostringstream output;
    clap::llama::Worker worker(input, output);
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
