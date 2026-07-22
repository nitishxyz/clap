#include "clap/llama/protocol.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace {

std::vector<nlohmann::json> fixture_lines(const char* path) {
  std::ifstream fixture(path);
  assert(fixture);
  std::vector<nlohmann::json> result;
  std::string line;
  while (std::getline(fixture, line)) {
    if (!line.empty()) result.push_back(nlohmann::json::parse(line));
  }
  return result;
}

}  // namespace

int main() {
  std::istringstream input("first\n{\"type\":\"shutdown\"}\n");
  clap::llama::StdinReader reader(input);
  std::string line;
  assert(reader.next(line) && line == "first");
  assert(reader.next(line) && line == "{\"type\":\"shutdown\"}");
  assert(!reader.next(line));
  assert(!reader.poll(line));

  const clap::llama::RequestError error("context_length_exceeded", "too long");
  assert(error.code == "context_length_exceeded");
  assert(std::string(error.what()) == "too long");

  const auto requests = fixture_lines(CLAP_LLAMA_PROTOCOL_V1_REQUEST_FIXTURE);
  assert(requests.size() == 6);
  for (const auto& fixture : requests) {
    const auto decoded = clap::llama::decode_v1_request(fixture.dump());
    assert(decoded.type == fixture["type"]);
    assert(decoded.request_id == fixture["request_id"]);
    if (decoded.type == "cancel") assert(decoded.target_request_id == fixture["target_request_id"]);
  }

  try {
    clap::llama::decode_v1_request(
        R"({"protocol":2,"type":"load","request_id":"bad","model":"m"})");
    assert(false);
  } catch (const clap::llama::V1DecodeError& failure) {
    assert(failure.code == "unsupported_protocol_version");
    assert(failure.request_id == "bad");
    assert(std::string(failure.what()).find("expected 1") != std::string::npos);
  }
  try {
    clap::llama::decode_v1_request(
        R"({"protocol":1,"type":"cancel","request_id":"cancel"})");
    assert(false);
  } catch (const clap::llama::V1DecodeError& failure) {
    assert(failure.code == "invalid_request");
    assert(failure.request_id == "cancel");
  }

  const auto event_fixtures = fixture_lines(CLAP_LLAMA_PROTOCOL_V1_EVENT_FIXTURE);
  std::ostringstream v1_output;
  clap::llama::ProtocolWriter writer(v1_output);
  writer.ready({{"backend", "fixture"}, {"streaming", true}}, {{"context_window", 4096}});
  assert(writer.accepted("req_generate"));
  assert(writer.started("req_generate"));
  assert(writer.token("req_generate", "Hello"));
  assert(writer.content("req_generate", {{"type", "text"}, {"text", "Hello"}}));
  assert(writer.prefill_progress("req_generate", 8, 16));
  assert(writer.completed("req_generate", {{"kind", "generated"}, {"content", "Hello"},
                                             {"finish_reason", "stop"}}));
  assert(!writer.token("req_generate", "late"));
  assert(!writer.completed("req_generate", {{"kind", "generated"}, {"content", "late"}}));
  const auto written = [&] {
    std::vector<nlohmann::json> values;
    std::istringstream stream(v1_output.str());
    std::string value;
    while (std::getline(stream, value)) values.push_back(nlohmann::json::parse(value));
    return values;
  }();
  assert(written.size() == 6);
  for (std::size_t index = 0; index < written.size(); ++index) {
    assert(written[index] == event_fixtures[index]);
  }
}
