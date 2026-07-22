#include "clap/llama/protocol.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <string>

namespace {

class SyncCountingBuffer : public std::stringbuf {
 public:
  int sync() override {
    ++sync_count;
    return std::stringbuf::sync();
  }

  int sync_count = 0;
};

nlohmann::json fixture_event(const std::string& id, const std::string& required_field = "") {
  std::ifstream fixture(CLAP_LLAMA_PROTOCOL_FIXTURE);
  assert(fixture);
  std::string line;
  while (std::getline(fixture, line)) {
    const auto event = nlohmann::json::parse(line);
    if (event.value("id", "") == id &&
        (required_field.empty() || event.contains(required_field))) return event;
  }
  assert(false);
  return {};
}

nlohmann::json emitted_event(const std::string& output) {
  assert(!output.empty() && output.back() == '\n');
  return nlohmann::json::parse(output);
}

}  // namespace

int main() {
  SyncCountingBuffer buffer;
  std::ostream output(&buffer);

  clap::llama::emit("req_chat", nlohmann::json{{"token", "Hello"}}, output);
  assert(buffer.sync_count == 1);
  assert(emitted_event(buffer.str()) == fixture_event("req_chat", "token"));

  buffer.str("");
  clap::llama::emit("", nlohmann::json{{"retention", nlohmann::json::object()}}, output);
  assert(!emitted_event(buffer.str()).contains("id"));

  buffer.str("");
  clap::llama::emit_error("req_error", "cache coordinator advance failed closed",
                          "cache_coordinator_error", output);
  assert(emitted_event(buffer.str()) == fixture_event("req_error"));

  buffer.str("");
  clap::llama::emit_error("req_uncoded_error", "chat.model is required", "", output);
  assert(emitted_event(buffer.str()) == fixture_event("req_uncoded_error"));

  buffer.str("");
  clap::llama::emit("invalid", nlohmann::json{{"token", std::string("bad\xFF", 4)}}, output);
  const auto replaced = emitted_event(buffer.str());
  assert(replaced["token"] == "bad\xEF\xBF\xBD");

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
}
