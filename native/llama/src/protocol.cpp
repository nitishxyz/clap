#include "clap/llama/protocol.h"

#include <cstdlib>
#include <iostream>
#include <utility>

namespace clap::llama {

StdinReader::StdinReader() : StdinReader(std::cin, true) {}

StdinReader::StdinReader(std::istream& input) : StdinReader(input, false) {}

StdinReader::StdinReader(std::istream& input, bool detach_on_destroy)
    : input_(input), detach_on_destroy_(detach_on_destroy), thread_([this] { run(); }) {}

StdinReader::~StdinReader() {
  if (!thread_.joinable()) return;
  if (detach_on_destroy_) {
    thread_.detach();
  } else {
    thread_.join();
  }
}

bool StdinReader::next(std::string& out) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !lines_.empty() || eof_; });
  if (lines_.empty()) return false;
  out = std::move(lines_.front());
  lines_.pop_front();
  return true;
}

bool StdinReader::poll(std::string& out) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (lines_.empty()) return false;
  out = std::move(lines_.front());
  lines_.pop_front();
  return true;
}

void StdinReader::run() {
  std::string line;
  while (std::getline(input_, line)) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      lines_.push_back(line);
    }
    cv_.notify_all();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    eof_ = true;
  }
  cv_.notify_all();
}

void emit(const std::string& id, nlohmann::json fields) {
  emit(id, std::move(fields), std::cout);
}

void emit(const std::string& id, nlohmann::json fields, std::ostream& output) {
  if (!id.empty()) fields["id"] = id;
  // error_handler_t::replace: never throw on stray invalid UTF-8 bytes from
  // byte-level BPE pieces; substitute U+FFFD instead of killing the request.
  output << fields.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
  output.flush();
}

void emit_error(const std::string& id, const std::string& message) {
  emit_error(id, message, std::cout);
}

void emit_error(const std::string& id, const std::string& message, std::ostream& output) {
  emit(id, nlohmann::json{{"error", message}}, output);
}

void emit_error(const std::string& id, const std::string& message, const std::string& code) {
  emit_error(id, message, code, std::cout);
}

void emit_error(const std::string& id, const std::string& message, const std::string& code,
                std::ostream& output) {
  if (code.empty()) {
    emit_error(id, message, output);
  } else {
    emit(id, nlohmann::json{{"error", message}, {"code", code}}, output);
  }
}

RequestError::RequestError(std::string error_code, const std::string& message)
    : std::runtime_error(message), code(std::move(error_code)) {}

ProtocolMode protocol_mode_from_environment() {
  const char* value = std::getenv("CLAP_WORKER_PROTOCOL");
  return value && std::string(value) == "legacy" ? ProtocolMode::Legacy : ProtocolMode::V1;
}

V1DecodeError::V1DecodeError(std::string error_code, std::string recoverable_request_id,
                             const std::string& message)
    : std::runtime_error(message), code(std::move(error_code)),
      request_id(std::move(recoverable_request_id)) {}

namespace {

std::string recoverable_id(const nlohmann::json& value) {
  if (value.is_object() && value.contains("request_id") && value["request_id"].is_string() &&
      !value["request_id"].get_ref<const std::string&>().empty()) {
    return value["request_id"].get<std::string>();
  }
  return "";
}

void require_string(const nlohmann::json& value, const char* key, const std::string& id) {
  if (!value.contains(key) || !value[key].is_string() || value[key].get_ref<const std::string&>().empty()) {
    throw V1DecodeError("invalid_request", id,
        std::string(key) + " must be a non-empty string");
  }
}

}  // namespace

V1Request decode_v1_request(const std::string& line) {
  nlohmann::json value;
  try {
    value = nlohmann::json::parse(line);
  } catch (const std::exception& error) {
    throw V1DecodeError("malformed_json", "",
        std::string("Malformed worker request JSON: ") + error.what());
  }
  if (!value.is_object()) throw V1DecodeError("invalid_request", "", "Worker request must be an object");
  const std::string id = recoverable_id(value);
  if (!value.contains("protocol") || !value["protocol"].is_number_integer() ||
      value["protocol"].get<int64_t>() != 1) {
    const std::string actual = value.contains("protocol") ? value["protocol"].dump() : "missing";
    throw V1DecodeError("unsupported_protocol_version", id,
        "Unsupported worker protocol version " + actual + "; expected 1");
  }
  require_string(value, "request_id", id);
  require_string(value, "type", id);
  const std::string type = value["type"].get<std::string>();
  if (type == "load") {
    require_string(value, "model", id);
  } else if (type == "generate") {
    if (!value.contains("prompt") || !value["prompt"].is_string()) {
      throw V1DecodeError("invalid_request", id, "prompt must be a string");
    }
  } else if (type == "cancel") {
    require_string(value, "target_request_id", id);
  } else if (type == "set_max_active") {
    if (!value.contains("max_active") || !value["max_active"].is_number_integer() ||
        value["max_active"].get<int64_t>() <= 0) {
      throw V1DecodeError("invalid_request", id, "max_active must be a positive integer");
    }
  } else if (type != "unload" && type != "shutdown") {
    throw V1DecodeError("unsupported_request_type", id,
        "Unsupported worker request type: " + type);
  }
  return {type, id, type == "cancel" ? value["target_request_id"].get<std::string>() : "",
          std::move(value)};
}

void ProtocolWriter::ready(nlohmann::json worker_capabilities,
                           nlohmann::json model_capabilities) {
  emit("", {{"protocol", 1}, {"type", "ready"},
      {"worker_capabilities", std::move(worker_capabilities)},
      {"model_capabilities", std::move(model_capabilities)}}, output_);
}

bool ProtocolWriter::scoped(const std::string& request_id, const char* type,
                            nlohmann::json fields, bool is_terminal) {
  if (request_id.empty() || terminals_.count(request_id)) return false;
  uint64_t& next = sequences_[request_id];
  fields["protocol"] = 1;
  fields["type"] = type;
  fields["request_id"] = request_id;
  fields["sequence"] = next++;
  emit("", std::move(fields), output_);
  if (is_terminal) terminals_.insert(request_id);
  return true;
}

bool ProtocolWriter::accepted(const std::string& id) {
  if (sequences_.count(id) || terminals_.count(id)) return false;
  return scoped(id, "accepted", {}, false);
}
bool ProtocolWriter::started(const std::string& id) { return scoped(id, "started", {}, false); }
bool ProtocolWriter::token(const std::string& id, const std::string& text) {
  return scoped(id, "token", {{"text", text}}, false);
}
bool ProtocolWriter::content(const std::string& id, nlohmann::json value) {
  return scoped(id, "content", {{"content", std::move(value)}}, false);
}
bool ProtocolWriter::prefill_progress(const std::string& id, uint64_t done, uint64_t total) {
  return scoped(id, "prefill_progress", {{"completed", done}, {"total", total}}, false);
}
bool ProtocolWriter::completed(const std::string& id, nlohmann::json result) {
  return scoped(id, "completed", {{"result", std::move(result)}}, true);
}
bool ProtocolWriter::failed(const std::string& id, const std::string& code,
                            const std::string& message, bool retryable, bool fatal,
                            nlohmann::json details) {
  nlohmann::json error{{"code", code}, {"message", message},
                       {"retryable", retryable}, {"fatal", fatal}};
  if (!details.is_null()) error["details"] = std::move(details);
  return scoped(id, "failed", {{"error", std::move(error)}}, true);
}
void ProtocolWriter::telemetry(nlohmann::json value) {
  emit("", {{"protocol", 1}, {"type", "telemetry"},
             {"telemetry", std::move(value)}}, output_);
}
bool ProtocolWriter::terminal(const std::string& id) const { return terminals_.count(id) != 0; }

}  // namespace clap::llama
