#include "clap/llama/protocol.h"

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

}  // namespace clap::llama
