#pragma once

#include <condition_variable>
#include <deque>
#include <istream>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace clap::llama {

class StdinReader {
 public:
  StdinReader();
  explicit StdinReader(std::istream& input);
  ~StdinReader();

  StdinReader(const StdinReader&) = delete;
  StdinReader& operator=(const StdinReader&) = delete;

  bool next(std::string& out);
  bool poll(std::string& out);

 private:
  StdinReader(std::istream& input, bool detach_on_destroy);
  void run();

  std::istream& input_;
  bool detach_on_destroy_;
  std::deque<std::string> lines_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool eof_ = false;
  std::thread thread_;
};

void emit(const std::string& id, nlohmann::json fields);
void emit(const std::string& id, nlohmann::json fields, std::ostream& output);
void emit_error(const std::string& id, const std::string& message);
void emit_error(const std::string& id, const std::string& message, std::ostream& output);
void emit_error(const std::string& id, const std::string& message, const std::string& code);
void emit_error(const std::string& id, const std::string& message, const std::string& code,
                std::ostream& output);

struct RequestError : std::runtime_error {
  std::string code;
  RequestError(std::string error_code, const std::string& message);
};

struct V1Request {
  std::string type;
  std::string request_id;
  std::string target_request_id;
  nlohmann::json body;
};

struct V1DecodeError : std::runtime_error {
  std::string code;
  std::string request_id;
  V1DecodeError(std::string error_code, std::string recoverable_request_id,
                const std::string& message);
};

V1Request decode_v1_request(const std::string& line);

class ProtocolWriter {
 public:
  explicit ProtocolWriter(std::ostream& output) : output_(output) {}

  void ready(nlohmann::json worker_capabilities, nlohmann::json model_capabilities);
  bool accepted(const std::string& request_id);
  bool started(const std::string& request_id);
  bool token(const std::string& request_id, const std::string& text);
  bool content(const std::string& request_id, nlohmann::json value);
  bool prefill_progress(const std::string& request_id, uint64_t completed, uint64_t total);
  bool completed(const std::string& request_id, nlohmann::json result);
  bool failed(const std::string& request_id, const std::string& code,
              const std::string& message, bool retryable = false, bool fatal = false,
              nlohmann::json details = nullptr);
  void retention_telemetry(nlohmann::json value);
  bool terminal(const std::string& request_id) const;

 private:
  bool scoped(const std::string& request_id, const char* type, nlohmann::json fields,
              bool terminal);

  std::ostream& output_;
  std::unordered_map<std::string, uint64_t> sequences_;
  std::unordered_set<std::string> terminals_;
};

}  // namespace clap::llama
