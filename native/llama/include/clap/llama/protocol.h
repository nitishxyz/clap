#pragma once

#include <condition_variable>
#include <deque>
#include <istream>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>

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

}  // namespace clap::llama
