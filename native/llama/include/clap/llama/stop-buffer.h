#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace clap::llama {

std::size_t utf8_incomplete_suffix(const std::string& text);
std::size_t find_stop(const std::string& text, const std::vector<std::string>& stops);
std::size_t partial_stop_suffix(const std::string& text,
                                const std::vector<std::string>& stops);

struct StopBufferResult {
  std::string visible;
  bool stop_complete = false;
};

class StopBuffer {
 public:
  StopBuffer() = default;
  explicit StopBuffer(std::vector<std::string> stops);

  void reset(std::vector<std::string> stops);
  StopBufferResult append(const std::string& text);
  std::string finish();

 private:
  std::vector<std::string> stops_;
  std::string held_;
};

}  // namespace clap::llama
