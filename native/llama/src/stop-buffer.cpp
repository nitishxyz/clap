#include "clap/llama/stop-buffer.h"

#include <algorithm>
#include <utility>

namespace clap::llama {

std::size_t utf8_incomplete_suffix(const std::string& text) {
  const std::size_t size = text.size();
  std::size_t cont = 0;
  while (cont < 3 && cont < size &&
         (static_cast<unsigned char>(text[size - 1 - cont]) & 0xC0) == 0x80) {
    cont += 1;
  }
  if (cont >= size) return 0;
  const unsigned char lead = static_cast<unsigned char>(text[size - 1 - cont]);
  std::size_t need = 0;
  if ((lead & 0x80) == 0) need = 1;
  else if ((lead & 0xE0) == 0xC0) need = 2;
  else if ((lead & 0xF0) == 0xE0) need = 3;
  else if ((lead & 0xF8) == 0xF0) need = 4;
  else return 0;
  const std::size_t have = cont + 1;
  return have < need ? have : 0;
}

std::size_t find_stop(const std::string& text, const std::vector<std::string>& stops) {
  std::size_t earliest = std::string::npos;
  for (const auto& stop : stops) {
    if (stop.empty()) continue;
    const std::size_t index = text.find(stop);
    if (index != std::string::npos && index < earliest) earliest = index;
  }
  return earliest;
}

std::size_t partial_stop_suffix(const std::string& text,
                                const std::vector<std::string>& stops) {
  std::size_t max_stop = 0;
  for (const auto& stop : stops) max_stop = std::max(max_stop, stop.size());
  const std::size_t max = std::min(text.size(), max_stop);
  for (std::size_t length = max; length > 0; --length) {
    const std::string suffix = text.substr(text.size() - length);
    for (const auto& stop : stops) {
      if (stop.size() > suffix.size() && stop.compare(0, suffix.size(), suffix) == 0) {
        return length;
      }
    }
  }
  return 0;
}

StopBuffer::StopBuffer(std::vector<std::string> stops) : stops_(std::move(stops)) {}

void StopBuffer::reset(std::vector<std::string> stops) {
  stops_ = std::move(stops);
  held_.clear();
}

StopBufferResult StopBuffer::append(const std::string& text) {
  held_ += text;
  if (!stops_.empty()) {
    const std::size_t stop_index = find_stop(held_, stops_);
    if (stop_index != std::string::npos) {
      std::string visible = held_.substr(0, stop_index);
      visible.resize(visible.size() - utf8_incomplete_suffix(visible));
      held_.clear();
      return {std::move(visible), true};
    }
    const std::size_t stop_hold = partial_stop_suffix(held_, stops_);
    const std::size_t utf8_hold = utf8_incomplete_suffix(held_);
    const std::size_t hold = std::max(stop_hold, utf8_hold);
    std::string visible = held_.substr(0, held_.size() - hold);
    held_ = held_.substr(visible.size());
    return {std::move(visible), false};
  }

  const std::size_t hold = utf8_incomplete_suffix(held_);
  std::string visible = held_.substr(0, held_.size() - hold);
  held_ = held_.substr(visible.size());
  return {std::move(visible), false};
}

std::string StopBuffer::finish() {
  held_.resize(held_.size() - utf8_incomplete_suffix(held_));
  std::string visible = std::move(held_);
  held_.clear();
  return visible;
}

}  // namespace clap::llama
