#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace clap::llama_native {

inline std::size_t utf8_incomplete_suffix(const std::string& text) {
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

inline std::size_t find_stop(const std::string& text,
                             const std::vector<std::string>& stops) {
  std::size_t earliest = std::string::npos;
  for (const auto& stop : stops) {
    if (stop.empty()) continue;
    const std::size_t index = text.find(stop);
    if (index != std::string::npos && index < earliest) earliest = index;
  }
  return earliest;
}

inline std::size_t partial_stop_suffix(const std::string& text,
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

enum class SchedulePhase { Prefill, Decode };

struct ScheduleRequest {
  SchedulePhase phase;
  bool runnable;
};

inline std::vector<std::size_t> decode_first_order(
    const std::vector<ScheduleRequest>& requests) {
  std::vector<std::size_t> order;
  order.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if (requests[index].runnable && requests[index].phase == SchedulePhase::Decode) {
      order.push_back(index);
    }
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if (requests[index].runnable && requests[index].phase == SchedulePhase::Prefill) {
      order.push_back(index);
    }
  }
  return order;
}

inline bool active_cancel_matches(const std::string& target, const std::string& id) {
  return target.empty() || target == id;
}

inline bool queued_cancel_matches(const std::string& target, const std::string& id) {
  return !target.empty() && target == id;
}

}  // namespace clap::llama_native
