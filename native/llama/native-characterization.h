#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace clap::llama_native {

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
