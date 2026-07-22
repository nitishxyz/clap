#include "native-characterization.h"

#include <cassert>
#include <string>
#include <vector>

using clap::llama_native::SchedulePhase;
using clap::llama_native::ScheduleRequest;

int main() {
  const std::vector<std::string> ids = {"same", "prefill", "same", "done"};
  const auto order = clap::llama_native::decode_first_order({
      {SchedulePhase::Prefill, true},
      {SchedulePhase::Prefill, true},
      {SchedulePhase::Decode, true},
      {SchedulePhase::Decode, false},
  });
  assert((order == std::vector<std::size_t>{2, 0, 1}));
  assert(ids[order[0]] == "same");
  assert(ids[order[1]] == "same");
  assert(ids[order[2]] == "prefill");

  assert(clap::llama_native::active_cancel_matches("", "active"));
  assert(!clap::llama_native::queued_cancel_matches("", "queued"));
  assert(clap::llama_native::active_cancel_matches("mixed-2", "mixed-2"));
  assert(clap::llama_native::queued_cancel_matches("mixed-2", "mixed-2"));
  assert(!clap::llama_native::queued_cancel_matches("mixed-2", "mixed-1"));
}
