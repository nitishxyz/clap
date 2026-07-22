#include "native-characterization.h"

#include <cassert>
#include <string>
#include <vector>

using clap::llama_native::SchedulePhase;
using clap::llama_native::ScheduleRequest;

int main() {
  const std::string euro = "\xE2\x82\xAC";
  assert(clap::llama_native::utf8_incomplete_suffix(euro.substr(0, 1)) == 1);
  assert(clap::llama_native::utf8_incomplete_suffix(euro.substr(0, 2)) == 2);
  assert(clap::llama_native::utf8_incomplete_suffix(euro) == 0);
  assert(clap::llama_native::utf8_incomplete_suffix("plain") == 0);

  const std::vector<std::string> stops = {"</stop>", "stop"};
  assert(clap::llama_native::partial_stop_suffix("answer</st", stops) == 4);
  assert(clap::llama_native::find_stop("answer</stop>tail", stops) == 6);
  assert(clap::llama_native::find_stop("no match", stops) == std::string::npos);

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
