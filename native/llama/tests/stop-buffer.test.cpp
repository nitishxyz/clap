#include "clap/llama/stop-buffer.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
  const std::string euro = "\xE2\x82\xAC";
  assert(clap::llama::utf8_incomplete_suffix(euro.substr(0, 1)) == 1);
  assert(clap::llama::utf8_incomplete_suffix(euro.substr(0, 2)) == 2);
  assert(clap::llama::utf8_incomplete_suffix(euro) == 0);
  assert(clap::llama::utf8_incomplete_suffix("plain") == 0);

  const std::vector<std::string> stops = {"</stop>", "stop"};
  assert(clap::llama::partial_stop_suffix("answer</st", stops) == 4);
  assert(clap::llama::find_stop("answer</stop>tail", stops) == 6);
  assert(clap::llama::find_stop("no match", stops) == std::string::npos);

  clap::llama::StopBuffer earliest({"later", "stop"});
  auto result = earliest.append("before stop and later");
  assert(result.visible == "before ");
  assert(result.stop_complete);
  assert(earliest.finish().empty());

  clap::llama::StopBuffer partial({"</stop>"});
  result = partial.append("answer</st");
  assert(result.visible == "answer");
  assert(!result.stop_complete);
  result = partial.append("op>ignored");
  assert(result.visible.empty());
  assert(result.stop_complete);

  clap::llama::StopBuffer utf8;
  result = utf8.append(euro.substr(0, 2));
  assert(result.visible.empty());
  assert(!result.stop_complete);
  result = utf8.append(euro.substr(2));
  assert(result.visible == euro);
  assert(!result.stop_complete);

  clap::llama::StopBuffer normal_completion({"done"});
  result = normal_completion.append("visibledo");
  assert(result.visible == "visible");
  assert(!result.stop_complete);
  assert(normal_completion.finish() == "do");

  clap::llama::StopBuffer cancellation({"done"});
  result = cancellation.append("visibledo");
  assert(result.visible == "visible");
  assert(!result.stop_complete);
  // Cancellation abandons the request without calling finish(), so "do" is not visible.
}
