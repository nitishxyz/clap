#include "clap/llama/prompt.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
  const auto input = clap::llama::prompt_input_from_request({
      {"messages", nlohmann::json::array({
          nlohmann::json{{"role", "system"}, {"content", "rules"}},
          nlohmann::json{{"role", "user"}, {"content", "hello"}},
          nlohmann::json{{"role", ""}, {"content", "ignored"}},
          4,
      })},
      {"prompt", "raw"},
      {"cache", {{"boundaries", nlohmann::json::array({
          nlohmann::json{{"kind", "messages"}, {"label", "turn"}, {"through_message", 1}},
          nlohmann::json{{"kind", "tools"}, {"label", "tools"}},
      })}}},
  });
  assert(input.messages.size() == 2);
  assert(input.messages[0].role == "system" && input.messages[0].content == "rules");
  assert(input.prompt == "raw");
  assert(input.boundaries.size() == 2);
  assert(input.boundaries[0].message_count == 2);
  assert(input.boundaries[1].message_count == 0);

  const std::vector<clap::llama::ChatEntry> entries = {
      {"system", "rules"}, {"assistant", "answer"}, {"tool", "result"}};
  assert(clap::llama::render_turn_prompt(entries, true) ==
      "<bos><|turn>system\nrules<turn|>\n<|turn>model\nanswer<turn|>\n"
      "<|turn>user\nresult<turn|>\n<|turn>model\n");
  assert(clap::llama::render_turn_prompt(entries, false).find("<|turn>model\n", 10) !=
      std::string::npos);
  assert(clap::llama::fallback_prompt(entries) == "rules\nanswer\nresult\n");

  const std::vector<llama_token> final = {1, 2, 3, 4};
  auto boundary = clap::llama::exact_prompt_boundary({1, 2}, final,
      [](llama_token) { return false; });
  assert(boundary && *boundary == 2);
  boundary = clap::llama::exact_prompt_boundary({1, 2, 99, 100}, final,
      [](llama_token token) { return token == 99 || token == 100; });
  assert(boundary && *boundary == 2);
  boundary = clap::llama::exact_prompt_boundary({1, 2, 99}, final,
      [](llama_token) { return false; });
  assert(!boundary);
  boundary = clap::llama::exact_prompt_boundary({}, final,
      [](llama_token) { return true; });
  assert(!boundary);
}
