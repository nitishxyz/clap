#pragma once

#include "llama.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace clap::llama {

class ModelRuntime;

struct ChatEntry {
  std::string role;
  std::string content;
};

struct PromptBoundaryRequest {
  std::string kind;
  std::string label;
  std::size_t message_count = 0;
};

struct PromptInput {
  std::vector<ChatEntry> messages;
  std::string prompt;
  std::vector<PromptBoundaryRequest> boundaries;
};

struct ResolvedPromptBoundary {
  std::size_t token_count = 0;
  std::string kind;
  std::string label;
  bool requested = false;
  std::string status;
  std::string skip_reason;
};

struct PreparedPrompt {
  std::vector<llama_token> tokens;
  std::vector<uint64_t> stable_boundaries;
  std::vector<std::size_t> structural_boundaries;
  std::vector<ResolvedPromptBoundary> resolved_boundaries;
};

PromptInput prompt_input_from_request(const nlohmann::json& request);
std::string render_turn_prompt(const std::vector<ChatEntry>& entries,
                               bool add_generation_prompt);
std::string fallback_prompt(const std::vector<ChatEntry>& entries);
std::optional<std::size_t> exact_prompt_boundary(
    const std::vector<llama_token>& prefix,
    const std::vector<llama_token>& final,
    const std::function<bool(llama_token)>& is_terminal);

class PromptRenderer {
 public:
  explicit PromptRenderer(const ModelRuntime& runtime) : runtime_(runtime) {}

  PreparedPrompt prepare(const PromptInput& input) const;

 private:
  std::string render(const std::vector<ChatEntry>& entries,
                     bool add_generation_prompt) const;
  std::vector<llama_token> tokenize(const std::string& prompt) const;

  const ModelRuntime& runtime_;
};

}  // namespace clap::llama
