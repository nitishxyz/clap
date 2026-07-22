#include "clap/llama/prompt.h"

#include "clap/llama/model-runtime.h"

#include <algorithm>
#include <stdexcept>

namespace clap::llama {

PromptInput prompt_input_from_request(const nlohmann::json& request) {
  PromptInput input;
  if (request.contains("messages") && request["messages"].is_array()) {
    for (const auto& message : request["messages"]) {
      if (!message.is_object()) continue;
      const std::string role = message.value("role", "");
      std::string content;
      if (message.contains("content") && message["content"].is_string()) {
        content = message["content"].get<std::string>();
      }
      if (role.empty()) continue;
      input.messages.push_back({role, content});
    }
  }
  if (request.contains("prompt") && request["prompt"].is_string()) {
    input.prompt = request["prompt"].get<std::string>();
  }
  if (request.contains("cache") && request["cache"].is_object() &&
      request["cache"].contains("boundaries") && request["cache"]["boundaries"].is_array()) {
    for (const auto& descriptor : request["cache"]["boundaries"]) {
      const std::string kind = descriptor.value("kind", "");
      input.boundaries.push_back({
          kind,
          descriptor.value("label", ""),
          kind == "messages"
              ? descriptor.value("through_message", input.messages.size()) + 1
              : 0,
      });
    }
  }
  return input;
}

std::string render_turn_prompt(const std::vector<ChatEntry>& entries,
                               bool add_generation_prompt) {
  if (entries.empty()) return "";
  std::string prompt = "<bos>";
  for (const auto& entry : entries) {
    std::string role = entry.role == "assistant" ? "model" : entry.role;
    if (role == "tool") role = "user";
    prompt += "<|turn>" + role + "\n" + entry.content + "<turn|>\n";
  }
  if (add_generation_prompt) prompt += "<|turn>model\n";
  return prompt;
}

std::string fallback_prompt(const std::vector<ChatEntry>& entries) {
  std::string prompt;
  for (const auto& entry : entries) {
    prompt += entry.content;
    prompt += "\n";
  }
  return prompt;
}

std::optional<std::size_t> exact_prompt_boundary(
    const std::vector<llama_token>& prefix,
    const std::vector<llama_token>& final,
    const std::function<bool(llama_token)>& is_terminal) {
  if (prefix.empty()) return std::nullopt;
  if (prefix.size() < final.size() &&
      std::equal(prefix.begin(), prefix.end(), final.begin())) return prefix.size();
  std::size_t shared = 0;
  while (shared < prefix.size() && shared < final.size() && prefix[shared] == final[shared]) {
    ++shared;
  }
  if (shared == 0 || shared == prefix.size() || shared >= final.size()) return std::nullopt;
  if (!std::all_of(prefix.begin() + static_cast<std::ptrdiff_t>(shared), prefix.end(),
                   is_terminal)) return std::nullopt;
  return shared;
}

std::string PromptRenderer::render(const std::vector<ChatEntry>& entries,
                                   bool add_generation_prompt) const {
  if (entries.empty()) return "";
  const char* tmpl = llama_model_chat_template(runtime_.model(), nullptr);
  if (tmpl && std::string(tmpl).find("<|turn>") != std::string::npos) {
    return render_turn_prompt(entries, add_generation_prompt);
  }

  std::vector<ChatEntry> normalized;
  normalized.reserve(entries.size());
  for (const auto& entry : entries) {
    normalized.push_back({entry.role == "tool" ? "user" : entry.role, entry.content});
  }
  std::vector<llama_chat_message> chat;
  chat.reserve(normalized.size());
  for (const auto& entry : normalized) chat.push_back({entry.role.c_str(), entry.content.c_str()});

  int32_t length = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
      add_generation_prompt, nullptr, 0);
  if (length <= 0) return "";
  std::vector<char> buffer(static_cast<std::size_t>(length) + 1);
  int32_t written = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
      add_generation_prompt, buffer.data(), buffer.size());
  if (written <= 0) return "";
  return std::string(buffer.data(), static_cast<std::size_t>(written));
}

std::vector<llama_token> PromptRenderer::tokenize(const std::string& prompt) const {
  const bool add_special = prompt.find("<bos>") == std::string::npos &&
      prompt.find("<|turn>") == std::string::npos;
  const llama_vocab* vocab = runtime_.vocab();
  const int count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
      nullptr, 0, add_special, true);
  if (count <= 0) throw std::runtime_error("failed to tokenize prompt");
  std::vector<llama_token> tokens(static_cast<std::size_t>(count));
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(),
      add_special, true) < 0) {
    throw std::runtime_error("failed to tokenize prompt");
  }
  return tokens;
}

PreparedPrompt PromptRenderer::prepare(const PromptInput& input) const {
  std::string prompt = render(input.messages, true);
  if (prompt.empty()) prompt = fallback_prompt(input.messages);
  if (prompt.empty()) prompt = input.prompt;
  if (prompt.empty()) throw std::runtime_error("chat request contains no messages or prompt");

  PreparedPrompt prepared;
  prepared.tokens = tokenize(prompt);
  const auto resolve = [&](std::size_t message_count, const std::string& kind,
                           const std::string& label, bool requested) {
    if (message_count == 0 || message_count > input.messages.size()) {
      if (requested) prepared.resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    const std::vector<ChatEntry> prefix_entries(
        input.messages.begin(), input.messages.begin() + message_count);
    const std::string prefix_prompt = render(prefix_entries, false);
    if (prefix_prompt.empty()) {
      if (requested) prepared.resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    std::vector<llama_token> prefix_tokens;
    try {
      prefix_tokens = tokenize(prefix_prompt);
    } catch (...) {
      if (requested) prepared.resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    const auto exact = exact_prompt_boundary(prefix_tokens, prepared.tokens,
        [this](llama_token token) { return llama_vocab_is_eog(runtime_.vocab(), token); });
    if (!exact) {
      if (requested) prepared.resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "non_prefix_template_boundary"});
      return;
    }
    const std::size_t boundary = *exact;
    prepared.stable_boundaries.push_back(boundary);
    if (kind != "prompt") prepared.structural_boundaries.push_back(boundary);
    const auto existing = std::find_if(prepared.resolved_boundaries.begin(),
        prepared.resolved_boundaries.end(), [boundary](const auto& value) {
          return value.token_count == boundary;
        });
    if (existing == prepared.resolved_boundaries.end()) {
      prepared.resolved_boundaries.push_back(
          {boundary, kind, label, requested, "resolved", ""});
    } else if (requested) {
      existing->kind = kind;
      existing->label = label;
      existing->requested = true;
      existing->status = "resolved";
      existing->skip_reason.clear();
    }
  };

  std::size_t leading_systems = 0;
  while (leading_systems < input.messages.size() &&
         input.messages[leading_systems].role == "system") ++leading_systems;
  for (std::size_t count = 1; count <= leading_systems; ++count) {
    resolve(count, "messages", "", false);
  }
  for (const auto& boundary : input.boundaries) {
    if (boundary.kind == "messages") {
      resolve(boundary.message_count, boundary.kind, boundary.label, true);
    } else if (boundary.kind == "tools") {
      prepared.resolved_boundaries.push_back(
          {0, boundary.kind, boundary.label, true, "skipped",
           "unsupported_template_boundary"});
    }
  }
  if (prepared.tokens.size() > 16) {
    prepared.stable_boundaries.push_back(prepared.tokens.size() - 1);
    prepared.resolved_boundaries.push_back(
        {prepared.tokens.size() - 1, "prompt", "", false, "resolved", ""});
  }
  std::sort(prepared.stable_boundaries.begin(), prepared.stable_boundaries.end());
  prepared.stable_boundaries.erase(std::unique(prepared.stable_boundaries.begin(),
      prepared.stable_boundaries.end()), prepared.stable_boundaries.end());
  std::sort(prepared.structural_boundaries.begin(), prepared.structural_boundaries.end());
  prepared.structural_boundaries.erase(std::unique(prepared.structural_boundaries.begin(),
      prepared.structural_boundaries.end()), prepared.structural_boundaries.end());
  return prepared;
}

}  // namespace clap::llama
