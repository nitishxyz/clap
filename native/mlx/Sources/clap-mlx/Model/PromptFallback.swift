import Foundation
import MLXLMCommon

typealias PromptEntry = (role: String, content: String)

func normalizedMessageEntries(_ messages: [ChatMessage]) -> [PromptEntry] {
  messages.compactMap { message in
    guard let content = messageText(message) else { return nil }
    if message.role == "tool" {
      return ("user", "Tool result:\n\(content)")
    }
    return (message.role, content)
  }
}

func structuredTemplateMessages(_ messages: [ChatMessage]) -> [[String: any Sendable]] {
  messages.compactMap { message in
    if message.role == "assistant", let structured = structuredToolCallMessage(message) {
      return structured
    }
    guard let content = messageText(message) else { return nil }
    if message.role == "tool" {
      return ["role": "user", "content": "Tool result:\n\(content)"]
    }
    return ["role": message.role, "content": content]
  }
}

func toolInstructionPatchedEntries(_ entries: [PromptEntry], tools: [ToolSpec]) -> [PromptEntry] {
  let toolJson = toolSpecJson(tools)
  let instructions = [
    "You may call tools by responding with JSON only.",
    "Use this exact shape when calling tools:",
    "{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}",
    "Do not include natural language when calling tools.",
    "Available tools: \(toolJson)",
  ].joined(separator: "\n")
  var patched = entries
  if let index = patched.firstIndex(where: { $0.role == "system" }) {
    patched[index] = ("system", instructions + "\n\n" + patched[index].content)
  } else {
    patched.insert(("system", instructions), at: 0)
  }
  return patched
}

func templateMessages(_ entries: [PromptEntry]) -> [[String: any Sendable]] {
  entries.map { ["role": $0.role, "content": $0.content] }
}

func plainTranscript(_ entries: [PromptEntry]) -> String {
  entries.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
}

func usesGemma4FallbackPrompt(_ modelDirectory: URL) -> Bool {
  let configURL = modelDirectory.appendingPathComponent("config.json")
  let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
  let jinjaURL = modelDirectory.appendingPathComponent("chat_template.jinja")
  if FileManager.default.fileExists(atPath: jinjaURL.path) { return false }
  guard let config = try? String(contentsOf: configURL, encoding: .utf8) else { return false }
  let tokenizerConfig = (try? String(contentsOf: tokenizerConfigURL, encoding: .utf8)) ?? ""
  return config.contains("\"model_type\": \"gemma4\"") && !tokenizerConfig.contains("\"chat_template\"")
}

func gemma4Prompt(entries: [PromptEntry]) -> String {
  var parts = ["<bos>"]
  for entry in entries {
    let role = entry.role == "assistant" ? "assistant" : "user"
    parts.append("<|turn>\(role)\n\(entry.content)")
  }
  parts.append("<|turn>assistant\n")
  return parts.joined()
}
