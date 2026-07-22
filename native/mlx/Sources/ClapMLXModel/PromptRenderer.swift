import ClapCachePolicy
import Foundation

public struct PromptTokenizerAdapter {
  public let eosTokenId: Int?
  private let encodeBody: (String, Bool) -> [Int]
  private let templateBody: ([[String: any Sendable]], [PromptToolSpec]?, [String: any Sendable]?) throws -> [Int]

  public init(eosTokenId: Int?, encode: @escaping (String, Bool) -> [Int],
              applyChatTemplate: @escaping ([[String: any Sendable]], [PromptToolSpec]?,
                [String: any Sendable]?) throws -> [Int]) {
    self.eosTokenId = eosTokenId
    encodeBody = encode
    templateBody = applyChatTemplate
  }

  func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
    encodeBody(text, addSpecialTokens)
  }

  func applyChatTemplate(messages: [[String: any Sendable]], tools: [PromptToolSpec]?,
                         additionalContext: [String: any Sendable]? = nil) throws -> [Int] {
    try templateBody(messages, tools, additionalContext)
  }
}

public enum PromptRenderer {
  public static func render(messages: [PromptMessage], tools: [PromptToolSpec]?,
                            boundaries: [PromptBoundaryDescriptor], modelDirectory: URL,
                            tokenizer: PromptTokenizerAdapter,
                            log: (String) -> Void = { _ in }) throws -> PreparedPrompt {
    let entries = normalizedEntries(messages)
    guard !entries.isEmpty else { throw PromptRendererError.noMessages }
    let structuredMessages = structuredMessages(messages)
    let usedFallback = usesGemma4FallbackPrompt(modelDirectory)
    let promptTokens: [Int]
    if usedFallback {
      promptTokens = tokenizer.encode(gemma4Prompt(entries), addSpecialTokens: false)
    } else {
      do {
        promptTokens = try tokenizer.applyChatTemplate(messages: structuredMessages, tools: tools)
      } catch {
        if tools != nil {
          let compatible = (tools ?? []).map(templateCompatibleToolSpec)
          if let tokens = try? tokenizer.applyChatTemplate(
            messages: structuredMessages, tools: compatible) {
            log("required a nullable-preserving compatibility view of all \(compatible.count) caller-provided tools")
            promptTokens = tokens
          } else {
            log("failed with all \(tools?.count ?? 0) caller-provided tools (\(error)); retrying with JSON tool instructions")
            let patched = toolInstructionPatchedEntries(entries, tools: tools ?? [])
            if let tokens = try? tokenizer.applyChatTemplate(
              messages: templateMessages(patched), tools: nil) {
              promptTokens = tokens
            } else {
              log("template retry without tools failed; using plain transcript")
              promptTokens = tokenizer.encode(plainTranscript(patched))
            }
          }
        } else {
          log("chat template failed (\(error)); using plain transcript")
          promptTokens = tokenizer.encode(plainTranscript(entries))
        }
      }
    }

    var stable: [Int] = []
    var resolved: [Int: ResolvedPromptBoundary] = [:]
    var structural: [ResolvedPromptBoundary] = []
    func resolve(_ prefix: [[String: any Sendable]], tools: [PromptToolSpec]?,
                 kind: String, label: String?, requested: Bool) {
      guard !prefix.isEmpty else {
        if requested { structural.append(skipped(kind, label, "unsupported_template_boundary")) }
        return
      }
      guard let rendered = try? tokenizer.applyChatTemplate(messages: prefix, tools: tools,
        additionalContext: ["add_generation_prompt": false]) else {
        if requested { structural.append(skipped(kind, label, "unsupported_template_boundary")) }
        return
      }
      guard let boundary = exactTemplateBoundary(prefix: rendered, final: promptTokens,
        eosToken: tokenizer.eosTokenId) else {
        if requested { structural.append(skipped(kind, label, "non_prefix_template_boundary")) }
        return
      }
      let value = ResolvedPromptBoundary(tokenCount: boundary, kind: kind, label: label,
        requested: requested, status: "resolved", skipReason: nil)
      stable.append(boundary)
      if resolved[boundary] == nil || requested { resolved[boundary] = value }
      if requested { structural.append(value) }
    }
    if !usedFallback {
      let leading = structuredMessages.prefix { ($0["role"] as? String) == "system" }.count
      if leading > 0 {
        for count in 1...leading {
          resolve(Array(structuredMessages.prefix(count)), tools: tools,
            kind: "messages", label: nil, requested: false)
        }
      }
      stable = Array(Set(stable)).sorted()
    }
    for descriptor in boundaries {
      guard !usedFallback else {
        structural.append(skipped(descriptor.kind, descriptor.label,
          "unsupported_template_boundary"))
        continue
      }
      if descriptor.kind == "tools" {
        structural.append(skipped(descriptor.kind, descriptor.label,
          "unsupported_template_boundary"))
      } else if descriptor.kind == "messages", let index = descriptor.throughMessage,
                index >= 0, index < structuredMessages.count {
        resolve(Array(structuredMessages.prefix(index + 1)), tools: tools,
          kind: "messages", label: descriptor.label, requested: true)
      }
    }
    let promptBoundary = promptTokens.count - 1
    if promptBoundary >= 16 {
      stable.append(promptBoundary)
      if resolved[promptBoundary] == nil {
        resolved[promptBoundary] = ResolvedPromptBoundary(tokenCount: promptBoundary,
          kind: "prompt", label: nil, requested: false, status: "resolved", skipReason: nil)
      }
      stable = Array(Set(stable)).sorted()
    }
    return PreparedPrompt(tokens: promptTokens, stableBoundaries: stable,
      structuralBoundaries: structural, resolvedBoundaries: resolved,
      usedFallback: usedFallback)
  }

  private static func skipped(_ kind: String, _ label: String?, _ reason: String)
    -> ResolvedPromptBoundary {
    ResolvedPromptBoundary(tokenCount: nil, kind: kind, label: label,
      requested: true, status: "skipped", skipReason: reason)
  }
}

private typealias PromptEntry = (role: String, content: String)

private func parsedArguments(_ raw: String?) -> [String: any Sendable] {
  guard let raw, let data = raw.data(using: .utf8),
        let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
  return value.mapValues(sendableValue)
}

private func sendableValue(_ value: Any) -> any Sendable {
  if value is NSNull { return NSNull() }
  if let value = value as? Bool { return value }
  if let value = value as? Int { return value }
  if let value = value as? Double { return value }
  if let value = value as? String { return value }
  if let value = value as? [Any] { return value.map(sendableValue) }
  if let value = value as? [String: Any] { return value.mapValues(sendableValue) }
  return String(describing: value)
}

private func encodedToolCalls(_ calls: [PromptToolCall]) -> String? {
  let object: [String: Any] = ["tool_calls": calls.map {
    ["name": $0.name, "arguments": parsedArguments($0.arguments)]
  }]
  guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return nil }
  return String(data: data, encoding: .utf8)
}

private func messageText(_ message: PromptMessage) -> String? {
  if let content = message.content, !content.isEmpty { return content }
  if let calls = message.toolCalls, !calls.isEmpty { return encodedToolCalls(calls) }
  return nil
}

private func normalizedEntries(_ messages: [PromptMessage]) -> [PromptEntry] {
  messages.compactMap { message in
    guard let content = messageText(message) else { return nil }
    return message.role == "tool" ? ("user", "Tool result:\n\(content)")
      : (message.role, content)
  }
}

private func structuredMessages(_ messages: [PromptMessage]) -> [[String: any Sendable]] {
  messages.compactMap { message in
    if message.role == "assistant", let calls = message.toolCalls, !calls.isEmpty {
      let rendered: [[String: any Sendable]] = calls.map { call in
        let arguments = parsedArguments(call.arguments)
        let function: [String: any Sendable] = ["name": call.name, "arguments": arguments]
        return ["type": "function", "function": function, "name": call.name,
          "arguments": arguments]
      }
      var result: [String: any Sendable] = ["role": "assistant", "tool_calls": rendered]
      result["content"] = message.content?.isEmpty == false ? message.content! : ""
      return result
    }
    guard let content = messageText(message) else { return nil }
    return message.role == "tool"
      ? ["role": "user", "content": "Tool result:\n\(content)"]
      : ["role": message.role, "content": content]
  }
}

private func templateCompatibleSchemaValue(_ value: any Sendable) -> any Sendable {
  if let dict = value as? [String: any Sendable] {
    var result: [String: any Sendable] = [:]
    var nullable = false
    for (key, entry) in dict {
      if key == "type", let list = entry as? [any Sendable] {
        let types = list.compactMap { $0 as? String }
        let nonNull = types.filter { $0 != "null" }
        if types.count == list.count, types.contains("null"), nonNull.count == 1 {
          result[key] = nonNull[0]
          nullable = true
          continue
        }
      }
      result[key] = templateCompatibleSchemaValue(entry)
    }
    if nullable { result["nullable"] = true }
    return result
  }
  if let list = value as? [any Sendable] { return list.map(templateCompatibleSchemaValue) }
  return value
}

private func templateCompatibleToolSpec(_ spec: PromptToolSpec) -> PromptToolSpec {
  var result = spec
  if let function = spec["function"] as? [String: any Sendable] {
    for key in ["name", "description", "parameters"] where result[key] == nil {
      result[key] = function[key]
    }
  }
  return templateCompatibleSchemaValue(result) as? PromptToolSpec ?? result
}

private func toolSpecJson(_ specs: [PromptToolSpec]) -> String {
  let trimmed = specs.map { spec -> [String: any Sendable] in
    let source = spec["function"] as? [String: any Sendable] ?? spec
    var entry: [String: any Sendable] = [:]
    for key in ["name", "description", "parameters"] {
      if let value = source[key] { entry[key] = value }
    }
    return entry
  }
  guard JSONSerialization.isValidJSONObject(trimmed),
        let data = try? JSONSerialization.data(withJSONObject: trimmed, options: [.sortedKeys]),
        let text = String(data: data, encoding: .utf8) else { return "[]" }
  return text
}

private func toolInstructionPatchedEntries(_ entries: [PromptEntry], tools: [PromptToolSpec])
  -> [PromptEntry] {
  let instructions = ["You may call tools by responding with JSON only.",
    "Use this exact shape when calling tools:",
    "{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}",
    "Do not include natural language when calling tools.",
    "Available tools: \(toolSpecJson(tools))"].joined(separator: "\n")
  var patched = entries
  if let index = patched.firstIndex(where: { $0.role == "system" }) {
    patched[index] = ("system", instructions + "\n\n" + patched[index].content)
  } else { patched.insert(("system", instructions), at: 0) }
  return patched
}

private func templateMessages(_ entries: [PromptEntry]) -> [[String: any Sendable]] {
  entries.map { ["role": $0.role, "content": $0.content] }
}

private func plainTranscript(_ entries: [PromptEntry]) -> String {
  entries.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
}

private func usesGemma4FallbackPrompt(_ directory: URL) -> Bool {
  if FileManager.default.fileExists(atPath: directory.appendingPathComponent("chat_template.jinja").path) {
    return false
  }
  guard let config = try? String(contentsOf: directory.appendingPathComponent("config.json"),
    encoding: .utf8) else { return false }
  let tokenizer = (try? String(contentsOf: directory.appendingPathComponent("tokenizer_config.json"),
    encoding: .utf8)) ?? ""
  return config.contains("\"model_type\": \"gemma4\"") && !tokenizer.contains("\"chat_template\"")
}

private func gemma4Prompt(_ entries: [PromptEntry]) -> String {
  var parts = ["<bos>"]
  for entry in entries {
    parts.append("<|turn>\(entry.role == "assistant" ? "assistant" : "user")\n\(entry.content)")
  }
  parts.append("<|turn>assistant\n")
  return parts.joined()
}
