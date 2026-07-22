import Foundation
import MLXLMCommon

struct ToolsEnvelope: Decodable {
  let tools: [JSONValue]?
}

private struct EmittedToolCall: Encodable {
  let name: String
  let arguments: [String: JSONValue]
}

private struct EmittedToolCalls: Encodable {
  let tool_calls: [EmittedToolCall]
}

// Canonical text form for an assistant message that carried structured
// tool_calls so chat templates see the call the model originally made.
func encodeIncomingToolCalls(_ calls: [IncomingToolCall]) -> String? {
  let converted = calls.map { call -> EmittedToolCall in
    var arguments: [String: JSONValue] = [:]
    if let raw = call.function.arguments, let data = raw.data(using: .utf8),
       let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: data) {
      arguments = decoded
    }
    return EmittedToolCall(name: call.function.name, arguments: arguments)
  }
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  guard let data = try? encoder.encode(EmittedToolCalls(tool_calls: converted)) else { return nil }
  return String(data: data, encoding: .utf8)
}

func messageText(_ message: ChatMessage) -> String? {
  if let content = message.content, !content.isEmpty { return content }
  if let calls = message.tool_calls, !calls.isEmpty { return encodeIncomingToolCalls(calls) }
  return nil
}

func sendableValue(_ value: JSONValue) -> any Sendable {
  switch value {
  case .null: return NSNull()
  case .bool(let v): return v
  case .int(let v): return v
  case .double(let v): return v
  case .string(let v): return v
  case .array(let items): return items.map { sendableValue($0) }
  case .object(let entries): return entries.mapValues { sendableValue($0) }
  }
}

// Structured template message for an assistant turn that carried tool_calls:
// letting the chat template render the model's trained tool-call format keeps
// the continuation prompt byte-identical to what the model generated, which
// preserves KV cache extension even for non-rewindable (sliding-window) caches.
func structuredToolCallMessage(_ message: ChatMessage) -> [String: any Sendable]? {
  guard let calls = message.tool_calls, !calls.isEmpty else { return nil }
  let rendered: [[String: any Sendable]] = calls.map { call in
    var arguments: [String: any Sendable] = [:]
    if let raw = call.function.arguments, let data = raw.data(using: .utf8),
       let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: data) {
      arguments = decoded.mapValues { sendableValue($0) }
    }
    let function: [String: any Sendable] = ["name": call.function.name, "arguments": arguments]
    return ["type": "function", "function": function, "name": call.function.name, "arguments": arguments]
  }
  var result: [String: any Sendable] = ["role": "assistant", "tool_calls": rendered]
  if let content = message.content, !content.isEmpty { result["content"] = content }
  else { result["content"] = "" }
  return result
}

// Some model templates expect a flattened tool and apply string filters to
// JSON-Schema `type`. Try the caller's exact OpenAI tool objects first; this
// compatibility view is only a second render attempt. Nullable single-type
// unions retain their semantics through the sibling `nullable` field used by
// those templates. Multi-type unions remain unchanged rather than being
// silently narrowed.
func templateCompatibleSchemaValue(_ value: any Sendable) -> any Sendable {
  if let dict = value as? [String: any Sendable] {
    var result: [String: any Sendable] = [:]
    var nullableUnion = false
    for (key, entry) in dict {
      if key == "type", let list = entry as? [any Sendable] {
        let types = list.compactMap { $0 as? String }
        let nonNull = types.filter { $0 != "null" }
        if types.count == list.count, types.contains("null"), nonNull.count == 1 {
          result[key] = nonNull[0]
          nullableUnion = true
          continue
        }
      }
      result[key] = templateCompatibleSchemaValue(entry)
    }
    if nullableUnion { result["nullable"] = true }
    return result
  }
  if let list = value as? [any Sendable] {
    return list.map { templateCompatibleSchemaValue($0) }
  }
  return value
}

func templateCompatibleToolSpec(_ spec: ToolSpec) -> ToolSpec {
  var result = spec
  if let function = spec["function"] as? [String: any Sendable] {
    for key in ["name", "description", "parameters"] where result[key] == nil {
      result[key] = function[key]
    }
  }
  return templateCompatibleSchemaValue(result) as? ToolSpec ?? result
}

// Compact JSON for tool specs used by the in-worker instruction fallback.
func toolSpecJson(_ specs: [ToolSpec]) -> String {
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
