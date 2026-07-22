import Foundation
import Testing
@testable import ClapMLXModel

@Suite("Prompt renderer")
struct PromptRendererTests {
  @Test("normalizes tool messages and structured assistant calls")
  func normalizesMessages() throws {
    let directory = try fixture(config: "{}", tokenizer: "{\"chat_template\":\"x\"}")
    defer { try? FileManager.default.removeItem(at: directory) }
    var captured: [[String: any Sendable]] = []
    let adapter = PromptTokenizerAdapter(eosTokenId: 99, encode: { _, _ in [] }) {
      messages, _, context in
      if context == nil { captured = messages }
      return context == nil ? Array(0..<20) : Array(0..<messages.count * 4)
    }
    let messages = [
      PromptMessage(role: "assistant", content: nil,
        toolCalls: [PromptToolCall(name: "lookup", arguments: "{\"z\":2,\"a\":1}")]),
      PromptMessage(role: "tool", content: "result")
    ]
    let result = try PromptRenderer.render(messages: messages, tools: nil, boundaries: [],
      modelDirectory: directory, tokenizer: adapter)
    #expect((captured[0]["role"] as? String) == "assistant")
    #expect((captured[1]["role"] as? String) == "user")
    #expect((captured[1]["content"] as? String) == "Tool result:\nresult")
    #expect(result.stableBoundaries == [19])
    #expect(!result.usedFallback)
  }

  @Test("uses exact Gemma fallback text and rejects requested boundaries")
  func gemmaFallback() throws {
    let directory = try fixture(config: "{\"model_type\": \"gemma4\"}")
    defer { try? FileManager.default.removeItem(at: directory) }
    var encoded = ""
    var special = true
    let adapter = PromptTokenizerAdapter(eosTokenId: nil, encode: { text, addSpecial in
      encoded = text
      special = addSpecial
      return Array(0..<18)
    }, applyChatTemplate: { _, _, _ in Issue.record("template should not be called"); return [] })
    let result = try PromptRenderer.render(
      messages: [PromptMessage(role: "system", content: "rules"),
        PromptMessage(role: "assistant", content: "ok")], tools: nil,
      boundaries: [PromptBoundaryDescriptor(kind: "messages", throughMessage: 0, label: "s")],
      modelDirectory: directory, tokenizer: adapter)
    #expect(encoded == "<bos><|turn>user\nrules<|turn>assistant\nok<|turn>assistant\n")
    #expect(!special)
    #expect(result.usedFallback)
    #expect(result.stableBoundaries == [17])
    #expect(result.structuralBoundaries == [ResolvedPromptBoundary(tokenCount: nil,
      kind: "messages", label: "s", requested: true, status: "skipped",
      skipReason: "unsupported_template_boundary")])
  }

  @Test("retries nullable-compatible tools before instruction fallback")
  func compatibleTools() throws {
    let directory = try fixture(config: "{}")
    defer { try? FileManager.default.removeItem(at: directory) }
    var attempts = 0
    var compatibleNullable = false
    let adapter = PromptTokenizerAdapter(eosTokenId: nil, encode: { _, _ in [] }) {
      _, tools, context in
      if context != nil { return [] }
      attempts += 1
      let parameters = tools?.first?["parameters"] as? [String: any Sendable]
      compatibleNullable = parameters?["nullable"] as? Bool == true
      if attempts == 1 { throw TestError.template }
      return [1, 2, 3]
    }
    let tools: [PromptToolSpec] = [["parameters": ["type": ["string", "null"] as [any Sendable]]]]
    let result = try PromptRenderer.render(
      messages: [PromptMessage(role: "user", content: "hi")], tools: tools,
      boundaries: [], modelDirectory: directory, tokenizer: adapter)
    #expect(attempts == 2)
    #expect(compatibleNullable)
    #expect(result.tokens == [1, 2, 3])
  }

  @Test("falls back through JSON instructions to exact plain transcript")
  func instructionTranscriptFallback() throws {
    let directory = try fixture(config: "{}")
    defer { try? FileManager.default.removeItem(at: directory) }
    var transcript = ""
    let adapter = PromptTokenizerAdapter(eosTokenId: nil, encode: { text, _ in
      transcript = text
      return [7]
    }, applyChatTemplate: { _, _, _ in throw TestError.template })
    let tools: [PromptToolSpec] = [["function": ["name": "lookup",
      "description": "Lookup", "parameters": ["type": "object"]] as [String: any Sendable]]]
    _ = try PromptRenderer.render(messages: [PromptMessage(role: "user", content: "question")],
      tools: tools, boundaries: [], modelDirectory: directory, tokenizer: adapter)
    #expect(transcript == "system: You may call tools by responding with JSON only.\nUse this exact shape when calling tools:\n{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}\nDo not include natural language when calling tools.\nAvailable tools: [{\"description\":\"Lookup\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]\n\nuser: question\n\nassistant:")
  }

  @Test("resolves, sorts, deduplicates, and rejects template boundaries")
  func boundaries() throws {
    let directory = try fixture(config: "{}")
    defer { try? FileManager.default.removeItem(at: directory) }
    let adapter = PromptTokenizerAdapter(eosTokenId: 99, encode: { _, _ in [] }) {
      messages, _, context in
      if context == nil { return Array(0..<20) }
      if messages.count == 1 { return [0, 1, 99] }
      return [88]
    }
    let boundaries = [
      PromptBoundaryDescriptor(kind: "messages", throughMessage: 0, label: "first"),
      PromptBoundaryDescriptor(kind: "messages", throughMessage: 1, label: "bad"),
      PromptBoundaryDescriptor(kind: "tools", throughMessage: nil, label: "tools"),
      PromptBoundaryDescriptor(kind: "messages", throughMessage: 99, label: "invalid")
    ]
    let result = try PromptRenderer.render(messages: [
      PromptMessage(role: "system", content: "s"), PromptMessage(role: "user", content: "u")],
      tools: nil, boundaries: boundaries, modelDirectory: directory, tokenizer: adapter)
    #expect(result.stableBoundaries == [2, 19])
    #expect(result.resolvedBoundaries[2]?.label == "first")
    #expect(result.structuralBoundaries.count == 3)
    #expect(result.structuralBoundaries[1].skipReason == "non_prefix_template_boundary")
    #expect(result.structuralBoundaries[2].skipReason == "unsupported_template_boundary")
  }

  private enum TestError: Error { case template }

  private func fixture(config: String, tokenizer: String? = nil) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
    if let tokenizer {
      try Data(tokenizer.utf8).write(to: directory.appendingPathComponent("tokenizer_config.json"))
    }
    return directory
  }
}
