import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

struct ChatMessage: Decodable {
  let role: String
  let content: String?
  let tool_calls: [IncomingToolCall]?
}

struct IncomingToolCall: Decodable {
  struct Function: Decodable {
    let name: String
    let arguments: String?
  }
  let function: Function
}

struct WorkerUsage: Encodable {
  let prompt_tokens: Int
  let completion_tokens: Int
}

struct WorkerCache: Encodable {
  let hit: Bool
  let reused_tokens: Int
  let side_request: Bool
}

struct WorkerPrefill: Encodable {
  let done: Int
  let total: Int
}

struct WorkerMessage: Encodable {
  let id: String?
  let token: String?
  let content: String?
  let loaded: Bool?
  let unloaded: Bool?
  let done: Bool?
  let error: String?
  let cancelled: Bool?
  let finish_reason: String?
  let usage: WorkerUsage?
  let cache: WorkerCache?
  let prefill: WorkerPrefill?
}

struct ControlRequest: Decodable {
  let id: String?
  let type: String?
  let model: String?
  let messages: [ChatMessage]?
  let stream: Bool?
  let max_tokens: Int?
  let temperature: Double?
}

func emit(id: String? = nil, token: String? = nil, content: String? = nil, loaded: Bool? = nil, unloaded: Bool? = nil, done: Bool? = nil, error: String? = nil, cancelled: Bool? = nil, finishReason: String? = nil, usage: WorkerUsage? = nil, cache: WorkerCache? = nil, prefill: WorkerPrefill? = nil) {
  let message = WorkerMessage(id: id, token: token, content: content, loaded: loaded, unloaded: unloaded, done: done, error: error, cancelled: cancelled, finish_reason: finishReason, usage: usage, cache: cache, prefill: prefill)
  let data = try! JSONEncoder().encode(message)
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data([0x0a]))
}

// Buffers stdin lines so the main loop can poll for cancel messages while a
// generation is streaming.
actor LineBuffer {
  private var lines: [String] = []
  private var waiter: CheckedContinuation<String?, Never>?
  private var finished = false

  func push(_ line: String) {
    if let waiter {
      self.waiter = nil
      waiter.resume(returning: line)
    } else {
      lines.append(line)
    }
  }

  func finish() {
    finished = true
    if let waiter {
      self.waiter = nil
      waiter.resume(returning: nil)
    }
  }

  func next() async -> String? {
    if !lines.isEmpty { return lines.removeFirst() }
    if finished { return nil }
    return await withCheckedContinuation { waiter = $0 }
  }

  func poll() -> String? {
    lines.isEmpty ? nil : lines.removeFirst()
  }
}

func isCancelMessage(_ line: String, activeId: String?) -> Bool {
  guard let data = line.data(using: .utf8),
        let control = try? JSONDecoder().decode(ControlRequest.self, from: data),
        control.type == "cancel" else { return false }
  guard let target = control.id, !target.isEmpty else { return true }
  return target == activeId
}

struct ToolsEnvelope: Decodable {
  let tools: [JSONValue]?
}

struct EmittedToolCall: Encodable {
  let name: String
  let arguments: [String: JSONValue]
}

struct EmittedToolCalls: Encodable {
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

// Chat templates apply string filters (e.g. gemma runs `| upper` on every
// parameter `type`), so JSON-Schema union types like ["string","null"] crash
// the whole render. Normalize any non-string `type` to its first string form.
func sanitizeSchemaValue(_ value: any Sendable) -> any Sendable {
  if let dict = value as? [String: any Sendable] {
    var result: [String: any Sendable] = [:]
    for (key, entry) in dict {
      if key == "type", !(entry is String) {
        if let list = entry as? [any Sendable], let first = list.compactMap({ $0 as? String }).first(where: { $0 != "null" }) ?? list.compactMap({ $0 as? String }).first {
          result[key] = first
        } else {
          result[key] = "string"
        }
      } else {
        result[key] = sanitizeSchemaValue(entry)
      }
    }
    return result
  }
  if let list = value as? [any Sendable] {
    return list.map { sanitizeSchemaValue($0) }
  }
  return value
}

// Compact JSON for tool specs used by the in-worker instruction fallback.
func toolSpecJson(_ specs: [ToolSpec]) -> String {
  let trimmed = specs.map { spec -> [String: any Sendable] in
    var entry: [String: any Sendable] = [:]
    for key in ["name", "description", "parameters"] {
      if let value = spec[key] { entry[key] = value }
    }
    return entry
  }
  guard JSONSerialization.isValidJSONObject(trimmed),
        let data = try? JSONSerialization.data(withJSONObject: trimmed, options: [.sortedKeys]),
        let text = String(data: data, encoding: .utf8) else { return "[]" }
  return text
}

func debugLog(_ message: String) {
  FileHandle.standardError.write(Data("[clap-mlx] \(message)\n".utf8))
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

func gemma4Prompt(entries: [(role: String, content: String)]) -> String {
  var parts = ["<bos>"]
  for entry in entries {
    let role = entry.role == "assistant" ? "assistant" : "user"
    parts.append("<|turn>\(role)\n\(entry.content)")
  }
  parts.append("<|turn>assistant\n")
  return parts.joined()
}

func validateModelDirectory(_ path: String) throws -> URL {
  let url = URL(fileURLWithPath: path)
  var isDirectory: ObjCBool = false
  guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
    throw WorkerError.invalidModelDirectory("MLX model directory not found: \(path)")
  }
  guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
    throw WorkerError.invalidModelDirectory("MLX model directory is missing config.json: \(path)")
  }
  return url
}

enum WorkerError: Error, CustomStringConvertible {
  case invalidModelDirectory(String)

  var description: String {
    switch self {
    case .invalidModelDirectory(let message): message
    }
  }
}

// ModelContext members (model, tokenizer) are not Sendable but are only used
// from this single-threaded main loop; ChatSession uses the same pattern.
struct UncheckedBox<T>: @unchecked Sendable {
  let value: T
}

func declaredEosTokenIds(_ url: URL) -> Set<Int> {
  guard let data = try? Data(contentsOf: url),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
  var ids: Set<Int> = []
  func collect(_ value: Any?) {
    if let id = value as? Int { ids.insert(id) }
    if let list = value as? [Any] { for item in list { collect(item) } }
  }
  collect(json["eos_token_id"])
  if let textConfig = json["text_config"] as? [String: Any] { collect(textConfig["eos_token_id"]) }
  return ids
}

func main() async {
  do {
    guard #available(macOS 14.0, *) else {
      emit(error: "clap-mlx requires macOS 14 or newer on Apple Silicon")
      exit(2)
    }
    #if !arch(arm64)
    emit(error: "clap-mlx requires Apple Silicon arm64")
    exit(2)
    #endif

    var loadedModel: String?
    var loadedDirectory: URL?
    var loadedContainer: ModelContainer?
    var languageModel: (any LanguageModel)?
    var tokenizer: (any MLXLMCommon.Tokenizer)?
    var eosTokenIds: Set<Int> = []

    // Multi-session token-level KV cache slots (mirrors the llama.cpp
    // worker): each slot remembers the exact token ids resident in its cache.
    // Requests pick the slot with the longest common prefix (any number of
    // interleaved sessions stay warm); misses recycle the LRU slot.
    final class KVSlot {
      var caches: [KVCache] = []
      var tokens: [Int] = []
      var lastUsed: UInt64 = 0
    }
    let slotLimit = max(1, Int(ProcessInfo.processInfo.environment["CLAP_MLX_SLOTS"] ?? "4") ?? 4)
    var kvSlots: [KVSlot] = []
    var kvUseCounter: UInt64 = 0
    func invalidateKVCache() {
      kvSlots = []
      kvUseCounter = 0
    }

    func loadModel(_ model: String, directory: URL) async throws {
      invalidateKVCache()
      let container = try await loadModelContainer(from: directory, using: #huggingFaceTokenizerLoader())
      let box = await container.perform { context in
        UncheckedBox(value: (context.model, context.tokenizer, context.configuration.extraEOSTokens))
      }
      loadedContainer = container
      languageModel = box.value.0
      tokenizer = box.value.1
      loadedModel = model
      loadedDirectory = directory
      eosTokenIds = []
      if let eos = box.value.1.eosTokenId { eosTokenIds.insert(eos) }
      for extra in box.value.2 {
        if let id = box.value.1.convertTokenToId(extra) { eosTokenIds.insert(id) }
      }
      // HF checkpoints declare turn-ending tokens (e.g. gemma's <turn|>) in
      // config.json / generation_config.json eos_token_id, not the tokenizer.
      for file in ["generation_config.json", "config.json"] {
        eosTokenIds.formUnion(declaredEosTokenIds(directory.appendingPathComponent(file)))
      }
      debugLog("model loaded; eos token ids: \(eosTokenIds.sorted())")
    }

    let buffer = LineBuffer()
    let readerTask = Task.detached {
      do {
        for try await line in FileHandle.standardInput.bytes.lines {
          await buffer.push(String(line))
        }
      } catch {}
      await buffer.finish()
    }
    defer { readerTask.cancel() }
    var deferred: [String] = []

    while true {
      let line: String
      if !deferred.isEmpty {
        line = deferred.removeFirst()
      } else if let next = await buffer.next() {
        line = next
      } else {
        break
      }
      if line.isEmpty { continue }
      guard let data = line.data(using: .utf8) else { continue }
      let control = try JSONDecoder().decode(ControlRequest.self, from: data)
      let id = control.id
      let type = control.type ?? "chat"

      if type == "shutdown" {
        emit(id: id, done: true)
        break
      }

      if type == "cancel" {
        continue  // request already finished; nothing to cancel
      }

      if type == "unload" {
        invalidateKVCache()
        loadedContainer = nil
        languageModel = nil
        tokenizer = nil
        loadedDirectory = nil
        loadedModel = nil
        emit(id: id, unloaded: true, done: true)
        continue
      }

      if type == "load" {
        guard let model = control.model else {
          emit(id: id, error: "load.model is required")
          continue
        }
        let modelDirectory = try validateModelDirectory(model)
        if loadedModel != model || languageModel == nil {
          try await loadModel(model, directory: modelDirectory)
        }
        emit(id: id, loaded: true, done: true)
        continue
      }

      guard let model = control.model else {
        emit(id: id, error: "chat.model is required")
        continue
      }
      let modelDirectory = try validateModelDirectory(model)
      if loadedModel != model || languageModel == nil {
        try await loadModel(model, directory: modelDirectory)
      }
      guard let lm = languageModel, let tok = tokenizer else {
        emit(id: id, error: "model is not loaded")
        continue
      }
      let maxTokens = control.max_tokens ?? 4096
      let temperature = control.temperature ?? 0.7
      // Extract OpenAI-style tool specs so the chat template can declare them
      // natively; the model then emits its trained tool-call format, which the
      // server parses from the raw text.
      var toolSpecs: [ToolSpec]? = nil
      if let envelope = try? JSONDecoder().decode(ToolsEnvelope.self, from: data),
         let rawTools = envelope.tools, !rawTools.isEmpty {
        toolSpecs = rawTools.compactMap { raw -> ToolSpec? in
          // Pass the raw OpenAI tool value through (anyValue keeps it as-is;
          // asSchema would wrongly describe its *shape*). Also flatten
          // name/description/parameters to the top level because some
          // templates (gemma) read them flat instead of under `function`.
          guard var spec = raw.anyValue as? [String: any Sendable] else { return nil }
          if let function = spec["function"] as? [String: any Sendable] {
            for key in ["name", "description", "parameters"] where spec[key] == nil {
              spec[key] = function[key]
            }
          }
          return sanitizeSchemaValue(spec) as? [String: any Sendable] ?? spec
        }
        if toolSpecs?.isEmpty == true { toolSpecs = nil }
      }
      let entries = (control.messages ?? []).compactMap { message -> (role: String, content: String)? in
        guard let content = messageText(message) else { return nil }
        // Many chat templates (gemma included) silently drop `tool` role
        // messages, so the model never sees tool results. Map them to user
        // turns with an explicit wrapper — same approach as the llama.cpp
        // worker — which every template renders.
        if message.role == "tool" {
          return ("user", "Tool result:\n\(content)")
        }
        return (message.role, content)
      }
      guard !entries.isEmpty else {
        emit(id: id, error: "chat request contains no messages")
        continue
      }
      // Structured template messages: assistant tool_calls stay structured so
      // the template renders the model's native tool-call format.
      let structuredMessages: [[String: any Sendable]] = (control.messages ?? []).compactMap { message in
        if message.role == "assistant", let structured = structuredToolCallMessage(message) {
          return structured
        }
        guard let content = messageText(message) else { return nil }
        if message.role == "tool" {
          return ["role": "user", "content": "Tool result:\n\(content)"]
        }
        return ["role": message.role, "content": content]
      }

      // Build the full prompt token ids.
      let usesFallback = usesGemma4FallbackPrompt(loadedDirectory ?? modelDirectory)
      var promptTokens: [Int]
      if usesFallback {
        promptTokens = tok.encode(text: gemma4Prompt(entries: entries), addSpecialTokens: false)
      } else {
        do {
          promptTokens = try tok.applyChatTemplate(messages: structuredMessages, tools: toolSpecs)
        } catch {
          // Template failed. If tools were involved, the schema may be beyond
          // what the template supports: retry without native tools and inject
          // JSON tool-call instructions so tool use still works.
          if toolSpecs != nil {
            debugLog("chat template failed with tools (\(error)); retrying with JSON tool instructions")
            let toolJson = toolSpecJson(toolSpecs ?? [])
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
            let retryMessages: [[String: any Sendable]] = patched.map { ["role": $0.role, "content": $0.content] }
            if let tokens = try? tok.applyChatTemplate(messages: retryMessages, tools: nil) {
              promptTokens = tokens
            } else {
              debugLog("template retry without tools failed; using plain transcript")
              let transcript = patched.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
              promptTokens = tok.encode(text: transcript)
            }
          } else {
            // Template rejected the conversation shape (strict templates);
            // fall back to a plain transcript so the request still succeeds.
            debugLog("chat template failed (\(error)); using plain transcript")
            let transcript = entries.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
            promptTokens = tok.encode(text: transcript)
          }
        }
      }
      if ProcessInfo.processInfo.environment["CLAP_MLX_DEBUG_PROMPT"] != nil {
        debugLog("prompt (\(promptTokens.count) tokens): \(tok.decode(tokenIds: promptTokens, skipSpecialTokens: false))")
      }

      // Multi-session KV slots: pick the slot whose cached tokens share the
      // longest prefix with this prompt; a weak match recycles the LRU slot.
      var bestSlot: KVSlot? = nil
      var bestPrefix = 0
      for candidate in kvSlots {
        var prefix = 0
        let maxPrefix = min(candidate.tokens.count, promptTokens.count)
        while prefix < maxPrefix && candidate.tokens[prefix] == promptTokens[prefix] { prefix += 1 }
        if prefix > bestPrefix {
          bestPrefix = prefix
          bestSlot = candidate
        }
      }
      let slot: KVSlot
      if let matched = bestSlot, bestPrefix >= 16 {
        slot = matched
      } else {
        bestPrefix = 0
        if kvSlots.count < slotLimit {
          slot = KVSlot()
          kvSlots.append(slot)
        } else {
          slot = kvSlots.min { $0.lastUsed < $1.lastUsed } ?? KVSlot()
          slot.caches = []
          slot.tokens = []
        }
      }
      kvUseCounter += 1
      slot.lastUsed = kvUseCounter
      var prefix = bestPrefix
      if prefix == promptTokens.count { prefix -= 1 }  // always feed at least one token for logits

      var caches: [KVCache]
      var fedTokens: [Int]
      var suffix: [Int]
      var reusedTokens = 0
      // Recurrent/hybrid models (linear attention) have fixed-size state that
      // cannot be rewound; they can only extend an exact continuation.
      let trimmable = !slot.caches.isEmpty && slot.caches.allSatisfy { $0.isTrimmable }
      let trimNeeded = slot.tokens.count - prefix
      if prefix > 0 && !slot.caches.isEmpty && (trimmable || trimNeeded == 0) {
        if trimNeeded > 0 {
          for cache in slot.caches { cache.trim(trimNeeded) }
        }
        caches = slot.caches
        fedTokens = Array(promptTokens.prefix(prefix))
        suffix = Array(promptTokens.dropFirst(prefix))
        reusedTokens = prefix
        debugLog("kv prefix reuse (slot \(kvSlots.firstIndex(where: { $0 === slot }) ?? -1)): \(prefix)/\(promptTokens.count) tokens cached, prefilling \(suffix.count)")
      } else {
        caches = lm.newCache(parameters: nil)
        fedTokens = []
        suffix = promptTokens
        if !slot.tokens.isEmpty {
          let reason = trimmable || slot.caches.isEmpty
            ? "no usable prefix"
            : "recurrent cache cannot rewind (matched \(prefix), needs trim of \(trimNeeded))"
          debugLog("kv cache miss: \(reason) (cached \(slot.tokens.count), prompt \(promptTokens.count))")
        }
      }
      // The slot's cache objects are mutated from here on; do not keep a
      // stale token list if this request is cancelled mid-prefill.
      slot.caches = caches
      slot.tokens = fedTokens

      let generateParameters = GenerateParameters(maxTokens: maxTokens, temperature: Float(temperature))
      let streaming = control.stream ?? true
      var collected = ""
      var cancelled = false
      var finishReason = "stop"
      var generatedCount = 0

      // Chunked prefill: feed all but the tail of the suffix in windows so we
      // can poll for cancellation and report progress during long prompts.
      let prefillChunk = 512
      var iterator: TokenIterator? = nil
      var pos = 0
      while suffix.count - pos > prefillChunk {
        let chunk = Array(suffix[pos ..< pos + prefillChunk])
        // Creating the iterator prefills the chunk into the shared cache; the
        // sampled token is discarded.
        _ = try TokenIterator(input: LMInput(tokens: MLXArray(chunk)), model: lm, cache: caches, parameters: generateParameters)
        pos += prefillChunk
        fedTokens.append(contentsOf: chunk)
        slot.tokens = fedTokens
        emit(id: id, prefill: WorkerPrefill(done: prefix + pos, total: promptTokens.count))
        while let pending = await buffer.poll() {
          if isCancelMessage(pending, activeId: id) {
            cancelled = true
          } else {
            deferred.append(pending)
          }
        }
        if cancelled { break }
      }

      if !cancelled {
        let tail = Array(suffix.dropFirst(pos))
        iterator = try TokenIterator(input: LMInput(tokens: MLXArray(tail)), model: lm, cache: caches, parameters: generateParameters)
        fedTokens.append(contentsOf: tail)
        slot.tokens = fedTokens
      }

      if var it = iterator, !cancelled {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tok)
        var sampledTokens: [Int] = []
        generation: while let token = it.next() {
          sampledTokens.append(token)
          if eosTokenIds.contains(token) {
            if ProcessInfo.processInfo.environment["CLAP_MLX_DEBUG_PROMPT"] != nil {
              debugLog("eos token \(token) (\(tok.convertIdToToken(token) ?? "?")) after \(generatedCount) tokens; eos set: \(eosTokenIds)")
            }
            finishReason = "stop"
            break generation
          }
          generatedCount += 1
          detokenizer.append(token: token)
          if let chunk = detokenizer.next(), !chunk.isEmpty {
            collected += chunk
            if streaming {
              emit(id: id, token: chunk)
            }
          }
          if generatedCount >= maxTokens {
            finishReason = "length"
            break generation
          }
          while let pending = await buffer.poll() {
            if isCancelMessage(pending, activeId: id) {
              cancelled = true
              break generation
            }
            deferred.append(pending)
          }
        }
        if it.tokenCount >= maxTokens && finishReason == "stop" && !cancelled && generatedCount >= maxTokens {
          finishReason = "length"
        }
        // Sync bookkeeping with the true cache length instead of inferring
        // which sampled tokens the iterator has already fed: the cache offset
        // is authoritative, and any mismatch here corrupts the next request's
        // prefix trim (observed as immediate-EOS or degraded generations).
        let full = fedTokens + sampledTokens
        let cacheLength = caches.first?.offset ?? full.count
        slot.tokens = Array(full.prefix(min(cacheLength, full.count)))
      }

      if !streaming && !collected.isEmpty && !cancelled {
        emit(id: id, content: collected)
      }
      let usage = WorkerUsage(prompt_tokens: promptTokens.count, completion_tokens: generatedCount)
      let cacheInfo = WorkerCache(hit: reusedTokens > 0, reused_tokens: reusedTokens, side_request: false)
      emit(id: id, done: true, cancelled: cancelled ? true : nil, finishReason: cancelled ? "cancel" : finishReason, usage: usage, cache: cacheInfo)
    }
  } catch {
    emit(error: String(describing: error))
    exit(1)
  }
}

await main()
