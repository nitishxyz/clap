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
  let reuse_kind: String?
  let reuse_scope: String?
  let side_request: Bool
}

struct WorkerPrefill: Encodable {
  let done: Int
  let total: Int
}

struct WorkerMemory: Encodable {
  let active_bytes: Int
  let cache_bytes: Int
  let peak_active_bytes: Int
}

struct WorkerMessage: Encodable {
  let id: String?
  let started: Bool?
  let token: String?
  let content: String?
  let loaded: Bool?
  let unloaded: Bool?
  let done: Bool?
  let error: String?
  let code: String?
  let cancelled: Bool?
  let finish_reason: String?
  let usage: WorkerUsage?
  let cache: WorkerCache?
  let prefill: WorkerPrefill?
  let memory: WorkerMemory?
}

// OpenAI-style stop can be a bare string or an array of strings.
enum StopField: Decodable {
  case none
  case sequences([String])

  init(from decoder: any Swift.Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let single = try? container.decode(String.self) {
      self = .sequences([single])
    } else if let list = try? container.decode([String].self) {
      self = .sequences(list)
    } else {
      self = .none
    }
  }

  var values: [String] {
    if case .sequences(let list) = self { return list.filter { !$0.isEmpty } }
    return []
  }
}

struct ControlRequest: Decodable {
  let id: String?
  let type: String?
  let model: String?
  let messages: [ChatMessage]?
  let stream: Bool?
  let max_tokens: Int?
  let temperature: Double?
  let top_p: Double?
  let top_k: Int?
  let min_p: Double?
  let seed: UInt64?
  let stop: StopField?
  let repetition_penalty: Double?
  let presence_penalty: Double?
  let frequency_penalty: Double?
}

func memorySnapshot() -> WorkerMemory {
  let snapshot = Memory.snapshot()
  return WorkerMemory(active_bytes: snapshot.activeMemory, cache_bytes: snapshot.cacheMemory, peak_active_bytes: snapshot.peakMemory)
}

func emit(id: String? = nil, started: Bool? = nil, token: String? = nil, content: String? = nil, loaded: Bool? = nil, unloaded: Bool? = nil, done: Bool? = nil, error: String? = nil, code: String? = nil, cancelled: Bool? = nil, finishReason: String? = nil, usage: WorkerUsage? = nil, cache: WorkerCache? = nil, prefill: WorkerPrefill? = nil, memory: WorkerMemory? = nil) {
  let message = WorkerMessage(id: id, started: started, token: token, content: content, loaded: loaded, unloaded: unloaded, done: done, error: error, code: code, cancelled: cancelled, finish_reason: finishReason, usage: usage, cache: cache, prefill: prefill, memory: memory)
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
    var modelContextLength = 0

    let env = ProcessInfo.processInfo.environment
    // Context window policy (parity with the llama worker): default to the
    // model's trained context; CLAP_MLX_CONTEXT pins it, and
    // CLAP_MLX_MAX_SESSION_CTX caps any single session's share.
    let contextOverride = Int(env["CLAP_MLX_CONTEXT"] ?? "") ?? 0
    let sessionCap = Int(env["CLAP_MLX_MAX_SESSION_CTX"] ?? "") ?? 0
    // KV cache quantization (parity with CLAP_LLAMA_KV_TYPE): q8_0/q4_0/f16.
    let kvBits: Int? = switch env["CLAP_MLX_KV_TYPE"] ?? "f16" {
    case "q8_0": 8
    case "q4_0": 4
    default: nil
    }

    func declaredContextLength(_ url: URL) -> Int {
      guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return 0 }
      func read(_ dict: [String: Any]) -> Int {
        for key in ["max_position_embeddings", "n_positions", "max_sequence_length"] {
          if let value = dict[key] as? Int, value > 0 { return value }
        }
        return 0
      }
      let direct = read(json)
      if direct > 0 { return direct }
      if let textConfig = json["text_config"] as? [String: Any] { return read(textConfig) }
      return 0
    }

    // Multi-session token-level KV cache slots (mirrors the llama.cpp
    // worker): each slot remembers the exact token ids resident in its cache.
    // Requests pick the slot with the longest common prefix (any number of
    // interleaved sessions stay warm); misses recycle the LRU slot.
    final class KVSlot {
      var caches: [KVCache] = []
      var tokens: [Int] = []
      var lastUsed: UInt64 = 0
      var busy = false
      // Anchor slots hold an exact-state snapshot of a shared prefix (e.g.
      // the org-wide system prompt) taken before rotating/sliding-window
      // caches rotate it away. They are only used via copy() (whole-copy
      // branching) and never extended in place.
      var isAnchor = false
      var anchorScope: String? = nil
      // A request-local end-of-prompt snapshot restored after decode. Unlike
      // a dedicated prefix anchor, this remains the session's normal slot.
      var isPromptBoundary = false
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
      modelContextLength = contextOverride > 0
        ? contextOverride
        : declaredContextLength(directory.appendingPathComponent("config.json"))
      if kvBits != nil { debugLog("kv cache quantization enabled: \(kvBits!)-bit") }
      debugLog("context length: \(modelContextLength > 0 ? String(modelContextLength) : "unknown")\(sessionCap > 0 ? ", session cap \(sessionCap)" : "")")
      debugLog("model loaded; eos token ids: \(eosTokenIds.sorted())")
      Memory.clearCache()
      let memory = memorySnapshot()
      debugLog("mlx memory after load: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
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
    // ---- Interleaved multi-request scheduler ----------------------------
    // Mirrors the llama.cpp worker's continuous batching at the scheduling
    // level: several requests are active at once, each stepped in round-robin
    // (one prefill chunk OR a few decode tokens per pass), so a long prefill
    // or generation never blocks other sessions' token streams. MLX evaluates
    // sequences one at a time on Metal (no fused multi-sequence batch yet),
    // so aggregate throughput is shared — but head-of-line blocking is gone.

    let prefillChunk = 512
    let decodeStepsPerPass = 6
    let parallelLimit = max(1, Int(ProcessInfo.processInfo.environment["CLAP_MLX_PARALLEL"] ?? "") ?? slotLimit)

    final class ActiveRequest {
      let id: String?
      let streaming: Bool
      let maxTokens: Int
      let promptTokens: [Int]
      let reusedTokens: Int
      let reuseKind: String?
      let reuseScope: String?
      let slot: KVSlot
      var caches: [KVCache]
      var promptBoundaryCaches: [KVCache]? = nil
      var continuationBoundary: Int? = nil
      var continuationBoundaryCaches: [KVCache]? = nil
      var fedTokens: [Int]
      var suffix: [Int]
      var pos = 0
      var iterator: TokenIterator?
      var detokenizer: NaiveStreamingDetokenizer
      var sampledTokens: [Int] = []
      var collected = ""
      var emitted = 0  // chars of `collected` already streamed (stop holdback)
      var generatedCount = 0
      var finishReason = "stop"
      var cancelled = false
      var completed = false
      var failed = false
      // Absolute prompt index where a prefix anchor should be captured
      // during this request's prefill (set when a donor existed but its
      // caches could not rewind to the shared boundary).
      var anchorPlantAt: Int? = nil
      var anchorPlantScope: String? = nil
      var anchorPlanted = false
      let parameters: GenerateParameters
      let stops: [String]
      let holdback: Int

      init(id: String?, streaming: Bool, maxTokens: Int, promptTokens: [Int], reusedTokens: Int, reuseKind: String?, reuseScope: String?, slot: KVSlot, caches: [KVCache], fedTokens: [Int], suffix: [Int], detokenizer: NaiveStreamingDetokenizer, parameters: GenerateParameters, stops: [String]) {
        self.id = id
        self.streaming = streaming
        self.maxTokens = maxTokens
        self.promptTokens = promptTokens
        self.reusedTokens = reusedTokens
        self.reuseKind = reuseKind
        self.reuseScope = reuseScope
        self.slot = slot
        self.caches = caches
        self.fedTokens = fedTokens
        self.suffix = suffix
        self.detokenizer = detokenizer
        self.parameters = parameters
        self.stops = stops
        self.holdback = stops.map { $0.count }.max().map { $0 - 1 } ?? 0
      }
    }

    var active: [ActiveRequest] = []
    var pendingChats: [(id: String?, control: ControlRequest, data: Data)] = []
    var controlBacklog: [String] = []
    var allocatorNeedsIdleClear = false

    // Recurrent caches such as Mamba keep state but intentionally report an
    // offset of zero. Hybrid models still have attention caches whose maximum
    // offset tracks the sequence length; pure recurrent models use the token
    // bookkeeping fallback supplied by the caller.
    func cacheSequenceLength(_ caches: [KVCache], fallback: Int) -> Int {
      let offset = caches.map(\.offset).max() ?? 0
      return offset > 0 ? offset : fallback
    }

    func finalize(_ req: ActiveRequest) {
      req.slot.busy = false
      if req.failed {
        req.slot.caches = []
        req.slot.tokens = []
        req.slot.isPromptBoundary = false
        req.slot.anchorScope = nil
        return
      }
      if let continuationBoundary = req.continuationBoundary,
         let continuationCaches = req.continuationBoundaryCaches {
        req.slot.caches = continuationCaches
        req.slot.tokens = Array(req.promptTokens.prefix(continuationBoundary))
        req.slot.isPromptBoundary = true
        req.slot.anchorScope = "conversation"
        debugLog("restored rolling conversation anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(continuationBoundary) tokens; discarded \(req.promptTokens.count - continuationBoundary) prompt suffix tokens and \(req.generatedCount) decoded tokens")
      } else if let promptBoundary = req.promptBoundaryCaches {
        req.slot.caches = promptBoundary
        req.slot.tokens = req.promptTokens
        req.slot.isPromptBoundary = true
        req.slot.anchorScope = "conversation"
        debugLog("restored prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens; discarded \(req.generatedCount) decoded tokens")
      } else {
        // The cache offset is authoritative for what is resident; any mismatch
        // here corrupts the next request's prefix trim.
        let full = req.fedTokens + req.sampledTokens
        let cacheLength = cacheSequenceLength(req.caches, fallback: full.count)
        req.slot.tokens = Array(full.prefix(min(cacheLength, full.count)))
        req.slot.isPromptBoundary = false
        req.slot.anchorScope = nil
      }
      // Flush any text held back for stop-sequence matching.
      if req.streaming && !req.cancelled && req.emitted < req.collected.count {
        let tail = String(req.collected.dropFirst(req.emitted))
        if !tail.isEmpty { emit(id: req.id, token: tail) }
        req.emitted = req.collected.count
      }
      if !req.streaming && !req.collected.isEmpty && !req.cancelled {
        emit(id: req.id, content: req.collected)
      }
      let usage = WorkerUsage(prompt_tokens: req.promptTokens.count, completion_tokens: req.generatedCount)
      let cacheInfo = WorkerCache(hit: req.reusedTokens > 0, reused_tokens: req.reusedTokens, reuse_kind: req.reuseKind, reuse_scope: req.reuseScope, side_request: false)
      emit(id: req.id, done: true, cancelled: req.cancelled ? true : nil, finishReason: req.cancelled ? "cancel" : req.finishReason, usage: usage, cache: cacheInfo)
    }

    func prepareRequest(id: String?, control: ControlRequest, data: Data) async -> ActiveRequest? {
      do {
        guard let model = control.model else {
          emit(id: id, error: "chat.model is required")
          return nil
        }
        let modelDirectory = try validateModelDirectory(model)
        if loadedModel != model || languageModel == nil {
          try await loadModel(model, directory: modelDirectory)
        }
        guard let lm = languageModel, let tok = tokenizer else {
          emit(id: id, error: "model is not loaded")
          return nil
        }
        let maxTokens = control.max_tokens ?? 4096
        let temperature = control.temperature ?? 0.7
        // Full sampling parity with the llama worker: top_p/top_k/min_p,
        // seed, repetition/presence/frequency penalties, and opt-in KV cache
        // quantization (CLAP_MLX_KV_TYPE=q8_0|q4_0).
        let generateParameters = GenerateParameters(
          maxTokens: maxTokens,
          kvBits: kvBits,
          temperature: Float(temperature),
          topP: Float(control.top_p ?? 1.0),
          topK: control.top_k ?? 0,
          minP: Float(control.min_p ?? 0.0),
          repetitionPenalty: control.repetition_penalty.map { Float($0) },
          presencePenalty: control.presence_penalty.map { Float($0) },
          frequencyPenalty: control.frequency_penalty.map { Float($0) },
          seed: control.seed
        )
        var toolSpecs: [ToolSpec]? = nil
        if let envelope = try? JSONDecoder().decode(ToolsEnvelope.self, from: data),
           let rawTools = envelope.tools, !rawTools.isEmpty {
          toolSpecs = rawTools.compactMap { raw -> ToolSpec? in
            raw.anyValue as? [String: any Sendable]
          }
          guard toolSpecs?.count == rawTools.count else {
            emit(id: id, error: "one or more caller-provided tools could not be represented for the chat template")
            return nil
          }
        }
        let entries = (control.messages ?? []).compactMap { message -> (role: String, content: String)? in
          guard let content = messageText(message) else { return nil }
          if message.role == "tool" {
            return ("user", "Tool result:\n\(content)")
          }
          return (message.role, content)
        }
        guard !entries.isEmpty else {
          emit(id: id, error: "chat request contains no messages")
          return nil
        }
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

        let usesFallback = usesGemma4FallbackPrompt(loadedDirectory ?? modelDirectory)
        var promptTokens: [Int]
        var renderedNatively = false
        var renderedMessages: [[String: any Sendable]]? = nil
        var renderedToolSpecs: [ToolSpec]? = nil
        if usesFallback {
          promptTokens = tok.encode(text: gemma4Prompt(entries: entries), addSpecialTokens: false)
        } else {
          do {
            promptTokens = try tok.applyChatTemplate(messages: structuredMessages, tools: toolSpecs)
            renderedNatively = true
            renderedMessages = structuredMessages
            renderedToolSpecs = toolSpecs
          } catch {
            if toolSpecs != nil {
              let compatible = (toolSpecs ?? []).map { templateCompatibleToolSpec($0) }
              if let tokens = try? tok.applyChatTemplate(messages: structuredMessages, tools: compatible) {
                debugLog("chat template for \(loadedModel ?? model) required a nullable-preserving compatibility view of all \(compatible.count) caller-provided tools")
                promptTokens = tokens
                renderedNatively = true
                renderedMessages = structuredMessages
                renderedToolSpecs = compatible
              } else {
                debugLog("chat template for \(loadedModel ?? model) failed with all \(toolSpecs?.count ?? 0) caller-provided tools (\(error)); retrying with JSON tool instructions")
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
                  renderedNatively = true
                  renderedMessages = retryMessages
                  renderedToolSpecs = nil
                } else {
                  debugLog("template retry without tools failed; using plain transcript")
                  let transcript = patched.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
                  promptTokens = tok.encode(text: transcript)
                }
              }
            } else {
              debugLog("chat template failed (\(error)); using plain transcript")
              let transcript = entries.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
              promptTokens = tok.encode(text: transcript)
            }
          }
        }
        var templateBoundary = 0
        if renderedNatively,
           let renderedMessages,
           let stable = try? tok.applyChatTemplate(
             messages: renderedMessages,
             tools: renderedToolSpecs,
             additionalContext: ["add_generation_prompt": false]
           ),
           stable.count >= 16,
           stable.count < promptTokens.count,
           promptTokens.starts(with: stable) {
          templateBoundary = stable.count
          debugLog("native template continuation boundary: \(templateBoundary)/\(promptTokens.count) tokens")
        }
        var systemBoundary = 0
        if renderedNatively, let renderedMessages {
          let leadingSystem = Array(renderedMessages.prefix { ($0["role"] as? String) == "system" })
          if !leadingSystem.isEmpty {
            let probeA = leadingSystem + [["role": "user", "content": "alpha_clap_cache_probe"]]
            let probeB = leadingSystem + [["role": "user", "content": "zeta_clap_cache_probe"]]
            if let tokensA = try? tok.applyChatTemplate(
                 messages: probeA,
                 tools: renderedToolSpecs,
                 additionalContext: ["add_generation_prompt": false]
               ),
               let tokensB = try? tok.applyChatTemplate(
                 messages: probeB,
                 tools: renderedToolSpecs,
                 additionalContext: ["add_generation_prompt": false]
               ) {
              let limit = min(tokensA.count, tokensB.count, promptTokens.count)
              while systemBoundary < limit,
                    tokensA[systemBoundary] == tokensB[systemBoundary],
                    tokensA[systemBoundary] == promptTokens[systemBoundary] {
                systemBoundary += 1
              }
              if systemBoundary >= 16 {
                debugLog("native template system boundary: \(systemBoundary)/\(promptTokens.count) tokens")
              }
            }
          }
        }
        if ProcessInfo.processInfo.environment["CLAP_MLX_DEBUG_PROMPT"] != nil {
          debugLog("prompt (\(promptTokens.count) tokens): \(tok.decode(tokenIds: promptTokens, skipSpecialTokens: false))")
        }

        // Admission control (parity with the llama worker): reject oversized
        // prompts before any prefill with a structured code the server maps
        // to an OpenAI-style 400.
        let outputReserve = max(1, min(maxTokens, 256))
        if modelContextLength > 0 && promptTokens.count + outputReserve >= modelContextLength {
          emit(id: id, error: "prompt exceeds context window; prompt tokens=\(promptTokens.count), context=\(modelContextLength), reserved output tokens=\(outputReserve). Increase CLAP_MLX_CONTEXT or reduce the prompt/tool history.", code: "context_length_exceeded")
          return nil
        }
        if sessionCap > 0 && promptTokens.count + outputReserve >= sessionCap {
          emit(id: id, error: "prompt exceeds the per-session context cap; prompt tokens=\(promptTokens.count), max_session_ctx=\(sessionCap), reserved output tokens=\(outputReserve). Reduce the prompt/tool history or raise max_session_ctx / CLAP_MLX_MAX_SESSION_CTX.", code: "context_length_exceeded")
          return nil
        }

        // Slot selection: longest shared prefix among non-busy slots; a weak
        // match recycles the LRU non-busy slot. parallelLimit <= slotLimit
        // guarantees a free slot exists.
        var bestSlot: KVSlot? = nil
        var bestPrefix = 0
        for candidate in kvSlots where !candidate.busy && !candidate.isAnchor {
          var prefix = 0
          let maxPrefix = min(candidate.tokens.count, promptTokens.count)
          while prefix < maxPrefix && candidate.tokens[prefix] == promptTokens[prefix] { prefix += 1 }
          if prefix > bestPrefix {
            bestPrefix = prefix
            bestSlot = candidate
          }
        }
        // A future tool turn can change template output shortly before the
        // generation suffix (Qwen's multi-step-tool formatting does this).
        // Snapshot a small safety margin before the no-generation boundary;
        // for large prompts this still preserves effectively all prefill work.
        let templateSafetyTail = 64
        let proactiveAnchor = systemBoundary >= 16
          ? systemBoundary
          : max(0, templateBoundary - templateSafetyTail)
        // Keep the normal session slot at a later, template-safe boundary.
        // The safety tail covers model templates that rewrite control tokens
        // immediately before a tool continuation. The persistent system
        // anchor remains separate for new conversations.
        let rollingBoundary = templateBoundary >= 16
          ? max(systemBoundary, templateBoundary - templateSafetyTail)
          : 0
        var anchorCandidate = proactiveAnchor
        // Demote a best match whose caches cannot rewind to the shared
        // boundary: same-slot reuse would fail, and keeping it as bestSlot
        // would both wipe a warm session and shadow anchor donors.
        if let matched = bestSlot, bestPrefix >= 16 {
          var adjusted = bestPrefix
          if adjusted == promptTokens.count { adjusted -= 1 }
          let rewindNeeded = matched.tokens.count - adjusted
          let trimmableBest = !matched.caches.isEmpty && matched.caches.allSatisfy { $0.isTrimmable }
          if !(trimmableBest || rewindNeeded == 0) {
            if proactiveAnchor < 16 { anchorCandidate = bestPrefix }
            bestSlot = nil
            bestPrefix = 0
          }
        }
        // Cross-slot prefix branching (dedup): if any OTHER slot — busy ones
        // and anchors included — holds a longer shared prefix (e.g. the
        // org-wide system prompt), clone its cache and trim to the shared
        // boundary instead of re-prefilling. Donors qualify when their caches
        // can trim to the boundary OR when the donor's entire cache is a
        // prefix of the new prompt (whole-copy, trim 0 — works even for
        // rotating/sliding-window caches that cannot rewind).
        var branchDonor: KVSlot? = nil
        var branchPrefix = bestPrefix
        for candidate in kvSlots where candidate !== bestSlot {
          guard !candidate.caches.isEmpty else { continue }
          var prefix = 0
          let maxPrefix = min(candidate.tokens.count, promptTokens.count)
          while prefix < maxPrefix && candidate.tokens[prefix] == promptTokens[prefix] { prefix += 1 }
          let wholeCopy = prefix == candidate.tokens.count
          if !wholeCopy && !candidate.caches.allSatisfy({ $0.isTrimmable }) {
            if proactiveAnchor < 16 && prefix > anchorCandidate { anchorCandidate = prefix }
            continue
          }
          if prefix > branchPrefix {
            branchPrefix = prefix
            branchDonor = candidate
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
          } else if let lru = kvSlots.filter({ !$0.busy && !$0.isAnchor }).min(by: { $0.lastUsed < $1.lastUsed }) ?? kvSlots.filter({ !$0.busy }).min(by: { $0.lastUsed < $1.lastUsed }) {
            slot = lru
            slot.caches = []
            slot.tokens = []
            slot.isAnchor = false
            slot.isPromptBoundary = false
            slot.anchorScope = nil
          } else {
            slot = KVSlot()
            kvSlots.append(slot)
          }
        }
        kvUseCounter += 1
        slot.lastUsed = kvUseCounter
        slot.busy = true
        var prefix = bestPrefix
        if prefix == promptTokens.count { prefix -= 1 }  // always feed at least one token for logits

        var caches: [KVCache]
        var fedTokens: [Int]
        var suffix: [Int]
        var reusedTokens = 0
        var reuseKind: String? = nil
        var reuseScope: String? = nil
        var branched = false
        if let donor = branchDonor, branchPrefix >= 16, branchPrefix > bestPrefix, slot !== donor {
          var sharedPrefix = branchPrefix
          if sharedPrefix == promptTokens.count { sharedPrefix -= 1 }
          let cloned = donor.caches.map { $0.copy() }
          let cloneOffset = cacheSequenceLength(cloned, fallback: donor.tokens.count)
          let trimNeeded = cloneOffset - sharedPrefix
          if trimNeeded >= 0 && (trimNeeded == 0 || cloned.allSatisfy({ $0.isTrimmable })) {
            if trimNeeded > 0 {
              for cache in cloned { cache.trim(trimNeeded) }
            }
            caches = cloned
            fedTokens = Array(promptTokens.prefix(sharedPrefix))
            suffix = Array(promptTokens.dropFirst(sharedPrefix))
            reusedTokens = sharedPrefix
            reuseKind = donor.isAnchor || donor.isPromptBoundary ? "anchor" : "branch"
            reuseScope = donor.anchorScope
            prefix = sharedPrefix
            branched = true
            debugLog("kv prefix branch: cloned \(sharedPrefix)/\(promptTokens.count) shared tokens from \(donor.isAnchor ? "an anchor" : "another slot"), prefilling \(suffix.count)")
          } else {
            caches = []
            fedTokens = []
            suffix = promptTokens
          }
        } else {
          caches = []
          fedTokens = []
          suffix = promptTokens
        }
        let trimmable = !slot.caches.isEmpty && slot.caches.allSatisfy { $0.isTrimmable }
        let trimNeeded = slot.tokens.count - prefix
        if branched {
          // cache assignment handled above
        } else if prefix > 0 && !slot.caches.isEmpty && (trimmable || trimNeeded == 0) {
          if trimNeeded > 0 {
            for cache in slot.caches { cache.trim(trimNeeded) }
          }
          caches = slot.caches
          fedTokens = Array(promptTokens.prefix(prefix))
          suffix = Array(promptTokens.dropFirst(prefix))
          reusedTokens = prefix
          reuseKind = slot.isPromptBoundary ? "anchor" : "slot"
          reuseScope = slot.anchorScope
          debugLog("kv prefix reuse (slot \(kvSlots.firstIndex(where: { $0 === slot }) ?? -1)): \(prefix)/\(promptTokens.count) tokens cached, prefilling \(suffix.count)")
        } else {
          if prefix >= 16 && !slot.caches.isEmpty && !trimmable {
            // Same-slot rewind failed too; remember the boundary for anchoring.
            if prefix > anchorCandidate { anchorCandidate = prefix }
          }
          caches = lm.newCache(parameters: generateParameters)
          fedTokens = []
          suffix = promptTokens
          if !slot.tokens.isEmpty {
            let reason = trimmable || slot.caches.isEmpty
              ? "no usable prefix"
              : "recurrent cache cannot rewind (matched \(prefix), needs trim of \(trimNeeded))"
            debugLog("kv cache miss: \(reason) (cached \(slot.tokens.count), prompt \(promptTokens.count))")
          }
        }
        slot.caches = caches
        slot.tokens = fedTokens
        slot.isPromptBoundary = false
        slot.anchorScope = nil

        let request = ActiveRequest(
          id: id,
          streaming: control.stream ?? true,
          maxTokens: maxTokens,
          promptTokens: promptTokens,
          reusedTokens: reusedTokens,
          reuseKind: reuseKind,
          reuseScope: reuseScope,
          slot: slot,
          caches: caches,
          fedTokens: fedTokens,
          suffix: suffix,
          detokenizer: NaiveStreamingDetokenizer(tokenizer: tok),
          parameters: generateParameters,
          stops: control.stop?.values ?? []
        )
        if rollingBoundary >= 16,
           rollingBoundary < promptTokens.count,
           rollingBoundary > systemBoundary {
          request.continuationBoundary = rollingBoundary
          if rollingBoundary == reusedTokens,
             request.caches.contains(where: { !$0.isTrimmable }) {
            request.continuationBoundaryCaches = request.caches.map { $0.copy() }
            debugLog("captured rolling conversation anchor from reused cache: \(rollingBoundary) tokens")
          }
        }
        // Plant an anchor at the native template's pre-generation boundary or
        // at a discovered unrecoverable shared boundary. Tool continuations
        // rerender the prompt before the generation suffix, so the full prompt
        // snapshot alone is not necessarily their exact token prefix.
        if reusedTokens == 0, anchorCandidate >= 16, anchorCandidate < promptTokens.count {
          let boundary = Array(promptTokens.prefix(anchorCandidate))
          let exists = kvSlots.contains { $0.tokens == boundary }
          if !exists {
            request.anchorPlantAt = anchorCandidate
            request.anchorPlantScope = anchorCandidate == systemBoundary ? "system" : "conversation"
          }
        }
        return request
      } catch {
        emit(id: id, error: String(describing: error))
        return nil
      }
    }

    @discardableResult
    func snapshotAnchor(_ req: ActiveRequest, at plant: Int, reason: String) -> Bool {
      let boundary = Array(req.promptTokens.prefix(plant))
      let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
      guard req.fedTokens == boundary, offset == plant else {
        debugLog("\(reason) anchor skipped: fed=\(req.fedTokens.count) offset=\(offset) plant=\(plant)")
        return false
      }
      // Dedupe against OTHER slots only: mid-prefill, the requester's own
      // slot tokens are exactly the boundary.
      guard !kvSlots.contains(where: { $0 !== req.slot && $0.tokens == boundary }) else { return true }
      // When the pool is full, a small side-request anchor must not evict an
      // expensive harness/system anchor. Replace only an anchor no larger
      // than the new one, preferring the smallest and then oldest. An empty
      // idle session slot is also safe; otherwise skip this snapshot.
      let anchor: KVSlot
      if kvSlots.count < slotLimit {
        anchor = KVSlot()
        kvSlots.append(anchor)
      } else if let victim = kvSlots.filter({
          !$0.busy && $0 !== req.slot && $0.isAnchor && $0.tokens.count <= plant
        }).min(by: {
          $0.tokens.count == $1.tokens.count ? $0.lastUsed < $1.lastUsed : $0.tokens.count < $1.tokens.count
        })
        ?? kvSlots.filter({ !$0.busy && $0 !== req.slot && !$0.isAnchor && $0.caches.isEmpty }).min(by: { $0.lastUsed < $1.lastUsed }) {
        anchor = victim
      } else {
        debugLog("\(reason) anchor skipped: no lower-value cache slot available for \(plant) tokens")
        return false
      }
      anchor.isAnchor = true
      anchor.anchorScope = req.anchorPlantScope
      anchor.caches = req.caches.map { $0.copy() }
      anchor.tokens = boundary
      kvUseCounter += 1
      anchor.lastUsed = kvUseCounter
      debugLog("planted \(reason) anchor: \(plant) tokens (exact-state snapshot for non-rewindable caches)")
      return true
    }

    func plantAnchor(_ req: ActiveRequest) {
      guard !req.anchorPlanted, let plant = req.anchorPlantAt else { return }
      req.anchorPlanted = true
      _ = snapshotAnchor(req, at: plant, reason: "prefix")
    }

    func captureContinuationBoundary(_ req: ActiveRequest) {
      guard req.continuationBoundaryCaches == nil,
            let boundary = req.continuationBoundary,
            boundary == req.fedTokens.count,
            req.caches.contains(where: { !$0.isTrimmable }) else { return }
      let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
      guard offset == boundary else {
        debugLog("rolling conversation anchor skipped: fed=\(req.fedTokens.count) offset=\(offset) boundary=\(boundary)")
        return
      }
      req.continuationBoundaryCaches = req.caches.map { $0.copy() }
      debugLog("captured rolling conversation anchor: \(boundary) tokens")
    }

    func step(_ req: ActiveRequest) {
      guard !req.cancelled, !req.completed, !req.failed, let lm = languageModel else { return }
      do {
        if req.iterator == nil {
          // Split chunk boundaries at the anchor plant point so the cache
          // state exactly at the shared boundary exists for snapshotting.
          var chunkEnd = min(req.pos + prefillChunk, req.suffix.count)
          if let plant = req.anchorPlantAt, !req.anchorPlanted {
            let rel = plant - req.reusedTokens
            if req.pos < rel && rel < chunkEnd { chunkEnd = rel }
          }
          if let boundary = req.continuationBoundary, req.continuationBoundaryCaches == nil {
            let rel = boundary - req.reusedTokens
            if req.pos < rel && rel < chunkEnd { chunkEnd = rel }
          }
          if chunkEnd < req.suffix.count {
            // Creating the iterator prefills the chunk into the shared cache;
            // the sampled token is discarded.
            let chunk = Array(req.suffix[req.pos ..< chunkEnd])
            _ = try TokenIterator(input: LMInput(tokens: MLXArray(chunk)), model: lm, cache: req.caches, parameters: req.parameters)
            req.pos = chunkEnd
            req.fedTokens.append(contentsOf: chunk)
            req.slot.tokens = req.fedTokens
            emit(id: req.id, prefill: WorkerPrefill(done: req.reusedTokens + req.pos, total: req.promptTokens.count))
            if let plant = req.anchorPlantAt, !req.anchorPlanted, plant - req.reusedTokens == req.pos {
              plantAnchor(req)
            }
            if let boundary = req.continuationBoundary, boundary - req.reusedTokens == req.pos {
              captureContinuationBoundary(req)
            }
          } else {
            if let plant = req.anchorPlantAt, !req.anchorPlanted, plant - req.reusedTokens == req.pos {
              plantAnchor(req)
            }
            if let boundary = req.continuationBoundary, boundary - req.reusedTokens == req.pos {
              captureContinuationBoundary(req)
            }
            let tail = Array(req.suffix.dropFirst(req.pos))
            req.iterator = try TokenIterator(input: LMInput(tokens: MLXArray(tail)), model: lm, cache: req.caches, parameters: req.parameters)
            req.pos = req.suffix.count
            req.fedTokens.append(contentsOf: tail)
            req.slot.tokens = req.fedTokens
            // Decode mutates recurrent cache state irreversibly. Preserve the
            // exact end-of-prompt state in this request and restore it into
            // the same slot at finalize, so even a one-slot worker can reuse
            // it for the next tool-result continuation.
            if req.continuationBoundaryCaches == nil && req.caches.contains(where: { !$0.isTrimmable }) {
              let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
              if offset == req.promptTokens.count {
                req.promptBoundaryCaches = req.caches.map { $0.copy() }
                debugLog("captured prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens")
              } else {
                debugLog("prompt-boundary anchor skipped: cache offset \(offset), prompt \(req.promptTokens.count)")
              }
            }
          }
          return
        }
        guard var it = req.iterator else { return }
        var steps = 0
        while steps < decodeStepsPerPass {
          guard let token = it.next() else {
            req.completed = true
            break
          }
          steps += 1
          req.sampledTokens.append(token)
          if eosTokenIds.contains(token) {
            if ProcessInfo.processInfo.environment["CLAP_MLX_DEBUG_PROMPT"] != nil {
              debugLog("eos token \(token) (\(tokenizer?.convertIdToToken(token) ?? "?")) after \(req.generatedCount) tokens; eos set: \(eosTokenIds)")
            }
            req.finishReason = "stop"
            req.completed = true
            break
          }
          req.generatedCount += 1
          req.detokenizer.append(token: token)
          if let chunk = req.detokenizer.next(), !chunk.isEmpty {
            req.collected += chunk
            // Stop sequences: scan collected text; on match truncate and
            // finish. While streaming, hold back enough of the tail that a
            // stop split across tokens is never emitted.
            if !req.stops.isEmpty {
              // Only the tail can contain a new match: the window covers the
              // fresh chunk plus a full stop length of earlier context.
              let windowStart = req.collected.index(
                req.collected.endIndex,
                offsetBy: -min(req.collected.count, chunk.count + req.holdback),
              )
              var earliest: Range<String.Index>? = nil
              for stop in req.stops {
                if let found = req.collected.range(of: stop, range: windowStart..<req.collected.endIndex),
                   earliest == nil || found.lowerBound < earliest!.lowerBound {
                  earliest = found
                }
              }
              if let match = earliest {
                req.collected = String(req.collected[..<match.lowerBound])
                if req.streaming && req.emitted < req.collected.count {
                  emit(id: req.id, token: String(req.collected.dropFirst(req.emitted)))
                  req.emitted = req.collected.count
                }
                req.finishReason = "stop"
                req.completed = true
                break
              }
            }
            if req.streaming {
              if req.stops.isEmpty {
                emit(id: req.id, token: chunk)
                req.emitted = req.collected.count
              } else {
                let safe = max(req.emitted, req.collected.count - req.holdback)
                if safe > req.emitted {
                  let start = req.collected.index(req.collected.startIndex, offsetBy: req.emitted)
                  let end = req.collected.index(req.collected.startIndex, offsetBy: safe)
                  emit(id: req.id, token: String(req.collected[start..<end]))
                  req.emitted = safe
                }
              }
            }
          }
          if req.generatedCount >= req.maxTokens {
            req.finishReason = "length"
            req.completed = true
            break
          }
        }
        req.iterator = it
      } catch {
        emit(id: req.id, error: String(describing: error))
        req.failed = true
      }
    }

    // Returns true when the worker should shut down.
    func handleLine(_ line: String) async -> Bool {
      guard !line.isEmpty, let data = line.data(using: .utf8),
            let control = try? JSONDecoder().decode(ControlRequest.self, from: data) else { return false }
      let id = control.id
      let type = control.type ?? "chat"

      if type == "shutdown" {
        for req in active {
          req.cancelled = true
          finalize(req)
        }
        active.removeAll()
        for pending in pendingChats {
          emit(id: pending.id, done: true, cancelled: true, finishReason: "cancel", usage: nil, cache: nil)
        }
        pendingChats.removeAll()
        emit(id: id, done: true)
        return true
      }

      if type == "cancel" {
        let target = control.id
        let matchesAll = target == nil || target?.isEmpty == true
        for req in active where matchesAll || req.id == target {
          req.cancelled = true
        }
        var remaining: [(id: String?, control: ControlRequest, data: Data)] = []
        for pending in pendingChats {
          if matchesAll || pending.id == target {
            emit(id: pending.id, done: true, cancelled: true, finishReason: "cancel", usage: nil, cache: nil)
          } else {
            remaining.append(pending)
          }
        }
        pendingChats = remaining
        return false
      }

      if type == "unload" || type == "load" {
        // Model mutations wait until in-flight requests drain.
        if !active.isEmpty || !pendingChats.isEmpty {
          controlBacklog.append(line)
          return false
        }
        if type == "unload" {
          invalidateKVCache()
          loadedContainer = nil
          languageModel = nil
          tokenizer = nil
          loadedDirectory = nil
          loadedModel = nil
          emit(id: id, unloaded: true, done: true)
          return false
        }
        guard let model = control.model else {
          emit(id: id, error: "load.model is required")
          return false
        }
        do {
          let modelDirectory = try validateModelDirectory(model)
          if loadedModel != model || languageModel == nil {
            try await loadModel(model, directory: modelDirectory)
          }
          emit(id: id, loaded: true, done: true, memory: memorySnapshot())
        } catch {
          emit(id: id, error: String(describing: error))
        }
        return false
      }

      pendingChats.append((id: id, control: control, data: data))
      return false
    }

    mainLoop: while true {
      // Idle: block on input (or drain deferred control work) instead of
      // spinning; busy: poll without blocking so generation keeps stepping.
      if active.isEmpty && pendingChats.isEmpty {
        if !controlBacklog.isEmpty {
          let line = controlBacklog.removeFirst()
          if await handleLine(line) { break mainLoop }
          continue mainLoop
        }
        guard let line = await buffer.next() else { break mainLoop }
        if await handleLine(line) { break mainLoop }
      }
      while let line = await buffer.poll() {
        if await handleLine(line) { break mainLoop }
      }

      // Admit pending chats up to the parallel limit. Requests for a
      // different model wait until the current model's requests drain.
      while active.count < parallelLimit, !pendingChats.isEmpty {
        let candidate = pendingChats[0]
        let needsLoad = loadedModel != candidate.control.model || languageModel == nil
        if needsLoad && !active.isEmpty { break }
        pendingChats.removeFirst()
        emit(id: candidate.id, started: true)
        if let request = await prepareRequest(id: candidate.id, control: candidate.control, data: candidate.data) {
          active.append(request)
          allocatorNeedsIdleClear = true
        }
      }

      // Round-robin: one prefill chunk or a few decode tokens per request.
      for request in active {
        step(request)
      }
      for request in active where request.completed || request.cancelled || request.failed {
        finalize(request)
      }
      active.removeAll { $0.completed || $0.cancelled || $0.failed }
      if active.isEmpty && pendingChats.isEmpty && allocatorNeedsIdleClear {
        Memory.clearCache()
        let memory = memorySnapshot()
        debugLog("mlx memory after idle clear: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
        emit(memory: memory)
        allocatorNeedsIdleClear = false
      }
    }
  } catch {
    emit(error: String(describing: error))
    exit(1)
  }
}

await main()
