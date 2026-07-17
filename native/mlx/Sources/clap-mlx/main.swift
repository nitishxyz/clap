import Foundation
import HuggingFace
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

struct ChatRequest: Decodable {
  let id: String?
  let type: String?
  let model: String
  let messages: [ChatMessage]
  let stream: Bool?
  let max_tokens: Int?
  let temperature: Double?
}

struct WorkerUsage: Encodable {
  let prompt_tokens: Int
  let completion_tokens: Int
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

func emit(id: String? = nil, token: String? = nil, content: String? = nil, loaded: Bool? = nil, unloaded: Bool? = nil, done: Bool? = nil, error: String? = nil, cancelled: Bool? = nil, finishReason: String? = nil, usage: WorkerUsage? = nil) {
  let message = WorkerMessage(id: id, token: token, content: content, loaded: loaded, unloaded: unloaded, done: done, error: error, cancelled: cancelled, finish_reason: finishReason, usage: usage)
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

func chatRole(_ role: String) -> Chat.Message.Role {
  switch role {
  case "assistant": .assistant
  case "system": .system
  default: .user
  }
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

func encodeToolCalls(_ calls: [ToolCall]) -> String? {
  let payload = EmittedToolCalls(tool_calls: calls.map { EmittedToolCall(name: $0.function.name, arguments: $0.function.arguments) })
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  guard let data = try? encoder.encode(payload) else { return nil }
  return String(data: data, encoding: .utf8)
}

// Canonical text form for an assistant message that carried structured
// tool_calls, matching what encodeToolCalls produced when we generated them.
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

func normalizedForMatch(_ text: String) -> String {
  text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
}

// The server cleans model output before returning it (strips protocol markers,
// reasoning blocks, whitespace), so the assistant text a client echoes back is
// often a cleaned subset of the raw text we generated. Treat the turn as
// matching when the echoed text is contained in the raw output.
func assistantEchoMatches(cachedRaw: String, incoming: String) -> Bool {
  let cached = normalizedForMatch(cachedRaw)
  let echoed = normalizedForMatch(incoming)
  if cached == echoed { return true }
  guard !echoed.isEmpty else { return false }
  return cached.contains(echoed)
}

func debugLog(_ message: String) {
  FileHandle.standardError.write(Data("[clap-mlx] \(message)\n".utf8))
}

func chatHistoryAndPrompt(from request: ChatRequest) -> (history: [Chat.Message], prompt: String, role: Chat.Message.Role) {
  let entries = request.messages.compactMap { message -> (role: String, content: String)? in
    guard let content = message.content, !content.isEmpty else { return nil }
    return (message.role, content)
  }
  guard let last = entries.last else { return ([], "", .user) }
  let history = entries.dropLast().map { entry in
    Chat.Message(role: chatRole(entry.role), content: entry.content)
  }
  return (Array(history), last.content, chatRole(last.role))
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

func gemma4Prompt(from request: ChatRequest) -> String {
  var parts = ["<bos>"]
  for message in request.messages {
    guard let content = message.content, !content.isEmpty else { continue }
    let role = message.role == "assistant" ? "assistant" : "user"
    parts.append("<|turn>\(role)\n\(content)")
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

    // Warm session cache: keep recent ChatSessions (and their KV caches) alive
    // so a continuing conversation only prefills the new suffix instead of the
    // entire history. Multiple slots because clients such as agent harnesses
    // interleave the main conversation with side requests (title generation,
    // summaries) that would otherwise evict the main session every turn.
    final class SessionCacheEntry {
      let session: ChatSession
      var entries: [(role: String, content: String)]
      let model: String
      let maxTokens: Int
      let temperature: Double
      let toolsKey: String
      init(session: ChatSession, entries: [(role: String, content: String)], model: String, maxTokens: Int, temperature: Double, toolsKey: String) {
        self.session = session
        self.entries = entries
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.toolsKey = toolsKey
      }
    }
    var sessionCache: [SessionCacheEntry] = []
    let sessionCacheLimit = 4
    func invalidateSessionCache() {
      sessionCache = []
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
        invalidateSessionCache()
        loadedContainer = nil
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
        if loadedModel != model {
          invalidateSessionCache()
          loadedContainer = try await loadModelContainer(from: modelDirectory, using: #huggingFaceTokenizerLoader())
          loadedModel = model
          loadedDirectory = modelDirectory
        }
        emit(id: id, loaded: true, done: true)
        continue
      }

      guard let model = control.model else {
        emit(id: id, error: "chat.model is required")
        continue
      }
      let modelDirectory = try validateModelDirectory(model)
      if loadedModel != model || loadedContainer == nil {
        invalidateSessionCache()
        loadedContainer = try await loadModelContainer(from: modelDirectory, using: #huggingFaceTokenizerLoader())
        loadedModel = model
        loadedDirectory = modelDirectory
      }
      guard let container = loadedContainer else {
        emit(id: id, error: "model is not loaded")
        continue
      }
      let request = ChatRequest(id: control.id, type: control.type, model: model, messages: control.messages ?? [], stream: control.stream, max_tokens: control.max_tokens, temperature: control.temperature)
      let maxTokens = request.max_tokens ?? 4096
      let temperature = request.temperature ?? 0.7
      // Extract OpenAI-style tool specs so the chat template can declare them
      // natively; mlx-swift-lm then parses the model's native tool-call tokens
      // into structured events instead of dropping them as special tokens.
      var toolSpecs: [ToolSpec]? = nil
      var toolsKey = ""
      if let envelope = try? JSONDecoder().decode(ToolsEnvelope.self, from: data),
         let rawTools = envelope.tools, !rawTools.isEmpty {
        toolSpecs = rawTools.map { $0.asSchema }
        let keyEncoder = JSONEncoder()
        keyEncoder.outputFormatting = [.sortedKeys]
        if let keyData = try? keyEncoder.encode(rawTools) {
          toolsKey = String(data: keyData, encoding: .utf8) ?? ""
        }
      }
      let generateParameters = GenerateParameters(
        maxTokens: maxTokens,
        temperature: Float(temperature)
      )
      let session: ChatSession
      let inputPrompt: String
      let inputRole: Chat.Message.Role
      let usesFallback = usesGemma4FallbackPrompt(loadedDirectory ?? modelDirectory)
      let entries = request.messages.compactMap { message -> (role: String, content: String)? in
        guard let content = messageText(message) else { return nil }
        return (message.role, content)
      }
      guard let last = entries.last else {
        emit(id: id, error: "chat request contains no messages")
        continue
      }
      func entriesContinue(_ cached: [(role: String, content: String)]) -> Bool {
        guard entries.count == cached.count + 1 else { return false }
        return zip(entries.prefix(cached.count), cached).allSatisfy { incoming, previous in
          guard incoming.role == previous.role else { return false }
          return previous.role == "assistant"
            ? assistantEchoMatches(cachedRaw: previous.content, incoming: incoming.content)
            : incoming.content == previous.content
        }
      }
      var reusedEntry: SessionCacheEntry? = nil
      for (index, candidate) in sessionCache.enumerated() {
        guard candidate.model == model, candidate.maxTokens == maxTokens,
              candidate.temperature == temperature, candidate.toolsKey == toolsKey,
              entriesContinue(candidate.entries) else { continue }
        reusedEntry = sessionCache.remove(at: index)
        break
      }
      let continuesCachedConversation = reusedEntry != nil
      if reusedEntry == nil && !sessionCache.isEmpty {
        let incomingShape = entries.map { "\($0.role):\($0.content.count)" }.joined(separator: ",")
        let cachedShapes = sessionCache.map { entry in entry.entries.map { "\($0.role):\($0.content.count)" }.joined(separator: ",") }.joined(separator: " | ")
        debugLog("session cache miss: incoming [\(incomingShape)] cached [\(cachedShapes)] tools \(toolsKey.count) chars")
      } else if reusedEntry != nil {
        debugLog("session cache hit: reusing warm KV cache (\(entries.count) msgs)")
      }
      if usesFallback {
        if continuesCachedConversation, let reused = reusedEntry {
          // The warm session's KV cache already covers the prior conversation
          // (including our own reply), so feed only the new turn's fragment.
          session = reused.session
          let role = last.role == "assistant" ? "assistant" : "user"
          inputPrompt = "<|turn>\(role)\n\(last.content)<|turn>assistant\n"
        } else {
          session = ChatSession(container, generateParameters: generateParameters)
          inputPrompt = gemma4Prompt(from: request)
        }
        inputRole = .user
      } else {
        if continuesCachedConversation, let reused = reusedEntry {
          session = reused.session
        } else {
          let history = entries.dropLast().map { Chat.Message(role: chatRole($0.role), content: $0.content) }
          session = ChatSession(container, history: history, generateParameters: generateParameters)
          session.tools = toolSpecs
        }
        inputPrompt = last.content
        inputRole = chatRole(last.role)
      }
      let streaming = request.stream ?? true
      var collected = ""
      var cancelled = false
      var usage: WorkerUsage? = nil
      var finishReason: String? = nil
      var nativeToolCalls: [ToolCall] = []
      generation: for try await item in session.streamDetails(to: inputPrompt, role: inputRole, images: [], videos: []) {
        while let pending = await buffer.poll() {
          if isCancelMessage(pending, activeId: id) {
            cancelled = true
            break generation
          }
          deferred.append(pending)
        }
        if let call = item.toolCall {
          nativeToolCalls.append(call)
          continue
        }
        if let info = item.info {
          usage = WorkerUsage(prompt_tokens: info.promptTokenCount, completion_tokens: info.generationTokenCount)
          finishReason = switch info.stopReason {
          case .length: "length"
          default: "stop"
          }
          continue
        }
        guard let chunk = item.chunk, !chunk.isEmpty else { continue }
        collected += chunk
        if streaming {
          emit(id: id, token: chunk)
        }
      }
      if !cancelled, !nativeToolCalls.isEmpty, let toolCallJson = encodeToolCalls(nativeToolCalls) {
        // Surface native tool-call events in the canonical JSON shape the
        // server already parses from plain text output.
        collected += toolCallJson
        if streaming {
          emit(id: id, token: toolCallJson)
        }
      }
      if !streaming && !collected.isEmpty && !cancelled {
        emit(id: id, content: collected)
      }
      if !cancelled {
        // The session's history now ends with this reply; remember it so the
        // next request (history + our reply + new message) is a prefix match
        // and reuses the warm KV cache. A cancelled generation leaves partial
        // tokens in the KV cache, so those sessions are dropped instead.
        var storedEntries = entries
        storedEntries.append(("assistant", collected))
        sessionCache.append(SessionCacheEntry(session: session, entries: storedEntries, model: model, maxTokens: maxTokens, temperature: temperature, toolsKey: toolsKey))
        if sessionCache.count > sessionCacheLimit {
          sessionCache.removeFirst()
        }
      }
      emit(id: id, done: true, cancelled: cancelled ? true : nil, finishReason: cancelled ? "cancel" : finishReason, usage: usage)
    }
  } catch {
    emit(error: String(describing: error))
    exit(1)
  }
}

await main()
