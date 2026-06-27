import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

struct ChatMessage: Decodable {
  let role: String
  let content: String?
}

struct ChatRequest: Decodable {
  let model: String
  let messages: [ChatMessage]
  let stream: Bool?
  let max_tokens: Int?
  let temperature: Double?
}

struct WorkerMessage: Encodable {
  let token: String?
  let content: String?
  let done: Bool?
  let error: String?
}

func emit(token: String? = nil, content: String? = nil, done: Bool? = nil, error: String? = nil) {
  let message = WorkerMessage(token: token, content: content, done: done, error: error)
  let data = try! JSONEncoder().encode(message)
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data([0x0a]))
}

func prompt(from request: ChatRequest, modelDirectory: URL) -> String {
  if usesGemma4FallbackPrompt(modelDirectory) {
    return gemma4Prompt(from: request)
  }
  if let lastUser = request.messages.last(where: { $0.role == "user" }), let content = lastUser.content, !content.isEmpty {
    return content
  }
  return request.messages.compactMap { message in
    guard let content = message.content, !content.isEmpty else { return nil }
    return "\(message.role): \(content)"
  }.joined(separator: "\n")
}

func usesGemma4FallbackPrompt(_ modelDirectory: URL) -> Bool {
  let configURL = modelDirectory.appendingPathComponent("config.json")
  let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
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

    guard let line = readLine(), let data = line.data(using: .utf8) else {
      emit(error: "expected one JSON request line on stdin")
      exit(2)
    }
    let request = try JSONDecoder().decode(ChatRequest.self, from: data)

    let modelDirectory = try validateModelDirectory(request.model)
    let container = try await loadModelContainer(from: modelDirectory, using: #huggingFaceTokenizerLoader())
    let session = ChatSession(
      container,
      generateParameters: GenerateParameters(
        maxTokens: request.max_tokens ?? 256,
        temperature: Float(request.temperature ?? 0.7)
      )
    )

    let inputPrompt = prompt(from: request, modelDirectory: modelDirectory)

    if request.stream ?? true {
      for try await chunk in session.streamResponse(to: inputPrompt) {
        if !chunk.isEmpty {
          emit(token: chunk)
        }
      }
    } else {
      let output = try await session.respond(to: inputPrompt)
      if !output.isEmpty {
        emit(content: output)
      }
    }
    emit(done: true)
  } catch {
    emit(error: String(describing: error))
    exit(1)
  }
}

await main()
