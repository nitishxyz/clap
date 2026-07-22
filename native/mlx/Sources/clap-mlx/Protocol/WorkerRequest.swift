import ClapMLXWorkerCore
import Foundation

enum WorkerProtocolMode {
  case legacy
  case v1

  static func fromEnvironment(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> Self {
    environment["CLAP_WORKER_PROTOCOL"] == "v1" ? .v1 : .legacy
  }
}

struct V1Request {
  let type: String
  let requestID: String
  let targetRequestID: String?
  let control: ControlRequest
  let controlData: Data
}

typealias V1DecodeError = V1EnvelopeDecodeError

func decodeV1Request(_ line: String) throws -> V1Request {
  let envelope = try decodeV1Envelope(line)
  let control: ControlRequest
  do {
    control = try JSONDecoder().decode(ControlRequest.self, from: envelope.legacyPayload)
  } catch {
    throw V1DecodeError(code: "invalid_request", requestID: envelope.requestID,
      description: "Invalid \(envelope.type) request payload: \(error)")
  }
  return V1Request(type: envelope.type, requestID: envelope.requestID,
    targetRequestID: envelope.targetRequestID, control: control,
    controlData: envelope.legacyPayload)
}

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

struct CacheBoundaryDescriptor: Decodable {
  let kind: String
  let through_message: Int?
  let label: String?
}

struct CacheIntent: Decodable {
  let namespace: String?
  let tenant: String?
  let project: String?
  let harness: String?
  let agent: String?
  let session: String?
  let priority: String?
  let side_request: Bool?
  let boundaries: [CacheBoundaryDescriptor]?
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
  let max_active: Int?
  let previous_max_active: Int?
  let limiting_reason: String?
  let last_adjustment_reason: String?
  let last_adjustment_at: String?
  let retained_growth_reserve_bytes: UInt64?
  let global_resident_memory_bytes: UInt64?
  let pressure_state: String?
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
  let cache: CacheIntent?
}
