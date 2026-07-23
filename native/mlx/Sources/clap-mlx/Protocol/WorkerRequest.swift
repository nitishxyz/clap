import ClapMLXWorkerCore
import ClapMLXCache
import Foundation

struct V1Request {
  let type: String
  let requestID: String
  let targetRequestID: String?
  let control: ControlRequest
  let controlData: Data
}

struct StructuredOutputRequest: Decodable {
  let kind: String
  let strength: String
  let schema: StructuredJSONValue?
}

enum StructuredJSONValue: Decodable {
  case object([String: StructuredJSONValue]), array([StructuredJSONValue]), string(String)
  case number(Double), boolean(Bool), null

  init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() { self = .null }
    else if let value = try? container.decode([String: StructuredJSONValue].self) { self = .object(value) }
    else if let value = try? container.decode([StructuredJSONValue].self) { self = .array(value) }
    else if let value = try? container.decode(String.self) { self = .string(value) }
    else if let value = try? container.decode(Bool.self) { self = .boolean(value) }
    else { self = .number(try container.decode(Double.self)) }
  }

  var foundationValue: Any {
    switch self {
    case .object(let value): return value.mapValues(\.foundationValue)
    case .array(let value): return value.map(\.foundationValue)
    case .string(let value): return value
    case .number(let value): return value
    case .boolean(let value): return value
    case .null: return NSNull()
    }
  }
}

typealias V1DecodeError = V1EnvelopeDecodeError

func decodeV1Request(_ line: String) throws -> V1Request {
  let envelope = try decodeV1Envelope(line)
  if envelope.structuredOutput?.strength == "required" {
    throw V1DecodeError(code: "structured_output_capability_required",
      requestID: envelope.requestID,
      description: "MLX supports structured output only as best_effort post-validation")
  }
  let control: ControlRequest
  do {
    control = try JSONDecoder().decode(ControlRequest.self, from: envelope.legacyPayload)
  } catch {
    let path: [any CodingKey] = switch error {
    case DecodingError.typeMismatch(_, let context): context.codingPath
    case DecodingError.valueNotFound(_, let context): context.codingPath
    case DecodingError.keyNotFound(_, let context): context.codingPath
    case DecodingError.dataCorrupted(let context): context.codingPath
    default: []
    }
    let identityError = path.contains { $0.stringValue == "cache_identity" }
    throw V1DecodeError(code: identityError ? "invalid_cache_identity" : "invalid_request",
      requestID: envelope.requestID,
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
  let cache_identity: OpaqueCacheIdentityInput?
  let structured_output: StructuredOutputRequest?
}
