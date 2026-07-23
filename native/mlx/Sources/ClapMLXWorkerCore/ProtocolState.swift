import CoreFoundation
import Foundation

public struct V1RequestEnvelope {
  public let type: String
  public let requestID: String
  public let targetRequestID: String?
  public let structuredOutput: V1StructuredOutput?
  public let legacyPayload: Data
}

public struct V1StructuredOutput: Equatable, Sendable {
  public let kind: String
  public let strength: String
  public let schemaJSON: Data?
}

public let structuredOutputMaxSchemaBytes = 64 * 1024
private let structuredOutputMaxSchemaDepth = 32
private let structuredOutputMaxSchemaProperties = 1024

public func resolveGenerateModel(requestModel: String?, residentModel: String?) -> String? {
  requestModel ?? residentModel
}

public struct V1EnvelopeDecodeError: Error, CustomStringConvertible {
  public let code: String
  public let requestID: String?
  public let description: String

  public init(code: String, requestID: String?, description: String) {
    self.code = code
    self.requestID = requestID
    self.description = description
  }
}

public func decodeV1Envelope(_ line: String) throws -> V1RequestEnvelope {
  let raw: Any
  do {
    raw = try JSONSerialization.jsonObject(with: Data(line.utf8))
  } catch {
    throw V1EnvelopeDecodeError(code: "malformed_json", requestID: nil,
      description: "Malformed worker request JSON: \(error)")
  }
  guard var object = raw as? [String: Any] else {
    throw V1EnvelopeDecodeError(code: "invalid_request", requestID: nil,
      description: "Worker request must be an object")
  }
  let recoveredID = nonemptyString(object["request_id"])
  guard let version = object["protocol"] as? NSNumber,
        CFGetTypeID(version) != CFBooleanGetTypeID(),
        version.intValue == 1, version.doubleValue == 1 else {
    let actual = object["protocol"].map { String(describing: $0) } ?? "missing"
    throw V1EnvelopeDecodeError(code: "unsupported_protocol_version", requestID: recoveredID,
      description: "Unsupported worker protocol version \(actual); expected 1")
  }
  let id = try requiredString(object, "request_id", recoveredID)
  let type = try requiredString(object, "type", id)
  var target: String?
  var structuredOutput: V1StructuredOutput?
  switch type {
  case "load": _ = try requiredString(object, "model", id)
  case "generate":
    guard object["prompt"] is String else {
      throw invalid(id, "prompt must be a string")
    }
    if let value = object["structured_output"] {
      structuredOutput = try decodeStructuredOutput(value, requestID: id)
    }
    guard object["cache_identity"] is [String: Any] else {
      throw V1EnvelopeDecodeError(code: "cache_identity_required", requestID: id,
        description: "cache_identity is required for generate requests")
    }
  case "cancel": target = try requiredString(object, "target_request_id", id)
  case "set_max_active":
    guard let value = object["max_active"] as? NSNumber,
          CFGetTypeID(value) != CFBooleanGetTypeID(), value.intValue > 0,
          Double(value.intValue) == value.doubleValue else {
      throw invalid(id, "max_active must be a positive integer")
    }
  case "unload", "shutdown": break
  default:
    throw V1EnvelopeDecodeError(code: "unsupported_request_type", requestID: id,
      description: "Unsupported worker request type: \(type)")
  }

  // Convert the canonical v1 envelope into the existing internal request shape.
  // Envelope routing fields never leak through to model request decoding.
  let prompt = object["prompt"] as? String
  let canonicalRequest = object["request"] as? [String: Any]
  object.removeValue(forKey: "protocol")
  object.removeValue(forKey: "request_id")
  object.removeValue(forKey: "target_request_id")
  object.removeValue(forKey: "prompt")
  object.removeValue(forKey: "request")
  object["id"] = type == "cancel" ? target : id
  object["type"] = type == "generate" ? "chat" : type
  if type == "generate", let canonicalRequest {
    for (key, value) in canonicalRequest where key != "cache_identity"
        && key != "structured_output" {
      object[key] = value
    }
  } else if let prompt {
    object["messages"] = [["role": "user", "content": prompt]]
  }
  let payload = try JSONSerialization.data(withJSONObject: object)
  return V1RequestEnvelope(type: type, requestID: id, targetRequestID: target,
    structuredOutput: structuredOutput,
    legacyPayload: payload)
}

private func decodeStructuredOutput(_ raw: Any, requestID: String) throws -> V1StructuredOutput {
  guard let value = raw as? [String: Any] else {
    throw structuredOutputInvalid(requestID, "structured_output must be an object")
  }
  let allowed = Set(["kind", "strength", "schema"])
  if let unsupported = value.keys.first(where: { !allowed.contains($0) }) {
    throw structuredOutputInvalid(requestID,
      "structured_output contains an unsupported field: \(unsupported)")
  }
  guard let kind = value["kind"] as? String,
        kind == "json_object" || kind == "json_schema" else {
    throw structuredOutputInvalid(requestID,
      "structured_output.kind must be json_object or json_schema")
  }
  guard let strength = value["strength"] as? String,
        strength == "best_effort" || strength == "required" else {
    throw structuredOutputInvalid(requestID,
      "structured_output.strength must be best_effort or required")
  }
  if kind == "json_object" {
    guard value["schema"] == nil else {
      throw structuredOutputInvalid(requestID,
        "structured_output.schema is not allowed for json_object")
    }
    return V1StructuredOutput(kind: kind, strength: strength, schemaJSON: nil)
  }
  guard let schema = value["schema"] as? [String: Any] else {
    throw structuredOutputInvalid(requestID,
      "structured_output.schema must be an object for json_schema")
  }
  let data = try JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys])
  guard data.count <= structuredOutputMaxSchemaBytes else {
    throw structuredOutputInvalid(requestID,
      "structured output schema exceeds the 64 KiB limit")
  }
  var properties = 0
  try validateSchema(schema, depth: 1, properties: &properties, requestID: requestID)
  return V1StructuredOutput(kind: kind, strength: strength, schemaJSON: data)
}

private func validateSchema(_ value: Any, depth: Int, properties: inout Int,
                            requestID: String) throws {
  guard depth <= structuredOutputMaxSchemaDepth else {
    throw structuredOutputInvalid(requestID,
      "structured output schema exceeds the maximum depth of 32")
  }
  if let array = value as? [Any] {
    for child in array {
      try validateSchema(child, depth: depth + 1, properties: &properties, requestID: requestID)
    }
  } else if let object = value as? [String: Any] {
    if let reference = object["$ref"] as? String, !reference.hasPrefix("#") {
      throw structuredOutputInvalid(requestID,
        "remote structured output schema references are not allowed")
    }
    if let declared = object["properties"] as? [String: Any] {
      properties += declared.count
      guard properties <= structuredOutputMaxSchemaProperties else {
        throw structuredOutputInvalid(requestID,
          "structured output schema exceeds the limit of 1024 properties")
      }
    }
    for child in object.values {
      try validateSchema(child, depth: depth + 1, properties: &properties, requestID: requestID)
    }
  }
}

private func structuredOutputInvalid(_ requestID: String,
                                     _ description: String) -> V1EnvelopeDecodeError {
  V1EnvelopeDecodeError(code: "invalid_structured_output", requestID: requestID,
    description: description)
}

private func nonemptyString(_ value: Any?) -> String? {
  (value as? String).flatMap { $0.isEmpty ? nil : $0 }
}

private func requiredString(_ object: [String: Any], _ key: String,
                            _ requestID: String?) throws -> String {
  guard let value = nonemptyString(object[key]) else {
    throw invalid(requestID, "\(key) must be a non-empty string")
  }
  return value
}

private func invalid(_ requestID: String?, _ description: String) -> V1EnvelopeDecodeError {
  V1EnvelopeDecodeError(code: "invalid_request", requestID: requestID,
    description: description)
}

public struct ProtocolSequence: Equatable, Sendable {
  public let requestID: String
  public let sequence: UInt64

  public init(requestID: String, sequence: UInt64) {
    self.requestID = requestID
    self.sequence = sequence
  }
}

public final class ProtocolTerminalTracker {
  private var sequences: [String: UInt64] = [:]
  private var terminals: Set<String> = []

  public init() {}

  public func accept(_ requestID: String) -> ProtocolSequence? {
    guard !requestID.isEmpty, sequences[requestID] == nil,
          !terminals.contains(requestID) else { return nil }
    sequences[requestID] = 1
    return ProtocolSequence(requestID: requestID, sequence: 0)
  }

  public func event(_ requestID: String) -> ProtocolSequence? {
    next(requestID, terminal: false)
  }

  public func terminal(_ requestID: String) -> ProtocolSequence? {
    next(requestID, terminal: true)
  }

  public func isTerminal(_ requestID: String) -> Bool {
    terminals.contains(requestID)
  }

  private func next(_ requestID: String, terminal: Bool) -> ProtocolSequence? {
    guard !requestID.isEmpty, !terminals.contains(requestID),
          let sequence = sequences[requestID] else { return nil }
    sequences[requestID] = sequence &+ 1
    if terminal { terminals.insert(requestID) }
    return ProtocolSequence(requestID: requestID, sequence: sequence)
  }
}
