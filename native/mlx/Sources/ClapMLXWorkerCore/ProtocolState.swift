import CoreFoundation
import Foundation

public enum WorkerProtocolMode: Equatable, Sendable {
  case legacy
  case v1

  public static func fromEnvironment(
    _ environment: [String: String] = ProcessInfo.processInfo.environment
  ) -> Self {
    environment["CLAP_WORKER_PROTOCOL"] == "legacy" ? .legacy : .v1
  }
}

public struct V1RequestEnvelope {
  public let type: String
  public let requestID: String
  public let targetRequestID: String?
  public let legacyPayload: Data
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
  switch type {
  case "load": _ = try requiredString(object, "model", id)
  case "generate":
    guard object["prompt"] is String else {
      throw invalid(id, "prompt must be a string")
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
  object.removeValue(forKey: "protocol")
  object.removeValue(forKey: "request_id")
  object.removeValue(forKey: "target_request_id")
  object.removeValue(forKey: "prompt")
  object["id"] = type == "cancel" ? target : id
  object["type"] = type == "generate" ? "chat" : type
  if let prompt {
    object["messages"] = [["role": "user", "content": prompt]]
  }
  let payload = try JSONSerialization.data(withJSONObject: object)
  return V1RequestEnvelope(type: type, requestID: id, targetRequestID: target,
    legacyPayload: payload)
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
