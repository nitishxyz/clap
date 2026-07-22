import CryptoKit
import Foundation

struct QuantizedLogit: Codable, Equatable {
  let token: Int32
  let value: Int32
}

struct ModelProbeObservation: Codable, Equatable {
  let scenario: String
  let operation: UInt32
  let reused: Int
  let generation: UInt64
  let logicalTokenSHA256: String
  let physicalStateSHA256: String
  let selectedNextToken: Int32
  let top16QuantizedLogitSHA256: String
  let top16QuantizedLogits: [QuantizedLogit]

  enum CodingKeys: String, CodingKey {
    case scenario, operation, reused, generation
    case logicalTokenSHA256 = "logical_token_sha256"
    case physicalStateSHA256 = "physical_state_sha256"
    case selectedNextToken = "selected_next_token"
    case top16QuantizedLogitSHA256 = "top16_quantized_logit_sha256"
    case top16QuantizedLogits = "top16_quantized_logits"
  }

  func canonicalJSON() throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(self), as: UTF8.self)
  }
}

enum ModelProbeFingerprint {
  static func sha256(_ bytes: [UInt8]) -> String {
    SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
  }

  static func tokens(_ tokens: [Int]) -> String {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(tokens.count * 4)
    for token in tokens { append(Int32(token), to: &bytes) }
    return sha256(bytes)
  }

  static func top16(_ logits: [Float]) -> [QuantizedLogit] {
    logits.enumerated().map { token, logit in
      let scaled = (Double(logit) * 1024).rounded(.toNearestOrAwayFromZero)
      let bounded = min(Double(Int32.max), max(Double(Int32.min), scaled))
      return QuantizedLogit(token: Int32(token), value: Int32(bounded))
    }.sorted {
      $0.value != $1.value ? $0.value > $1.value : $0.token < $1.token
    }.prefix(16).map { $0 }
  }

  static func logits(_ values: [QuantizedLogit]) -> String {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(values.count * 8)
    for value in values {
      append(value.token, to: &bytes)
      append(value.value, to: &bytes)
    }
    return sha256(bytes)
  }

  static func physical(offsets: [Int], componentStateCounts: [Int]) -> String {
    var bytes: [UInt8] = []
    for value in offsets { append(Int32(value), to: &bytes) }
    for value in componentStateCounts { append(Int32(value), to: &bytes) }
    return sha256(bytes)
  }

  private static func append(_ value: Int32, to bytes: inout [UInt8]) {
    let word = UInt32(bitPattern: value)
    for shift in stride(from: 0, to: 32, by: 8) {
      bytes.append(UInt8(truncatingIfNeeded: word >> UInt32(shift)))
    }
  }
}
