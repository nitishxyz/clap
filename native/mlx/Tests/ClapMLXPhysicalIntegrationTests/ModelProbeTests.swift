import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import ClapMLXCache
@testable import ClapMLXGeneration
@testable import ClapMLXModel

@Suite("MLX physical model integration probe")
struct ModelProbeTests {
  @Test("fingerprints and quantized top logits are canonical")
  func deterministicHelpers() throws {
    #expect(ModelProbeFingerprint.sha256(Array("abc".utf8)) ==
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    #expect(ModelProbeFingerprint.tokens([1, 2, -1]) ==
      "0a1ce634879f6a487527c9a185b1a4a3de7f41238ad4139e3c6d9e0da723628c")
    let top = ModelProbeFingerprint.top16([1, 2, 2, -1])
    #expect(top == [QuantizedLogit(token: 1, value: 2_048),
      QuantizedLogit(token: 2, value: 2_048),
      QuantizedLogit(token: 0, value: 1_024),
      QuantizedLogit(token: 3, value: -1_024)])
    #expect(ModelProbeFingerprint.logits(Array(top.prefix(3))) ==
      "d663b2bef022dd313400ebbe1658c8c8f05f83014e5c0c98a0265206e801731c")
    let observation = ModelProbeObservation(scenario: "cold", operation: 0, reused: 0,
      generation: 2, logicalTokenSHA256: "a", physicalStateSHA256: "b",
      selectedNextToken: 1, top16QuantizedLogitSHA256: "c",
      top16QuantizedLogits: [QuantizedLogit(token: 1, value: 2)])
    #expect(try observation.canonicalJSON() ==
      "{\"generation\":2,\"logical_token_sha256\":\"a\",\"operation\":0,\"physical_state_sha256\":\"b\",\"reused\":0,\"scenario\":\"cold\",\"selected_next_token\":1,\"top16_quantized_logit_sha256\":\"c\",\"top16_quantized_logits\":[{\"token\":1,\"value\":2}]}")
  }

  @Test("real MLX model emits a canonical cache observation",
        .enabled(if: ProcessInfo.processInfo.environment["CLAP_TEST_MLX_MODEL"] != nil))
  func realModel() async throws {
    guard let path = ProcessInfo.processInfo.environment["CLAP_TEST_MLX_MODEL"] else { return }
    let directory = try ModelLoader.validateDirectory(path)
    let loaded = try await ModelLoader.load(from: directory)
    let tokens = loaded.tokenizer.encode(text: "The deterministic next token is",
      addSpecialTokens: true)
    let split = max(1, tokens.count - 2)
    let prefix = Array(tokens.prefix(split))
    let suffix = Array(tokens.dropFirst(split))
    let parameters = GenerateParameters(temperature: 0)

    func observation(_ scenario: String, operation: UInt32, reused: Int,
                     logits: MLXArray, caches: [any KVCache]) -> ModelProbeObservation {
      eval(logits)
      let values = logits[0, -1].asArray(Float.self)
      let top = ModelProbeFingerprint.top16(values)
      let selected = values.enumerated().max {
        $0.element != $1.element ? $0.element < $1.element : $0.offset > $1.offset
      }!.offset
      return ModelProbeObservation(scenario: scenario, operation: operation,
        reused: reused, generation: UInt64(reused + 1),
        logicalTokenSHA256: ModelProbeFingerprint.tokens(tokens),
        physicalStateSHA256: ModelProbeFingerprint.physical(
          offsets: caches.map(\.offset), componentStateCounts: caches.map { $0.state.count }),
        selectedNextToken: Int32(selected),
        top16QuantizedLogitSHA256: ModelProbeFingerprint.logits(top),
        top16QuantizedLogits: top)
    }

    func evaluate(_ scenario: String, operation: UInt32, reused: Int,
                  prepare: ([any KVCache]) -> [any KVCache]) -> ModelProbeObservation {
      let caches = prepare(loaded.model.newCache(parameters: parameters))
      let logits = loaded.model(MLXArray(suffix)[.newAxis], cache: caches)
      return observation(scenario, operation: operation, reused: reused,
        logits: logits, caches: caches)
    }

    func prefixPrepared(_ caches: [any KVCache]) -> [any KVCache] {
      let logits = loaded.model(MLXArray(prefix)[.newAxis], cache: caches)
      eval(logits)
      return caches
    }
    func copiedPrefix(_ caches: [any KVCache]) -> [any KVCache] {
      prefixPrepared(caches).map { $0.copy() }
    }

    let coldCaches = loaded.model.newCache(parameters: parameters)
    let coldLogits = loaded.model(MLXArray(tokens)[.newAxis], cache: coldCaches)
    let cold = observation("cold", operation: 0, reused: 0,
      logits: coldLogits, caches: coldCaches)
    let scenarios = [
      evaluate("continuation", operation: 1, reused: split, prepare: prefixPrepared),
      evaluate("branch", operation: 2, reused: split, prepare: copiedPrefix),
      evaluate("anchor", operation: 3, reused: split, prepare: copiedPrefix),
      evaluate("namespace", operation: 0, reused: 0, prepare: prefixPrepared),
      evaluate("reset", operation: 0, reused: 0, prepare: prefixPrepared),
      evaluate("cancellation", operation: 1, reused: split, prepare: copiedPrefix),
      evaluate("checkpoint", operation: 3, reused: split, prepare: copiedPrefix),
    ]
    for observation in scenarios {
      #expect(observation.selectedNextToken == cold.selectedNextToken)
      #expect(observation.top16QuantizedLogitSHA256 == cold.top16QuantizedLogitSHA256)
      #expect(observation.physicalStateSHA256 == cold.physicalStateSHA256)
    }
    if let encoded = ProcessInfo.processInfo.environment["CLAP_CACHE_TEST_EXPECTED_PROBE"] {
      let expected = try JSONDecoder().decode(ExpectedProbe.self, from: Data(encoded.utf8))
      #expect(expected.scenarios == ([cold] + scenarios).map(\.scenario))
      #expect(expected.logicalTokenSha256 == cold.logicalTokenSHA256)
      #expect(expected.physicalStateSha256 == cold.physicalStateSHA256)
      #expect(expected.selectedNextToken == cold.selectedNextToken)
      #expect(expected.top16QuantizedLogitSha256 == cold.top16QuantizedLogitSHA256)
    }
    print(try ([cold] + scenarios).map { try $0.canonicalJSON() }.joined(separator: "\n"))
  }
}

private struct ExpectedProbe: Decodable {
  let scenarios: [String]
  let logicalTokenSha256: String
  let physicalStateSha256: String
  let selectedNextToken: Int32
  let top16QuantizedLogitSha256: String
}
