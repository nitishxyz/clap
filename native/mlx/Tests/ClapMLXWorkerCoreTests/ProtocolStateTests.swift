import Foundation
import Testing
@testable import ClapMLXWorkerCore

@Suite("Worker protocol sequence and terminal state")
struct ProtocolStateTests {
  @Test("shared v1 request fixtures decode into strict envelopes")
  func sharedRequestFixtures() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let root = testFile.deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let fixture = root.appendingPathComponent(
      "packages/worker-protocol/fixtures/v1/requests/all.jsonl")
    let lines = try String(contentsOf: fixture, encoding: .utf8).split(separator: "\n")
    let requests = try lines.map { try decodeV1Envelope(String($0)) }
    #expect(requests.map(\.type) == ["load", "generate", "cancel", "set_max_active",
      "unload", "shutdown"])
    #expect(requests[2].requestID == "req_cancel")
    #expect(requests[2].targetRequestID == "req_generate")

    let generated = try JSONSerialization.jsonObject(with: requests[1].legacyPayload)
      as? [String: Any]
    #expect(generated?["id"] as? String == "req_generate")
    #expect(generated?["type"] as? String == "chat")
    let messages = generated?["messages"] as? [[String: Any]]
    #expect(messages?.first?["content"] as? String == "Hello")
    #expect(generated?["protocol"] == nil)
    #expect(requests[1].structuredOutput?.kind == "json_schema")
    #expect(requests[1].structuredOutput?.strength == "required")
    #expect(requests[1].structuredOutput?.schemaJSON != nil)
  }

  @Test("canonical generation uses the resident model when its envelope omits model")
  func residentGenerateModel() {
    #expect(resolveGenerateModel(requestModel: nil, residentModel: "resident") == "resident")
    #expect(resolveGenerateModel(requestModel: "legacy", residentModel: "resident") == "legacy")
    #expect(resolveGenerateModel(requestModel: nil, residentModel: nil) == nil)
  }

  @Test("structured output validates typed contracts and schema bounds")
  func structuredOutputContracts() throws {
    let identity = try fixtureIdentity()
    let bestEffort = try generateEnvelope(identity: identity, structuredOutput: [
      "kind": "json_object", "strength": "best_effort",
    ])
    #expect(bestEffort.structuredOutput == V1StructuredOutput(
      kind: "json_object", strength: "best_effort", schemaJSON: nil))

    let localReference = try generateEnvelope(identity: identity, structuredOutput: [
      "kind": "json_schema", "strength": "best_effort",
      "schema": ["$defs": ["item": ["type": "string"]],
        "type": "array", "items": ["$ref": "#/$defs/item"]],
    ])
    #expect(localReference.structuredOutput?.schemaJSON != nil)

    let malformed: [Any] = [
      "bad",
      ["kind": "grammar", "strength": "best_effort"],
      ["kind": "json_object", "strength": "strict"],
      ["kind": "json_schema", "strength": "best_effort"],
      ["kind": "json_object", "strength": "best_effort", "schema": [:]],
      ["kind": "json_object", "strength": "best_effort", "extra": true],
      ["kind": "json_schema", "strength": "best_effort",
        "schema": ["$ref": "https://example.com/schema.json"]],
    ]
    for contract in malformed {
      do {
        _ = try generateEnvelope(identity: identity, structuredOutput: contract)
        Issue.record("invalid structured output decoded: \(contract)")
      } catch let error as V1EnvelopeDecodeError {
        #expect(error.code == "invalid_structured_output")
      }
    }

    do {
      _ = try generateEnvelope(identity: identity, structuredOutput: [
        "kind": "json_schema", "strength": "best_effort",
        "schema": ["description": String(repeating: "x", count: 65 * 1024)],
      ])
      Issue.record("oversized schema decoded")
    } catch let error as V1EnvelopeDecodeError {
      #expect(error.code == "invalid_structured_output")
      #expect(error.description.contains("64 KiB"))
    }

    var nested: Any = ["type": "string"]
    for _ in 0..<33 { nested = [nested] }
    do {
      _ = try generateEnvelope(identity: identity, structuredOutput: [
        "kind": "json_schema", "strength": "best_effort", "schema": ["allOf": nested],
      ])
      Issue.record("overly deep schema decoded")
    } catch let error as V1EnvelopeDecodeError {
      #expect(error.code == "invalid_structured_output")
      #expect(error.description.contains("depth"))
    }

    let properties = Dictionary(uniqueKeysWithValues: (0..<1025).map {
      ("p\($0)", ["type": "string"])
    })
    do {
      _ = try generateEnvelope(identity: identity, structuredOutput: [
        "kind": "json_schema", "strength": "best_effort",
        "schema": ["type": "object", "properties": properties],
      ])
      Issue.record("schema with too many properties decoded")
    } catch let error as V1EnvelopeDecodeError {
      #expect(error.code == "invalid_structured_output")
      #expect(error.description.contains("1024 properties"))
    }
  }

  private func fixtureIdentity() throws -> [String: Any] {
    let testFile = URL(fileURLWithPath: #filePath)
    let root = testFile.deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let fixture = root.appendingPathComponent(
      "packages/worker-protocol/fixtures/v1/cache-identity-vector.json")
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as! [String: Any]
    return object["identity"] as! [String: Any]
  }

  private func generateEnvelope(identity: [String: Any], structuredOutput: Any) throws
      -> V1RequestEnvelope {
    let object: [String: Any] = [
      "protocol": 1, "type": "generate", "request_id": "structured",
      "prompt": "hello", "cache_identity": identity,
      "structured_output": structuredOutput,
    ]
    let data = try JSONSerialization.data(withJSONObject: object)
    return try decodeV1Envelope(String(decoding: data, as: UTF8.self))
  }

  @Test("malformed, invalid version, cancel, and shutdown envelopes are classified")
  func envelopeFailuresAndControls() throws {
    do {
      _ = try decodeV1Envelope("{bad")
      Issue.record("malformed JSON decoded")
    } catch let error as V1EnvelopeDecodeError {
      #expect(error.code == "malformed_json")
      #expect(error.requestID == nil)
    }
    do {
      _ = try decodeV1Envelope(
        #"{"protocol":2,"type":"shutdown","request_id":"bad-version"}"#)
      Issue.record("unsupported version decoded")
    } catch let error as V1EnvelopeDecodeError {
      #expect(error.code == "unsupported_protocol_version")
      #expect(error.requestID == "bad-version")
    }
    let cancel = try decodeV1Envelope(
      #"{"protocol":1,"type":"cancel","request_id":"cancel","target_request_id":"target"}"#)
    #expect(cancel.requestID == "cancel")
    #expect(cancel.targetRequestID == "target")
    let shutdown = try decodeV1Envelope(
      #"{"protocol":1,"type":"shutdown","request_id":"shutdown"}"#)
    #expect(shutdown.type == "shutdown")
  }

  @Test("accepted starts at zero and events increase monotonically")
  func sequencing() {
    let tracker = ProtocolTerminalTracker()
    #expect(tracker.accept("request") == ProtocolSequence(requestID: "request", sequence: 0))
    #expect(tracker.event("request")?.sequence == 1)
    #expect(tracker.event("request")?.sequence == 2)
    #expect(tracker.terminal("request")?.sequence == 3)
  }

  @Test("accepted and terminal are emitted exactly once")
  func exactlyOnce() {
    let tracker = ProtocolTerminalTracker()
    #expect(tracker.accept("request") != nil)
    #expect(tracker.accept("request") == nil)
    #expect(tracker.terminal("request") != nil)
    #expect(tracker.terminal("request") == nil)
    #expect(tracker.event("request") == nil)
    #expect(tracker.isTerminal("request"))
  }

  @Test("unaccepted and empty requests cannot emit scoped events")
  func requiresAcceptance() {
    let tracker = ProtocolTerminalTracker()
    #expect(tracker.event("missing") == nil)
    #expect(tracker.terminal("missing") == nil)
    #expect(tracker.accept("") == nil)
  }
}
