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
