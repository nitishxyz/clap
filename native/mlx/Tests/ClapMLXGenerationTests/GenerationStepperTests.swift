import ClapMLXCache
import Testing
@testable import ClapMLXGeneration

@Suite("Bounded generation stepper")
struct GenerationStepperTests {
  @Test("prefill consumes one bounded chunk split at nearest anchor")
  func boundedPrefill() {
    let fixture = Fixture(tokens: [], prompt: Array(1...8), reused: 0,
      anchors: [3, 6])
    let events = fixture.step(prefillQuantum: 5)
    #expect(events == [.prefill(done: 3, total: 8)])
    #expect(fixture.request.pos == 3)
    #expect(fixture.request.fedTokens == [1, 2, 3])
    #expect(fixture.request.prefillChunks == 1)
    #expect(fixture.plants == [3])
  }

  @Test("continuation boundary wins when nearer and captures exact snapshot")
  func continuationSplit() {
    let fixture = Fixture(tokens: [], prompt: Array(1...8), reused: 0,
      anchors: [6])
    fixture.request.continuationBoundary = 4
    _ = fixture.step(prefillQuantum: 8)
    #expect(fixture.request.pos == 4)
    #expect(fixture.continuationCaptures == [4])
  }

  @Test("final prefill persists iterator but decodes nothing in same turn")
  func finalPrefill() {
    let fixture = Fixture(tokens: [9, 10], prompt: [1, 2])
    let events = fixture.step(prefillQuantum: 8)
    #expect(events.isEmpty)
    #expect(fixture.request.iterator != nil)
    #expect(fixture.request.generatedCount == 0)
    #expect(fixture.request.sampledTokens.isEmpty)
    #expect(fixture.promptCaptures == 1)
  }

  @Test("decode is bounded and iterator persists across turns")
  func boundedDecode() {
    let fixture = Fixture(tokens: Array(10...20), prompt: [1])
    _ = fixture.step(prefillQuantum: 8)
    let first = fixture.step(decodeLimit: 6)
    let second = fixture.step(decodeLimit: 6)
    #expect(first.count == 6)
    #expect(second.count == 5)
    #expect(fixture.request.generatedCount == 11)
    #expect(fixture.request.sampledTokens == Array(10...20))
  }

  @Test("EOS is sampled but not counted or emitted")
  func eos() {
    let fixture = Fixture(tokens: [10, 99, 11], prompt: [1], eos: [99])
    _ = fixture.step()
    let events = fixture.step()
    #expect(events == [.token("10")])
    #expect(fixture.request.completed)
    #expect(fixture.request.generatedCount == 1)
    #expect(fixture.request.sampledTokens == [10, 99])
  }

  @Test("length stops exactly at max tokens")
  func length() {
    let fixture = Fixture(tokens: [1, 2, 3], prompt: [7], maxTokens: 2)
    _ = fixture.step()
    _ = fixture.step()
    #expect(fixture.request.completed)
    #expect(fixture.request.finishReason == "length")
    #expect(fixture.request.generatedCount == 2)
  }

  @Test("split stop is held back then truncated without leaking")
  func stopHoldback() {
    let fixture = Fixture(tokens: [1, 2, 3], prompt: [7], stops: ["bc"])
    fixture.text = [1: "a", 2: "b", 3: "c"]
    _ = fixture.step()
    let events = fixture.step()
    #expect(events == [.token("a")])
    #expect(fixture.request.completed)
    #expect(fixture.request.collected == "a")
  }

  @Test("backend error emits one error and terminal failure")
  func backendError() {
    let fixture = Fixture(tokens: [1], prompt: [7])
    fixture.failure = TestFailure.backend
    let events = fixture.step()
    #expect(events == [.error("backend")])
    #expect(fixture.request.failed)
    #expect(fixture.step().isEmpty)
  }
}

private enum TestFailure: Error, CustomStringConvertible {
  case backend
  var description: String { "backend" }
}

private final class Fixture {
  let request: ActiveRequest<TestCache, TestIterator, TestDetokenizer, TestParameters>
  var plants: [Int] = []
  var continuationCaptures: [Int] = []
  var promptCaptures = 0
  var text: [Int: String] = [:]
  var failure: TestFailure?
  private var clock: UInt64 = 1_000_000
  private let eos: Set<Int>

  init(tokens: [Int], prompt: [Int], reused: Int = 0, anchors: [Int] = [],
       maxTokens: Int = 100, stops: [String] = [], eos: Set<Int> = []) {
    self.eos = eos
    let identity = CacheIdentity(domain: "model", input: CacheIdentityInput(
      namespace: "tenant", tenant: nil, project: nil, harness: nil, agent: nil,
      session: "session", priority: nil, sideRequest: false), telemetryKey: "test")
    let parameters = TestParameters(tokens: tokens)
    let prepared = PreparedRequest(id: "id", admissionOrder: 1, admittedNs: 0,
      receivedToAdmittedMs: 0, templateTokenizeMs: 0, coordinatorPlanMs: 0,
      coordinatorApplyMs: 0, cacheMaterializeMs: 0, streaming: true,
      maxTokens: maxTokens, promptTokens: prompt, reusedTokens: reused,
      reuseKind: nil, reuseScope: nil, cacheIdentity: identity, cacheDecision: nil,
      cacheCandidates: [], cacheEvictions: [], cacheFallback: nil,
      parameters: parameters, stops: stops, anchorPlantAt: anchors,
      anchorPlantScopes: Dictionary(uniqueKeysWithValues: anchors.map { ($0, identity.scope) }),
      resolvedBoundaries: Dictionary(uniqueKeysWithValues: anchors.map {
        ($0, BoundaryInfo(tokenCount: $0, kind: "message", label: nil,
          requested: true, status: "authorized", skipReason: nil))
      }))
    let slot = CacheSlot<TestCache>()
    request = ActiveRequest(prepared: prepared,
      cache: GenerationCacheContext(slotIndex: 0, slot: slot, caches: [TestCache()]),
      fedTokens: Array(prompt.prefix(reused)), suffix: Array(prompt.dropFirst(reused)),
      detokenizer: TestDetokenizer())
  }

  func step(prefillQuantum: Int = 8, decodeLimit: Int = 6) -> [GenerationEvent] {
    GenerationStepper.step(request, prefillQuantum: prefillQuantum,
      decodeLimit: decodeLimit, eosTokenIds: eos, backend: backend())
  }

  private func backend() -> GenerationBackend<TestCache, TestIterator,
                                                TestDetokenizer, TestParameters> {
    GenerationBackend(prefill: { [weak self] _, _, parameters in
      if let failure = self?.failure { throw failure }
      return TestIterator(tokens: parameters.tokens)
    }, nextToken: { [weak self] iterator in
      if let failure = self?.failure { throw failure }
      guard iterator.index < iterator.tokens.count else { return nil }
      defer { iterator.index += 1 }
      return iterator.tokens[iterator.index]
    }, appendToken: { detokenizer, token in
      detokenizer.pending.append(token)
    }, nextText: { [weak self] detokenizer in
      guard !detokenizer.pending.isEmpty else { return nil }
      let token = detokenizer.pending.removeFirst()
      return self?.text[token] ?? String(token)
    }, appendAndAdvance: { _, slot, _, fed, tokens in
      fed.append(contentsOf: tokens)
      slot.tokens = fed
    }, plantAnchor: { [weak self] plant, _, _, _, _, _ in
      self?.plants.append(plant)
      return AnchorResult(materialized: true, evictedVictims: false, materializeMs: 1)
    }, captureContinuation: { [weak self] snapshots, boundary, caches, _ in
      self?.continuationCaptures.append(boundary)
      snapshots.continuationBoundary = boundary
      snapshots.continuation = caches
      return 1
    }, capturePromptBoundary: { [weak self] snapshots, _, caches, _ in
      self?.promptCaptures += 1
      snapshots.promptBoundary = caches
      return 1
    }, now: { [weak self] in
      self?.clock += 1_000_000
      return self?.clock ?? 0
    })
  }
}

private final class TestCache {}
private struct TestIterator {
  let tokens: [Int]
  var index = 0
}
private struct TestDetokenizer { var pending: [Int] = [] }
private struct TestParameters { let tokens: [Int] }
