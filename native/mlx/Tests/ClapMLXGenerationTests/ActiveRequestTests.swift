import Testing
import ClapMLXCache
@testable import ClapMLXGeneration

@Suite("Generation request state")
struct ActiveRequestTests {
  @Test("prepared request keeps immutable admission facts and holdback")
  func immutableFacts() {
    let prepared = makePrepared(stops: ["stop", "longer"])
    #expect(prepared.id == "request")
    #expect(prepared.promptTokens == [1, 2, 3])
    #expect(prepared.reusedTokens == 2)
    #expect(prepared.cacheEvictions == [3])
    #expect(prepared.holdback == 5)
    #expect(prepared.parameters.value == 42)
    #expect(prepared.anchorPlantAt == [3])
    #expect(prepared.resolvedBoundaries[3]?.kind == "message")
  }

  @Test("active state initializes counters cache and timing")
  func initialization() {
    let request = makeRequest()
    #expect(request.status == .active)
    #expect(request.pos == 0)
    #expect(request.sampledTokens.isEmpty)
    #expect(request.generatedCount == 0)
    #expect(request.cacheMaterializeMs == 4)
    #expect(request.lastStepFinishedNs == 100)
    #expect(request.caches.first?.value == 7)
    #expect(request.fedTokens == [1, 2])
    #expect(request.suffix == [3])
  }

  @Test("active request and cache context preserve reference semantics")
  func referenceSemantics() {
    let request = makeRequest()
    let moved = request
    moved.generatedCount = 2
    moved.caches[0].value = 9
    #expect(request.generatedCount == 2)
    #expect(request.caches[0].value == 9)
    #expect(request.cache === moved.cache)
  }

  @Test("terminal status transitions exactly once")
  func terminalStatus() {
    let request = makeRequest()
    #expect(request.transition(to: .completed))
    #expect(!request.transition(to: .cancelled))
    request.failed = true
    request.cancelled = true
    #expect(request.status == .completed)
    #expect(request.completed)
    #expect(!request.failed)
    #expect(!request.cancelled)
  }

  @Test("mutable counters and snapshots remain request-local")
  func countersAndSnapshots() {
    let first = makeRequest()
    let second = makeRequest()
    first.prefillTokens = 3
    first.prefillChunks = 2
    first.anchorPlanted.insert(3)
    first.cacheSnapshots.continuationBoundary = 2
    first.cacheSnapshots.continuation = [TestCache(11)]
    #expect(first.prefillTokens == 3)
    #expect(first.anchorPlanted == [3])
    #expect(first.cacheSnapshots.continuation?.first?.value == 11)
    #expect(second.prefillTokens == 0)
    #expect(second.anchorPlanted.isEmpty)
    #expect(second.cacheSnapshots.continuation == nil)
  }

  private func makePrepared(stops: [String] = ["end"]) -> PreparedRequest<TestParameters> {
    PreparedRequest(id: "request", admissionOrder: 5, admittedNs: 100,
      receivedToAdmittedMs: 1, templateTokenizeMs: 2, coordinatorPlanMs: 3,
      coordinatorApplyMs: 4, cacheMaterializeMs: 4, streaming: true,
      maxTokens: 10, promptTokens: [1, 2, 3], reusedTokens: 2,
      reuseKind: "continue", reuseScope: "session", cacheIdentity: identity(),
      cacheDecision: nil, cacheCandidates: [], cacheEvictions: [3],
      cacheFallback: nil, parameters: TestParameters(value: 42), stops: stops,
      anchorPlantAt: [3], anchorPlantScopes: [3: 1],
      resolvedBoundaries: [3: BoundaryInfo(tokenCount: 3, kind: "message",
        label: "m", requested: true, status: "materialized", skipReason: nil)],
      boundaryTelemetry: [], automaticCheckpointProposed: 1,
      automaticCheckpointDeduped: 0)
  }

  private func makeRequest() -> ActiveRequest<TestCache, TestIterator, TestDetokenizer, TestParameters> {
    let slot = CacheSlot<TestCache>(caches: [TestCache(7)], tokens: [1, 2])
    return ActiveRequest(prepared: makePrepared(),
      cache: GenerationCacheContext(slotIndex: 1, slot: slot, caches: slot.caches),
      fedTokens: [1, 2], suffix: [3], detokenizer: TestDetokenizer())
  }

  private func identity() -> CacheIdentity {
    CacheIdentity(domain: "model", input: CacheIdentityInput(namespace: "tenant",
      tenant: nil, project: nil, harness: nil, agent: nil, session: "session",
      priority: nil, sideRequest: false), telemetryKey: "test")
  }
}

private final class TestCache {
  var value: Int
  init(_ value: Int) { self.value = value }
}
private struct TestIterator {}
private struct TestDetokenizer {}
private struct TestParameters {
  let value: Int
}
