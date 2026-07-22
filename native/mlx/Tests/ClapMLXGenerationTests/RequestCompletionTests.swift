import ClapMLXCache
import Testing
@testable import ClapMLXGeneration

@Suite("Generation request completion")
struct RequestCompletionTests {
  @Test("streaming success flushes held tail before completion facts")
  func streamFlush() {
    let fixture = Fixture(streaming: true)
    fixture.request.collected = "hello"
    fixture.request.emitted = 2
    fixture.request.generatedCount = 3
    fixture.request.completed = true
    guard case .completion(let completion)? = fixture.finalize() else {
      Issue.record("expected completion")
      return
    }
    #expect(completion.outputs == [.token("llo")])
    #expect(completion.finishReason == "stop")
    #expect(completion.usage == CompletionUsageFacts(promptTokens: 3,
      completionTokens: 3))
    #expect(fixture.request.emitted == 5)
    #expect(fixture.finalize() == nil)
  }

  @Test("nonstream success emits content before completion")
  func nonstreamContent() {
    let fixture = Fixture(streaming: false)
    fixture.request.collected = "complete"
    fixture.request.completed = true
    guard case .completion(let completion)? = fixture.finalize() else {
      Issue.record("expected completion")
      return
    }
    #expect(completion.outputs == [.content("complete")])
    #expect(completion.content == "complete")
  }

  @Test("cancellation suppresses held and nonstream content")
  func cancellationSuppressesContent() {
    for streaming in [true, false] {
      let fixture = Fixture(streaming: streaming)
      fixture.request.collected = "secret"
      fixture.request.emitted = 1
      fixture.request.cancelled = true
      guard case .completion(let completion)? = fixture.finalize() else {
        Issue.record("expected cancelled completion")
        continue
      }
      #expect(completion.outputs.isEmpty)
      #expect(completion.status == .cancelled)
      #expect(completion.finishReason == "cancel")
    }
  }

  @Test("failure finalizes cache but produces no normal completion")
  func failure() {
    let fixture = Fixture(streaming: true)
    fixture.request.collected = "partial"
    fixture.request.failed = true
    guard case .failure(let failure)? = fixture.finalize() else {
      Issue.record("expected failure")
      return
    }
    #expect(failure.status == .failed)
    #expect(fixture.finalizedFailures == [true])
  }

  @Test("active finalization transitions once and preserves telemetry facts")
  func factsAndTransition() {
    let fixture = Fixture(streaming: true)
    fixture.request.schedulerWaitMs = 1
    fixture.request.cacheMaterializeMs = 2
    fixture.request.prefillMs = 3
    fixture.request.prefillTokens = 2
    fixture.request.prefillChunks = 1
    fixture.request.firstDecodeMs = 4
    fixture.request.firstEmitMs = 5
    fixture.request.materializedAnchors = [2]
    guard case .completion(let completion)? = fixture.finalize() else {
      Issue.record("expected completion")
      return
    }
    #expect(fixture.request.status == .completed)
    #expect(completion.timing.schedulerWaitMs == 1)
    #expect(completion.timing.cacheMaterializeMs == 2)
    #expect(completion.timing.prefillMs == 3)
    #expect(completion.cache.promptTokens == [1, 2, 3])
    #expect(completion.cache.materializedAnchors == [2])
    #expect(fixture.finalizedFailures == [false])
  }

  @Test("snapshot priority facts reach cache finalizer unchanged")
  func cacheSelectionInputs() {
    let fixture = Fixture(streaming: true)
    fixture.request.cacheSnapshots.continuationBoundary = 2
    fixture.request.cacheSnapshots.continuation = [CompletionCache(2)]
    fixture.request.cacheSnapshots.promptBoundary = [CompletionCache(3)]
    fixture.request.completed = true
    _ = fixture.finalize()
    #expect(fixture.snapshotContinuationCounts == [1])
    #expect(fixture.snapshotPromptCounts == [1])
  }
}

private final class Fixture {
  let request: ActiveRequest<CompletionCache, CompletionIterator,
    CompletionDetokenizer, CompletionParameters>
  var finalizedFailures: [Bool] = []
  var snapshotContinuationCounts: [Int] = []
  var snapshotPromptCounts: [Int] = []

  init(streaming: Bool) {
    let identity = CacheIdentity(domain: "model", input: CacheIdentityInput(
      namespace: "tenant", tenant: nil, project: nil, harness: nil, agent: nil,
      session: "session", priority: nil, sideRequest: false), telemetryKey: "test")
    let boundary = BoundaryInfo(tokenCount: 2, kind: "automatic_token", label: nil,
      requested: false, status: "authorized", skipReason: nil)
    let prepared = PreparedRequest(id: "id", admissionOrder: 1, admittedNs: 10,
      receivedToAdmittedMs: 0.1, templateTokenizeMs: 0.2,
      coordinatorPlanMs: 0.3, coordinatorApplyMs: 0.4,
      cacheMaterializeMs: 0.5, streaming: streaming, maxTokens: 8,
      promptTokens: [1, 2, 3], reusedTokens: 1, reuseKind: "continue",
      reuseScope: "session", cacheIdentity: identity, cacheDecision: nil,
      cacheCandidates: [], cacheEvictions: [4], cacheFallback: nil,
      parameters: CompletionParameters(), stops: ["stop"], anchorPlantAt: [2],
      anchorPlantScopes: [2: identity.scope], resolvedBoundaries: [2: boundary],
      boundaryTelemetry: [boundary], automaticCheckpointProposed: 1,
      automaticCheckpointDeduped: 0)
    let slot = CacheSlot<CompletionCache>(caches: [CompletionCache(1)], tokens: [1])
    request = ActiveRequest(prepared: prepared,
      cache: GenerationCacheContext(slotIndex: 0, slot: slot, caches: slot.caches),
      fedTokens: [1], suffix: [2, 3], detokenizer: CompletionDetokenizer())
  }

  func finalize() -> RequestFinalization? {
    request.finalize(using: GenerationFinalizer { [weak self] _, _, _, snapshots,
      _, _, _, _, failed in
      self?.finalizedFailures.append(failed)
      self?.snapshotContinuationCounts.append(snapshots.continuation?.count ?? 0)
      self?.snapshotPromptCounts.append(snapshots.promptBoundary?.count ?? 0)
    })
  }
}

private final class CompletionCache {
  let value: Int
  init(_ value: Int) { self.value = value }
}
private struct CompletionIterator {}
private struct CompletionDetokenizer {}
private struct CompletionParameters {}
