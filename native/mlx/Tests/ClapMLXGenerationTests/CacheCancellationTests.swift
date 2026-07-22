import ClapCachePolicy
import ClapMLXCache
import Testing
@testable import ClapMLXGeneration

@Suite("MLX cache cancellation boundaries")
struct CacheCancellationTests {
  @Test("cancellation before prefill releases exactly once without publishing tokens")
  func beforePrefill() {
    let fixture = Fixture()
    fixture.request.cancelled = true
    let completion = fixture.finalize()
    #expect(completion?.status == .cancelled)
    #expect(completion?.outputs.isEmpty == true)
    #expect(fixture.request.fedTokens.isEmpty)
    #expect(fixture.finalizedBoundaries == [[]])
    #expect(fixture.finalize() == nil)
  }

  @Test("cancellation after one prefill chunk retains only the cold-equivalent prefix")
  func afterPrefillChunk() {
    let fixture = Fixture()
    let events = GenerationStepper.step(fixture.request, prefillQuantum: 2,
      decodeLimit: 1, eosTokenIds: [], backend: fixture.backend)
    #expect(events == [.prefill(done: 2, total: 4)])
    #expect(fixture.request.fedTokens == [1, 2])
    fixture.request.cancelled = true
    let completion = fixture.finalize()
    #expect(completion?.status == .cancelled)
    #expect(fixture.finalizedBoundaries == [[1, 2]])
    #expect(fixture.finalize() == nil)
  }

  @Test("cancellation after first decode suppresses held text and unconfirmed token")
  func afterFirstDecode() {
    let fixture = Fixture()
    _ = GenerationStepper.step(fixture.request, prefillQuantum: 8,
      decodeLimit: 1, eosTokenIds: [], backend: fixture.backend)
    #expect(fixture.request.fedTokens == [1, 2, 3, 4])
    let events = GenerationStepper.step(fixture.request, prefillQuantum: 8,
      decodeLimit: 1, eosTokenIds: [], backend: fixture.backend)
    #expect(events.isEmpty)
    #expect(fixture.request.sampledTokens == [70])
    #expect(fixture.request.collected == "hel")
    #expect(fixture.request.emitted == 0)
    #expect(!fixture.request.fedTokens.contains(70))

    fixture.request.cancelled = true
    let completion = fixture.finalize()
    #expect(completion?.status == .cancelled)
    #expect(completion?.outputs.isEmpty == true)
    #expect(fixture.finalizedBoundaries == [[1, 2, 3, 4]])
    #expect(fixture.finalize() == nil)
  }
}

private final class Fixture {
  let request: ActiveRequest<CancellationCache, CancellationIterator,
    CancellationDetokenizer, CancellationParameters>
  let registry = RetainedRegistry<CacheSlot<CancellationCache>>(maxActive: 1, hardCeiling: 1)
  var finalizedBoundaries: [[Int]] = []
  var now: UInt64 = 20

  init() {
    let identity = testCacheIdentity()
    let prepared = PreparedRequest(id: "cancel", admissionOrder: 1, admittedNs: 10,
      receivedToAdmittedMs: 0, templateTokenizeMs: 0, coordinatorPlanMs: 0,
      coordinatorApplyMs: 0, cacheMaterializeMs: 0, streaming: true, maxTokens: 8,
      promptTokens: [1, 2, 3, 4], reusedTokens: 0, reuseKind: nil, reuseScope: nil,
      cacheIdentity: identity, cacheDecision: nil, cacheCandidates: [], cacheEvictions: [],
      cacheFallback: nil, parameters: CancellationParameters(), stops: ["hello"],
      anchorPlantAt: [], anchorPlantScopes: [:], resolvedBoundaries: [:],
      boundaryTelemetry: [], automaticCheckpointProposed: 0,
      automaticCheckpointDeduped: 0)
    let slot = CacheSlot<CancellationCache>(caches: [CancellationCache()], tokens: [])
    try! registry.register(slotID: 0, entry: slot)
    try! registry.activate(slotID: 0)
    slot.busy = true
    request = ActiveRequest(prepared: prepared,
      cache: GenerationCacheContext(slotIndex: 0, slot: slot, caches: slot.caches),
      fedTokens: [], suffix: [1, 2, 3, 4], detokenizer: CancellationDetokenizer())
  }

  var backend: GenerationBackend<CancellationCache, CancellationIterator,
    CancellationDetokenizer, CancellationParameters> {
    GenerationBackend(prefill: { tokens, caches, _ in
      for cache in caches { cache.resident += tokens.count }
      return CancellationIterator(tokens: [70])
    }, nextToken: { iterator in
      guard !iterator.tokens.isEmpty else { return nil }
      return iterator.tokens.removeFirst()
    }, appendToken: { detokenizer, token in
      detokenizer.pending = token == 70 ? "hel" : String(token)
    }, nextText: { detokenizer in
      defer { detokenizer.pending = nil }
      return detokenizer.pending
    }, appendAndAdvance: { _, slot, caches, fed, tokens in
      fed.append(contentsOf: tokens)
      slot.tokens = fed
      slot.caches = caches
    }, plantAnchor: { _, _, _, _, _, _ in
      AnchorResult(materialized: false, evictedVictims: false, materializeMs: 0)
    }, captureContinuation: { _, _, _, _ in 0 },
      capturePromptBoundary: { _, _, _, _ in 0 }, now: { [weak self] in
        guard let self else { return 0 }
        now += 1
        return now
      })
  }

  func finalize() -> RequestCompletion? {
    guard case .completion(let completion)? = request.finalize(using:
      GenerationFinalizer { [weak self] slotIndex, slot, caches, snapshots,
        prompt, fed, sampled, generated, failed in
        guard let self else { return }
        CacheExecutor.finalize(coordinator: nil, registry: registry,
          slotIndex: slotIndex, slot: slot, caches: &caches, snapshots: snapshots,
          promptTokens: prompt, fedTokens: fed, sampledTokens: sampled,
          generatedCount: generated, failed: failed,
          operations: CacheOperations(isTrimmable: { _ in true },
            copy: { CancellationCache(resident: $0.resident) },
            trim: { $0.resident -= $1 },
            sequenceLength: { caches, fallback in
              caches.map(\.resident).max() ?? fallback
            }, create: { [CancellationCache()] },
            physicalBytes: { UInt64(max(1, $0.count * 64)) }))
        finalizedBoundaries.append(slot.tokens)
      }) else { return nil }
    return completion
  }
}

private final class CancellationCache {
  var resident: Int
  init(resident: Int = 0) { self.resident = resident }
}
private struct CancellationIterator { var tokens: [Int] }
private struct CancellationDetokenizer { var pending: String? }
private struct CancellationParameters {}
