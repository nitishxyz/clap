import ClapCacheBridge
import ClapCachePolicy
import Testing
@testable import ClapMLXCache

@Suite("MLX cache lifecycle")
struct AnchorManagerTests {
  @Test("retained anchor materializes once and exact repeat is a no-op")
  func anchorNoOp() throws {
    let (coordinator, initial): (CacheCoordinator, RetainedRegistry<CacheSlot<FakeLifecycleCache>>) =
      try initialized()
    var registry = initial
    var counter: UInt64 = 0
    var copies = 0
    let tokens = Array(1...64)
    let identity = makeIdentity()
    let operations = makeOperations(copy: {
      copies += 1
      return FakeLifecycleCache(offset: $0.offset)
    })
    let first = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, boundary: tokens, sourceCaches: [FakeLifecycleCache(offset: 64)],
      sourceFedTokens: tokens, identity: identity, scope: identity.scope,
      structural: true, useCounter: &counter, operations: operations)
    let second = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, boundary: tokens, sourceCaches: [FakeLifecycleCache(offset: 64)],
      sourceFedTokens: tokens, identity: identity, scope: identity.scope,
      structural: true, useCounter: &counter, operations: operations)
    #expect(first.materialized)
    #expect(second.materialized)
    #expect(copies == 1)
  }

  @Test("anchor and request snapshot copy failures publish nothing")
  func copyFailures() throws {
    let (coordinator, initial): (CacheCoordinator, RetainedRegistry<CacheSlot<FakeLifecycleCache>>) =
      try initialized()
    var registry = initial
    var counter: UInt64 = 0
    let tokens = Array(1...64)
    let identity = makeIdentity()
    let operations = makeOperations(copy: { _ in throw LifecycleFailure.copy })
    let anchor = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, boundary: tokens, sourceCaches: [FakeLifecycleCache(offset: 64)],
      sourceFedTokens: tokens, identity: identity, scope: identity.scope,
      structural: false, useCounter: &counter, operations: operations)
    let snapshots = CacheSnapshots<FakeLifecycleCache>()
    let snapshotMs = AnchorManager.capturePromptBoundary(snapshots: snapshots,
      promptTokens: tokens, caches: [FakeLifecycleCache(offset: 64)], fedTokens: tokens,
      operations: operations)
    #expect(!anchor.materialized)
    #expect(snapshotMs == 0)
    #expect(snapshots.promptBoundary == nil)
    #expect(registry.slotIDs.compactMap { registry.entry(for: $0) }
      .filter { $0.isAnchor }.allSatisfy { $0.caches.isEmpty })
  }

  @Test("finalize restores exact continuation boundary and releases activation")
  func exactBoundaryFinalize() throws {
    let (coordinator, initial): (CacheCoordinator, RetainedRegistry<CacheSlot<FakeLifecycleCache>>) =
      try initialized()
    var registry = initial
    var counter: UInt64 = 0
    let identity = makeIdentity()
    let prompt = Array(1...8)
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, promptTokens: prompt, identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false, useCounter: &counter,
      operations: makeOperations())
    let snapshots = CacheSnapshots<FakeLifecycleCache>()
    snapshots.continuationBoundary = 4
    snapshots.continuation = [FakeLifecycleCache(offset: 4)]
    var caches = [FakeLifecycleCache(offset: 10)]
    CacheExecutor.finalize(coordinator: coordinator, registry: registry,
      slotIndex: admission.slotIndex, slot: admission.slot, caches: &caches,
      snapshots: snapshots, promptTokens: prompt, fedTokens: prompt,
      sampledTokens: [9, 10], generatedCount: 2, failed: false,
      operations: makeOperations())
    #expect(admission.slot.tokens == Array(prompt.prefix(4)))
    #expect(admission.slot.isPromptBoundary)
    #expect(caches.first?.offset == 4)
    #expect(!registry.isActive(slotID: UInt32(admission.slotIndex)))
  }

  @Test("advance and confirm failures invalidate generation and release")
  func metadataFailures() throws {
    let (coordinator, initial): (CacheCoordinator, RetainedRegistry<CacheSlot<FakeLifecycleCache>>) =
      try initialized()
    var registry = initial
    var counter: UInt64 = 0
    let identity = makeIdentity()
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, promptTokens: [1, 2], identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false, useCounter: &counter,
      operations: makeOperations())
    _ = try coordinator.invalidate(slot: admission.slotIndex,
      generation: admission.slot.coordinatorGeneration)
    var fed = admission.fedTokens
    CacheExecutor.appendAndAdvance(coordinator: coordinator, slotIndex: admission.slotIndex,
      slot: admission.slot, caches: admission.caches, fedTokens: &fed, tokens: [1],
      operations: makeOperations())
    #expect(fed == [1])
    #expect(admission.slot.coordinatorGeneration == 0)
    var caches = admission.caches
    CacheExecutor.finalize(coordinator: coordinator, registry: registry,
      slotIndex: admission.slotIndex, slot: admission.slot, caches: &caches,
      snapshots: CacheSnapshots(), promptTokens: [1, 2], fedTokens: fed,
      sampledTokens: [], generatedCount: 0, failed: false, operations: makeOperations())
    #expect(!registry.isActive(slotID: UInt32(admission.slotIndex)))
    #expect(admission.slot.coordinatorGeneration == 0)
  }

  @Test("full reset destroys coordinator registry and use counter")
  func reset() throws {
    var (coordinatorValue, registry):
      (CacheCoordinator, RetainedRegistry<CacheSlot<FakeLifecycleCache>>) = try initialized()
    var coordinator: CacheCoordinator? = coordinatorValue
    var counter: UInt64 = 12
    CacheExecutor.reset(coordinator: &coordinator, registry: &registry,
      maxActive: 2, hardCeiling: 3, useCounter: &counter)
    #expect(coordinator == nil)
    #expect(registry.count == 0)
    #expect(registry.activeCount == 0)
    #expect(counter == 0)
    _ = coordinatorValue
  }

  private func initialized<Cache>() throws
    -> (CacheCoordinator, RetainedRegistry<CacheSlot<Cache>>) {
    try CacheExecutor.initialize(retention: RetentionConfiguration(initialEntries: 1,
      hardCeiling: 3, physicalByteBudget: 1_000_000,
      highWatermarkBytes: 900_000, lowWatermarkBytes: 700_000), maxActive: 2,
      capacity: 1024, checkpoints: CoordinatorCheckpointConfiguration(enabled: false,
        minimumTokens: 2_048, intervalTokens: 2_048, maximum: 8,
        budgetBasisPoints: 2_500, budgetBytes: 0))
  }

  private func makeIdentity() -> CacheIdentity {
    CacheIdentity(domain: "model", input: CacheIdentityInput(namespace: "tenant",
      tenant: nil, project: "project", harness: nil, agent: nil, session: "session",
      priority: nil, sideRequest: false), telemetryKey: "test")
  }

  private func makeOperations(
    copy: @escaping (FakeLifecycleCache) throws -> FakeLifecycleCache = {
      FakeLifecycleCache(offset: $0.offset)
    }) -> CacheOperations<FakeLifecycleCache> {
    CacheOperations(isTrimmable: { _ in true }, copy: copy,
      trim: { $0.offset -= $1 }, sequenceLength: { caches, fallback in
        let offset = caches.map(\.offset).max() ?? 0
        return offset > 0 ? offset : fallback
      }, create: { [FakeLifecycleCache(offset: 0)] },
      physicalBytes: { UInt64(max(1, $0.count * 64)) })
  }
}

private enum LifecycleFailure: Error { case copy }
private final class FakeLifecycleCache {
  var offset: Int
  init(offset: Int) { self.offset = offset }
}
