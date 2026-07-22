import ClapCacheBridge
import ClapCachePolicy
import Testing
@testable import ClapMLXCache

@Suite("MLX automatic checkpoint lifecycle")
struct CheckpointLifecycleTests {
  @Test("automatic proposals obey interval budget maximum and explicit deduplication")
  func proposals() throws {
    let (coordinator, initial) = try initialized(checkpointBudget: 384)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let prompt = Array(1...22)
    let identity = testCacheIdentity()
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 6, promptTokens: prompt, identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [16], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: physical.cacheOperations())
    #expect(admission.anchorBoundaries == [16])
    #expect(Set(admission.anchorBoundaries).count == admission.anchorBoundaries.count)
    #expect((try coordinator.retentionTelemetry()).automatic_checkpoint_byte_budget == 384)

    let (unbudgetedCoordinator, unbudgetedInitial) = try initialized(checkpointBudget: 0)
    var unbudgetedRegistry = unbudgetedInitial
    var unbudgetedCounter: UInt64 = 0
    let intervals = try CacheExecutor.admit(coordinator: unbudgetedCoordinator,
      registry: &unbudgetedRegistry, hardCeiling: 6, promptTokens: prompt,
      identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &unbudgetedCounter, operations: physical.cacheOperations())
    #expect(intervals.anchorBoundaries == [16])
    #expect(intervals.anchorBoundaries.allSatisfy { $0 >= 8 && $0 % 4 == 0 })
    #expect(intervals.anchorBoundaries.count <= 3)
  }

  @Test("exact physical boundary materializes once and reset clears checkpoints")
  func materializeAndReset() throws {
    let (coordinator, initial) = try initialized(checkpointBudget: 0)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let boundary = Array(1...12)
    let identity = testCacheIdentity()
    let source = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 6, promptTokens: boundary, identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: physical.cacheOperations())
    source.caches[0].residentLength = boundary.count
    source.caches[0].storedLength = boundary.count
    source.slot.tokens = boundary
    source.slot.coordinatorGeneration = try coordinator.advance(slot: source.slotIndex,
      generation: source.slot.coordinatorGeneration, tokens: boundary,
      state: UInt32(CC_SLOT_SESSION), busy: false, physicalBytes: 64)
    source.slot.busy = false
    registry.release(slotID: UInt32(source.slotIndex))

    let first = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 6, boundary: boundary, sourceCaches: source.slot.caches,
      sourceFedTokens: boundary, identity: identity, scope: identity.scope,
      structural: false, useCounter: &counter, operations: physical.cacheOperations())
    #expect(first.materialized)
    let anchor = registry.slotIDs.compactMap { registry.entry(for: $0) }
      .first { $0.isAnchor }
    #expect(anchor?.tokens == boundary)
    #expect(anchor?.caches.first?.residentLength == boundary.count)
    let copyCount = physical.operations.filter { if case .copy = $0 { true } else { false } }.count
    let duplicate = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 6, boundary: boundary, sourceCaches: source.slot.caches,
      sourceFedTokens: boundary, identity: identity, scope: identity.scope,
      structural: false, useCounter: &counter, operations: physical.cacheOperations())
    #expect(duplicate.materialized)
    #expect(physical.operations.filter { if case .copy = $0 { true } else { false } }.count == copyCount)

    var optional: CacheCoordinator? = coordinator
    CacheExecutor.reset(coordinator: &optional, registry: &registry,
      maxActive: 2, hardCeiling: 6, useCounter: &counter)
    #expect(optional == nil)
    #expect(registry.count == 0)
    #expect(counter == 0)
  }

  @Test("failed checkpoint copy publishes no logical anchor")
  func copyFailure() throws {
    let (coordinator, initial) = try initialized(checkpointBudget: 0)
    var registry = initial
    let physical = PhysicalCacheState()
    physical.failure = .copy
    var counter: UInt64 = 0
    let boundary = Array(1...12)
    let identity = testCacheIdentity()
    let source = try physical.make(residentLength: boundary.count)
    let result = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 6, boundary: boundary, sourceCaches: [source],
      sourceFedTokens: boundary, identity: identity, scope: identity.scope,
      structural: false, useCounter: &counter, operations: physical.cacheOperations())
    #expect(!result.materialized)
    #expect((try coordinator.retentionTelemetry()).anchor_slots == 0)
    #expect(registry.slotIDs.compactMap { registry.entry(for: $0) }
      .allSatisfy { !$0.isAnchor || $0.caches.isEmpty })
  }

  private func initialized(checkpointBudget: UInt64) throws
    -> (CacheCoordinator, RetainedRegistry<CacheSlot<PhysicalCacheModel>>) {
    try CacheExecutor.initialize(retention: RetentionConfiguration(initialEntries: 2,
      hardCeiling: 6, physicalByteBudget: 1_536,
      highWatermarkBytes: 1_400, lowWatermarkBytes: 1_000), maxActive: 2,
      capacity: 512, checkpoints: CoordinatorCheckpointConfiguration(enabled: true,
        minimumTokens: 8, intervalTokens: 4, maximum: 3,
        budgetBasisPoints: 2_500, budgetBytes: checkpointBudget))
  }
}
