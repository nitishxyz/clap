import ClapCacheBridge
import ClapCachePolicy
import Testing
@testable import ClapMLXCache

@Suite("MLX cache executor")
struct CacheExecutorTests {
  @Test("fresh admission creates, commits, refreshes generation, and activates")
  func freshAdmission() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    var counter: UInt64 = 0
    let identity = makeIdentity(session: "one")
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 2, promptTokens: [1, 2, 3], identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: operations())
    #expect(admission.slotIndex == 0)
    #expect(admission.fedTokens.isEmpty)
    #expect(admission.suffix == [1, 2, 3])
    #expect(admission.reusedTokens == 0)
    #expect(admission.slot.caches.count == 1)
    #expect(admission.slot.coordinatorGeneration != 0)
    #expect(registry.isActive(slotID: 0))
  }

  @Test("create failure aborts and releases and invalidates target")
  func createFailure() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    var counter: UInt64 = 0
    let identity = makeIdentity(session: "one")
    #expect(throws: FakeFailure.create) {
      try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
        hardCeiling: 1, promptTokens: [1, 2], identity: identity,
        physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
        stableBoundaries: [], outputReserve: 1, kvQuantized: false,
        useCounter: &counter, operations: operations(create: { throw FakeFailure.create }))
    }
    #expect(registry.activeCount == 0)
    let slot = registry.entry(for: 0)
    #expect(slot?.caches.isEmpty == true)
    #expect(slot?.tokens.isEmpty == true)
    #expect(slot?.busy == false)
  }

  @Test("generation and identity mismatch cannot materialize a stale donor")
  func staleDonorFallsBackFresh() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    let slot = registry.entry(for: 0)!
    slot.caches = [FakeCache(offset: 3)]
    slot.tokens = [1, 2, 3]
    slot.coordinatorGeneration = 999
    slot.cacheIdentity = PhysicalCacheIdentity(fingerprint: [UInt8](repeating: 7, count: 32))
    var creates = 0
    var counter: UInt64 = 0
    let identity = makeIdentity(session: "new")
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 1, promptTokens: [1, 2, 3], identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: operations(create: {
        creates += 1
        return [FakeCache(offset: 0)]
      }))
    #expect(creates == 1)
    #expect(admission.reusedTokens == 0)
  }

  @Test("resident length mismatch cannot materialize a stale donor")
  func residentLengthMismatchFallsBackFresh() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    let identity = makeIdentity(session: "one")
    let slot = registry.entry(for: 0)!
    slot.caches = [FakeCache(offset: 1)]
    slot.tokens = [1]
    slot.cacheIdentity = PhysicalCacheIdentity(fingerprint: identity.fingerprint)
    var creates = 0
    var counter: UInt64 = 0
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 1, promptTokens: [1, 2], identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: operations(create: {
        creates += 1
        return [FakeCache(offset: 0)]
      }))
    #expect(creates == 1)
    #expect(admission.reusedTokens == 0)
  }

  @Test("continuation trim failure clears invalidates and releases target")
  func trimFailure() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    var counter: UInt64 = 0
    let identity = makeIdentity(session: "one")
    try seedResident(coordinator: coordinator, registry: &registry,
      identity: identity, counter: &counter)
    #expect(throws: FakeFailure.trim) {
      try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
        hardCeiling: 2, promptTokens: Array(1...63) + [99], identity: identity,
        physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
        stableBoundaries: [], outputReserve: 1, kvQuantized: false,
        useCounter: &counter, operations: operations(trim: { _, _ in throw FakeFailure.trim }))
    }
    #expect(registry.activeCount == 0)
    #expect(registry.entry(for: 0)?.caches.isEmpty == true)
    #expect(registry.entry(for: 0)?.busy == false)
  }

  @Test("branch copy failure preserves donor and clears the new target")
  func copyFailure() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    var counter: UInt64 = 0
    let donorIdentity = makeIdentity(session: "donor")
    try seedResident(coordinator: coordinator, registry: &registry,
      identity: donorIdentity, counter: &counter)
    let branchIdentity = makeIdentity(session: "branch")
    #expect(throws: FakeFailure.copy) {
      try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
        hardCeiling: 2, promptTokens: Array(1...63) + [99], identity: branchIdentity,
        physicalIdentity: PhysicalCacheIdentity(fingerprint: branchIdentity.fingerprint),
        stableBoundaries: [], outputReserve: 1, kvQuantized: false,
        useCounter: &counter, operations: operations(copy: { _ in throw FakeFailure.copy }))
    }
    #expect(registry.activeCount == 0)
    #expect(registry.entry(for: 0)?.tokens == Array(1...64))
    #expect(registry.entry(for: 0)?.caches.count == 1)
    #expect(registry.entry(for: 1)?.caches.isEmpty == true)
    #expect(registry.entry(for: 1)?.busy == false)
  }

  @Test("commit failure releases activation and clears the target")
  func commitFailure() throws {
    let coordinator = try makeCoordinator()
    var registry = try makeRegistry(coordinator: coordinator)
    var counter: UInt64 = 0
    let identity = makeIdentity(session: "one")
    let generation = try coordinator.slot(0).generation
    #expect(throws: (any Error).self) {
      try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
        hardCeiling: 1, promptTokens: [1, 2], identity: identity,
        physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
        stableBoundaries: [], outputReserve: 1, kvQuantized: false,
        useCounter: &counter, operations: operations(create: {
          _ = try coordinator.invalidate(slot: 0, generation: generation)
          return [FakeCache(offset: 0)]
        }))
    }
    #expect(registry.activeCount == 0)
    #expect(registry.entry(for: 0)?.caches.isEmpty == true)
    #expect(registry.entry(for: 0)?.busy == false)
  }

  private func makeCoordinator() throws -> CacheCoordinator {
    try CacheCoordinator(retention: RetentionConfiguration(initialEntries: 1,
      hardCeiling: 2, physicalByteBudget: 1_000_000,
      highWatermarkBytes: 900_000, lowWatermarkBytes: 700_000), capacity: 1024,
      checkpoints: CoordinatorCheckpointConfiguration(enabled: false,
        minimumTokens: 2_048, intervalTokens: 2_048, maximum: 8,
        budgetBasisPoints: 2_500, budgetBytes: 0))
  }

  private func makeRegistry(coordinator: CacheCoordinator)
    throws -> RetainedRegistry<CacheSlot<FakeCache>> {
    let registry = RetainedRegistry<CacheSlot<FakeCache>>(maxActive: 1, hardCeiling: 2)
    let slot = CacheSlot<FakeCache>()
    slot.coordinatorGeneration = try coordinator.slot(0).generation
    try registry.register(slotID: 0, entry: slot)
    return registry
  }

  private func makeIdentity(session: String) -> CacheIdentity {
    CacheIdentity(domain: "model", input: CacheIdentityInput(namespace: "tenant",
      tenant: nil, project: nil, harness: nil, agent: nil, session: session,
      priority: nil, sideRequest: false), telemetryKey: "test")
  }

  private func seedResident(coordinator: CacheCoordinator,
                            registry: inout RetainedRegistry<CacheSlot<FakeCache>>,
                            identity: CacheIdentity, counter: inout UInt64) throws {
    let tokens = Array(1...64)
    let admission = try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 2, promptTokens: tokens, identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: operations())
    registry.release(slotID: UInt32(admission.slotIndex))
    admission.slot.busy = false
    admission.slot.coordinatorGeneration = try coordinator.advance(slot: admission.slotIndex,
      generation: admission.slot.coordinatorGeneration, tokens: tokens,
      state: UInt32(CC_SLOT_SESSION), busy: false, physicalBytes: 64)
    admission.slot.tokens = tokens
    admission.slot.caches = [FakeCache(offset: tokens.count)]
  }

  private func operations(copy: @escaping (FakeCache) throws -> FakeCache = {
    FakeCache(offset: $0.offset)
  }, trim: @escaping (FakeCache, Int) throws -> Void = {
    $0.offset -= $1
  }, create: @escaping () throws -> [FakeCache] = {
    [FakeCache(offset: 0)]
  }) -> CacheOperations<FakeCache> {
    CacheOperations(isTrimmable: { _ in true }, copy: copy,
      trim: trim, sequenceLength: { caches, fallback in
        let value = caches.map(\.offset).max() ?? 0
        return value > 0 ? value : fallback
      }, create: create, physicalBytes: { UInt64(max(1, $0.count * 64)) })
  }
}

private enum FakeFailure: Error { case copy, create, trim }
private final class FakeCache {
  var offset: Int
  init(offset: Int) { self.offset = offset }
}
