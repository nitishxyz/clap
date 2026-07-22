import ClapCacheBridge
import ClapCachePolicy
import Testing
@testable import ClapMLXCache

@Suite("MLX cache architecture shapes")
struct CacheArchitectureShapeTests {
  @Test("standard cache trims continuation and copies branch prefix")
  func standardTrimAndCopy() throws {
    let fixture = try Fixture()
    let identity = testCacheIdentity(sessionByte: "a1")
    let donor = try fixture.seed(identity: identity,
      caches: [try fixture.physical.make(residentLength: 32)])
    var continuation = donor.slot.tokens
    continuation[31] = 900
    let continued = try fixture.admit(prompt: continuation, identity: identity)
    #expect(continued.reuseKind == "slot")
    #expect(fixture.physical.operations.contains {
      if case .trim = $0 { true } else { false }
    })
    fixture.release(continued)

    var branch = continuation
    branch[31] = 901
    let branched = try fixture.admit(prompt: branch,
      identity: testCacheIdentity(sessionByte: "a2"))
    #expect(branched.reuseKind == "branch")
    #expect(fixture.physical.operations.contains {
      if case .copy = $0 { true } else { false }
    })
  }

  @Test("sliding cache reports absolute logical length while retaining its stored window")
  func slidingAbsoluteLength() throws {
    let fixture = try Fixture()
    let identity = testCacheIdentity(sessionByte: "b1")
    let sliding = try fixture.physical.make(residentLength: 32,
      storedLength: 8, shape: .sliding(window: 8))
    let donor = try fixture.seed(identity: identity, caches: [sliding])
    #expect((try fixture.coordinator.slot(donor.slotIndex)).resident_len == 32)
    #expect(sliding.storedLength == 8)

    var continuation = donor.slot.tokens
    continuation[31] = 902
    let admission = try fixture.admit(prompt: continuation, identity: identity)
    #expect(admission.reusedTokens == 31)
    #expect(admission.caches[0].residentLength == 31)
    #expect(admission.caches[0].storedLength == 8)
  }

  @Test("recurrent cache falls back fresh when trimming is required and restores whole state")
  func recurrentFallbackAndRestore() throws {
    let fixture = try Fixture()
    let identity = testCacheIdentity(sessionByte: "c1")
    let recurrent = try fixture.physical.make(residentLength: 32, shape: .recurrent)
    let donor = try fixture.seed(identity: identity, caches: [recurrent])
    var changed = donor.slot.tokens
    changed[31] = 903
    let fresh = try fixture.admit(prompt: changed, identity: identity)
    #expect(fresh.reusedTokens == 0)
    #expect(fresh.caches.first?.id != recurrent.id)
    fixture.release(fresh)

    let anchor = AnchorManager.materialize(coordinator: fixture.coordinator,
      registry: &fixture.registry, hardCeiling: 4, boundary: donor.slot.tokens,
      sourceCaches: donor.slot.caches, sourceFedTokens: donor.slot.tokens,
      identity: identity, scope: identity.scope, structural: false,
      useCounter: &fixture.counter, operations: fixture.physical.cacheOperations())
    #expect(anchor.materialized)
    let boundary = donor.slot.tokens
    let generation = donor.slot.coordinatorGeneration
    donor.slot.clear()
    donor.slot.coordinatorGeneration = try fixture.coordinator.invalidate(
      slot: donor.slotIndex, generation: generation)
    var restoredPrompt = boundary
    restoredPrompt.append(904)
    let restored = try fixture.admit(prompt: restoredPrompt,
      identity: testCacheIdentity(sessionByte: "c2"))
    #expect(restored.reuseKind == "anchor")
    #expect(restored.reusedTokens == boundary.count)
    #expect(restored.caches.first?.residentLength == boundary.count)
  }

  @Test("hybrid ordered components copy all state in order")
  func hybridCopiesInOrder() throws {
    let fixture = try Fixture()
    let components = [
      try fixture.physical.make(residentLength: 32, shape: .hybridComponent),
      try fixture.physical.make(residentLength: 32, shape: .recurrent),
    ]
    let donor = try fixture.seed(identity: testCacheIdentity(sessionByte: "d1"),
      caches: components)
    var prompt = donor.slot.tokens
    prompt.append(905)
    let admission = try fixture.admit(prompt: prompt,
      identity: testCacheIdentity(sessionByte: "d2"))
    #expect(admission.reuseKind == "branch")
    #expect(admission.caches.count == 2)
    let copies = fixture.physical.operations.compactMap { operation -> Int? in
      if case .copy(let source, _) = operation { return source }
      return nil
    }
    #expect(Array(copies.suffix(2)) == components.map(\.id))
  }

  @Test("hybrid second-component copy failure clears the complete target")
  func hybridSecondCopyFailure() throws {
    let fixture = try Fixture()
    let components = [
      try fixture.physical.make(residentLength: 32, shape: .hybridComponent),
      try fixture.physical.make(residentLength: 32, shape: .recurrent),
    ]
    let donor = try fixture.seed(identity: testCacheIdentity(sessionByte: "e1"),
      caches: components)
    fixture.physical.failCopyNumber = 2
    var prompt = donor.slot.tokens
    prompt.append(906)
    #expect(throws: PhysicalCacheModelFailure.copy) {
      try fixture.admit(prompt: prompt, identity: testCacheIdentity(sessionByte: "e2"))
    }
    let targets = fixture.registry.slotIDs.compactMap { fixture.registry.entry(for: $0) }
      .filter { $0 !== donor.slot }
    #expect(targets.allSatisfy { $0.caches.isEmpty && $0.tokens.isEmpty && !$0.busy })
    #expect(donor.slot.caches.map(\.id) == components.map(\.id))
  }
}

private final class Fixture {
  let coordinator: CacheCoordinator
  var registry: RetainedRegistry<CacheSlot<PhysicalCacheModel>>
  let physical = PhysicalCacheState()
  var counter: UInt64 = 0

  init() throws {
    let initialized: (CacheCoordinator, RetainedRegistry<CacheSlot<PhysicalCacheModel>>) =
      try CacheExecutor.initialize(retention: RetentionConfiguration(initialEntries: 2,
        hardCeiling: 4, physicalByteBudget: 1_000_000,
        highWatermarkBytes: 900_000, lowWatermarkBytes: 700_000), maxActive: 2,
        capacity: 512, checkpoints: CoordinatorCheckpointConfiguration(enabled: false,
          minimumTokens: 2_048, intervalTokens: 2_048, maximum: 8,
          budgetBasisPoints: 2_500, budgetBytes: 0))
    coordinator = initialized.0
    registry = initialized.1
  }

  func seed(identity: CacheIdentity, caches: [PhysicalCacheModel]) throws
    -> CacheAdmission<PhysicalCacheModel> {
    let prompt = Array(1...32)
    let admission = try admit(prompt: prompt, identity: identity)
    admission.slot.caches = caches
    admission.slot.tokens = prompt
    admission.slot.cacheIdentity = PhysicalCacheIdentity(fingerprint: identity.fingerprint)
    admission.slot.coordinatorGeneration = try coordinator.advance(slot: admission.slotIndex,
      generation: admission.slot.coordinatorGeneration, tokens: prompt,
      state: UInt32(CC_SLOT_SESSION), busy: false, physicalBytes: 64)
    admission.slot.busy = false
    registry.release(slotID: UInt32(admission.slotIndex))
    return admission
  }

  func admit(prompt: [Int], identity: CacheIdentity) throws
    -> CacheAdmission<PhysicalCacheModel> {
    try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: 4, promptTokens: prompt, identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: physical.cacheOperations())
  }

  func release(_ admission: CacheAdmission<PhysicalCacheModel>) {
    try? coordinator.setBusy(slot: admission.slotIndex,
      generation: admission.slot.coordinatorGeneration, busy: false)
    admission.slot.busy = false
    registry.release(slotID: UInt32(admission.slotIndex))
  }
}
