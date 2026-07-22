import ClapCacheBridge
import ClapCachePolicy
import Testing
@testable import ClapMLXCache

@Suite("MLX physical and logical cache transaction boundaries")
struct CacheTransactionBoundaryTests {
  @Test("cold continuation and branch keep physical state aligned")
  func coldContinuationAndBranch() throws {
    let (coordinator, initial) = try initialized(initialEntries: 2, hardCeiling: 3)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let base = Array(1...32)
    let firstIdentity = testCacheIdentity(sessionByte: "a1")

    let cold = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: base, identity: firstIdentity, hardCeiling: 3)
    #expect(cold.reusedTokens == 0)
    #expect(cold.caches.first?.residentLength == 0)
    try publish(cold, coordinator: coordinator, registry: registry,
      physical: physical, prompt: base)
    #expect(cold.slot.caches.first?.residentLength == base.count)

    var continuedPrompt = base
    continuedPrompt[continuedPrompt.count - 1] = 900
    let continued = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: continuedPrompt, identity: firstIdentity, hardCeiling: 3)
    #expect(continued.reuseKind == "slot")
    #expect(continued.slotIndex == cold.slotIndex)
    #expect(continued.reusedTokens > 0)
    #expect(continued.caches.first?.residentLength == continued.reusedTokens)
    try publish(continued, coordinator: coordinator, registry: registry,
      physical: physical, prompt: continuedPrompt)

    let donor = continued.slot
    let donorID = donor.caches.first!.id
    let donorLength = donor.caches.first!.residentLength
    var branchPrompt = continuedPrompt
    branchPrompt[branchPrompt.count - 1] = 901
    let branchIdentity = testCacheIdentity(sessionByte: "a2")
    let branch = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: branchPrompt, identity: branchIdentity, hardCeiling: 3)
    #expect(branch.reuseKind == "branch")
    #expect(branch.slotIndex != continued.slotIndex)
    #expect(donor.caches.first?.id == donorID)
    #expect(donor.caches.first?.residentLength == donorLength)
    #expect(branch.caches.first?.residentLength == branch.reusedTokens)
  }

  @Test("anchor materialization restores once and exact repeat is a physical no-op")
  func anchorRestoreAndNoOp() throws {
    let (coordinator, initial) = try initialized(initialEntries: 2, hardCeiling: 3)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let boundary = Array(101...132)
    let identity = testCacheIdentity(sessionByte: "b1")
    let source = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: boundary, identity: identity, hardCeiling: 3)
    try publish(source, coordinator: coordinator, registry: registry,
      physical: physical, prompt: boundary)

    let anchor = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, boundary: boundary, sourceCaches: source.slot.caches,
      sourceFedTokens: boundary, identity: identity, scope: identity.scope,
      structural: false, useCounter: &counter, operations: physical.cacheOperations())
    #expect(anchor.materialized)
    let copies = physical.operations.filter {
      if case .copy = $0 { return true }; return false
    }.count
    let duplicate = AnchorManager.materialize(coordinator: coordinator, registry: &registry,
      hardCeiling: 3, boundary: boundary, sourceCaches: source.slot.caches,
      sourceFedTokens: boundary, identity: identity, scope: identity.scope,
      structural: false, useCounter: &counter, operations: physical.cacheOperations())
    #expect(duplicate.materialized)
    #expect(physical.operations.filter {
      if case .copy = $0 { return true }; return false
    }.count == copies)

    let oldGeneration = source.slot.coordinatorGeneration
    source.slot.clear()
    source.slot.coordinatorGeneration = try coordinator.invalidate(slot: source.slotIndex,
      generation: oldGeneration)
    var restoredPrompt = boundary
    restoredPrompt.append(999)
    let restored = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: restoredPrompt,
      identity: testCacheIdentity(sessionByte: "b2"), hardCeiling: 3)
    #expect(restored.reuseKind == "anchor")
    #expect(restored.reusedTokens == boundary.count)
    #expect(restored.caches.first?.residentLength == boundary.count)
  }

  @Test("authenticated namespaces isolate identical token streams")
  func authenticatedNamespaceIsolation() throws {
    let (coordinator, initial) = try initialized(initialEntries: 2, hardCeiling: 2)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let prompt = Array(201...232)
    let first = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: prompt,
      identity: testCacheIdentity(namespaceByte: "40", sessionByte: "c1"), hardCeiling: 2)
    try publish(first, coordinator: coordinator, registry: registry,
      physical: physical, prompt: prompt)
    let firstID = first.slot.caches.first!.id

    let isolated = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: prompt,
      identity: testCacheIdentity(namespaceByte: "41", sessionByte: "c1"), hardCeiling: 2)
    #expect(isolated.reusedTokens == 0)
    #expect(isolated.reuseKind == nil)
    #expect(first.slot.caches.first?.id == firstID)
  }

  @Test("physical preparation sees old logical state and failures publish nothing")
  func preparationAndFailures() throws {
    let (coordinator, initial) = try initialized(initialEntries: 1, hardCeiling: 1)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let originalIdentity = testCacheIdentity(namespaceByte: "42", sessionByte: "d1")
    let original = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: Array(301...332), identity: originalIdentity, hardCeiling: 1)
    try publish(original, coordinator: coordinator, registry: registry,
      physical: physical, prompt: Array(301...332))
    let oldLogical = try coordinator.slot(0)
    var observedOldLogical = false
    physical.beforeMutation = { operation in
      guard case .create = operation else { return }
      let during = try coordinator.slot(0)
      observedOldLogical = during.generation == oldLogical.generation &&
        during.resident_len == oldLogical.resident_len
    }
    let replacement = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: Array(401...432),
      identity: testCacheIdentity(namespaceByte: "43", sessionByte: "d2"), hardCeiling: 1)
    #expect(observedOldLogical)
    try publish(replacement, coordinator: coordinator, registry: registry,
      physical: physical, prompt: Array(401...432))

    let failureLogical = try coordinator.slot(0)
    physical.beforeMutation = nil
    physical.failure = .create
    #expect(throws: PhysicalCacheModelFailure.create) {
      try admit(coordinator, registry: &registry, physical: physical,
        counter: &counter, prompt: Array(501...532),
        identity: testCacheIdentity(namespaceByte: "44", sessionByte: "d3"), hardCeiling: 1)
    }
    let afterFailure = try coordinator.slot(0)
    #expect(afterFailure.generation != failureLogical.generation)
    #expect(afterFailure.resident_len == 0)
    #expect(registry.activeCount == 0)
    #expect(registry.entry(for: 0)?.busy == false)
    #expect(registry.entry(for: 0)?.caches.isEmpty == true)
  }

  @Test("coordinator commit failure invalidates target and leaves no busy lease")
  func commitFailure() throws {
    let (coordinator, initial) = try initialized(initialEntries: 1, hardCeiling: 1)
    var registry = initial
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let generation = try coordinator.slot(0).generation
    physical.beforeMutation = { operation in
      guard case .create = operation else { return }
      _ = try coordinator.invalidate(slot: 0, generation: generation)
    }
    #expect(throws: (any Error).self) {
      try admit(coordinator, registry: &registry, physical: physical,
        counter: &counter, prompt: Array(601...632),
        identity: testCacheIdentity(namespaceByte: "45", sessionByte: "e1"), hardCeiling: 1)
    }
    #expect((try coordinator.slot(0)).resident_len == 0)
    #expect(registry.activeCount == 0)
    #expect(registry.entry(for: 0)?.busy == false)
    #expect(registry.entry(for: 0)?.caches.isEmpty == true)
  }

  @Test("pressure victim is released only after logical commit")
  func victimAfterCommit() throws {
    let retention = RetentionConfiguration(initialEntries: 2, hardCeiling: 2,
      physicalByteBudget: 64, highWatermarkBytes: 50, lowWatermarkBytes: 20)
    let initialized: (CacheCoordinator, RetainedRegistry<CacheSlot<PhysicalCacheModel>>) =
      try CacheExecutor.initialize(retention: retention, maxActive: 1, capacity: 1024,
        checkpoints: checkpoints())
    let coordinator = initialized.0
    var registry = initialized.1
    let physical = PhysicalCacheState()
    var counter: UInt64 = 0
    let victimID = try seed(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: [1, 2],
      identity: testCacheIdentity(namespaceByte: "46", sessionByte: "f1"), hardCeiling: 2)
    let generations = try [coordinator.slot(0).generation, coordinator.slot(1).generation]
    var releasedAfterCommit = false
    physical.onRelease = { id in
      guard id == victimID else { return }
      let current = try? [coordinator.slot(0).generation, coordinator.slot(1).generation]
      releasedAfterCommit = current != generations
    }

    let second = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: [9, 10],
      identity: testCacheIdentity(namespaceByte: "47", sessionByte: "f2"), hardCeiling: 2)
    #expect(second.evictedVictims)
    #expect(releasedAfterCommit)
  }

  private func initialized(initialEntries: Int, hardCeiling: Int) throws
    -> (CacheCoordinator, RetainedRegistry<CacheSlot<PhysicalCacheModel>>) {
    try CacheExecutor.initialize(retention: RetentionConfiguration(initialEntries: initialEntries,
      hardCeiling: hardCeiling, physicalByteBudget: 1_000_000,
      highWatermarkBytes: 900_000, lowWatermarkBytes: 700_000), maxActive: 2,
      capacity: 1024, checkpoints: checkpoints())
  }

  private func checkpoints() -> CoordinatorCheckpointConfiguration {
    CoordinatorCheckpointConfiguration(enabled: false, minimumTokens: 2_048,
      intervalTokens: 2_048, maximum: 8, budgetBasisPoints: 2_500, budgetBytes: 0)
  }

  private func admit(_ coordinator: CacheCoordinator,
                     registry: inout RetainedRegistry<CacheSlot<PhysicalCacheModel>>,
                     physical: PhysicalCacheState, counter: inout UInt64,
                     prompt: [Int], identity: CacheIdentity,
                     hardCeiling: Int) throws -> CacheAdmission<PhysicalCacheModel> {
    try CacheExecutor.admit(coordinator: coordinator, registry: &registry,
      hardCeiling: hardCeiling, promptTokens: prompt, identity: identity,
      physicalIdentity: PhysicalCacheIdentity(fingerprint: identity.fingerprint),
      stableBoundaries: [], outputReserve: 1, kvQuantized: false,
      useCounter: &counter, operations: physical.cacheOperations())
  }

  private func publish(_ admission: CacheAdmission<PhysicalCacheModel>,
                       coordinator: CacheCoordinator,
                       registry: RetainedRegistry<CacheSlot<PhysicalCacheModel>>,
                       physical: PhysicalCacheState, prompt: [Int]) throws {
    let suffix = admission.suffix
    for cache in admission.caches { cache.residentLength += suffix.count }
    var fed = admission.fedTokens
    CacheExecutor.appendAndAdvance(coordinator: coordinator, slotIndex: admission.slotIndex,
      slot: admission.slot, caches: admission.caches, fedTokens: &fed, tokens: suffix,
      operations: physical.cacheOperations())
    var caches = admission.caches
    CacheExecutor.finalize(coordinator: coordinator, registry: registry,
      slotIndex: admission.slotIndex, slot: admission.slot, caches: &caches,
      snapshots: CacheSnapshots(), promptTokens: prompt, fedTokens: fed,
      sampledTokens: [], generatedCount: 0, failed: false,
      operations: physical.cacheOperations())
  }

  private func seed(_ coordinator: CacheCoordinator,
                    registry: inout RetainedRegistry<CacheSlot<PhysicalCacheModel>>,
                    physical: PhysicalCacheState, counter: inout UInt64,
                    prompt: [Int], identity: CacheIdentity, hardCeiling: Int) throws -> Int {
    let admission = try admit(coordinator, registry: &registry, physical: physical,
      counter: &counter, prompt: prompt, identity: identity, hardCeiling: hardCeiling)
    let id = admission.caches[0].id
    try publish(admission, coordinator: coordinator, registry: registry,
      physical: physical, prompt: prompt)
    return id
  }
}
