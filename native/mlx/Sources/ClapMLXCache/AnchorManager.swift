import ClapCacheBridge
import ClapCachePolicy
import Foundation

public final class CacheSnapshots<Cache> {
  public var promptBoundary: [Cache]?
  public var continuationBoundary: Int?
  public var continuation: [Cache]?

  public init() {}
}

public struct AnchorResult {
  public let materialized: Bool
  public let evictedVictims: Bool
  public let materializeMs: Double
}

public enum AnchorManager {
  public static func materialize<Cache>(coordinator: CacheCoordinator,
                                        registry: inout RetainedRegistry<CacheSlot<Cache>>,
                                        hardCeiling: Int, boundary: [Int],
                                        sourceCaches: [Cache], sourceFedTokens: [Int],
                                        identity: CacheIdentity, scope: UInt32,
                                        structural: Bool, useCounter: inout UInt64,
                                        operations: CacheOperations<Cache>) -> AnchorResult {
    let plant = boundary.count
    let offset = operations.sequenceLength(sourceCaches, sourceFedTokens.count)
    guard sourceFedTokens == boundary, offset == plant else {
      operations.log("anchor skipped: fed=\(sourceFedTokens.count) offset=\(offset) plant=\(plant)")
      return AnchorResult(materialized: false, evictedVictims: false, materializeMs: 0)
    }
    var slots = registry.slotIDs.compactMap { registry.entry(for: $0) }
    do {
      if !slots.contains(where: { !$0.busy && $0.caches.isEmpty }), registry.count < hardCeiling {
        let registered = try coordinator.registerSlot()
        try registry.register(slotID: UInt32(registered.slot),
          entry: CacheSlot<Cache>(coordinatorGeneration: registered.generation))
        slots = registry.slotIDs.compactMap { registry.entry(for: $0) }
      }
      let plan = try coordinator.plan(tokens: boundary, identity: identity,
        capabilities: UInt64(CC_CAP_WHOLE_STATE_COPY) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT),
        slots: slots.map { CacheSlotMaterialization(materialized: false, writable: !$0.busy,
          partialSuffixTrim: false, copyable: false) }, outputReserve: 0,
        state: UInt32(CC_SLOT_ANCHOR), scope: scope,
        estimatedBytesPerToken: operations.physicalBytes(sourceCaches) / UInt64(max(plant, 1)))
      guard plan.view.target < slots.count, !slots[plan.view.target].busy else {
        try? plan.abort()
        return AnchorResult(materialized: false, evictedVictims: false, materializeMs: 0)
      }
      let target = slots[plan.view.target]
      if plan.view.operation == UInt32(CC_OPERATION_NOOP) {
        _ = try plan.commit(residentTokens: plant, state: UInt32(CC_SLOT_ANCHOR),
          physicalBytes: operations.physicalBytes(target.caches))
        return AnchorResult(materialized: true, evictedVictims: false, materializeMs: 0)
      }
      let started = DispatchTime.now().uptimeNanoseconds
      do {
        let copied = try sourceCaches.map(operations.copy)
        target.clear()
        target.isAnchor = true
        target.anchorScope = cacheScopeName(scope)
        target.caches = copied
        target.tokens = boundary
        target.cacheIdentity = PhysicalCacheIdentity(fingerprint: identity.fingerprint)
        let index = plan.view.target
        let victims = plan.view.evictions.filter { $0 != index }.map(UInt32.init)
        try registry.validateEvictions(victims)
        _ = try plan.commit(residentTokens: plant, state: UInt32(CC_SLOT_ANCHOR),
          physicalBytes: operations.physicalBytes(copied))
        let logical = try coordinator.slot(index)
        target.coordinatorGeneration = logical.generation
        if structural {
          try coordinator.setAnchorProtected(slot: index, generation: logical.generation,
            protected: true)
        }
        registry.reconcileEvictions(victims) { _, victim in victim.clear() }
        useCounter += 1
        target.lastUsed = useCounter
        return AnchorResult(materialized: true, evictedVictims: !victims.isEmpty,
          materializeMs: Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)
      } catch {
        target.clear()
        try? plan.abort()
        operations.log("coordinated anchor commit failed: \(error)")
        return AnchorResult(materialized: false, evictedVictims: false,
          materializeMs: Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)
      }
    } catch {
      operations.log("coordinated anchor skipped: \(error)")
      return AnchorResult(materialized: false, evictedVictims: false, materializeMs: 0)
    }
  }

  public static func captureContinuation<Cache>(snapshots: CacheSnapshots<Cache>,
                                                 boundary: Int, caches: [Cache],
                                                 fedTokens: [Int],
                                                 operations: CacheOperations<Cache>) -> Double {
    guard snapshots.continuation == nil, boundary == fedTokens.count,
          caches.contains(where: { !operations.isTrimmable($0) }),
          operations.sequenceLength(caches, fedTokens.count) == boundary else { return 0 }
    let started = DispatchTime.now().uptimeNanoseconds
    do { snapshots.continuation = try caches.map(operations.copy) }
    catch { operations.log("rolling conversation anchor copy failed: \(error)"); return 0 }
    snapshots.continuationBoundary = boundary
    return Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
  }

  public static func capturePromptBoundary<Cache>(snapshots: CacheSnapshots<Cache>,
                                                   promptTokens: [Int], caches: [Cache],
                                                   fedTokens: [Int],
                                                   operations: CacheOperations<Cache>) -> Double {
    guard snapshots.continuation == nil,
          caches.contains(where: { !operations.isTrimmable($0) }),
          operations.sequenceLength(caches, fedTokens.count) == promptTokens.count else { return 0 }
    let started = DispatchTime.now().uptimeNanoseconds
    do { snapshots.promptBoundary = try caches.map(operations.copy) }
    catch { operations.log("prompt-boundary anchor copy failed: \(error)"); return 0 }
    return Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
  }
}
