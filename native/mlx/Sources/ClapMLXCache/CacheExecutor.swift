import ClapCacheBridge
import ClapCachePolicy
import Foundation

public struct CacheAdmission<Cache> {
  public let slotIndex: Int
  public let slot: CacheSlot<Cache>
  public let caches: [Cache]
  public let fedTokens: [Int]
  public let suffix: [Int]
  public let reusedTokens: Int
  public let reuseKind: String?
  public let reuseScope: String?
  public let decision: CacheDecision
  public let candidates: [CacheCandidateEvaluation]
  public let evictions: [Int]
  public let anchorBoundaries: [Int]
  public let coordinatorPlanMs: Double
  public let coordinatorApplyMs: Double
  public let cacheMaterializeMs: Double
  public let evictedVictims: Bool
}

public struct CacheOperations<Cache> {
  public let isTrimmable: (Cache) -> Bool
  public let copy: (Cache) throws -> Cache
  public let trim: (Cache, Int) throws -> Void
  public let sequenceLength: ([Cache], Int) -> Int
  public let create: () throws -> [Cache]
  public let physicalBytes: ([Cache]) -> UInt64
  public let log: (String) -> Void

  public init(isTrimmable: @escaping (Cache) -> Bool, copy: @escaping (Cache) throws -> Cache,
              trim: @escaping (Cache, Int) throws -> Void,
              sequenceLength: @escaping ([Cache], Int) -> Int,
              create: @escaping () throws -> [Cache], physicalBytes: @escaping ([Cache]) -> UInt64,
              log: @escaping (String) -> Void = { _ in }) {
    self.isTrimmable = isTrimmable
    self.copy = copy
    self.trim = trim
    self.sequenceLength = sequenceLength
    self.create = create
    self.physicalBytes = physicalBytes
    self.log = log
  }
}

public enum CacheExecutor {
  public static func admit<Cache>(coordinator: CacheCoordinator,
                                  registry: inout RetainedRegistry<CacheSlot<Cache>>,
                                  hardCeiling: Int, promptTokens: [Int],
                                  identity: CacheIdentity,
                                  physicalIdentity: PhysicalCacheIdentity,
                                  stableBoundaries: [Int], outputReserve: Int,
                                  kvQuantized: Bool, useCounter: inout UInt64,
                                  operations: CacheOperations<Cache>) throws -> CacheAdmission<Cache> {
    var slots = registry.slotIDs.compactMap { registry.entry(for: $0) }
    if !slots.contains(where: { !$0.busy && $0.caches.isEmpty }), registry.count < hardCeiling {
      let registered = try coordinator.registerSlot()
      let slot = CacheSlot<Cache>(coordinatorGeneration: registered.generation)
      try registry.register(slotID: UInt32(registered.slot), entry: slot)
      slots = registry.slotIDs.compactMap { registry.entry(for: $0) }
    }
    let materializations = slots.enumerated().map { index, slot in
      let logical = try? coordinator.slot(index)
      let physical = PhysicalSlotRecord(identity: slot.cacheIdentity, tokens: slot.tokens,
        generation: slot.coordinatorGeneration, hasCaches: !slot.caches.isEmpty,
        isAnchor: slot.isAnchor)
      let materialized = logical.map {
        physical.isMaterialized(for: physicalIdentity, logicalGeneration: $0.generation,
          logicalResidentLength: Int($0.resident_len), logicalState: $0.state,
          anchorState: UInt32(CC_SLOT_ANCHOR))
      } ?? false
      if !slot.caches.isEmpty && !materialized {
        var rejected: [String] = []
        if logical?.generation != slot.coordinatorGeneration { rejected.append("generation") }
        if logical.map({ Int($0.resident_len) }) != slot.tokens.count {
          rejected.append("resident_length")
        }
        let stateMatches = logical.map {
          (slot.isAnchor && $0.state == UInt32(CC_SLOT_ANCHOR)) ||
            (!slot.isAnchor && $0.state != UInt32(CC_SLOT_ANCHOR))
        } ?? false
        if !stateMatches { rejected.append("state") }
        if slot.cacheIdentity?.isCompatible(with: physicalIdentity) != true {
          rejected.append("namespace_identity")
        }
        operations.log("cache donor rejected: slot=\(index) reasons=\(rejected.joined(separator: ",")) physical_generation=\(slot.coordinatorGeneration) logical_generation=\(logical?.generation ?? 0) physical_tokens=\(slot.tokens.count) logical_resident=\(logical?.resident_len ?? 0) logical_state=\(logical?.state ?? 0) anchor=\(slot.isAnchor)")
      }
      return CacheSlotMaterialization(materialized: materialized, writable: !slot.busy,
        partialSuffixTrim: materialized && slot.caches.allSatisfy(operations.isTrimmable),
        copyable: materialized)
    }
    var capabilities = UInt64(CC_CAP_WHOLE_STATE_COPY) |
      UInt64(CC_CAP_PARTIAL_SUFFIX_TRIM) | UInt64(CC_CAP_PARTIAL_PREFIX_BRANCH) |
      UInt64(CC_CAP_SAFE_BUSY_DONOR) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT) |
      UInt64(CC_CAP_RELIABLE_RESIDENT_LENGTH) | UInt64(CC_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS)
    if materializations.contains(where: { $0.materialized && !$0.partialSuffixTrim }) {
      capabilities |= UInt64(CC_CAP_SLIDING_WINDOW) | UInt64(CC_CAP_RECURRENT_OR_HYBRID)
    }
    if kvQuantized { capabilities |= UInt64(CC_CAP_KV_QUANTIZED) }
    let bytesPerToken = slots.compactMap { slot -> UInt64? in
      guard !slot.tokens.isEmpty, !slot.caches.isEmpty else { return nil }
      return operations.physicalBytes(slot.caches) / UInt64(slot.tokens.count)
    }.max() ?? 0
    let planStarted = DispatchTime.now().uptimeNanoseconds
    let plan = try coordinator.plan(tokens: promptTokens, identity: identity,
      capabilities: capabilities, slots: materializations, stableBoundaries: stableBoundaries,
      outputReserve: outputReserve, estimatedBytesPerToken: bytesPerToken)
    let planMs = Double(DispatchTime.now().uptimeNanoseconds - planStarted) / 1_000_000
    let applyStarted = DispatchTime.now().uptimeNanoseconds
    let view = plan.view
    operations.log("cache coordinator plan: operation=\(view.operation) reuse=\(view.reuseTokens) donor=\(view.donor.map(String.init) ?? "none") target=\(view.target)")
    guard view.target < slots.count, view.donor == nil || view.donor! < slots.count else {
      try plan.abort(); throw CacheCoordinatorError.unavailable
    }
    let target = slots[view.target]
    var prefix = view.operation == UInt32(CC_OPERATION_CONTINUE) ? view.reuseTokens : 0
    var donor: CacheSlot<Cache>?
    var branchPrefix = 0
    if view.operation == UInt32(CC_OPERATION_CONTINUE) {
      let trimNeeded = target.tokens.count - view.reuseTokens
      guard view.donor == view.target, !target.caches.isEmpty, trimNeeded >= 0,
            trimNeeded == 0 || target.caches.allSatisfy(operations.isTrimmable) else {
        try plan.abort(); throw CacheCoordinatorError.unavailable
      }
    } else if view.operation == UInt32(CC_OPERATION_BRANCH) || view.operation == UInt32(CC_OPERATION_RESTORE) {
      guard let donorIndex = view.donor, donorIndex != view.target else {
        try plan.abort(); throw CacheCoordinatorError.unavailable
      }
      donor = slots[donorIndex]
      branchPrefix = view.reuseTokens
      let offset = operations.sequenceLength(donor!.caches, donor!.tokens.count)
      let trimNeeded = offset - branchPrefix
      guard !donor!.caches.isEmpty, trimNeeded >= 0,
            trimNeeded == 0 || donor!.caches.allSatisfy(operations.isTrimmable) else {
        try plan.abort(); throw CacheCoordinatorError.unavailable
      }
    }
    if view.operation != UInt32(CC_OPERATION_CONTINUE) { target.clear() }
    useCounter += 1
    target.lastUsed = useCounter
    target.busy = true
    let slotIndex = view.target
    var completed = false
    defer {
      if !completed {
        registry.release(slotID: UInt32(slotIndex))
        let generation = target.coordinatorGeneration
        target.clear()
        target.busy = false
        _ = try? coordinator.invalidate(slot: slotIndex, generation: generation)
      }
    }
    if prefix == promptTokens.count { prefix -= 1 }
    var caches: [Cache]
    var fed: [Int]
    var suffix: [Int]
    var reused = 0
    var reuseScope: String?
    var materializeMs = 0.0
    if let donor {
      let started = DispatchTime.now().uptimeNanoseconds
      var shared = branchPrefix
      if shared == promptTokens.count { shared -= 1 }
      caches = try donor.caches.map(operations.copy)
      let trimNeeded = operations.sequenceLength(caches, donor.tokens.count) - shared
      if trimNeeded > 0 { for cache in caches { try operations.trim(cache, trimNeeded) } }
      fed = Array(promptTokens.prefix(shared)); suffix = Array(promptTokens.dropFirst(shared))
      reused = shared; reuseScope = donor.anchorScope
      materializeMs += Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    } else if prefix > 0 {
      let started = DispatchTime.now().uptimeNanoseconds
      let trimNeeded = target.tokens.count - prefix
      if trimNeeded > 0 { for cache in target.caches { try operations.trim(cache, trimNeeded) } }
      caches = target.caches
      fed = Array(promptTokens.prefix(prefix)); suffix = Array(promptTokens.dropFirst(prefix))
      reused = prefix; reuseScope = target.anchorScope
      materializeMs += Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    } else {
      caches = try operations.create(); fed = []; suffix = promptTokens
    }
    target.caches = caches; target.tokens = fed; target.isPromptBoundary = false
    target.anchorScope = nil; target.cacheIdentity = physicalIdentity
    try registry.activate(slotID: UInt32(slotIndex))
    let victims = view.evictions.filter { $0 != slotIndex }.map(UInt32.init)
    try registry.validateEvictions(victims)
    let decision = try plan.commit(residentTokens: reused, state: UInt32(CC_SLOT_SESSION),
      physicalBytes: operations.physicalBytes(caches))
    target.coordinatorGeneration = try coordinator.slot(slotIndex).generation
    registry.reconcileEvictions(victims) { _, victim in victim.clear() }
    reused = decision.realizedReuseTokens
    reuseScope = cacheScopeName(decision.scope)
    completed = true
    return CacheAdmission(slotIndex: slotIndex, slot: target, caches: caches,
      fedTokens: fed, suffix: suffix, reusedTokens: reused,
      reuseKind: normalizedCacheReuseKind(operation: view.operation), reuseScope: reuseScope,
      decision: decision, candidates: plan.candidates, evictions: view.evictions,
      anchorBoundaries: view.anchorBoundaries, coordinatorPlanMs: planMs,
      coordinatorApplyMs: Double(DispatchTime.now().uptimeNanoseconds - applyStarted) / 1_000_000,
      cacheMaterializeMs: materializeMs, evictedVictims: !victims.isEmpty)
  }
}
