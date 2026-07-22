import ClapCacheBridge
import ClapCachePolicy
import Foundation

let cacheTelemetryKey = ProcessInfo.processInfo.environment["CLAP_TELEMETRY_HMAC_KEY"]
  ?? UUID().uuidString + UUID().uuidString

struct CacheIdentity {
  let fingerprint: [UInt8]
  let tenant: UInt64
  let project: UInt64
  let harness: UInt64
  let agent: UInt64
  let session: UInt64
  let scope: UInt32
  let priority: UInt32
  let sideRequest: Bool
  let exportedNamespace: String?

  init(domain: String, requestId: String?, intent: CacheIntent?) {
    let isolation = intent?.tenant ?? intent?.namespace ?? "local"
    let keyed = cacheTelemetryKey + "|"
    var bytes = [UInt8](repeating: 0, count: 32)
    (keyed + domain + "|tenant=" + isolation).withCString { cc_fingerprint_string($0, &bytes) }
    func hash(_ value: String?) -> UInt64 {
      (keyed + (value ?? "")).withCString { cc_hash_string($0) }
    }
    fingerprint = bytes
    tenant = hash(isolation)
    project = hash(intent?.project)
    harness = hash(intent?.harness)
    agent = hash(intent?.agent)
    session = intent?.session?.isEmpty == false ? hash(intent?.session) : 0
    if intent?.session?.isEmpty == false { scope = UInt32(CC_SCOPE_SESSION) }
    else if intent?.agent?.isEmpty == false { scope = UInt32(CC_SCOPE_AGENT) }
    else if intent?.project?.isEmpty == false { scope = UInt32(CC_SCOPE_PROJECT) }
    else if intent?.harness?.isEmpty == false { scope = UInt32(CC_SCOPE_HARNESS) }
    else { scope = UInt32(CC_SCOPE_TENANT) }
    priority = intent?.priority == "background"
      ? UInt32(CC_PRIORITY_BACKGROUND) : UInt32(CC_PRIORITY_INTERACTIVE)
    sideRequest = intent?.side_request ?? false
    exportedNamespace = intent?.namespace ?? intent?.tenant
  }
}

struct CachePlanView {
  let operation: UInt32
  let target: Int
  let targetGeneration: UInt64
  let donor: Int?
  let reuseTokens: Int
  let anchorTokens: Int
  let anchorBoundaries: [Int]
  let decisionUs: UInt64
  let evictions: [Int]
}

struct CacheCandidateEvaluation {
  let slot: Int
  let generation: UInt64
  let state: UInt32
  let sharedPrefixTokens: Int
  let namespaceCompatible: Bool
  let modelCompatible: Bool
  let sessionCompatible: Bool
  let generationCompatible: Bool
  let busyEligible: Bool
  let leaseEligible: Bool
  let materialized: Bool
  let trimEligible: Bool
  let copyEligible: Bool
  let eligible: Bool
  let selected: Bool
  let rejection: UInt32
}

struct CacheSlotMaterialization {
  let materialized: Bool
  let writable: Bool
  let partialSuffixTrim: Bool
  let copyable: Bool

  var flags: UInt8 {
    (materialized ? UInt8(CC_SLOT_MATERIALIZED) : 0) |
      (writable ? UInt8(CC_SLOT_WRITABLE) : 0) |
      (partialSuffixTrim ? UInt8(CC_SLOT_PARTIAL_SUFFIX_TRIM) : 0) |
      (copyable ? UInt8(CC_SLOT_COPY) : 0)
  }
}

struct CacheDecision {
  let hit: Bool
  let operation: UInt32
  let scope: UInt32
  let target: Int
  let donor: Int?
  let plannedReuseTokens: Int
  let realizedReuseTokens: Int
  let decisionUs: UInt64
}

enum CacheCoordinatorError: Error, CustomStringConvertible {
  case status(String, Int32)
  case unavailable

  var description: String {
    switch self {
    case .status(let operation, let status): return "\(operation) failed with cache status \(status)"
    case .unavailable: return "cache coordinator unavailable"
    }
  }

}

final class CachePlan {
  private var handle: OpaquePointer?
  private unowned let owner: CacheCoordinator
  let view: CachePlanView
  let candidates: [CacheCandidateEvaluation]

  init(owner: CacheCoordinator, handle: OpaquePointer) throws {
    self.owner = owner
    self.handle = handle
    var raw = cc_plan_view_t()
    try CacheCoordinator.check("cc_plan_view", cc_plan_view(handle, &raw))
    var evictions: [Int] = []
    for index in 0..<raw.eviction_count {
      var slot: UInt32 = 0
      var generation: UInt64 = 0
      try CacheCoordinator.check("cc_plan_eviction", cc_plan_eviction(handle, index, &slot, &generation))
      evictions.append(Int(slot))
    }
    var anchorBoundaryCount: UInt32 = 0
    try CacheCoordinator.check("cc_plan_anchor_boundary_count",
      cc_plan_anchor_boundary_count(handle, &anchorBoundaryCount))
    var anchorBoundaries: [Int] = []
    for index in 0..<anchorBoundaryCount {
      var tokenCount: UInt64 = 0
      try CacheCoordinator.check("cc_plan_anchor_boundary",
        cc_plan_anchor_boundary(handle, index, &tokenCount))
      anchorBoundaries.append(Int(tokenCount))
    }
    var candidateCount: UInt32 = 0
    try CacheCoordinator.check("cc_plan_candidate_count",
      cc_plan_candidate_count(handle, &candidateCount))
    var candidates: [CacheCandidateEvaluation] = []
    for index in 0..<candidateCount {
      var candidate = cc_candidate_t()
      try CacheCoordinator.check("cc_plan_candidate", cc_plan_candidate(handle, index, &candidate))
      candidates.append(CacheCandidateEvaluation(
        slot: Int(candidate.slot), generation: candidate.generation, state: candidate.state,
        sharedPrefixTokens: Int(candidate.shared_prefix_tokens),
        namespaceCompatible: candidate.namespace_compatible != 0,
        modelCompatible: candidate.model_compatible != 0,
        sessionCompatible: candidate.session_compatible != 0,
        generationCompatible: candidate.generation_compatible != 0,
        busyEligible: candidate.busy_eligible != 0,
        leaseEligible: candidate.lease_eligible != 0,
        materialized: candidate.materialized != 0,
        trimEligible: candidate.trim_eligible != 0,
        copyEligible: candidate.copy_eligible != 0,
        eligible: candidate.eligible != 0, selected: candidate.selected != 0,
        rejection: candidate.rejection))
    }
    self.candidates = candidates
    view = CachePlanView(
      operation: raw.operation,
      target: Int(raw.target_slot),
      targetGeneration: raw.target_generation,
      donor: raw.has_donor != 0 ? Int(raw.donor_slot) : nil,
      reuseTokens: Int(raw.reuse_tokens),
      anchorTokens: Int(raw.anchor_tokens),
      anchorBoundaries: anchorBoundaries,
      decisionUs: raw.decision_us,
      evictions: evictions
    )
  }

  deinit {
    if let handle {
      _ = cc_plan_abort(handle)
      cc_plan_destroy(handle)
    }
  }

  func commit(residentTokens: Int, state: UInt32, physicalBytes: UInt64 = 0) throws -> CacheDecision {
    guard let handle else { throw CacheCoordinatorError.unavailable }
    var raw = cc_decision_t()
    try CacheCoordinator.check("cc_plan_commit", cc_plan_commit(handle, UInt64(residentTokens), state, physicalBytes, &raw))
    cc_plan_destroy(handle)
    self.handle = nil
    return CacheDecision(
      hit: raw.hit != 0,
      operation: raw.operation,
      scope: raw.scope,
      target: Int(raw.target_slot),
      donor: raw.has_donor != 0 ? Int(raw.donor_slot) : nil,
      plannedReuseTokens: Int(raw.planned_reuse_tokens),
      realizedReuseTokens: Int(raw.realized_reuse_tokens),
      decisionUs: raw.decision_us
    )
  }

  func abort() throws {
    guard let handle else { return }
    try CacheCoordinator.check("cc_plan_abort", cc_plan_abort(handle))
    cc_plan_destroy(handle)
    self.handle = nil
  }
}

final class CacheCoordinator {
  private let handle: OpaquePointer

  init(retention: RetentionConfiguration, capacity: Int,
       environment: [String: String] = ProcessInfo.processInfo.environment) throws {
    let enabled = environment["CLAP_CACHE_CHECKPOINTS_ENABLED"] != "0"
    let minimum = UInt64(environment["CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS"] ?? "") ?? 2_048
    let interval = UInt64(environment["CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS"] ?? "") ?? 2_048
    let maximum = UInt32(environment["CLAP_CACHE_CHECKPOINT_MAX"] ?? "") ?? 8
    let fraction = UInt32(environment["CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS"] ?? "") ?? 2_500
    let cap = UInt64(environment["CLAP_CACHE_CHECKPOINT_BUDGET_BYTES"] ?? "") ?? 0
    guard let handle = cc_manager_create_with_retention(
      UInt32(retention.initialEntries), 16, UInt64(max(capacity, 1)),
      UInt32(retention.hardCeiling), UInt32(retention.hardCeiling),
      retention.physicalByteBudget, retention.highWatermarkBytes,
      retention.lowWatermarkBytes, enabled ? 1 : 0, minimum, interval, maximum, fraction, cap) else {
      throw CacheCoordinatorError.unavailable
    }
    self.handle = handle
  }

  deinit { cc_manager_destroy(handle) }

  static func check(_ operation: String, _ status: Int32) throws {
    if status != CC_OK { throw CacheCoordinatorError.status(operation, status) }
  }

  func plan(tokens: [Int], identity: CacheIdentity, capabilities: UInt64,
            slots: [CacheSlotMaterialization],
            stableBoundaries: [Int] = [],
            outputReserve: Int, state: UInt32 = UInt32(CC_SLOT_SESSION),
            scope: UInt32? = nil, estimatedBytesPerToken: UInt64 = 0) throws -> CachePlan {
    let token32 = tokens.map(Int32.init)
    let slotCapabilities = slots.map(\.flags)
    let boundaries = stableBoundaries.map(UInt64.init)
    let raw = identity.fingerprint.withUnsafeBufferPointer { fingerprint in
      token32.withUnsafeBufferPointer { tokens in
        slotCapabilities.withUnsafeBufferPointer { slots in
          boundaries.withUnsafeBufferPointer { boundaries in
            cc_manager_plan(handle, tokens.baseAddress, tokens.count, fingerprint.baseAddress,
              identity.tenant, identity.project, identity.harness, identity.agent,
              identity.session, scope ?? identity.scope, identity.priority,
              identity.sideRequest ? 1 : 0, capabilities, slots.baseAddress, slots.count,
              boundaries.baseAddress, boundaries.count, UInt64(outputReserve),
              estimatedBytesPerToken, state)
          }
        }
      }
    }
    guard let raw else {
      throw CacheCoordinatorError.status("cc_manager_plan", cc_manager_last_status(handle))
    }
    do { return try CachePlan(owner: self, handle: raw) }
    catch { cc_plan_destroy(raw); throw error }
  }

  func slot(_ id: Int) throws -> cc_slot_info_t {
    var info = cc_slot_info_t()
    try Self.check("cc_manager_slot", cc_manager_slot(handle, UInt32(id), &info))
    return info
  }

  func registerSlot() throws -> (slot: Int, generation: UInt64) {
    var slot: UInt32 = 0
    var generation: UInt64 = 0
    try Self.check("cc_manager_register_slot",
      cc_manager_register_slot(handle, &slot, &generation))
    return (Int(slot), generation)
  }

  func retentionTelemetry() throws -> cc_retention_telemetry_t {
    var telemetry = cc_retention_telemetry_t()
    try Self.check("cc_manager_retention_telemetry",
      cc_manager_retention_telemetry(handle, &telemetry))
    return telemetry
  }

  func advance(slot: Int, generation: UInt64, tokens: [Int], state: UInt32,
               busy: Bool, physicalBytes: UInt64) throws -> UInt64 {
    let token32 = tokens.map(Int32.init)
    var next: UInt64 = 0
    let status = token32.withUnsafeBufferPointer {
      cc_manager_advance(handle, UInt32(slot), generation, $0.baseAddress, $0.count,
                         state, busy ? 1 : 0, physicalBytes, &next)
    }
    try Self.check("cc_manager_advance", status)
    return next
  }

  func confirm(slot: Int, generation: UInt64, tokens: [Int], state: UInt32,
               busy: Bool, physicalBytes: UInt64) throws -> UInt64 {
    let token32 = tokens.map(Int32.init)
    var next: UInt64 = 0
    let status = token32.withUnsafeBufferPointer {
      cc_manager_confirm(handle, UInt32(slot), generation, $0.baseAddress, $0.count,
                         state, busy ? 1 : 0, physicalBytes, &next)
    }
    try Self.check("cc_manager_confirm", status)
    return next
  }

  func setBusy(slot: Int, generation: UInt64, busy: Bool) throws {
    try Self.check("cc_manager_set_busy", cc_manager_set_busy(handle, UInt32(slot), generation, busy ? 1 : 0))
  }

  func setAnchorProtected(slot: Int, generation: UInt64, protected: Bool) throws {
    try Self.check("cc_manager_set_anchor_protected",
      cc_manager_set_anchor_protected(handle, UInt32(slot), generation, protected ? 1 : 0))
  }

  func invalidate(slot: Int, generation: UInt64) throws -> UInt64 {
    var next: UInt64 = 0
    try Self.check("cc_manager_invalidate", cc_manager_invalidate(handle, UInt32(slot), generation, &next))
    return next
  }

  func reset() throws {
    var epoch: UInt64 = 0
    try Self.check("cc_manager_reset", cc_manager_reset(handle, &epoch))
  }
}

func cacheScopeName(_ scope: UInt32) -> String {
  switch scope {
  case UInt32(CC_SCOPE_SESSION): return "session"
  case UInt32(CC_SCOPE_AGENT): return "agent"
  case UInt32(CC_SCOPE_PROJECT): return "project"
  case UInt32(CC_SCOPE_HARNESS): return "harness"
  case UInt32(CC_SCOPE_TENANT): return "tenant"
  default: return "none"
  }
}
