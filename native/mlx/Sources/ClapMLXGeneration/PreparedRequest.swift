import ClapMLXCache

public struct BoundaryInfo: Equatable, Sendable {
  public let tokenCount: Int?
  public let kind: String
  public let label: String?
  public let requested: Bool
  public let status: String
  public let skipReason: String?

  public init(tokenCount: Int?, kind: String, label: String?, requested: Bool,
              status: String, skipReason: String?) {
    self.tokenCount = tokenCount
    self.kind = kind
    self.label = label
    self.requested = requested
    self.status = status
    self.skipReason = skipReason
  }
}

public struct PreparedRequest<Parameters> {
  public let id: String?
  public let admissionOrder: UInt64
  public let admittedNs: UInt64
  public let receivedToAdmittedMs: Double
  public let templateTokenizeMs: Double
  public let coordinatorPlanMs: Double
  public let coordinatorApplyMs: Double
  public let initialCacheMaterializeMs: Double
  public let streaming: Bool
  public let maxTokens: Int
  public let promptTokens: [Int]
  public let reusedTokens: Int
  public let reuseKind: String?
  public let reuseScope: String?
  public let cacheIdentity: CacheIdentity
  public let cacheDecision: CacheDecision?
  public let cacheCandidates: [CacheCandidateEvaluation]
  public let cacheEvictions: [Int]
  public let cacheFallback: String?
  public let parameters: Parameters
  public let stops: [String]
  public let holdback: Int
  public let anchorPlantAt: [Int]
  public let anchorPlantScopes: [Int: UInt32]
  public let resolvedBoundaries: [Int: BoundaryInfo]
  public let boundaryTelemetry: [BoundaryInfo]
  public let automaticCheckpointProposed: Int
  public let automaticCheckpointDeduped: Int

  public init(id: String?, admissionOrder: UInt64, admittedNs: UInt64,
              receivedToAdmittedMs: Double, templateTokenizeMs: Double,
              coordinatorPlanMs: Double, coordinatorApplyMs: Double,
              cacheMaterializeMs: Double, streaming: Bool, maxTokens: Int,
              promptTokens: [Int], reusedTokens: Int, reuseKind: String?,
              reuseScope: String?, cacheIdentity: CacheIdentity,
              cacheDecision: CacheDecision?, cacheCandidates: [CacheCandidateEvaluation],
              cacheEvictions: [Int], cacheFallback: String?, parameters: Parameters,
              stops: [String], anchorPlantAt: [Int] = [],
              anchorPlantScopes: [Int: UInt32] = [:],
              resolvedBoundaries: [Int: BoundaryInfo] = [:],
              boundaryTelemetry: [BoundaryInfo] = [],
              automaticCheckpointProposed: Int = 0,
              automaticCheckpointDeduped: Int = 0) {
    self.id = id
    self.admissionOrder = admissionOrder
    self.admittedNs = admittedNs
    self.receivedToAdmittedMs = receivedToAdmittedMs
    self.templateTokenizeMs = templateTokenizeMs
    self.coordinatorPlanMs = coordinatorPlanMs
    self.coordinatorApplyMs = coordinatorApplyMs
    self.initialCacheMaterializeMs = cacheMaterializeMs
    self.streaming = streaming
    self.maxTokens = maxTokens
    self.promptTokens = promptTokens
    self.reusedTokens = reusedTokens
    self.reuseKind = reuseKind
    self.reuseScope = reuseScope
    self.cacheIdentity = cacheIdentity
    self.cacheDecision = cacheDecision
    self.cacheCandidates = cacheCandidates
    self.cacheEvictions = cacheEvictions
    self.cacheFallback = cacheFallback
    self.parameters = parameters
    self.stops = stops
    self.holdback = stops.map(\.count).max().map { $0 - 1 } ?? 0
    self.anchorPlantAt = anchorPlantAt
    self.anchorPlantScopes = anchorPlantScopes
    self.resolvedBoundaries = resolvedBoundaries
    self.boundaryTelemetry = boundaryTelemetry
    self.automaticCheckpointProposed = automaticCheckpointProposed
    self.automaticCheckpointDeduped = automaticCheckpointDeduped
  }
}
