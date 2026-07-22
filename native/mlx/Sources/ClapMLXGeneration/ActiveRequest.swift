import ClapMLXCache

public final class GenerationCacheContext<Cache> {
  public let slotIndex: Int
  public let slot: CacheSlot<Cache>
  public var caches: [Cache]
  public let snapshots = CacheSnapshots<Cache>()

  public init(slotIndex: Int, slot: CacheSlot<Cache>, caches: [Cache]) {
    self.slotIndex = slotIndex
    self.slot = slot
    self.caches = caches
  }
}

public final class ActiveRequest<Cache, Iterator, Detokenizer, Parameters> {
  public let prepared: PreparedRequest<Parameters>
  public let cache: GenerationCacheContext<Cache>
  public var continuationBoundary: Int?
  public var fedTokens: [Int]
  public var suffix: [Int]
  public var pos = 0
  public var iterator: Iterator?
  public var detokenizer: Detokenizer
  public var sampledTokens: [Int] = []
  public var collected = ""
  public var emitted = 0
  public var generatedCount = 0
  public var finishReason = "stop"
  public private(set) var status: RequestStatus = .active
  public var anchorPlanted: Set<Int> = []
  public var materializedAnchors: Set<Int> = []
  public var schedulerWaitMs = 0.0
  public var cacheMaterializeMs: Double
  public var prefillMs = 0.0
  public var prefillTokens = 0
  public var prefillChunks = 0
  public var firstDecodeMs = 0.0
  public var firstEmitMs = 0.0
  public var lastStepFinishedNs: UInt64

  public init(prepared: PreparedRequest<Parameters>, cache: GenerationCacheContext<Cache>,
              fedTokens: [Int], suffix: [Int], detokenizer: Detokenizer) {
    self.prepared = prepared
    self.cache = cache
    self.fedTokens = fedTokens
    self.suffix = suffix
    self.detokenizer = detokenizer
    cacheMaterializeMs = prepared.initialCacheMaterializeMs
    lastStepFinishedNs = prepared.admittedNs
  }

  @discardableResult
  public func transition(to terminal: RequestStatus) -> Bool {
    guard terminal.isTerminal, status == .active else { return false }
    status = terminal
    return true
  }

  public var completed: Bool {
    get { status == .completed }
    set { if newValue { _ = transition(to: .completed) } }
  }
  public var cancelled: Bool {
    get { status == .cancelled }
    set { if newValue { _ = transition(to: .cancelled) } }
  }
  public var failed: Bool {
    get { status == .failed }
    set { if newValue { _ = transition(to: .failed) } }
  }

  public var id: String? { prepared.id }
  public var admissionOrder: UInt64 { prepared.admissionOrder }
  public var admittedNs: UInt64 { prepared.admittedNs }
  public var receivedToAdmittedMs: Double { prepared.receivedToAdmittedMs }
  public var templateTokenizeMs: Double { prepared.templateTokenizeMs }
  public var coordinatorPlanMs: Double { prepared.coordinatorPlanMs }
  public var coordinatorApplyMs: Double { prepared.coordinatorApplyMs }
  public var streaming: Bool { prepared.streaming }
  public var maxTokens: Int { prepared.maxTokens }
  public var promptTokens: [Int] { prepared.promptTokens }
  public var reusedTokens: Int { prepared.reusedTokens }
  public var reuseKind: String? { prepared.reuseKind }
  public var reuseScope: String? { prepared.reuseScope }
  public var cacheIdentity: CacheIdentity { prepared.cacheIdentity }
  public var cacheDecision: CacheDecision? { prepared.cacheDecision }
  public var cacheCandidates: [CacheCandidateEvaluation] { prepared.cacheCandidates }
  public var cacheEvictions: [Int] { prepared.cacheEvictions }
  public var cacheFallback: String? { prepared.cacheFallback }
  public var parameters: Parameters { prepared.parameters }
  public var stops: [String] { prepared.stops }
  public var holdback: Int { prepared.holdback }
  public var anchorPlantAt: [Int] { prepared.anchorPlantAt }
  public var anchorPlantScopes: [Int: UInt32] { prepared.anchorPlantScopes }
  public var resolvedBoundaries: [Int: BoundaryInfo] { prepared.resolvedBoundaries }
  public var boundaryTelemetry: [BoundaryInfo] { prepared.boundaryTelemetry }
  public var automaticCheckpointProposed: Int { prepared.automaticCheckpointProposed }
  public var automaticCheckpointDeduped: Int { prepared.automaticCheckpointDeduped }
  public var slotIndex: Int { cache.slotIndex }
  public var slot: CacheSlot<Cache> { cache.slot }
  public var caches: [Cache] {
    get { cache.caches }
    set { cache.caches = newValue }
  }
  public var cacheSnapshots: CacheSnapshots<Cache> { cache.snapshots }
}
