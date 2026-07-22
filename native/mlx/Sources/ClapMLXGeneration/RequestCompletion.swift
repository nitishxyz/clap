import ClapMLXCache

public enum RequestStatus: Equatable, Sendable {
  case active
  case completed
  case cancelled
  case failed

  public var isTerminal: Bool { self != .active }
}

public enum CompletionOutput: Equatable, Sendable {
  case token(String)
  case content(String)
}

public struct CompletionUsageFacts: Equatable, Sendable {
  public let promptTokens: Int
  public let completionTokens: Int
}

public struct CompletionTimingFacts: Equatable, Sendable {
  public let receivedToAdmittedMs: Double
  public let templateTokenizeMs: Double
  public let coordinatorPlanMs: Double
  public let coordinatorApplyMs: Double
  public let schedulerWaitMs: Double
  public let cacheMaterializeMs: Double
  public let prefillMs: Double
  public let promptTokens: Int
  public let reusedTokens: Int
  public let prefillTokens: Int
  public let prefillChunks: Int
  public let firstDecodeMs: Double
  public let firstEmitMs: Double
}

public struct CompletionCacheFacts {
  public let identity: CacheIdentity
  public let decision: CacheDecision?
  public let candidates: [CacheCandidateEvaluation]
  public let evictions: [Int]
  public let fallback: String?
  public let slotIndex: Int
  public let promptTokens: [Int]
  public let reusedTokens: Int
  public let reuseKind: String?
  public let reuseScope: String?
  public let anchorPlantAt: [Int]
  public let resolvedBoundaries: [Int: BoundaryInfo]
  public let boundaryTelemetry: [BoundaryInfo]
  public let materializedAnchors: Set<Int>
  public let automaticCheckpointProposed: Int
  public let automaticCheckpointDeduped: Int
}

public struct RequestCompletion {
  public let status: RequestStatus
  public let finishReason: String
  public let content: String
  public let generatedTokens: Int
  public let outputs: [CompletionOutput]
  public let usage: CompletionUsageFacts
  public let timing: CompletionTimingFacts
  public let cache: CompletionCacheFacts
}

public struct RequestFailure: Equatable, Sendable {
  public let status: RequestStatus
  public init() { status = .failed }
}

public enum RequestFinalization {
  case completion(RequestCompletion)
  case failure(RequestFailure)
}

public struct GenerationFinalizer<Cache> {
  public let finalizeCache: (Int, CacheSlot<Cache>, inout [Cache], CacheSnapshots<Cache>,
                             [Int], [Int], [Int], Int, Bool) -> Void

  public init(finalizeCache: @escaping (Int, CacheSlot<Cache>, inout [Cache],
                                        CacheSnapshots<Cache>, [Int], [Int], [Int],
                                        Int, Bool) -> Void) {
    self.finalizeCache = finalizeCache
  }
}
