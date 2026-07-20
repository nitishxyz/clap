public struct LatencySchedulerRequest: Equatable, Sendable {
  public let id: String
  public let admissionOrder: UInt64
  public let residualPrefillTokens: Int
  public let decoding: Bool
  public let emittedFirstToken: Bool
  public let cancelled: Bool

  public init(id: String, admissionOrder: UInt64, residualPrefillTokens: Int,
              decoding: Bool, emittedFirstToken: Bool, cancelled: Bool = false) {
    self.id = id
    self.admissionOrder = admissionOrder
    self.residualPrefillTokens = max(0, residualPrefillTokens)
    self.decoding = decoding
    self.emittedFirstToken = emittedFirstToken
    self.cancelled = cancelled
  }
}

public struct LatencySchedulerStep: Equatable, Sendable {
  public let id: String
  public let prefillQuantum: Int
  public let turns: Int
}

public enum LatencyScheduler {
  public static let normalPrefillQuantum = 512
  public static let contendedPrefillQuantum = 96
  public static let nearFirstTokenThreshold = 256

  // Every runnable request appears exactly once in each round. Priority only
  // changes order within the round, so a stream of short arrivals cannot
  // starve older prefills or decode work.
  public static func round(_ requests: [LatencySchedulerRequest]) -> [LatencySchedulerStep] {
    let runnable = requests.filter { !$0.cancelled }
    let contended = runnable.count > 1
    let quantum = contended ? contendedPrefillQuantum : normalPrefillQuantum
    return runnable.sorted {
      let lhs = priority($0)
      let rhs = priority($1)
      return lhs == rhs ? $0.admissionOrder < $1.admissionOrder : lhs < rhs
    }.map { request in
      let boostedTurns = contended && !request.decoding && !request.emittedFirstToken &&
        request.residualPrefillTokens <= nearFirstTokenThreshold
        ? max(1, (request.residualPrefillTokens + quantum - 1) / quantum + 2) : 1
      return LatencySchedulerStep(id: request.id, prefillQuantum: quantum,
        turns: min(5, boostedTurns))
    }
  }

  private static func priority(_ request: LatencySchedulerRequest) -> Int {
    if !request.emittedFirstToken && request.residualPrefillTokens <= nearFirstTokenThreshold {
      return 0
    }
    if request.decoding && !request.emittedFirstToken { return 1 }
    if request.decoding { return 2 }
    return 3
  }
}
