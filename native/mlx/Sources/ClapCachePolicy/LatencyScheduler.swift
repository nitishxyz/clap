public enum SchedulingPriority: Int, Equatable, Sendable {
  case background = 0
  case normal = 1
  case interactive = 2
}

public struct LatencySchedulerRequest: Equatable, Sendable {
  public let id: String
  public let admissionOrder: UInt64
  public let residualPrefillTokens: Int
  public let decoding: Bool
  public let emittedFirstToken: Bool
  public let cancelled: Bool
  public let priority: SchedulingPriority

  public init(id: String, admissionOrder: UInt64, residualPrefillTokens: Int,
              decoding: Bool, emittedFirstToken: Bool, cancelled: Bool = false,
              priority: SchedulingPriority = .normal) {
    self.id = id
    self.admissionOrder = admissionOrder
    self.residualPrefillTokens = max(0, residualPrefillTokens)
    self.decoding = decoding
    self.emittedFirstToken = emittedFirstToken
    self.cancelled = cancelled
    self.priority = priority
  }
}

public struct LatencySchedulerStep: Equatable, Sendable {
  public let id: String
  public let prefillQuantum: Int
  public let turns: Int
}

public enum LatencyScheduler {
  public static let normalPrefillQuantum = 512
  public static let interactivePrefillQuantum = 192
  public static let normalContendedPrefillQuantum = 96
  public static let backgroundPrefillQuantum = 48
  public static let nearFirstTokenThreshold = 256

  // Every runnable request appears exactly once in each round. Priority only
  // changes order within the round, so a stream of short arrivals cannot
  // starve older prefills or decode work.
  public static func round(_ requests: [LatencySchedulerRequest]) -> [LatencySchedulerStep] {
    let runnable = requests.filter { !$0.cancelled }
    let contended = runnable.count > 1
    let ordered = runnable.sorted {
      if $0.priority != $1.priority { return $0.priority.rawValue > $1.priority.rawValue }
      let lhs = phasePriority($0), rhs = phasePriority($1)
      return lhs == rhs ? $0.admissionOrder < $1.admissionOrder : lhs < rhs
    }
    let step = { (request: LatencySchedulerRequest) in
      LatencySchedulerStep(id: request.id, prefillQuantum: contended ? quantum(request.priority) : normalPrefillQuantum,
        turns: 1)
    }
    guard contended else { return ordered.map(step) }
    var result = ordered.map(step) // Every runnable request advances before extras.
    for request in ordered {
      let extras = request.priority == .interactive ? 3 : request.priority == .normal ? 1 : 0
      result.append(contentsOf: Array(repeating: step(request), count: extras))
    }
    return result
  }

  private static func quantum(_ priority: SchedulingPriority) -> Int {
    switch priority {
    case .interactive: interactivePrefillQuantum
    case .normal: normalContendedPrefillQuantum
    case .background: backgroundPrefillQuantum
    }
  }

  private static func phasePriority(_ request: LatencySchedulerRequest) -> Int {
    if !request.emittedFirstToken && request.residualPrefillTokens <= nearFirstTokenThreshold {
      return 0
    }
    if request.decoding && !request.emittedFirstToken { return 1 }
    if request.decoding { return 2 }
    return 3
  }
}
