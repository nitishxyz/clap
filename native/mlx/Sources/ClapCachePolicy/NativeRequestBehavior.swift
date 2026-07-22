public struct StopSequenceScan: Equatable, Sendable {
  public let matchOffset: Int?
  public let safeCount: Int
}

public enum StopSequencePolicy {
  public static func scan(collected: String, appendedCount: Int, stops: [String],
                          emittedCount: Int, holdback: Int) -> StopSequenceScan {
    guard !stops.isEmpty else {
      return StopSequenceScan(matchOffset: nil, safeCount: collected.count)
    }
    let windowStart = collected.index(
      collected.endIndex,
      offsetBy: -min(collected.count, appendedCount + holdback)
    )
    var earliest: Range<String.Index>?
    for stop in stops {
      if let found = collected.range(of: stop, range: windowStart..<collected.endIndex),
         earliest == nil || found.lowerBound < earliest!.lowerBound {
        earliest = found
      }
    }
    let matchOffset = earliest.map {
      collected.distance(from: collected.startIndex, to: $0.lowerBound)
    }
    return StopSequenceScan(
      matchOffset: matchOffset,
      safeCount: max(emittedCount, collected.count - holdback)
    )
  }
}

public enum RequestCancellationPolicy {
  public static func matches(target: String?, requestID: String?) -> Bool {
    target == nil || target?.isEmpty == true || requestID == target
  }
}
