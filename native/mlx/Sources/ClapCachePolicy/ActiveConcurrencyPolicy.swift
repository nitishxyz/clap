import Foundation

public struct ActiveConcurrencyInputs: Equatable, Sendable {
  public let explicitMax: Int?
  public let physicalMemoryBytes: UInt64
  public let startupAvailableMemoryBytes: UInt64?
  public let modelActiveBytes: UInt64?
  public let retainedBudgetBytes: UInt64
  public let retainedBytes: UInt64
  public let recentRetainedGrowthBytes: UInt64
  public let retainedGrowthMinimumBytes: UInt64
  public let retainedGrowthReservePercent: UInt64
  public let retainedCeiling: Int
  public let processorCount: Int
  public let isHybridOrRecurrent: Bool

  public init(explicitMax: Int? = nil, physicalMemoryBytes: UInt64,
              startupAvailableMemoryBytes: UInt64?, modelActiveBytes: UInt64?,
              retainedBudgetBytes: UInt64, retainedBytes: UInt64 = 0,
              recentRetainedGrowthBytes: UInt64 = 0,
              retainedGrowthMinimumBytes: UInt64 = 64 * 1_048_576,
              retainedGrowthReservePercent: UInt64 = 10, retainedCeiling: Int,
              processorCount: Int, isHybridOrRecurrent: Bool = false) {
    self.explicitMax = explicitMax
    self.physicalMemoryBytes = physicalMemoryBytes
    self.startupAvailableMemoryBytes = startupAvailableMemoryBytes
    self.modelActiveBytes = modelActiveBytes
    self.retainedBudgetBytes = retainedBudgetBytes
    self.retainedBytes = retainedBytes
    self.recentRetainedGrowthBytes = recentRetainedGrowthBytes
    self.retainedGrowthMinimumBytes = retainedGrowthMinimumBytes
    self.retainedGrowthReservePercent = min(100, retainedGrowthReservePercent)
    self.retainedCeiling = retainedCeiling
    self.processorCount = processorCount
    self.isHybridOrRecurrent = isHybridOrRecurrent
  }
}

public struct ActiveConcurrencyDecision: Equatable, Sendable {
  public let mode: String
  public let selectedMax: Int
  public let backendCeiling: Int
  public let hardwareCeiling: Int
  public let modelCeiling: Int
  public let memoryCeiling: Int
  public let reason: String
  public let osReserveBytes: UInt64
  public let retainedBytes: UInt64
  public let retainedGrowthReserveBytes: UInt64
  public let perActiveReserveBytes: UInt64
  public let usableRuntimeBytes: UInt64
}

public enum ActiveConcurrencyPolicy {
  private static let gib: UInt64 = 1_073_741_824
  private static let mib: UInt64 = 1_048_576

  public static func selectMLX(_ input: ActiveConcurrencyInputs) -> ActiveConcurrencyDecision {
    let backendCeiling = 16
    let modelCeiling = input.isHybridOrRecurrent ? 2 : backendCeiling
    let hardwareCeiling: Int
    switch max(1, input.processorCount) {
    case 1...4: hardwareCeiling = 2
    case 5...8: hardwareCeiling = 4
    case 9...12: hardwareCeiling = 8
    default: hardwareCeiling = backendCeiling
    }

    let osReserve = min(input.physicalMemoryBytes,
      max(2 * gib, input.physicalMemoryBytes / 10))
    let startupAvailable = input.startupAvailableMemoryBytes ??
      input.physicalMemoryBytes.subtractingReportingOverflow(osReserve).partialValue
    let modelActive = input.modelActiveBytes ?? input.physicalMemoryBytes / 2
    let retained = min(input.retainedBytes, input.retainedBudgetBytes)
    let remainingRetainedBudget = input.retainedBudgetBytes - retained
    let fractionalReserve = remainingRetainedBudget / 100 * input.retainedGrowthReservePercent
    let growthReserve = min(remainingRetainedBudget, max(input.recentRetainedGrowthBytes,
      input.retainedGrowthMinimumBytes, fractionalReserve))
    let committed = saturatingAdd(modelActive,
      saturatingAdd(retained, saturatingAdd(growthReserve, osReserve)))
    let usable = startupAvailable > committed ? startupAvailable - committed : 0
    let perActive = max(256 * mib, modelActive / 16)
    let measuredMemoryCeiling = max(1,
      min(backendCeiling, Int(usable / max(1, perActive))))
    let memoryCeiling = input.startupAvailableMemoryBytes == nil || input.modelActiveBytes == nil
      ? min(2, measuredMemoryCeiling) : measuredMemoryCeiling
    let safeCeiling = max(1, min(backendCeiling, hardwareCeiling, modelCeiling,
      memoryCeiling, max(1, input.retainedCeiling)))
    let requested = input.explicitMax.flatMap { $0 > 0 ? $0 : nil }
    let selected = requested.map { min($0, safeCeiling) } ?? min(8, safeCeiling)
    let reason = requested == nil ? limitingReason(selected: selected,
      ceilings: [("memory", memoryCeiling), ("model", modelCeiling),
        ("hardware", hardwareCeiling), ("retained", max(1, input.retainedCeiling))],
      fallback: "bounded_backend_default")
      : selected == requested ? "explicit_override" : "explicit_override_clamped_to_safe_ceiling"
    return ActiveConcurrencyDecision(mode: requested == nil ? "auto" : "fixed",
      selectedMax: max(1, selected), backendCeiling: backendCeiling,
      hardwareCeiling: hardwareCeiling, modelCeiling: modelCeiling,
      memoryCeiling: memoryCeiling, reason: reason, osReserveBytes: osReserve,
      retainedBytes: retained, retainedGrowthReserveBytes: growthReserve,
      perActiveReserveBytes: perActive, usableRuntimeBytes: usable)
  }

  private static func saturatingAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
    let result = lhs.addingReportingOverflow(rhs)
    return result.overflow ? UInt64.max : result.partialValue
  }

  private static func limitingReason(selected: Int, ceilings: [(String, Int)],
                                     fallback: String) -> String {
    ceilings.first { $0.1 == selected }?.0.appending("_ceiling") ?? fallback
  }
}
