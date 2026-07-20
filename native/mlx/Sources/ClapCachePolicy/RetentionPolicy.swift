import Foundation

public struct RetentionConfiguration: Equatable, Sendable {
  public let initialEntries: Int
  public let hardCeiling: Int
  public let physicalByteBudget: UInt64
  public let highWatermarkBytes: UInt64
  public let lowWatermarkBytes: UInt64

  public init(initialEntries: Int, hardCeiling: Int, physicalByteBudget: UInt64,
              highWatermarkBytes: UInt64, lowWatermarkBytes: UInt64) {
    self.initialEntries = initialEntries
    self.hardCeiling = hardCeiling
    self.physicalByteBudget = physicalByteBudget
    self.highWatermarkBytes = highWatermarkBytes
    self.lowWatermarkBytes = lowWatermarkBytes
  }

  public static func fromEnvironment(_ environment: [String: String],
                                     physicalMemoryBytes: UInt64,
                                     startupAvailableMemoryBytes: UInt64? = nil) -> RetentionConfiguration {
    func positiveInt(_ names: [String], default fallback: Int) -> Int {
      for name in names {
        if let text = environment[name], let value = Int(text), value > 0 { return value }
      }
      return fallback
    }
    func positiveDouble(_ name: String) -> Double? {
      guard let text = environment[name], let value = Double(text), value > 0 else { return nil }
      return value
    }

    let initial = positiveInt(["CLAP_MLX_RETAINED_INITIAL"], default: 4)

    let budget: UInt64
    if let bytes = environment["CLAP_MLX_RETAINED_BUDGET_BYTES"].flatMap(UInt64.init), bytes > 0 {
      budget = bytes
    } else if let gib = positiveDouble("CLAP_MLX_RETAINED_BUDGET_GIB") {
      budget = UInt64(min(gib * 1_073_741_824, Double(UInt64.max)))
    } else if let percent = positiveDouble("CLAP_MLX_RETAINED_BUDGET_PERCENT") {
      budget = UInt64(Double(physicalMemoryBytes) * min(percent, 100) / 100)
    } else {
      // This is a conservative startup allowance, not a live runtime budget.
      // Reserve at least 8 GiB and 25% of physical memory for the OS, model,
      // and runtime, then clamp against the memory available before model load.
      let gib: UInt64 = 1_073_741_824
      let proportionalReserve = physicalMemoryBytes / 4
      let reserve = min(physicalMemoryBytes, max(8 * gib, proportionalReserve))
      let availableAfterReserve = physicalMemoryBytes - reserve
      let physicalBudget = min(availableAfterReserve / 4, physicalMemoryBytes / 5)
      let headroomBudget = startupAvailableMemoryBytes.map { $0 / 8 } ?? physicalBudget
      budget = min(physicalBudget, headroomBudget)
    }
    let defaultCeiling = budget > 0 ? 256 : initial
    let ceiling = max(initial, positiveInt(["CLAP_MLX_RETAINED_MAX"], default: defaultCeiling))

    guard budget > 0 else {
      return RetentionConfiguration(initialEntries: initial, hardCeiling: ceiling,
        physicalByteBudget: 0, highWatermarkBytes: 0, lowWatermarkBytes: 0)
    }
    func watermark(bytesName: String, percentName: String, defaultPercent: Double) -> UInt64 {
      if let bytes = environment[bytesName].flatMap(UInt64.init), bytes > 0 {
        return min(bytes, budget)
      }
      let percent = positiveDouble(percentName) ?? defaultPercent
      return UInt64(Double(budget) * min(percent, 100) / 100)
    }
    let high = watermark(bytesName: "CLAP_MLX_RETAINED_HIGH_BYTES",
      percentName: "CLAP_MLX_RETAINED_HIGH_PERCENT", defaultPercent: 90)
    let low = min(high, watermark(bytesName: "CLAP_MLX_RETAINED_LOW_BYTES",
      percentName: "CLAP_MLX_RETAINED_LOW_PERCENT", defaultPercent: 75))
    return RetentionConfiguration(initialEntries: initial, hardCeiling: ceiling,
      physicalByteBudget: budget, highWatermarkBytes: high, lowWatermarkBytes: low)
  }
}

public enum RetainedRegistryError: Error, Equatable {
  case hardCeiling
  case duplicateSlot
  case unknownSlot
  case maxActive
}

public final class RetainedRegistry<Entry: AnyObject> {
  public private(set) var maxActive: Int
  public let hardCeiling: Int
  private var entries: [UInt32: Entry] = [:]
  private var orderedSlotIDs: [UInt32] = []
  private var activeSlotIDs: Set<UInt32> = []

  public init(maxActive: Int, hardCeiling: Int) {
    self.maxActive = max(1, maxActive)
    self.hardCeiling = max(1, hardCeiling)
  }

  public var count: Int { entries.count }
  public var activeCount: Int { activeSlotIDs.count }
  public var slotIDs: [UInt32] { orderedSlotIDs }

  public func updateMaxActive(_ value: Int) {
    maxActive = max(1, value)
  }

  public func register(slotID: UInt32, entry: Entry) throws {
    guard entries.count < hardCeiling else { throw RetainedRegistryError.hardCeiling }
    guard entries[slotID] == nil else { throw RetainedRegistryError.duplicateSlot }
    entries[slotID] = entry
    orderedSlotIDs.append(slotID)
  }

  public func entry(for slotID: UInt32) -> Entry? { entries[slotID] }

  public func activate(slotID: UInt32) throws {
    guard entries[slotID] != nil else { throw RetainedRegistryError.unknownSlot }
    if activeSlotIDs.contains(slotID) { return }
    guard activeSlotIDs.count < maxActive else { throw RetainedRegistryError.maxActive }
    activeSlotIDs.insert(slotID)
  }

  public func release(slotID: UInt32) {
    activeSlotIDs.remove(slotID)
  }

  public func isActive(slotID: UInt32) -> Bool { activeSlotIDs.contains(slotID) }

  public func validateEvictions(_ slotIDs: [UInt32]) throws {
    for slotID in slotIDs {
      guard entries[slotID] != nil else { throw RetainedRegistryError.unknownSlot }
      guard !activeSlotIDs.contains(slotID) else { throw RetainedRegistryError.maxActive }
    }
  }

  public func reconcileEvictions(_ slotIDs: [UInt32],
                                 clear: (UInt32, Entry) -> Void) {
    for slotID in slotIDs {
      guard let entry = entries[slotID] else { continue }
      clear(slotID, entry)
    }
  }

  public func removeAll() {
    activeSlotIDs.removeAll()
    orderedSlotIDs.removeAll()
    entries.removeAll()
  }
}

public struct CacheArrayDescriptor: Sendable {
  public let storageIdentity: UInt64?
  public let elementCount: Int
  public let itemSize: Int
  public let allocatedBytes: Int?

  public init(storageIdentity: UInt64? = nil, elementCount: Int, itemSize: Int,
              allocatedBytes: Int? = nil) {
    self.storageIdentity = storageIdentity
    self.elementCount = elementCount
    self.itemSize = itemSize
    self.allocatedBytes = allocatedBytes
  }
}

public enum PhysicalCacheByteEstimator {
  public static func estimate(arrays: [CacheArrayDescriptor]) -> UInt64 {
    var seen: Set<UInt64> = []
    var total: UInt64 = 0
    for array in arrays {
      if let identity = array.storageIdentity, !seen.insert(identity).inserted { continue }
      let logical = max(0, array.elementCount).multipliedReportingOverflow(by: max(0, array.itemSize))
      let logicalBytes = logical.overflow ? Int.max : logical.partialValue
      let bytes = max(0, array.allocatedBytes ?? logicalBytes)
      total = total.addingReportingOverflow(UInt64(bytes)).overflow
        ? UInt64.max : total.addingReportingOverflow(UInt64(bytes)).partialValue
    }
    return total
  }
}
