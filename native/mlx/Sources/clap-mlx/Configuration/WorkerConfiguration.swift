import ClapCachePolicy
import Darwin
import Foundation

struct CheckpointConfiguration {
  let enabled: Bool
  let minimumTokens: Int
  let intervalTokens: Int
  let maximum: Int
  let coordinatorMinimumTokens: UInt64
  let coordinatorIntervalTokens: UInt64
  let coordinatorMaximum: UInt32
  let budgetBasisPoints: UInt32
  let budgetBytes: UInt64

  func offsets(promptTokens: Int) -> [Int] {
    guard enabled, promptTokens >= minimumTokens, promptTokens > 1,
          intervalTokens > 0, maximum > 0 else { return [] }
    let reusable = promptTokens - 1
    let multiplier = max(1, ((reusable / intervalTokens) + maximum - 1) / maximum)
    let interval = intervalTokens * multiplier
    return Array(stride(from: interval, through: reusable, by: interval).prefix(maximum))
  }
}

struct WorkerConfiguration {
  let physicalMemoryBytes: UInt64
  let availableMemoryAtStartup: UInt64?
  let processorCount: Int
  let retention: RetentionConfiguration
  let retainedGrowthMinimumBytes: UInt64
  let retainedGrowthReservePercent: UInt64
  let contextOverride: Int
  let sessionCap: Int
  let kvBits: Int?
  let outputOverride: Int
  let explicitMaxActive: Int?
  let checkpoints: CheckpointConfiguration
  let debugPrompt: Bool

  init(environment: [String: String], physicalMemoryBytes: UInt64,
       startupAvailableMemoryBytes: UInt64?, processorCount: Int) {
    self.physicalMemoryBytes = physicalMemoryBytes
    availableMemoryAtStartup = startupAvailableMemoryBytes
    self.processorCount = processorCount
    retention = RetentionConfiguration.fromEnvironment(environment,
      physicalMemoryBytes: physicalMemoryBytes,
      startupAvailableMemoryBytes: startupAvailableMemoryBytes)
    retainedGrowthMinimumBytes = UInt64(environment["CLAP_MLX_RETAINED_GROWTH_MIN_BYTES"] ?? "")
      ?? 64 * 1_048_576
    retainedGrowthReservePercent = UInt64(environment["CLAP_MLX_RETAINED_GROWTH_RESERVE_PERCENT"] ?? "")
      ?? 10
    contextOverride = Int(environment["CLAP_MLX_CONTEXT"] ?? "") ?? 0
    sessionCap = Int(environment["CLAP_MLX_MAX_SESSION_CTX"] ?? "") ?? 0
    kvBits = switch environment["CLAP_MLX_KV_TYPE"] ?? "f16" {
    case "q8_0": 8
    case "q4_0": 4
    default: nil
    }
    outputOverride = Int(environment["CLAP_MLX_MAX_OUTPUT"] ?? "") ?? 0
    explicitMaxActive = Int(environment["CLAP_MAX_ACTIVE"] ?? "")
    checkpoints = CheckpointConfiguration(
      enabled: environment["CLAP_CACHE_CHECKPOINTS_ENABLED"] != "0",
      minimumTokens: Int(environment["CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS"] ?? "") ?? 2_048,
      intervalTokens: Int(environment["CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS"] ?? "") ?? 2_048,
      maximum: Int(environment["CLAP_CACHE_CHECKPOINT_MAX"] ?? "") ?? 8,
      coordinatorMinimumTokens: UInt64(environment["CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS"] ?? "") ?? 2_048,
      coordinatorIntervalTokens: UInt64(environment["CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS"] ?? "") ?? 2_048,
      coordinatorMaximum: UInt32(environment["CLAP_CACHE_CHECKPOINT_MAX"] ?? "") ?? 8,
      budgetBasisPoints: UInt32(environment["CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS"] ?? "") ?? 2_500,
      budgetBytes: UInt64(environment["CLAP_CACHE_CHECKPOINT_BUDGET_BYTES"] ?? "") ?? 0)
    debugPrompt = environment["CLAP_MLX_DEBUG_PROMPT"] != nil
  }

  static func current() -> WorkerConfiguration {
    WorkerConfiguration(environment: ProcessInfo.processInfo.environment,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
      startupAvailableMemoryBytes: startupAvailableMemoryBytes(),
      processorCount: ProcessInfo.processInfo.processorCount)
  }
}

private func startupAvailableMemoryBytes() -> UInt64? {
  var statistics = vm_statistics64()
  var count = mach_msg_type_number_t(
    MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
  let status = withUnsafeMutablePointer(to: &statistics) { pointer in
    pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
    }
  }
  var pageSize: vm_size_t = 0
  guard status == KERN_SUCCESS,
        host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return nil }
  let pages = UInt64(statistics.free_count) + UInt64(statistics.inactive_count) +
    UInt64(statistics.purgeable_count)
  let bytes = pages.multipliedReportingOverflow(by: UInt64(pageSize))
  return bytes.overflow ? nil : bytes.partialValue
}
