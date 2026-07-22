import ClapCachePolicy
import ClapMLXCache
import ClapMLXModel
import Foundation
import MLX
import MLXLMCommon

final class WorkerState {
  typealias KVSlot = CacheSlot<KVCache>

  let modelRuntime = ModelRuntime()
  let configuration: WorkerConfiguration
  let physicalMemoryBytes: UInt64
  let availableMemoryAtStartup: UInt64?
  let retentionConfig: RetentionConfiguration
  let retainedGrowthMinimumBytes: UInt64
  let retainedGrowthReservePercent: UInt64
  let contextOverride: Int
  let sessionCap: Int
  let kvBits: Int?

  var activePolicy: ActiveConcurrencyDecision
  var maxActive: Int
  var retainedRegistry: RetainedRegistry<KVSlot>
  var kvUseCounter: UInt64 = 0
  var cacheCoordinator: CacheCoordinator?
  var cacheDomain = ""
  var lastEvictionReason: String?
  var previousMaxActive: Int?
  var coordinatedLimitingReason: String?
  var lastAdjustmentReason: String?
  var lastAdjustmentAt: String?
  var coordinatedGrowthReserveBytes: UInt64?
  var globalResidentMemoryBytes: UInt64?
  var pressureState: String?
  var activePolicyModelBytes: UInt64?
  var allocatorNeedsIdleClear = false

  init(configuration: WorkerConfiguration = .current()) {
    self.configuration = configuration
    physicalMemoryBytes = configuration.physicalMemoryBytes
    availableMemoryAtStartup = configuration.availableMemoryAtStartup
    retentionConfig = configuration.retention
    retainedGrowthMinimumBytes = configuration.retainedGrowthMinimumBytes
    retainedGrowthReservePercent = configuration.retainedGrowthReservePercent
    contextOverride = configuration.contextOverride
    sessionCap = configuration.sessionCap
    kvBits = configuration.kvBits
    activePolicy = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: configuration.physicalMemoryBytes,
      startupAvailableMemoryBytes: configuration.availableMemoryAtStartup,
      modelActiveBytes: nil,
      retainedBudgetBytes: configuration.retention.physicalByteBudget,
      retainedGrowthMinimumBytes: configuration.retainedGrowthMinimumBytes,
      retainedGrowthReservePercent: configuration.retainedGrowthReservePercent,
      retainedCeiling: configuration.retention.hardCeiling,
      processorCount: configuration.processorCount))
    maxActive = activePolicy.selectedMax
    retainedRegistry = RetainedRegistry(maxActive: activePolicy.selectedMax,
      hardCeiling: configuration.retention.hardCeiling)
  }

  var kvSlots: [KVSlot] {
    retainedRegistry.slotIDs.compactMap { retainedRegistry.entry(for: $0) }
  }

  func cacheOperations(create: @escaping () throws -> [KVCache] = { [] })
    -> CacheOperations<KVCache> {
    mlxCacheOperations(create: create, log: debugLog)
  }

  func invalidateKVCache() {
    CacheExecutor.reset(coordinator: &cacheCoordinator, registry: &retainedRegistry,
      maxActive: maxActive, hardCeiling: retentionConfig.hardCeiling,
      useCounter: &kvUseCounter)
    lastEvictionReason = nil
  }

  func retentionSnapshot(queued: Int = 0) -> WorkerRetention? {
    guard let coordinator = cacheCoordinator,
          let telemetry = try? coordinator.retentionTelemetry() else { return nil }
    return workerRetention(RetentionTelemetryFacts(telemetry: telemetry,
      configuration: configuration, activePolicy: activePolicy,
      maxActive: maxActive, queued: queued, previousMaxActive: previousMaxActive,
      limitingReason: coordinatedLimitingReason,
      lastAdjustmentReason: lastAdjustmentReason, lastAdjustmentAt: lastAdjustmentAt,
      coordinatedGrowthReserveBytes: coordinatedGrowthReserveBytes,
      globalResidentMemoryBytes: globalResidentMemoryBytes, pressureState: pressureState,
      modelActiveBytes: activePolicyModelBytes,
      hybridOrRecurrent: modelRuntime.tokenCapabilities.hybridOrRecurrent,
      activeCount: retainedRegistry.activeCount, lastEvictionReason: lastEvictionReason))
  }

  func loadModel(_ model: String, directory: URL) async throws {
    invalidateKVCache()
    try await modelRuntime.load(identifier: model, directory: directory,
      contextOverride: contextOverride, sessionCap: sessionCap,
      outputOverride: configuration.outputOverride)
    let metadata = modelRuntime.metadata!
    cacheDomain = "\(model)|mlx|ctx=\(modelRuntime.tokenCapabilities.effectiveContextLength)|kv=\(kvBits.map(String.init) ?? "f16")|layout=1"
    Memory.clearCache()
    let memory = memorySnapshot()
    activePolicyModelBytes = memory.active_bytes > 0 ? UInt64(memory.active_bytes) : nil
    activePolicy = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      explicitMax: configuration.explicitMaxActive,
      physicalMemoryBytes: physicalMemoryBytes,
      startupAvailableMemoryBytes: availableMemoryAtStartup,
      modelActiveBytes: activePolicyModelBytes,
      retainedBudgetBytes: retentionConfig.physicalByteBudget,
      retainedGrowthMinimumBytes: retainedGrowthMinimumBytes,
      retainedGrowthReservePercent: retainedGrowthReservePercent,
      retainedCeiling: retentionConfig.hardCeiling,
      processorCount: configuration.processorCount,
      isHybridOrRecurrent: modelRuntime.tokenCapabilities.hybridOrRecurrent))
    maxActive = activePolicy.selectedMax
    do {
      let initialized: (CacheCoordinator, RetainedRegistry<KVSlot>) = try CacheExecutor.initialize(
        retention: retentionConfig, maxActive: maxActive, capacity: Int.max / 4,
        checkpoints: configuration.checkpoints.coordinatorConfiguration)
      cacheCoordinator = initialized.0
      retainedRegistry = initialized.1
    } catch {
      cacheCoordinator = nil
      debugLog("cache coordinator unavailable; cache admission fails closed: \(error)")
    }
    if kvBits != nil { debugLog("kv cache quantization enabled: \(kvBits!)-bit") }
    debugLog("declared metadata: architecture=\(metadata.architecture ?? "unknown") model_type=\(metadata.modelType ?? "unknown") context_source=\(modelRuntime.tokenCapabilities.contextLengthSource ?? "unknown") sliding_window=\(metadata.slidingWindow?.value.description ?? "unknown") output_source=\(modelRuntime.tokenCapabilities.maxOutputTokensSource ?? "unknown")")
    debugLog("context length: \(modelRuntime.tokenCapabilities.effectiveContextLength > 0 ? String(modelRuntime.tokenCapabilities.effectiveContextLength) : "unknown")\(sessionCap > 0 ? ", session cap \(sessionCap)" : "")")
    debugLog("model loaded; eos token ids: \(modelRuntime.eosTokenIds.sorted())")
    debugLog("active concurrency: mode=\(activePolicy.mode) selected=\(activePolicy.selectedMax) reason=\(activePolicy.reason) memory_ceiling=\(activePolicy.memoryCeiling)")
    debugLog("mlx memory after load: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
  }

  func updateMaxActive(_ requested: Int, control: ControlRequest) {
    let old = maxActive
    maxActive = min(requested, activePolicy.backendCeiling, activePolicy.hardwareCeiling,
      activePolicy.modelCeiling, max(1, retentionConfig.hardCeiling))
    previousMaxActive = control.previous_max_active ?? old
    coordinatedLimitingReason = control.limiting_reason
    lastAdjustmentReason = control.last_adjustment_reason
    lastAdjustmentAt = control.last_adjustment_at
    coordinatedGrowthReserveBytes = control.retained_growth_reserve_bytes
    globalResidentMemoryBytes = control.global_resident_memory_bytes
    pressureState = control.pressure_state
    retainedRegistry.updateMaxActive(maxActive)
  }

  func clearAllocatorIfIdle(activeEmpty: Bool, pendingEmpty: Bool) {
    guard activeEmpty, pendingEmpty, allocatorNeedsIdleClear else { return }
    Memory.clearCache()
    let memory = memorySnapshot()
    debugLog("mlx memory after idle clear: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
    emit(memory: memory, retention: retentionSnapshot())
    allocatorNeedsIdleClear = false
  }
}
