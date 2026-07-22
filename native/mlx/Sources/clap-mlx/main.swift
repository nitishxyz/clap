import Foundation
import Darwin
import ClapCacheBridge
import ClapCachePolicy
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

func tokenFingerprint(_ tokens: [Int], count: Int) -> String {
  let limit = min(max(count, 0), tokens.count)
  var bytes = Array((cacheTelemetryKey + "|tokens-v1|\(limit)|").utf8)
  for token in tokens.prefix(limit) {
    var value = UInt32(truncatingIfNeeded: token).littleEndian
    withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
  }
  return (0..<4).map { domain -> String in
    var hash = UInt64(1_469_598_103_934_665_603) ^ UInt64(domain)
    for byte in bytes {
      hash ^= UInt64(byte)
      hash &*= 1_099_511_628_211
    }
    return String(format: "%016llx", hash)
  }.joined()
}

func cacheCandidateState(_ state: UInt32) -> String {
  switch state {
  case UInt32(CC_SLOT_SESSION): return "session"
  case UInt32(CC_SLOT_PROMPT_BOUNDARY): return "prompt_boundary"
  case UInt32(CC_SLOT_ANCHOR): return "anchor"
  default: return "empty"
  }
}

func cacheCandidateRejection(_ rejection: UInt32) -> String? {
  [1: "namespace", 2: "model_domain", 3: "generation", 4: "busy_lease",
   5: "materialization", 6: "session", 7: "nontrim", 8: "capability",
   9: "min_prefix", 10: "capacity", 11: "absent_anchor", 12: "lower_rank"][rejection]
}

func memorySnapshot() -> WorkerMemory {
  let snapshot = Memory.snapshot()
  return WorkerMemory(active_bytes: snapshot.activeMemory, cache_bytes: snapshot.cacheMemory, peak_active_bytes: snapshot.peakMemory)
}

func debugLog(_ message: String) {
  FileHandle.standardError.write(Data("[clap-mlx] \(message)\n".utf8))
}

func validateModelDirectory(_ path: String) throws -> URL {
  let url = URL(fileURLWithPath: path)
  var isDirectory: ObjCBool = false
  guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
    throw WorkerError.invalidModelDirectory("MLX model directory not found: \(path)")
  }
  guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
    throw WorkerError.invalidModelDirectory("MLX model directory is missing config.json: \(path)")
  }
  return url
}

// ModelContext members (model, tokenizer) are not Sendable but are only used
// from this single-threaded main loop; ChatSession uses the same pattern.
struct UncheckedBox<T>: @unchecked Sendable {
  let value: T
}

func declaredEosTokenIds(_ url: URL) -> Set<Int> {
  guard let data = try? Data(contentsOf: url),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
  var ids: Set<Int> = []
  func collect(_ value: Any?) {
    if let id = value as? Int { ids.insert(id) }
    if let list = value as? [Any] { for item in list { collect(item) } }
  }
  collect(json["eos_token_id"])
  if let textConfig = json["text_config"] as? [String: Any] { collect(textConfig["eos_token_id"]) }
  return ids
}

func main() async {
    guard #available(macOS 14.0, *) else {
      emit(error: "clap-mlx requires macOS 14 or newer on Apple Silicon")
      exit(2)
    }
    #if !arch(arm64)
    emit(error: "clap-mlx requires Apple Silicon arm64")
    exit(2)
    #endif

    var loadedModel: String?
    var loadedDirectory: URL?
    var languageModel: (any LanguageModel)?
    var tokenizer: (any MLXLMCommon.Tokenizer)?
    var eosTokenIds: Set<Int> = []
    var declaredModelContextLength = 0
    var modelContextLength = 0
    var modelMaxOutputTokens = 0
    var modelContextLengthSource: String?
    var modelMaxOutputTokensSource: String?

    let configuration = WorkerConfiguration.current()
    let physicalMemoryBytes = configuration.physicalMemoryBytes
    let availableMemoryAtStartup = configuration.availableMemoryAtStartup
    let retentionConfig = configuration.retention
    let retainedGrowthMinimumBytes = configuration.retainedGrowthMinimumBytes
    let retainedGrowthReservePercent = configuration.retainedGrowthReservePercent
    var activePolicy = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: physicalMemoryBytes,
      startupAvailableMemoryBytes: availableMemoryAtStartup, modelActiveBytes: nil,
      retainedBudgetBytes: retentionConfig.physicalByteBudget,
      retainedGrowthMinimumBytes: retainedGrowthMinimumBytes,
      retainedGrowthReservePercent: retainedGrowthReservePercent,
      retainedCeiling: retentionConfig.hardCeiling,
      processorCount: configuration.processorCount))
    var maxActive = activePolicy.selectedMax
    // Context window policy (parity with the llama worker): default to the
    // model's trained context; CLAP_MLX_CONTEXT pins it, and
    // CLAP_MLX_MAX_SESSION_CTX caps any single session's share.
    let contextOverride = configuration.contextOverride
    let sessionCap = configuration.sessionCap
    // KV cache quantization (parity with CLAP_LLAMA_KV_TYPE): q8_0/q4_0/f16.
    let kvBits = configuration.kvBits

    // Physical KV slots mirror Rust coordinator state. The worker executes
    // authorized operations but does not independently choose cache policy.
    final class KVSlot {
      var caches: [KVCache] = []
      var tokens: [Int] = []
      var lastUsed: UInt64 = 0
      var busy = false
      // Anchor slots hold an exact-state snapshot of a shared prefix (e.g.
      // the org-wide system prompt) taken before rotating/sliding-window
      // caches rotate it away. They are only used via copy() (whole-copy
      // branching) and never extended in place.
      var isAnchor = false
      var anchorScope: String? = nil
      // A request-local end-of-prompt snapshot restored after decode. Unlike
      // a dedicated prefix anchor, this remains the session's normal slot.
      var isPromptBoundary = false
      var coordinatorGeneration: UInt64 = 0
      var cacheIdentity: PhysicalCacheIdentity? = nil
    }
    var retainedRegistry = RetainedRegistry<KVSlot>(maxActive: maxActive,
      hardCeiling: retentionConfig.hardCeiling)
    var kvSlots: [KVSlot] {
      retainedRegistry.slotIDs.compactMap { retainedRegistry.entry(for: $0) }
    }
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
    var activePolicyHybrid = false
    func invalidateKVCache() {
      try? cacheCoordinator?.reset()
      cacheCoordinator = nil
      retainedRegistry = RetainedRegistry<KVSlot>(maxActive: maxActive,
        hardCeiling: retentionConfig.hardCeiling)
      kvUseCounter = 0
      lastEvictionReason = nil
    }

    func clearPhysicalSlot(_ slot: KVSlot) {
      slot.caches = []
      slot.tokens = []
      slot.isAnchor = false
      slot.isPromptBoundary = false
      slot.anchorScope = nil
      slot.cacheIdentity = nil
      slot.coordinatorGeneration = 0
    }

    func physicalCacheBytes(_ caches: [KVCache]) -> UInt64 {
      let arrays = caches.flatMap(\.state).map {
        CacheArrayDescriptor(storageIdentity: UInt64(bitPattern:
          Int64(ObjectIdentifier($0).hashValue)),
          elementCount: $0.size, itemSize: $0.itemSize,
          allocatedBytes: $0.nbytes)
      }
      return max(1, PhysicalCacheByteEstimator.estimate(arrays: arrays))
    }

    func retentionSnapshot(queued: Int = 0) -> WorkerRetention? {
      guard let coordinator = cacheCoordinator,
            let telemetry = try? coordinator.retentionTelemetry() else { return nil }
      let retainedBytes = min(telemetry.total_bytes, retentionConfig.physicalByteBudget)
      let remainingBudget = retentionConfig.physicalByteBudget - retainedBytes
      let growthReserve = min(remainingBudget, max(retainedGrowthMinimumBytes,
        remainingBudget / 100 * min(100, retainedGrowthReservePercent)))
      let policy = WorkerActivePolicy(mode: activePolicy.mode,
        selected_max: maxActive, backend_ceiling: activePolicy.backendCeiling,
        hardware_ceiling: activePolicy.hardwareCeiling, model_ceiling: activePolicy.modelCeiling,
        memory_ceiling: activePolicy.memoryCeiling,
        reason: coordinatedLimitingReason ?? activePolicy.reason,
        inputs: WorkerActivePolicyInputs(physical_memory_bytes: physicalMemoryBytes,
          startup_available_bytes: availableMemoryAtStartup,
          model_active_bytes: activePolicyModelBytes,
          retained_budget_bytes: retentionConfig.physicalByteBudget,
          retained_bytes: retainedBytes,
          retained_growth_reserve_bytes: growthReserve,
          os_reserve_bytes: activePolicy.osReserveBytes,
          usable_runtime_bytes: activePolicy.usableRuntimeBytes,
          per_active_reserve_bytes: activePolicy.perActiveReserveBytes,
          processor_count: configuration.processorCount,
          hybrid_or_recurrent: activePolicyHybrid))
      return WorkerRetention(max_active: maxActive, queued: queued,
        previous_max_active: previousMaxActive,
        last_adjustment_reason: lastAdjustmentReason,
        last_adjustment_at: lastAdjustmentAt,
        retained_growth_reserve_bytes: coordinatedGrowthReserveBytes ?? growthReserve,
        global_resident_memory_bytes: globalResidentMemoryBytes,
        pressure_state: pressureState, active_policy: policy,
        active: retainedRegistry.activeCount,
        retained_total: Int(telemetry.total_slots), retained_sessions: Int(telemetry.session_slots),
        retained_anchors: Int(telemetry.anchor_slots), retained_bytes: telemetry.total_bytes,
        session_bytes: telemetry.session_bytes, anchor_bytes: telemetry.anchor_bytes,
        automatic_checkpoint_count: Int(telemetry.automatic_checkpoint_slots),
        automatic_checkpoint_bytes: telemetry.automatic_checkpoint_bytes,
        automatic_checkpoint_budget_bytes: telemetry.automatic_checkpoint_byte_budget,
        automatic_checkpoints_enabled: configuration.checkpoints.enabled,
        automatic_checkpoint_minimum_tokens: configuration.checkpoints.minimumTokens,
        automatic_checkpoint_interval_tokens: configuration.checkpoints.intervalTokens,
        automatic_checkpoint_max: configuration.checkpoints.maximum,
        budget_bytes: telemetry.physical_byte_budget,
        high_watermark_bytes: telemetry.high_watermark_bytes,
        low_watermark_bytes: telemetry.low_watermark_bytes,
        under_pressure: telemetry.under_pressure != 0,
        hard_ceiling: retentionConfig.hardCeiling, eviction_reason: lastEvictionReason,
        eviction_count: telemetry.evictions)
    }

    func loadModel(_ model: String, directory: URL) async throws {
      invalidateKVCache()
      let container = try await loadModelContainer(from: directory, using: #huggingFaceTokenizerLoader())
      let box = await container.perform { context in
        UncheckedBox(value: (context.model, context.tokenizer, context.configuration.extraEOSTokens))
      }
      languageModel = box.value.0
      tokenizer = box.value.1
      loadedModel = model
      loadedDirectory = directory
      eosTokenIds = []
      if let eos = box.value.1.eosTokenId { eosTokenIds.insert(eos) }
      for extra in box.value.2 {
        if let id = box.value.1.convertTokenToId(extra) { eosTokenIds.insert(id) }
      }
      // HF checkpoints declare turn-ending tokens (e.g. gemma's <turn|>) in
      // config.json / generation_config.json eos_token_id, not the tokenizer.
      for file in ["generation_config.json", "config.json"] {
        eosTokenIds.formUnion(declaredEosTokenIds(directory.appendingPathComponent(file)))
      }
      let metadata = DeclaredModelMetadata.load(from: directory)
      declaredModelContextLength = metadata.context?.value ?? 0
      modelContextLengthSource = metadata.context?.source
      let knownContextCaps = [declaredModelContextLength, contextOverride, sessionCap].filter { $0 > 0 }
      modelContextLength = knownContextCaps.min() ?? 0
      modelMaxOutputTokens = metadata.maxOutputTokens?.value ?? 0
      modelMaxOutputTokensSource = metadata.maxOutputTokens?.source
      let outputOverride = configuration.outputOverride
      if outputOverride > 0 {
        if modelMaxOutputTokens == 0 || outputOverride < modelMaxOutputTokens {
          modelMaxOutputTokens = outputOverride
          modelMaxOutputTokensSource = "environment:CLAP_MLX_MAX_OUTPUT"
        }
      }
      cacheDomain = "\(model)|mlx|ctx=\(modelContextLength)|kv=\(kvBits.map(String.init) ?? "f16")|layout=1"
      Memory.clearCache()
      let memory = memorySnapshot()
      activePolicyModelBytes = memory.active_bytes > 0 ? UInt64(memory.active_bytes) : nil
      let capabilityText = [metadata.architecture, metadata.modelType]
        .compactMap { $0?.lowercased() }.joined(separator: " ")
      activePolicyHybrid = ["hybrid", "recurrent", "mamba", "deltanet", "ssm"]
        .contains { capabilityText.contains($0) }
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
        isHybridOrRecurrent: activePolicyHybrid))
      maxActive = activePolicy.selectedMax
      retainedRegistry = RetainedRegistry<KVSlot>(maxActive: maxActive,
        hardCeiling: retentionConfig.hardCeiling)
      do {
        cacheCoordinator = try CacheCoordinator(retention: retentionConfig,
          capacity: Int.max / 4, checkpoints: configuration.checkpoints)
        for slotID in 0..<retentionConfig.initialEntries {
          let slot = KVSlot()
          slot.coordinatorGeneration = try cacheCoordinator?.slot(slotID).generation ?? 0
          try retainedRegistry.register(slotID: UInt32(slotID), entry: slot)
        }
      } catch {
        cacheCoordinator = nil
        debugLog("cache coordinator unavailable; cache admission fails closed: \(error)")
      }
      if kvBits != nil { debugLog("kv cache quantization enabled: \(kvBits!)-bit") }
      debugLog("declared metadata: architecture=\(metadata.architecture ?? "unknown") model_type=\(metadata.modelType ?? "unknown") context_source=\(modelContextLengthSource ?? "unknown") sliding_window=\(metadata.slidingWindow?.value.description ?? "unknown") output_source=\(modelMaxOutputTokensSource ?? "unknown")")
      debugLog("context length: \(modelContextLength > 0 ? String(modelContextLength) : "unknown")\(sessionCap > 0 ? ", session cap \(sessionCap)" : "")")
      debugLog("model loaded; eos token ids: \(eosTokenIds.sorted())")
      debugLog("active concurrency: mode=\(activePolicy.mode) selected=\(activePolicy.selectedMax) reason=\(activePolicy.reason) memory_ceiling=\(activePolicy.memoryCeiling)")
      debugLog("mlx memory after load: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
    }

    let buffer = LineBuffer()
    let readerTask = Task.detached {
      do {
        for try await line in FileHandle.standardInput.bytes.lines {
          await buffer.push(String(line))
        }
      } catch {}
      await buffer.finish()
    }
    defer { readerTask.cancel() }
    // ---- Interleaved multi-request scheduler ----------------------------
    // Mirrors the llama.cpp worker's continuous batching at the scheduling
    // level: several requests are active at once, each stepped in round-robin
    // (one prefill chunk OR a few decode tokens per pass), so a long prefill
    // or generation never blocks other sessions' token streams. MLX evaluates
    // sequences one at a time on Metal (no fused multi-sequence batch yet),
    // so aggregate throughput is shared — but head-of-line blocking is gone.

    let decodeStepsPerPass = 6

    final class ActiveRequest {
      struct BoundaryInfo {
        let tokenCount: Int?
        let kind: String
        let label: String?
        let requested: Bool
        let status: String
        let skipReason: String?
      }
      let id: String?
      let admissionOrder: UInt64
      let streaming: Bool
      let maxTokens: Int
      let promptTokens: [Int]
      let reusedTokens: Int
      let reuseKind: String?
      let reuseScope: String?
      let cacheIdentity: CacheIdentity
      let cacheDecision: CacheDecision?
      let cacheCandidates: [CacheCandidateEvaluation]
      let cacheEvictions: [Int]
      let cacheFallback: String?
      let slotIndex: Int
      let slot: KVSlot
      var caches: [KVCache]
      var promptBoundaryCaches: [KVCache]? = nil
      var continuationBoundary: Int? = nil
      var continuationBoundaryCaches: [KVCache]? = nil
      var fedTokens: [Int]
      var suffix: [Int]
      var pos = 0
      var iterator: TokenIterator?
      var detokenizer: NaiveStreamingDetokenizer
      var sampledTokens: [Int] = []
      var collected = ""
      var emitted = 0  // chars of `collected` already streamed (stop holdback)
      var generatedCount = 0
      var finishReason = "stop"
      var cancelled = false
      var completed = false
      var failed = false
      // Exact prompt indices authorized by Rust for physical snapshotting.
      var anchorPlantAt: [Int] = []
      var anchorPlantScopes: [Int: UInt32] = [:]
      var resolvedBoundaries: [Int: BoundaryInfo] = [:]
      var boundaryTelemetry: [BoundaryInfo] = []
      var anchorPlanted: Set<Int> = []
      var materializedAnchors: Set<Int> = []
      var automaticCheckpointProposed = 0
      var automaticCheckpointDeduped = 0
      let admittedNs: UInt64
      let receivedToAdmittedMs: Double
      let templateTokenizeMs: Double
      let coordinatorPlanMs: Double
      let coordinatorApplyMs: Double
      var schedulerWaitMs = 0.0
      var cacheMaterializeMs = 0.0
      var prefillMs = 0.0
      var prefillTokens = 0
      var prefillChunks = 0
      var firstDecodeMs = 0.0
      var firstEmitMs = 0.0
      var lastStepFinishedNs: UInt64
      let parameters: GenerateParameters
      let stops: [String]
      let holdback: Int

      init(id: String?, admissionOrder: UInt64, admittedNs: UInt64, receivedToAdmittedMs: Double, templateTokenizeMs: Double, coordinatorPlanMs: Double, coordinatorApplyMs: Double, cacheMaterializeMs: Double, streaming: Bool, maxTokens: Int, promptTokens: [Int], reusedTokens: Int, reuseKind: String?, reuseScope: String?, cacheIdentity: CacheIdentity, cacheDecision: CacheDecision?, cacheCandidates: [CacheCandidateEvaluation], cacheEvictions: [Int], cacheFallback: String?, slotIndex: Int, slot: KVSlot, caches: [KVCache], fedTokens: [Int], suffix: [Int], detokenizer: NaiveStreamingDetokenizer, parameters: GenerateParameters, stops: [String]) {
        self.id = id
        self.admissionOrder = admissionOrder
        self.admittedNs = admittedNs
        self.receivedToAdmittedMs = receivedToAdmittedMs
        self.templateTokenizeMs = templateTokenizeMs
        self.coordinatorPlanMs = coordinatorPlanMs
        self.coordinatorApplyMs = coordinatorApplyMs
        self.cacheMaterializeMs = cacheMaterializeMs
        self.lastStepFinishedNs = admittedNs
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
        self.slotIndex = slotIndex
        self.slot = slot
        self.caches = caches
        self.fedTokens = fedTokens
        self.suffix = suffix
        self.detokenizer = detokenizer
        self.parameters = parameters
        self.stops = stops
        self.holdback = stops.map { $0.count }.max().map { $0 - 1 } ?? 0
      }
    }

    var active: [ActiveRequest] = []
    var pendingChats: [(id: String?, control: ControlRequest, data: Data, receivedNs: UInt64)] = []
    var controlBacklog: [String] = []
    var allocatorNeedsIdleClear = false
    var nextAdmissionOrder: UInt64 = 0

    // Recurrent caches such as Mamba keep state but intentionally report an
    // offset of zero. Hybrid models still have attention caches whose maximum
    // offset tracks the sequence length; pure recurrent models use the token
    // bookkeeping fallback supplied by the caller.
    func cacheSequenceLength(_ caches: [KVCache], fallback: Int) -> Int {
      let offset = caches.map(\.offset).max() ?? 0
      return offset > 0 ? offset : fallback
    }

    func ensureAdmissionSlot() throws {
      guard !kvSlots.contains(where: { !$0.busy && $0.caches.isEmpty }) else { return }
      guard retainedRegistry.count < retentionConfig.hardCeiling,
            let coordinator = cacheCoordinator else { return }
      let registered = try coordinator.registerSlot()
      let slot = KVSlot()
      slot.coordinatorGeneration = registered.generation
      try retainedRegistry.register(slotID: UInt32(registered.slot), entry: slot)
    }

    func finalize(_ req: ActiveRequest) {
      req.slot.busy = false
      retainedRegistry.release(slotID: UInt32(req.slotIndex))
      if req.failed {
        let generation = req.slot.coordinatorGeneration
        clearPhysicalSlot(req.slot)
        if let coordinator = cacheCoordinator, generation != 0 {
          req.slot.coordinatorGeneration = (try? coordinator.invalidate(
            slot: req.slotIndex, generation: generation)) ?? 0
        }
        return
      }
      if let continuationBoundary = req.continuationBoundary,
         let continuationCaches = req.continuationBoundaryCaches {
        req.slot.caches = continuationCaches
        req.slot.tokens = Array(req.promptTokens.prefix(continuationBoundary))
        req.slot.isPromptBoundary = true
        req.slot.anchorScope = "conversation"
        debugLog("restored rolling conversation anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(continuationBoundary) tokens; discarded \(req.promptTokens.count - continuationBoundary) prompt suffix tokens and \(req.generatedCount) decoded tokens")
      } else if let promptBoundary = req.promptBoundaryCaches {
        req.slot.caches = promptBoundary
        req.slot.tokens = req.promptTokens
        req.slot.isPromptBoundary = true
        req.slot.anchorScope = "conversation"
        debugLog("restored prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens; discarded \(req.generatedCount) decoded tokens")
      } else {
        // The cache offset is authoritative for what is resident; any mismatch
        // here corrupts the next request's prefix trim.
        let full = req.fedTokens + req.sampledTokens
        let cacheLength = cacheSequenceLength(req.caches, fallback: full.count)
        req.slot.tokens = Array(full.prefix(min(cacheLength, full.count)))
        req.slot.isPromptBoundary = false
        req.slot.anchorScope = nil
      }
      if let coordinator = cacheCoordinator, req.slot.coordinatorGeneration != 0 {
        do {
          req.slot.coordinatorGeneration = try coordinator.confirm(
            slot: req.slotIndex, generation: req.slot.coordinatorGeneration,
            tokens: req.slot.tokens,
            state: req.slot.isPromptBoundary ? UInt32(CC_SLOT_PROMPT_BOUNDARY) : UInt32(CC_SLOT_SESSION),
            busy: true, physicalBytes: physicalCacheBytes(req.slot.caches))
          try coordinator.setBusy(slot: req.slotIndex,
                                  generation: req.slot.coordinatorGeneration, busy: false)
        } catch {
          let generation = req.slot.coordinatorGeneration
          if generation != 0 {
            _ = try? coordinator.invalidate(slot: req.slotIndex, generation: generation)
          }
          req.slot.coordinatorGeneration = 0
          debugLog("cache finalize metadata failed: \(error)")
        }
      }
      // Flush any text held back for stop-sequence matching.
      if req.streaming && !req.cancelled && req.emitted < req.collected.count {
        let tail = String(req.collected.dropFirst(req.emitted))
        if !tail.isEmpty { emit(id: req.id, token: tail) }
        req.emitted = req.collected.count
      }
      if !req.streaming && !req.collected.isEmpty && !req.cancelled {
        emit(id: req.id, content: req.collected)
      }
      let usage = WorkerUsage(prompt_tokens: req.promptTokens.count, completion_tokens: req.generatedCount)
      let timing = WorkerTiming(
        received_to_admitted_ms: req.receivedToAdmittedMs,
        template_tokenize_ms: req.templateTokenizeMs,
        coordinator_wait_ms: req.receivedToAdmittedMs,
        coordinator_plan_ms: req.coordinatorPlanMs,
        coordinator_apply_ms: req.coordinatorApplyMs,
        scheduler_wait_ms: req.schedulerWaitMs,
        cache_materialize_ms: req.cacheMaterializeMs,
        prefill_ms: req.prefillMs,
        residual_prefill_tokens: req.promptTokens.count - req.reusedTokens,
        prefill_tokens: req.prefillTokens,
        prefill_chunks: req.prefillChunks,
        first_decode_ms: req.firstDecodeMs,
        first_emit_ms: req.firstEmitMs,
        normal_prefill_quantum: LatencyScheduler.normalPrefillQuantum,
        contended_prefill_quantum: LatencyScheduler.contendedPrefillQuantum)
      let cacheInfo = WorkerCache(
        hit: req.reusedTokens > 0,
        reused_tokens: req.reusedTokens,
        reuse_kind: req.reuseKind,
        reuse_scope: req.reuseScope,
        side_request: req.cacheIdentity.sideRequest,
        namespace: req.cacheIdentity.exportedNamespace,
        donor_slot: req.cacheDecision?.donor,
        target_slot: req.slotIndex,
        evicted_slots: req.cacheEvictions,
        decision_us: req.cacheDecision?.decisionUs ?? 0,
        planned_reuse_tokens: req.cacheDecision?.plannedReuseTokens ?? req.reusedTokens,
        realized_reuse_tokens: req.cacheDecision?.realizedReuseTokens ?? req.reusedTokens,
        fallback: req.cacheFallback,
        miss_reason: req.reusedTokens > 0 ? nil : "no_shared_prefix",
        candidates: req.cacheCandidates.map { candidate in
          WorkerCacheCandidate(
            slot: candidate.slot, generation: candidate.generation,
            state: cacheCandidateState(candidate.state),
            shared_prefix_tokens: candidate.sharedPrefixTokens,
            namespace_compatible: candidate.namespaceCompatible,
            model_compatible: candidate.modelCompatible,
            session_compatible: candidate.sessionCompatible,
            generation_compatible: candidate.generationCompatible,
            busy_eligible: candidate.busyEligible,
            lease_eligible: candidate.leaseEligible,
            materialized: candidate.materialized,
            trim_eligible: candidate.trimEligible,
            copy_eligible: candidate.copyEligible,
            eligible: candidate.eligible, selected: candidate.selected,
            rejection: cacheCandidateRejection(candidate.rejection))
        },
        prompt_token_hash: tokenFingerprint(req.promptTokens, count: req.promptTokens.count),
        prompt_token_count: req.promptTokens.count,
        stable_boundary_token_hash: req.materializedAnchors.max().map {
          tokenFingerprint(req.promptTokens, count: $0)
        },
        stable_boundary_token_count: req.materializedAnchors.max() ?? 0,
        stable_boundary_kind: req.materializedAnchors.isEmpty ? nil : "prompt",
        automatic_checkpoint_proposed: req.automaticCheckpointProposed,
        automatic_checkpoint_authorized: req.anchorPlantAt.filter {
          req.resolvedBoundaries[$0]?.kind == "automatic_token"
        }.count,
        automatic_checkpoint_materialized: req.materializedAnchors.filter {
          req.resolvedBoundaries[$0]?.kind == "automatic_token"
        }.count,
        automatic_checkpoint_deduped: req.automaticCheckpointDeduped,
        automatic_checkpoint_skipped: max(0, req.automaticCheckpointProposed
          - req.automaticCheckpointDeduped - req.anchorPlantAt.filter {
            req.resolvedBoundaries[$0]?.kind == "automatic_token"
          }.count),
        stable_boundaries: req.boundaryTelemetry.map { boundary in
          return WorkerCacheBoundary(
            token_hash: boundary.tokenCount.map { tokenFingerprint(req.promptTokens, count: $0) },
            token_count: boundary.tokenCount,
            kind: boundary.kind,
            label: boundary.label,
            requested: boundary.requested,
            status: boundary.status,
            skip_reason: boundary.skipReason,
            materialized: boundary.tokenCount.map { req.materializedAnchors.contains($0) })
        }
      )
      emit(id: req.id, done: true, cancelled: req.cancelled ? true : nil, finishReason: req.cancelled ? "cancel" : req.finishReason, usage: usage, cache: cacheInfo, timing: timing)
    }

    func prepareRequest(id: String?, control: ControlRequest, data: Data, receivedNs: UInt64) async -> ActiveRequest? {
      do {
        nextAdmissionOrder &+= 1
        let admissionOrder = nextAdmissionOrder
        let admittedNs = DispatchTime.now().uptimeNanoseconds
        let receivedToAdmittedMs = Double(admittedNs - receivedNs) / 1_000_000
        let templateStartNs = admittedNs
        guard let model = control.model else {
          emit(id: id, error: "chat.model is required")
          return nil
        }
        let modelDirectory = try validateModelDirectory(model)
        if loadedModel != model || languageModel == nil {
          try await loadModel(model, directory: modelDirectory)
        }
        guard let lm = languageModel, let tok = tokenizer else {
          emit(id: id, error: "model is not loaded")
          return nil
        }
        let requestedMaxTokens = control.max_tokens
        let temperature = control.temperature ?? 0.7
        // Full sampling parity with the llama worker: top_p/top_k/min_p,
        // seed, repetition/presence/frequency penalties, and opt-in KV cache
        // quantization (CLAP_MLX_KV_TYPE=q8_0|q4_0).
        var toolSpecs: [ToolSpec]? = nil
        if let envelope = try? JSONDecoder().decode(ToolsEnvelope.self, from: data),
           let rawTools = envelope.tools, !rawTools.isEmpty {
          toolSpecs = rawTools.compactMap { raw -> ToolSpec? in
            raw.anyValue as? [String: any Sendable]
          }
          guard toolSpecs?.count == rawTools.count else {
            emit(id: id, error: "one or more caller-provided tools could not be represented for the chat template")
            return nil
          }
        }
        let entries = normalizedMessageEntries(control.messages ?? [])
        guard !entries.isEmpty else {
          emit(id: id, error: "chat request contains no messages")
          return nil
        }
        let structuredMessages = structuredTemplateMessages(control.messages ?? [])

        let usesFallback = usesGemma4FallbackPrompt(loadedDirectory ?? modelDirectory)
        var promptTokens: [Int]
        if usesFallback {
          promptTokens = tok.encode(text: gemma4Prompt(entries: entries), addSpecialTokens: false)
        } else {
          do {
            promptTokens = try tok.applyChatTemplate(messages: structuredMessages, tools: toolSpecs)
          } catch {
            if toolSpecs != nil {
              let compatible = (toolSpecs ?? []).map { templateCompatibleToolSpec($0) }
              if let tokens = try? tok.applyChatTemplate(messages: structuredMessages, tools: compatible) {
                debugLog("chat template for \(loadedModel ?? model) required a nullable-preserving compatibility view of all \(compatible.count) caller-provided tools")
                promptTokens = tokens
              } else {
                debugLog("chat template for \(loadedModel ?? model) failed with all \(toolSpecs?.count ?? 0) caller-provided tools (\(error)); retrying with JSON tool instructions")
                let patched = toolInstructionPatchedEntries(entries, tools: toolSpecs ?? [])
                let retryMessages = templateMessages(patched)
                if let tokens = try? tok.applyChatTemplate(messages: retryMessages, tools: nil) {
                  promptTokens = tokens
                } else {
                  debugLog("template retry without tools failed; using plain transcript")
                  promptTokens = tok.encode(text: plainTranscript(patched))
                }
              }
            } else {
              debugLog("chat template failed (\(error)); using plain transcript")
              promptTokens = tok.encode(text: plainTranscript(entries))
            }
          }
        }
        if configuration.debugPrompt {
          debugLog("prompt (\(promptTokens.count) tokens): \(tok.decode(tokenIds: promptTokens, skipSpecialTokens: false))")
        }
        let templateTokenizeMs = Double(DispatchTime.now().uptimeNanoseconds - templateStartNs) / 1_000_000

        // Admission control (parity with the llama worker): reject oversized
        // prompts before any prefill with a structured code the server maps
        // to an OpenAI-style 400.
        if modelContextLength > 0 && promptTokens.count >= modelContextLength {
          emit(id: id, error: "prompt is too long for the loaded model; prompt_tokens=\(promptTokens.count), max_input_tokens=\(modelContextLength - 1), effective_context_window=\(modelContextLength).", code: "context_length_exceeded")
          return nil
        }
        if let requestedMaxTokens, modelMaxOutputTokens > 0, requestedMaxTokens > modelMaxOutputTokens {
          emit(id: id, error: "requested max_tokens=\(requestedMaxTokens) exceeds the loaded model maximum output tokens=\(modelMaxOutputTokens).", code: "max_output_tokens_exceeded")
          return nil
        }
        if requestedMaxTokens == nil && modelContextLength == 0 && modelMaxOutputTokens == 0 {
          emit(id: id, error: "max_tokens is required because this model does not declare token limits.", code: "token_capability_unknown")
          return nil
        }
        let availableOutput = modelContextLength > 0 ? modelContextLength - promptTokens.count : modelMaxOutputTokens
        let maxTokens = requestedMaxTokens
          ?? (modelMaxOutputTokens > 0 ? min(modelMaxOutputTokens, availableOutput) : availableOutput)
        if modelContextLength > 0 && promptTokens.count + maxTokens > modelContextLength {
          emit(id: id, error: "prompt plus requested output exceeds the loaded model context; prompt_tokens=\(promptTokens.count), requested_output_tokens=\(maxTokens), effective_context_window=\(modelContextLength).", code: "context_length_exceeded")
          return nil
        }
        let generateParameters = GenerateParameters(
          maxTokens: maxTokens,
          kvBits: kvBits,
          temperature: Float(temperature),
          topP: Float(control.top_p ?? 1.0),
          topK: control.top_k ?? 0,
          minP: Float(control.min_p ?? 0.0),
          repetitionPenalty: control.repetition_penalty.map { Float($0) },
          presencePenalty: control.presence_penalty.map { Float($0) },
          frequencyPenalty: control.frequency_penalty.map { Float($0) },
          seed: control.seed
        )
        let outputReserve = maxTokens

        let cacheIdentity = CacheIdentity(domain: cacheDomain, requestId: id, intent: control.cache)
        let physicalIdentity = PhysicalCacheIdentity(fingerprint: cacheIdentity.fingerprint)
        let automaticProposals = configuration.checkpoints.offsets(promptTokens: promptTokens.count)
        let automaticDeduped = automaticProposals.filter { checkpoint in
          kvSlots.contains { slot in
            slot.isAnchor && slot.cacheIdentity?.isCompatible(with: physicalIdentity) == true &&
              slot.tokens.count == checkpoint &&
              slot.tokens.elementsEqual(promptTokens.prefix(checkpoint))
          }
        }.count
        var stableBoundaries: [Int] = []
        var stableBoundaryScopes: [Int: UInt32] = [:]
        var resolvedBoundaries: [Int: ActiveRequest.BoundaryInfo] = [:]
        var boundaryTelemetry: [ActiveRequest.BoundaryInfo] = []
        func resolveBoundary(messages: [[String: any Sendable]], tools: [ToolSpec]?, kind: String,
                             label: String?, requested: Bool) {
          guard !messages.isEmpty else {
            if requested {
              boundaryTelemetry.append(.init(tokenCount: nil, kind: kind, label: label,
                requested: true, status: "skipped", skipReason: "unsupported_template_boundary"))
            }
            return
          }
          let rendered: [Int]
          do {
            rendered = try tok.applyChatTemplate(
              messages: messages, tools: tools,
              additionalContext: ["add_generation_prompt": false])
          } catch {
            if requested {
              boundaryTelemetry.append(.init(tokenCount: nil, kind: kind, label: label,
                requested: true, status: "skipped", skipReason: "unsupported_template_boundary"))
            }
            return
          }
          guard let boundary = exactTemplateBoundary(
            prefix: rendered, final: promptTokens, eosToken: tok.eosTokenId) else {
            if requested {
              boundaryTelemetry.append(.init(tokenCount: nil, kind: kind, label: label,
                requested: true, status: "skipped", skipReason: "non_prefix_template_boundary"))
            }
            return
          }
          stableBoundaries.append(boundary)
          stableBoundaryScopes[boundary] = cacheIdentity.scope
          if resolvedBoundaries[boundary] == nil || requested {
            resolvedBoundaries[boundary] = .init(tokenCount: boundary, kind: kind, label: label,
              requested: requested, status: "resolved", skipReason: nil)
          }
          if requested {
            boundaryTelemetry.append(.init(tokenCount: boundary, kind: kind, label: label,
              requested: true, status: "resolved", skipReason: nil))
          }
        }
        if !usesFallback {
          let leadingSystemCount = structuredMessages.prefix { ($0["role"] as? String) == "system" }.count
          let systemPrefixCounts = leadingSystemCount > 0 ? Array(1...leadingSystemCount) : []
          for count in systemPrefixCounts {
            let prefixMessages = Array(structuredMessages.prefix(count))
            resolveBoundary(messages: prefixMessages, tools: toolSpecs,
              kind: "messages", label: nil, requested: false)
          }
          stableBoundaries = Array(Set(stableBoundaries)).sorted()
        }
        for descriptor in control.cache?.boundaries ?? [] {
          guard !usesFallback else {
            boundaryTelemetry.append(.init(tokenCount: nil, kind: descriptor.kind,
              label: descriptor.label, requested: true, status: "skipped",
              skipReason: "unsupported_template_boundary"))
            continue
          }
          if descriptor.kind == "tools" {
            // MLX's tokenizer bridge cannot render tools independently from
            // messages or expose a tools-end offset. Rendering with a first
            // message would incorrectly include message content, so fail the
            // optimization closed without invoking an empty-message template.
            boundaryTelemetry.append(.init(tokenCount: nil, kind: descriptor.kind,
              label: descriptor.label, requested: true, status: "skipped",
              skipReason: "unsupported_template_boundary"))
          } else if descriptor.kind == "messages", let index = descriptor.through_message,
                    index >= 0, index < structuredMessages.count {
            resolveBoundary(messages: Array(structuredMessages.prefix(index + 1)), tools: toolSpecs,
              kind: "messages", label: descriptor.label, requested: true)
          }
        }
        let promptBoundary = promptTokens.count - 1
        if promptBoundary >= 16 {
          stableBoundaries.append(promptBoundary)
          if stableBoundaryScopes[promptBoundary] == nil {
            stableBoundaryScopes[promptBoundary] = cacheIdentity.scope
          }
          if resolvedBoundaries[promptBoundary] == nil {
            resolvedBoundaries[promptBoundary] = .init(tokenCount: promptBoundary, kind: "prompt",
              label: nil, requested: false, status: "resolved", skipReason: nil)
          }
          stableBoundaries = Array(Set(stableBoundaries)).sorted()
        }
        var coordinatorPlan: CachePlan? = nil
        var coordinatorPlanMs = 0.0
        var coordinatorApplyStartedNs = DispatchTime.now().uptimeNanoseconds
        var cacheMaterializeMs = 0.0
        var cacheFallback: String? = cacheCoordinator == nil ? "coordinator_unavailable_no_cache" : nil
        if let coordinator = cacheCoordinator {
          try ensureAdmissionSlot()
          let slotMaterializations = kvSlots.enumerated().map { index, slot in
            let logical = try? coordinator.slot(index)
            let physical = PhysicalSlotRecord(identity: slot.cacheIdentity, tokens: slot.tokens,
              generation: slot.coordinatorGeneration, hasCaches: !slot.caches.isEmpty,
              isAnchor: slot.isAnchor)
            let generationMatches = logical.map { $0.generation == slot.coordinatorGeneration } ?? false
            let residentMatches = logical.map { Int($0.resident_len) == slot.tokens.count } ?? false
            let stateMatches = logical.map {
              (slot.isAnchor && $0.state == UInt32(CC_SLOT_ANCHOR)) ||
                (!slot.isAnchor && $0.state != UInt32(CC_SLOT_ANCHOR))
            } ?? false
            let identityMatches = slot.cacheIdentity?.isCompatible(with: physicalIdentity) ?? false
            let materialized = logical.map {
              physical.isMaterialized(for: physicalIdentity, logicalGeneration: $0.generation,
                logicalResidentLength: Int($0.resident_len), logicalState: $0.state,
                anchorState: UInt32(CC_SLOT_ANCHOR))
            } ?? false
            if !slot.caches.isEmpty && !materialized {
              var rejected: [String] = []
              if !generationMatches { rejected.append("generation") }
              if !residentMatches { rejected.append("resident_length") }
              if !stateMatches { rejected.append("state") }
              if !identityMatches { rejected.append("namespace_identity") }
              debugLog("cache donor rejected: slot=\(index) reasons=\(rejected.joined(separator: ",")) physical_generation=\(slot.coordinatorGeneration) logical_generation=\(logical?.generation ?? 0) physical_tokens=\(slot.tokens.count) logical_resident=\(logical?.resident_len ?? 0) logical_state=\(logical?.state ?? 0) anchor=\(slot.isAnchor)")
            }
            return CacheSlotMaterialization(
              materialized: materialized,
              writable: !slot.busy,
              partialSuffixTrim: materialized && slot.caches.allSatisfy(\.isTrimmable),
              copyable: materialized
            )
          }
          var capabilities = UInt64(CC_CAP_WHOLE_STATE_COPY) |
            UInt64(CC_CAP_PARTIAL_SUFFIX_TRIM) | UInt64(CC_CAP_PARTIAL_PREFIX_BRANCH) |
            UInt64(CC_CAP_SAFE_BUSY_DONOR) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT) |
            UInt64(CC_CAP_RELIABLE_RESIDENT_LENGTH) |
            UInt64(CC_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS)
          if slotMaterializations.contains(where: { $0.materialized && !$0.partialSuffixTrim }) {
            capabilities |= UInt64(CC_CAP_SLIDING_WINDOW) | UInt64(CC_CAP_RECURRENT_OR_HYBRID)
          }
          if kvBits != nil { capabilities |= UInt64(CC_CAP_KV_QUANTIZED) }
          let estimatedBytesPerToken = kvSlots.compactMap { slot -> UInt64? in
            guard !slot.tokens.isEmpty, !slot.caches.isEmpty else { return nil }
            return physicalCacheBytes(slot.caches) / UInt64(slot.tokens.count)
          }.max() ?? 0
          do {
            let planStartedNs = DispatchTime.now().uptimeNanoseconds
            coordinatorPlan = try coordinator.plan(tokens: promptTokens, identity: cacheIdentity,
              capabilities: capabilities, slots: slotMaterializations,
              stableBoundaries: stableBoundaries, outputReserve: outputReserve,
              estimatedBytesPerToken: estimatedBytesPerToken)
            coordinatorPlanMs += Double(DispatchTime.now().uptimeNanoseconds - planStartedNs) / 1_000_000
            coordinatorApplyStartedNs = DispatchTime.now().uptimeNanoseconds
            if let view = coordinatorPlan?.view {
              debugLog("cache coordinator plan: operation=\(view.operation) reuse=\(view.reuseTokens) donor=\(view.donor.map(String.init) ?? "none") target=\(view.target)")
            }
          } catch CacheCoordinatorError.status(_, let status)
              where status == CC_NO_CAPACITY || status == CC_SLOT_BUSY {
            pendingChats.insert((id: id, control: control, data: data, receivedNs: receivedNs), at: 0)
            debugLog("cache admission deferred: coordinator status=\(status)")
            return nil
          } catch {
            cacheFallback = "coordinator_plan_failed_closed"
            debugLog("cache coordinator plan failed closed: \(error)")
            throw error
          }
          guard coordinatorPlan != nil else { throw CacheCoordinatorError.unavailable }
        }
        guard coordinatorPlan != nil else { throw CacheCoordinatorError.unavailable }
        if sessionCap > 0 && promptTokens.count + outputReserve > sessionCap {
          emit(id: id, error: "prompt exceeds the per-session context cap; prompt tokens=\(promptTokens.count), max_session_ctx=\(sessionCap), reserved output tokens=\(outputReserve). Reduce the prompt/tool history or raise max_session_ctx / CLAP_MLX_MAX_SESSION_CTX.", code: "context_length_exceeded")
          return nil
        }

        // Rust is the sole cache policy authority. These variables describe
        // only the coordinator-selected physical operation.
        var bestPrefix = 0
        var branchDonor: KVSlot? = nil
        var branchPrefix = bestPrefix
        if let planned = coordinatorPlan {
          let view = planned.view
          guard view.target < kvSlots.count,
                view.donor == nil || view.donor! < kvSlots.count else {
            try planned.abort()
            throw CacheCoordinatorError.unavailable
          }
          let targetSlot = kvSlots[view.target]
          if view.operation == UInt32(CC_OPERATION_CONTINUE) {
            let trimNeeded = targetSlot.tokens.count - view.reuseTokens
            guard view.donor == view.target, !targetSlot.caches.isEmpty, trimNeeded >= 0,
                  trimNeeded == 0 || targetSlot.caches.allSatisfy(\.isTrimmable) else {
              try planned.abort()
              throw CacheCoordinatorError.unavailable
            }
          } else if view.operation == UInt32(CC_OPERATION_BRANCH) ||
                    view.operation == UInt32(CC_OPERATION_RESTORE) {
            guard let donorIndex = view.donor, donorIndex != view.target else {
              try planned.abort()
              throw CacheCoordinatorError.unavailable
            }
            let donorSlot = kvSlots[donorIndex]
            let donorOffset = cacheSequenceLength(donorSlot.caches, fallback: donorSlot.tokens.count)
            let trimNeeded = donorOffset - view.reuseTokens
            guard !donorSlot.caches.isEmpty, trimNeeded >= 0,
                  trimNeeded == 0 || donorSlot.caches.allSatisfy(\.isTrimmable) else {
              try planned.abort()
              throw CacheCoordinatorError.unavailable
            }
          }
          if view.operation == UInt32(CC_OPERATION_CONTINUE) {
            bestPrefix = view.reuseTokens
            branchDonor = nil
          } else if view.operation == UInt32(CC_OPERATION_BRANCH) ||
                    view.operation == UInt32(CC_OPERATION_RESTORE) {
            bestPrefix = 0
            branchDonor = view.donor.map { kvSlots[$0] }
            branchPrefix = view.reuseTokens
          } else {
            bestPrefix = 0
            branchDonor = nil
            branchPrefix = 0
          }
        }
        guard let planned = coordinatorPlan else { throw CacheCoordinatorError.unavailable }
        let slot = kvSlots[planned.view.target]
        if planned.view.operation != UInt32(CC_OPERATION_CONTINUE) {
          clearPhysicalSlot(slot)
        }
        kvUseCounter += 1
        slot.lastUsed = kvUseCounter
        slot.busy = true
        var prefix = bestPrefix
        if prefix == promptTokens.count { prefix -= 1 }  // always feed at least one token for logits

        var caches: [KVCache]
        var fedTokens: [Int]
        var suffix: [Int]
        var reusedTokens = 0
        let reuseKind = normalizedCacheReuseKind(operation: planned.view.operation)
        var reuseScope: String? = nil
        var branched = false
        if let donor = branchDonor {
          let materializeStartedNs = DispatchTime.now().uptimeNanoseconds
          var sharedPrefix = branchPrefix
          if sharedPrefix == promptTokens.count { sharedPrefix -= 1 }
          let cloned = donor.caches.map { $0.copy() }
          let cloneOffset = cacheSequenceLength(cloned, fallback: donor.tokens.count)
          let trimNeeded = cloneOffset - sharedPrefix
          if trimNeeded > 0 {
            for cache in cloned { cache.trim(trimNeeded) }
          }
          caches = cloned
          fedTokens = Array(promptTokens.prefix(sharedPrefix))
          suffix = Array(promptTokens.dropFirst(sharedPrefix))
          reusedTokens = sharedPrefix
          reuseScope = donor.anchorScope
          prefix = sharedPrefix
          branched = true
          cacheMaterializeMs += Double(DispatchTime.now().uptimeNanoseconds - materializeStartedNs) / 1_000_000
          debugLog("kv prefix branch: cloned \(sharedPrefix)/\(promptTokens.count) shared tokens from \(donor.isAnchor ? "an anchor" : "another slot"), prefilling \(suffix.count)")
        } else {
          caches = []
          fedTokens = []
          suffix = promptTokens
        }
        let trimmable = !slot.caches.isEmpty && slot.caches.allSatisfy { $0.isTrimmable }
        let trimNeeded = slot.tokens.count - prefix
        if branched {
          // cache assignment handled above
        } else if prefix > 0 {
          let materializeStartedNs = DispatchTime.now().uptimeNanoseconds
          if trimNeeded > 0 {
            for cache in slot.caches { cache.trim(trimNeeded) }
          }
          cacheMaterializeMs += Double(DispatchTime.now().uptimeNanoseconds - materializeStartedNs) / 1_000_000
          caches = slot.caches
          fedTokens = Array(promptTokens.prefix(prefix))
          suffix = Array(promptTokens.dropFirst(prefix))
          reusedTokens = prefix
          reuseScope = slot.anchorScope
          debugLog("kv prefix reuse (slot \(kvSlots.firstIndex(where: { $0 === slot }) ?? -1)): \(prefix)/\(promptTokens.count) tokens cached, prefilling \(suffix.count)")
        } else {
          caches = lm.newCache(parameters: generateParameters)
          fedTokens = []
          suffix = promptTokens
          if !slot.tokens.isEmpty {
            let reason = trimmable || slot.caches.isEmpty
              ? "no usable prefix"
              : "recurrent cache cannot rewind (matched \(prefix), needs trim of \(trimNeeded))"
            debugLog("kv cache miss: \(reason) (cached \(slot.tokens.count), prompt \(promptTokens.count))")
          }
        }
        slot.caches = caches
        slot.tokens = fedTokens
        slot.isPromptBoundary = false
        slot.anchorScope = nil
        slot.cacheIdentity = physicalIdentity

        var cacheDecision: CacheDecision? = nil
        let slotIndex = planned.view.target
        try retainedRegistry.activate(slotID: UInt32(slotIndex))
        if let planned = coordinatorPlan {
          do {
            let victims = planned.view.evictions.filter { $0 != slotIndex }.map(UInt32.init)
            try retainedRegistry.validateEvictions(victims)
            cacheDecision = try planned.commit(residentTokens: reusedTokens,
                                               state: UInt32(CC_SLOT_SESSION),
                                               physicalBytes: physicalCacheBytes(caches))
            slot.coordinatorGeneration = try cacheCoordinator?.slot(
              slotIndex).generation ?? 0
            retainedRegistry.reconcileEvictions(victims) { _, victim in
              clearPhysicalSlot(victim)
            }
            if !victims.isEmpty {
              lastEvictionReason = retentionConfig.physicalByteBudget > 0
                ? "byte_pressure" : "retained_capacity"
            }
            reusedTokens = cacheDecision?.realizedReuseTokens ?? reusedTokens
            reuseScope = cacheScopeName(cacheDecision?.scope ?? cacheIdentity.scope)
          } catch {
            retainedRegistry.release(slotID: UInt32(slotIndex))
            clearPhysicalSlot(slot)
            throw error
          }
        }

        for checkpoint in coordinatorPlan?.view.anchorBoundaries ?? []
          where resolvedBoundaries[checkpoint] == nil {
          resolvedBoundaries[checkpoint] = .init(tokenCount: checkpoint, kind: "automatic_token",
            label: nil, requested: false, status: "authorized", skipReason: nil)
        }
        let request = ActiveRequest(
          id: id,
          admissionOrder: admissionOrder,
          admittedNs: admittedNs,
          receivedToAdmittedMs: receivedToAdmittedMs,
          templateTokenizeMs: templateTokenizeMs,
          coordinatorPlanMs: coordinatorPlanMs,
          coordinatorApplyMs: Double(DispatchTime.now().uptimeNanoseconds - coordinatorApplyStartedNs) / 1_000_000,
          cacheMaterializeMs: cacheMaterializeMs,
          streaming: control.stream ?? true,
          maxTokens: maxTokens,
          promptTokens: promptTokens,
          reusedTokens: reusedTokens,
          reuseKind: reuseKind,
          reuseScope: reuseScope,
          cacheIdentity: cacheIdentity,
          cacheDecision: cacheDecision,
          cacheCandidates: coordinatorPlan?.candidates ?? [],
          cacheEvictions: coordinatorPlan?.view.evictions ?? [],
          cacheFallback: cacheFallback,
          slotIndex: slotIndex,
          slot: slot,
          caches: caches,
          fedTokens: fedTokens,
          suffix: suffix,
          detokenizer: NaiveStreamingDetokenizer(tokenizer: tok),
          parameters: generateParameters,
          stops: control.stop?.values ?? []
        )
        request.anchorPlantAt = coordinatorPlan?.view.anchorBoundaries ?? []
        request.anchorPlantScopes = stableBoundaryScopes
        request.resolvedBoundaries = resolvedBoundaries
        request.automaticCheckpointProposed = automaticProposals.count
        request.automaticCheckpointDeduped = automaticDeduped
        request.boundaryTelemetry = boundaryTelemetry + resolvedBoundaries.values
          .filter { !$0.requested }.sorted { ($0.tokenCount ?? 0) < ($1.tokenCount ?? 0) }
        return request
      } catch {
        emit(id: id, error: String(describing: error))
        return nil
      }
    }

    @discardableResult
    func snapshotAnchor(_ req: ActiveRequest, at plant: Int, reason: String) -> Bool {
      let boundary = Array(req.promptTokens.prefix(plant))
      let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
      guard req.fedTokens == boundary, offset == plant else {
        debugLog("\(reason) anchor skipped: fed=\(req.fedTokens.count) offset=\(offset) plant=\(plant)")
        return false
      }
      // Rust owns anchor deduplication, target choice, and eviction policy.
      guard let coordinator = cacheCoordinator else { return false }
      let anchor: KVSlot
      let anchorPlan: CachePlan
      do {
        try ensureAdmissionSlot()
        let slotMaterializations = kvSlots.map {
          CacheSlotMaterialization(
            materialized: false,
            writable: !$0.busy,
            partialSuffixTrim: false,
            copyable: false
          )
        }
        anchorPlan = try coordinator.plan(tokens: boundary, identity: req.cacheIdentity,
          capabilities: UInt64(CC_CAP_WHOLE_STATE_COPY) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT),
          slots: slotMaterializations,
          outputReserve: 0, state: UInt32(CC_SLOT_ANCHOR),
          scope: req.anchorPlantScopes[plant] ?? req.cacheIdentity.scope,
          estimatedBytesPerToken: physicalCacheBytes(req.caches) / UInt64(max(plant, 1)))
        guard anchorPlan.view.target < kvSlots.count,
              !kvSlots[anchorPlan.view.target].busy else {
          try? anchorPlan.abort()
          debugLog("\(reason) coordinated anchor skipped: target unavailable")
          return false
        }
        anchor = kvSlots[anchorPlan.view.target]
      } catch {
        debugLog("\(reason) coordinated anchor skipped: \(error)")
        return false
      }
      if anchorPlan.view.operation == UInt32(CC_OPERATION_NOOP) {
        do {
          _ = try anchorPlan.commit(residentTokens: plant, state: UInt32(CC_SLOT_ANCHOR),
            physicalBytes: physicalCacheBytes(anchor.caches))
          debugLog("coordinated \(reason) anchor already exists: slot=\(anchorPlan.view.target)")
          return true
        } catch {
          debugLog("\(reason) coordinated anchor no-op failed: \(error)")
          return false
        }
      }
      anchor.isAnchor = true
      let structural = req.resolvedBoundaries[plant].map {
        $0.kind != "prompt" && $0.kind != "automatic_token"
      } ?? false
      anchor.anchorScope = cacheScopeName(req.anchorPlantScopes[plant] ?? req.cacheIdentity.scope)
      let snapshotStartedNs = DispatchTime.now().uptimeNanoseconds
      anchor.caches = req.caches.map { $0.copy() }
      req.cacheMaterializeMs += Double(DispatchTime.now().uptimeNanoseconds - snapshotStartedNs) / 1_000_000
      anchor.tokens = boundary
      anchor.cacheIdentity = PhysicalCacheIdentity(fingerprint: req.cacheIdentity.fingerprint)
      do {
        let index = anchorPlan.view.target
        let victims = anchorPlan.view.evictions.filter { $0 != index }.map(UInt32.init)
        try retainedRegistry.validateEvictions(victims)
        _ = try anchorPlan.commit(residentTokens: plant, state: UInt32(CC_SLOT_ANCHOR),
          physicalBytes: physicalCacheBytes(anchor.caches))
        let logical = try coordinator.slot(index)
        anchor.coordinatorGeneration = logical.generation
        if structural {
          try coordinator.setAnchorProtected(slot: index,
            generation: logical.generation, protected: true)
        }
        retainedRegistry.reconcileEvictions(victims) { _, victim in
          clearPhysicalSlot(victim)
        }
        if !victims.isEmpty {
          lastEvictionReason = retentionConfig.physicalByteBudget > 0
            ? "byte_pressure" : "retained_capacity"
        }
        let fingerprint = req.cacheIdentity.fingerprint.map { String(format: "%02x", $0) }.joined()
        let flags = CacheSlotMaterialization(materialized: !anchor.caches.isEmpty,
          writable: !anchor.busy, partialSuffixTrim: anchor.caches.allSatisfy(\.isTrimmable),
          copyable: !anchor.caches.isEmpty).flags
        debugLog("coordinated \(reason) anchor committed: slot=\(index) logical_state=\(logical.state) logical_generation=\(logical.generation) logical_resident=\(logical.resident_len) namespace=\(fingerprint) physical_generation=\(anchor.coordinatorGeneration) physical_tokens=\(anchor.tokens.count) physical_identity=\(anchor.cacheIdentity != nil) flags=\(flags)")
      } catch {
        clearPhysicalSlot(anchor)
        debugLog("\(reason) coordinated anchor commit failed: \(error)")
        return false
      }
      kvUseCounter += 1
      anchor.lastUsed = kvUseCounter
      debugLog("planted \(reason) anchor: \(plant) tokens (exact-state snapshot for non-rewindable caches)")
      return true
    }

    func plantAnchor(_ req: ActiveRequest) {
      for plant in req.anchorPlantAt where plant == req.fedTokens.count && !req.anchorPlanted.contains(plant) {
        req.anchorPlanted.insert(plant)
        if snapshotAnchor(req, at: plant, reason: "prefix") {
          req.materializedAnchors.insert(plant)
        }
      }
    }

    func captureContinuationBoundary(_ req: ActiveRequest) {
      guard req.continuationBoundaryCaches == nil,
            let boundary = req.continuationBoundary,
            boundary == req.fedTokens.count,
            req.caches.contains(where: { !$0.isTrimmable }) else { return }
      let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
      guard offset == boundary else {
        debugLog("rolling conversation anchor skipped: fed=\(req.fedTokens.count) offset=\(offset) boundary=\(boundary)")
        return
      }
      let snapshotStartedNs = DispatchTime.now().uptimeNanoseconds
      req.continuationBoundaryCaches = req.caches.map { $0.copy() }
      req.cacheMaterializeMs += Double(DispatchTime.now().uptimeNanoseconds - snapshotStartedNs) / 1_000_000
      debugLog("captured rolling conversation anchor: \(boundary) tokens")
    }

    func advanceCoordinator(_ req: ActiveRequest, tokens: [Int]) {
      guard !tokens.isEmpty, let coordinator = cacheCoordinator,
            req.slot.coordinatorGeneration != 0 else { return }
      do {
        req.slot.coordinatorGeneration = try coordinator.advance(
          slot: req.slotIndex, generation: req.slot.coordinatorGeneration,
          tokens: tokens, state: UInt32(CC_SLOT_SESSION), busy: true,
          physicalBytes: physicalCacheBytes(req.caches))
      } catch {
        let generation = req.slot.coordinatorGeneration
        if generation != 0 {
          _ = try? coordinator.invalidate(slot: req.slotIndex, generation: generation)
        }
        req.slot.coordinatorGeneration = 0
        debugLog("cache metadata advance reconciled after error: \(error)")
      }
    }

    func step(_ req: ActiveRequest, prefillQuantum: Int) {
      guard !req.cancelled, !req.completed, !req.failed, let lm = languageModel else { return }
      let stepStartedNs = DispatchTime.now().uptimeNanoseconds
      req.schedulerWaitMs += Double(stepStartedNs - req.lastStepFinishedNs) / 1_000_000
      defer { req.lastStepFinishedNs = DispatchTime.now().uptimeNanoseconds }
      do {
        if req.iterator == nil {
          // Split chunks at every Rust-authorized boundary so physical state
          // exists at each exact nested prefix.
          var chunkEnd = min(req.pos + prefillQuantum, req.suffix.count)
          if let plant = req.anchorPlantAt.first(where: { !req.anchorPlanted.contains($0) && $0 > req.reusedTokens + req.pos }) {
            let rel = plant - req.reusedTokens
            if req.pos < rel && rel < chunkEnd { chunkEnd = rel }
          }
          if let boundary = req.continuationBoundary, req.continuationBoundaryCaches == nil {
            let rel = boundary - req.reusedTokens
            if req.pos < rel && rel < chunkEnd { chunkEnd = rel }
          }
          if chunkEnd < req.suffix.count {
            // Creating the iterator prefills the chunk into the shared cache;
            // the sampled token is discarded.
            let chunk = Array(req.suffix[req.pos ..< chunkEnd])
            let prefillStartedNs = DispatchTime.now().uptimeNanoseconds
            _ = try TokenIterator(input: LMInput(tokens: MLXArray(chunk)), model: lm, cache: req.caches, parameters: req.parameters)
            req.prefillMs += Double(DispatchTime.now().uptimeNanoseconds - prefillStartedNs) / 1_000_000
            req.prefillTokens += chunk.count
            req.prefillChunks += 1
            req.pos = chunkEnd
            req.fedTokens.append(contentsOf: chunk)
            req.slot.tokens = req.fedTokens
            advanceCoordinator(req, tokens: chunk)
            emit(id: req.id, prefill: WorkerPrefill(done: req.reusedTokens + req.pos, total: req.promptTokens.count))
            plantAnchor(req)
            if let boundary = req.continuationBoundary, boundary - req.reusedTokens == req.pos {
              captureContinuationBoundary(req)
            }
          } else {
            plantAnchor(req)
            if let boundary = req.continuationBoundary, boundary - req.reusedTokens == req.pos {
              captureContinuationBoundary(req)
            }
            let tail = Array(req.suffix.dropFirst(req.pos))
            let prefillStartedNs = DispatchTime.now().uptimeNanoseconds
            req.iterator = try TokenIterator(input: LMInput(tokens: MLXArray(tail)), model: lm, cache: req.caches, parameters: req.parameters)
            req.prefillMs += Double(DispatchTime.now().uptimeNanoseconds - prefillStartedNs) / 1_000_000
            req.prefillTokens += tail.count
            req.prefillChunks += 1
            req.pos = req.suffix.count
            req.fedTokens.append(contentsOf: tail)
            req.slot.tokens = req.fedTokens
            advanceCoordinator(req, tokens: tail)
            // Decode mutates recurrent cache state irreversibly. Preserve the
            // exact end-of-prompt state in this request and restore it into
            // the same slot at finalize, so even a one-slot worker can reuse
            // it for the next tool-result continuation.
            if req.continuationBoundaryCaches == nil && req.caches.contains(where: { !$0.isTrimmable }) {
              let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
              if offset == req.promptTokens.count {
                let snapshotStartedNs = DispatchTime.now().uptimeNanoseconds
                req.promptBoundaryCaches = req.caches.map { $0.copy() }
                req.cacheMaterializeMs += Double(DispatchTime.now().uptimeNanoseconds - snapshotStartedNs) / 1_000_000
                debugLog("captured prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens")
              } else {
                debugLog("prompt-boundary anchor skipped: cache offset \(offset), prompt \(req.promptTokens.count)")
              }
            }
          }
          return
        }
        guard var it = req.iterator else { return }
        var steps = 0
        while steps < decodeStepsPerPass {
          let firstDecodeStartedNs = req.generatedCount == 0
            ? DispatchTime.now().uptimeNanoseconds : 0
          guard let token = it.next() else {
            req.completed = true
            break
          }
          if firstDecodeStartedNs != 0 && req.firstDecodeMs == 0 {
            req.firstDecodeMs = Double(DispatchTime.now().uptimeNanoseconds - firstDecodeStartedNs) / 1_000_000
          }
          steps += 1
          req.sampledTokens.append(token)
          if eosTokenIds.contains(token) {
            if configuration.debugPrompt {
              debugLog("eos token \(token) (\(tokenizer?.convertIdToToken(token) ?? "?")) after \(req.generatedCount) tokens; eos set: \(eosTokenIds)")
            }
            req.finishReason = "stop"
            req.completed = true
            break
          }
          req.generatedCount += 1
          req.detokenizer.append(token: token)
          if let chunk = req.detokenizer.next(), !chunk.isEmpty {
            req.collected += chunk
            // Stop sequences: scan collected text; on match truncate and
            // finish. While streaming, hold back enough of the tail that a
            // stop split across tokens is never emitted.
            if !req.stops.isEmpty {
              let scan = StopSequencePolicy.scan(
                collected: req.collected, appendedCount: chunk.count,
                stops: req.stops, emittedCount: req.emitted, holdback: req.holdback)
              if let matchOffset = scan.matchOffset {
                req.collected = String(req.collected.prefix(matchOffset))
                if req.streaming && req.emitted < req.collected.count {
                  emit(id: req.id, token: String(req.collected.dropFirst(req.emitted)))
                  req.emitted = req.collected.count
                }
                req.finishReason = "stop"
                req.completed = true
                break
              }
            }
            if req.streaming {
              if req.stops.isEmpty {
                if req.firstEmitMs == 0 {
                  req.firstEmitMs = Double(DispatchTime.now().uptimeNanoseconds - req.admittedNs) / 1_000_000
                }
                emit(id: req.id, token: chunk)
                req.emitted = req.collected.count
              } else {
                let safe = StopSequencePolicy.scan(
                  collected: req.collected, appendedCount: chunk.count,
                  stops: req.stops, emittedCount: req.emitted, holdback: req.holdback).safeCount
                if safe > req.emitted {
                  if req.firstEmitMs == 0 {
                    req.firstEmitMs = Double(DispatchTime.now().uptimeNanoseconds - req.admittedNs) / 1_000_000
                  }
                  let start = req.collected.index(req.collected.startIndex, offsetBy: req.emitted)
                  let end = req.collected.index(req.collected.startIndex, offsetBy: safe)
                  emit(id: req.id, token: String(req.collected[start..<end]))
                  req.emitted = safe
                }
              }
            }
          }
          if req.generatedCount >= req.maxTokens {
            req.finishReason = "length"
            req.completed = true
            break
          }
        }
        req.iterator = it
      } catch {
        emit(id: req.id, error: String(describing: error))
        req.failed = true
      }
    }

    // Returns true when the worker should shut down.
    func handleLine(_ line: String) async -> Bool {
      guard !line.isEmpty, let data = line.data(using: .utf8),
            let control = try? JSONDecoder().decode(ControlRequest.self, from: data) else { return false }
      let id = control.id
      let type = control.type ?? "chat"

      if type == "shutdown" {
        for req in active {
          req.cancelled = true
          finalize(req)
        }
        active.removeAll()
        for pending in pendingChats {
          emit(id: pending.id, done: true, cancelled: true, finishReason: "cancel", usage: nil, cache: nil)
        }
        pendingChats.removeAll()
        emit(id: id, done: true)
        return true
      }

      if type == "cancel" {
        let target = control.id
        for req in active where RequestCancellationPolicy.matches(
          target: target, requestID: req.id) {
          req.cancelled = true
        }
        var remaining: [(id: String?, control: ControlRequest, data: Data, receivedNs: UInt64)] = []
        for pending in pendingChats {
          if RequestCancellationPolicy.matches(target: target, requestID: pending.id) {
            emit(id: pending.id, done: true, cancelled: true, finishReason: "cancel", usage: nil, cache: nil)
          } else {
            remaining.append(pending)
          }
        }
        pendingChats = remaining
        return false
      }

      if type == "set_max_active" {
        guard let requested = control.max_active, requested > 0 else {
          emit(id: id, error: "set_max_active.max_active must be positive")
          return false
        }
        let oldMaxActive = maxActive
        maxActive = min(requested, activePolicy.backendCeiling,
          activePolicy.hardwareCeiling, activePolicy.modelCeiling,
          max(1, retentionConfig.hardCeiling))
        previousMaxActive = control.previous_max_active ?? oldMaxActive
        coordinatedLimitingReason = control.limiting_reason
        lastAdjustmentReason = control.last_adjustment_reason
        lastAdjustmentAt = control.last_adjustment_at
        coordinatedGrowthReserveBytes = control.retained_growth_reserve_bytes
        globalResidentMemoryBytes = control.global_resident_memory_bytes
        pressureState = control.pressure_state
        retainedRegistry.updateMaxActive(maxActive)
        emit(id: id, done: true, retention: retentionSnapshot(queued: pendingChats.count))
        return false
      }

      if type == "unload" || type == "load" {
        // Model mutations wait until in-flight requests drain.
        if !active.isEmpty || !pendingChats.isEmpty {
          controlBacklog.append(line)
          return false
        }
        if type == "unload" {
          invalidateKVCache()
          languageModel = nil
          tokenizer = nil
          loadedDirectory = nil
          loadedModel = nil
          emit(id: id, unloaded: true, done: true)
          return false
        }
        guard let model = control.model else {
          emit(id: id, error: "load.model is required")
          return false
        }
        do {
          let modelDirectory = try validateModelDirectory(model)
          if loadedModel != model || languageModel == nil {
            try await loadModel(model, directory: modelDirectory)
          }
          emit(id: id, loaded: true, done: true, memory: memorySnapshot(),
            retention: retentionSnapshot(), tokenCapabilities: WorkerTokenCapabilities(
            model_context_window: declaredModelContextLength > 0 ? declaredModelContextLength : nil,
            effective_context_window: modelContextLength > 0 ? modelContextLength : nil,
            max_input_tokens: modelContextLength > 0 ? modelContextLength - 1 : nil,
            max_output_tokens: modelMaxOutputTokens > 0 ? modelMaxOutputTokens : nil,
            model_context_window_source: modelContextLengthSource,
            max_output_tokens_source: modelMaxOutputTokensSource,
            backend_allocation_cap: contextOverride > 0 ? contextOverride : (declaredModelContextLength > 0 ? declaredModelContextLength : nil),
            user_configured_override: contextOverride > 0 ? contextOverride : nil
          ))
        } catch {
          emit(id: id, error: String(describing: error))
        }
        return false
      }

      pendingChats.append((id: id, control: control, data: data,
        receivedNs: DispatchTime.now().uptimeNanoseconds))
      emit(retention: retentionSnapshot(queued: pendingChats.count))
      return false
    }

    mainLoop: while true {
      // Idle: block on input (or drain deferred control work) instead of
      // spinning; busy: poll without blocking so generation keeps stepping.
      if active.isEmpty && pendingChats.isEmpty {
        if !controlBacklog.isEmpty {
          let line = controlBacklog.removeFirst()
          if await handleLine(line) { break mainLoop }
          continue mainLoop
        }
        guard let line = await buffer.next() else { break mainLoop }
        if await handleLine(line) { break mainLoop }
      }
      while let line = await buffer.poll() {
        if await handleLine(line) { break mainLoop }
      }

      // Admit pending chats up to the parallel limit. Requests for a
      // different model wait until the current model's requests drain.
      while active.count < maxActive, !pendingChats.isEmpty {
        if retainedRegistry.count >= retentionConfig.hardCeiling && kvSlots.allSatisfy(\.busy) { break }
        let candidate = pendingChats[0]
        let needsLoad = loadedModel != candidate.control.model || languageModel == nil
        if needsLoad && !active.isEmpty { break }
        pendingChats.removeFirst()
        emit(id: candidate.id, started: true)
        if let request = await prepareRequest(id: candidate.id, control: candidate.control,
          data: candidate.data, receivedNs: candidate.receivedNs) {
          active.append(request)
          allocatorNeedsIdleClear = true
          emit(retention: retentionSnapshot(queued: pendingChats.count))
        } else if pendingChats.first?.id == candidate.id {
          break
        }
      }

      // Every runnable request gets one bounded Metal turn per round. Short
      // not-yet-emitting requests run first, while all requests remain present
      // exactly once so the priority boost cannot starve long prefills.
      let schedule = LatencyScheduler.round(active.map { request in
        LatencySchedulerRequest(
          id: String(request.admissionOrder), admissionOrder: request.admissionOrder,
          residualPrefillTokens: max(0, request.suffix.count - request.pos),
          decoding: request.iterator != nil, emittedFirstToken: request.emitted > 0,
          cancelled: request.cancelled)
      })
      for turn in schedule {
        if let request = active.first(where: { String($0.admissionOrder) == turn.id }) {
          for _ in 0..<turn.turns where !request.completed && !request.cancelled && !request.failed {
            step(request, prefillQuantum: turn.prefillQuantum)
          }
        }
      }
      for request in active where request.completed || request.cancelled || request.failed {
        finalize(request)
      }
      active.removeAll { $0.completed || $0.cancelled || $0.failed }
      if active.isEmpty && pendingChats.isEmpty && allocatorNeedsIdleClear {
        Memory.clearCache()
        let memory = memorySnapshot()
        debugLog("mlx memory after idle clear: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
        emit(memory: memory, retention: retentionSnapshot())
        allocatorNeedsIdleClear = false
      }
      if !pendingChats.isEmpty {
        try? await Task.sleep(nanoseconds: 10_000_000)
      }
    }
}

await main()
