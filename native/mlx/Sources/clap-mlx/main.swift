import Foundation
import Darwin
import ClapCacheBridge
import ClapCachePolicy
import ClapMLXCache
import ClapMLXGeneration
import ClapMLXModel
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

func debugLog(_ message: String) {
  FileHandle.standardError.write(Data("[clap-mlx] \(message)\n".utf8))
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

    let modelRuntime = ModelRuntime()

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
    typealias KVSlot = CacheSlot<KVCache>
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
    func invalidateKVCache() {
      CacheExecutor.reset(coordinator: &cacheCoordinator, registry: &retainedRegistry,
        maxActive: maxActive, hardCeiling: retentionConfig.hardCeiling,
        useCounter: &kvUseCounter)
      lastEvictionReason = nil
    }

    func cacheOperations(create: @escaping () throws -> [KVCache] = { [] })
      -> CacheOperations<KVCache> {
      mlxCacheOperations(create: create, log: debugLog)
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
          retention: retentionConfig, maxActive: maxActive,
          capacity: Int.max / 4,
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

    typealias ActiveRequest = ClapMLXGeneration.ActiveRequest<KVCache, TokenIterator,
      NaiveStreamingDetokenizer, GenerateParameters>

    var active: [ActiveRequest] = []
    var pendingChats: [(id: String?, control: ControlRequest, data: Data, receivedNs: UInt64)] = []
    var controlBacklog: [String] = []
    var allocatorNeedsIdleClear = false
    var nextAdmissionOrder: UInt64 = 0

    func finalize(_ req: ActiveRequest) {
      CacheExecutor.finalize(coordinator: cacheCoordinator, registry: retainedRegistry,
        slotIndex: req.slotIndex, slot: req.slot, caches: &req.caches,
        snapshots: req.cacheSnapshots, promptTokens: req.promptTokens,
        fedTokens: req.fedTokens, sampledTokens: req.sampledTokens,
        generatedCount: req.generatedCount, failed: req.failed,
        operations: cacheOperations())
      if req.failed {
        return
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
      let usage = workerUsage(promptTokens: req.promptTokens.count,
        completionTokens: req.generatedCount)
      let timing = workerTiming(TimingTelemetryFacts(
        receivedToAdmittedMs: req.receivedToAdmittedMs,
        templateTokenizeMs: req.templateTokenizeMs,
        coordinatorPlanMs: req.coordinatorPlanMs,
        coordinatorApplyMs: req.coordinatorApplyMs,
        schedulerWaitMs: req.schedulerWaitMs,
        cacheMaterializeMs: req.cacheMaterializeMs,
        prefillMs: req.prefillMs,
        promptTokens: req.promptTokens.count,
        reusedTokens: req.reusedTokens,
        prefillTokens: req.prefillTokens,
        prefillChunks: req.prefillChunks,
        firstDecodeMs: req.firstDecodeMs,
        firstEmitMs: req.firstEmitMs))
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
        let modelDirectory = try ModelLoader.validateDirectory(model)
        if modelRuntime.modelIdentifier != model || !modelRuntime.isLoaded {
          try await loadModel(model, directory: modelDirectory)
        }
        guard let lm = modelRuntime.languageModel, let tok = modelRuntime.tokenizer else {
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
        let descriptors = (control.cache?.boundaries ?? []).map {
          PromptBoundaryDescriptor(kind: $0.kind, throughMessage: $0.through_message,
            label: $0.label)
        }
        let prepared: PreparedPrompt
        do {
          prepared = try PromptRenderer.render(messages: promptMessages(control.messages ?? []),
            tools: toolSpecs, boundaries: descriptors,
            modelDirectory: modelRuntime.directory ?? modelDirectory,
            tokenizer: promptTokenizerAdapter(tok)) { message in
              if message.hasPrefix("required ") || message.hasPrefix("failed with ") {
                debugLog("chat template for \(modelRuntime.modelIdentifier ?? model) \(message)")
              } else {
                debugLog(message)
              }
            }
        } catch PromptRendererError.noMessages {
          emit(id: id, error: "chat request contains no messages")
          return nil
        }
        let promptTokens = prepared.tokens
        if configuration.debugPrompt {
          debugLog("prompt (\(promptTokens.count) tokens): \(tok.decode(tokenIds: promptTokens, skipSpecialTokens: false))")
        }
        let templateTokenizeMs = Double(DispatchTime.now().uptimeNanoseconds - templateStartNs) / 1_000_000

        // Admission control (parity with the llama worker): reject oversized
        // prompts before any prefill with a structured code the server maps
        // to an OpenAI-style 400.
        let maxTokens: Int
        switch modelRuntime.tokenCapabilities.resolveOutputTokens(
          promptTokens: promptTokens.count, requestedMaxTokens: requestedMaxTokens) {
        case .success(let resolved):
          maxTokens = resolved
        case .failure(let error):
          emit(id: id, error: error.message, code: error.code)
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

        let cacheIdentity = CacheIdentity(domain: cacheDomain,
          input: control.cache?.identityInput ?? CacheIdentityInput(namespace: nil, tenant: nil,
            project: nil, harness: nil, agent: nil, session: nil, priority: nil,
            sideRequest: false))
        let physicalIdentity = PhysicalCacheIdentity(fingerprint: cacheIdentity.fingerprint)
        let automaticProposals = configuration.checkpoints.offsets(promptTokens: promptTokens.count)
        let automaticDeduped = automaticProposals.filter { checkpoint in
          kvSlots.contains { slot in
            slot.isAnchor && slot.cacheIdentity?.isCompatible(with: physicalIdentity) == true &&
              slot.tokens.count == checkpoint &&
              slot.tokens.elementsEqual(promptTokens.prefix(checkpoint))
          }
        }.count
        let stableBoundaries = prepared.stableBoundaries
        let stableBoundaryScopes = Dictionary(uniqueKeysWithValues:
          stableBoundaries.map { ($0, cacheIdentity.scope) })
        var resolvedBoundaries = prepared.resolvedBoundaries.mapValues { boundary in
          BoundaryInfo(tokenCount: boundary.tokenCount, kind: boundary.kind,
            label: boundary.label, requested: boundary.requested, status: boundary.status,
            skipReason: boundary.skipReason)
        }
        let boundaryTelemetry = prepared.structuralBoundaries.map { boundary in
          BoundaryInfo(tokenCount: boundary.tokenCount, kind: boundary.kind,
            label: boundary.label, requested: boundary.requested, status: boundary.status,
            skipReason: boundary.skipReason)
        }
        guard let coordinator = cacheCoordinator else { throw CacheCoordinatorError.unavailable }
        if sessionCap > 0 && promptTokens.count + outputReserve > sessionCap {
          emit(id: id, error: "prompt exceeds the per-session context cap; prompt tokens=\(promptTokens.count), max_session_ctx=\(sessionCap), reserved output tokens=\(outputReserve). Reduce the prompt/tool history or raise max_session_ctx / CLAP_MLX_MAX_SESSION_CTX.", code: "context_length_exceeded")
          return nil
        }
        let admission: CacheAdmission<KVCache>
        do {
          admission = try CacheExecutor.admit(coordinator: coordinator,
            registry: &retainedRegistry, hardCeiling: retentionConfig.hardCeiling,
            promptTokens: promptTokens, identity: cacheIdentity,
            physicalIdentity: physicalIdentity, stableBoundaries: stableBoundaries,
            outputReserve: outputReserve, kvQuantized: kvBits != nil,
            useCounter: &kvUseCounter,
            operations: cacheOperations(create: {
              lm.newCache(parameters: generateParameters)
            }))
        } catch CacheCoordinatorError.status(_, let status)
            where status == CC_NO_CAPACITY || status == CC_SLOT_BUSY {
          pendingChats.insert((id: id, control: control, data: data, receivedNs: receivedNs), at: 0)
          debugLog("cache admission deferred: coordinator status=\(status)")
          return nil
        } catch {
          debugLog("cache coordinator plan failed closed: \(error)")
          throw error
        }
        if admission.evictedVictims {
          lastEvictionReason = retentionConfig.physicalByteBudget > 0
            ? "byte_pressure" : "retained_capacity"
        }
        for checkpoint in admission.anchorBoundaries where resolvedBoundaries[checkpoint] == nil {
          resolvedBoundaries[checkpoint] = .init(tokenCount: checkpoint, kind: "automatic_token",
            label: nil, requested: false, status: "authorized", skipReason: nil)
        }
        let preparedRequest = PreparedRequest(id: id, admissionOrder: admissionOrder,
          admittedNs: admittedNs, receivedToAdmittedMs: receivedToAdmittedMs,
          templateTokenizeMs: templateTokenizeMs,
          coordinatorPlanMs: admission.coordinatorPlanMs,
          coordinatorApplyMs: admission.coordinatorApplyMs,
          cacheMaterializeMs: admission.cacheMaterializeMs,
          streaming: control.stream ?? true, maxTokens: maxTokens,
          promptTokens: promptTokens, reusedTokens: admission.reusedTokens,
          reuseKind: admission.reuseKind, reuseScope: admission.reuseScope,
          cacheIdentity: cacheIdentity, cacheDecision: admission.decision,
          cacheCandidates: admission.candidates, cacheEvictions: admission.evictions,
          cacheFallback: nil, parameters: generateParameters,
          stops: control.stop?.values ?? [], anchorPlantAt: admission.anchorBoundaries,
          anchorPlantScopes: stableBoundaryScopes, resolvedBoundaries: resolvedBoundaries,
          boundaryTelemetry: boundaryTelemetry + resolvedBoundaries.values
            .filter { !$0.requested }.sorted { ($0.tokenCount ?? 0) < ($1.tokenCount ?? 0) },
          automaticCheckpointProposed: automaticProposals.count,
          automaticCheckpointDeduped: automaticDeduped)
        return ActiveRequest(prepared: preparedRequest,
          cache: GenerationCacheContext(slotIndex: admission.slotIndex,
            slot: admission.slot, caches: admission.caches),
          fedTokens: admission.fedTokens, suffix: admission.suffix,
          detokenizer: NaiveStreamingDetokenizer(tokenizer: tok))
      } catch {
        emit(id: id, error: String(describing: error))
        return nil
      }
    }

    @discardableResult
    func snapshotAnchor(_ req: ActiveRequest, at plant: Int, reason: String) -> Bool {
      guard let coordinator = cacheCoordinator else { return false }
      let structural = req.resolvedBoundaries[plant].map {
        $0.kind != "prompt" && $0.kind != "automatic_token"
      } ?? false
      let result = AnchorManager.materialize(coordinator: coordinator,
        registry: &retainedRegistry, hardCeiling: retentionConfig.hardCeiling,
        boundary: Array(req.promptTokens.prefix(plant)), sourceCaches: req.caches,
        sourceFedTokens: req.fedTokens, identity: req.cacheIdentity,
        scope: req.anchorPlantScopes[plant] ?? req.cacheIdentity.scope,
        structural: structural, useCounter: &kvUseCounter, operations: cacheOperations())
      req.cacheMaterializeMs += result.materializeMs
      if result.evictedVictims {
        lastEvictionReason = retentionConfig.physicalByteBudget > 0
          ? "byte_pressure" : "retained_capacity"
      }
      if result.materialized {
        debugLog("planted \(reason) anchor: \(plant) tokens (exact-state snapshot for non-rewindable caches)")
      }
      return result.materialized
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
      guard let boundary = req.continuationBoundary else { return }
      req.cacheMaterializeMs += AnchorManager.captureContinuation(
        snapshots: req.cacheSnapshots, boundary: boundary, caches: req.caches,
        fedTokens: req.fedTokens, operations: cacheOperations())
    }

    func advanceCoordinator(_ req: ActiveRequest, tokens: [Int]) {
      CacheExecutor.appendAndAdvance(coordinator: cacheCoordinator,
        slotIndex: req.slotIndex, slot: req.slot, caches: req.caches,
        fedTokens: &req.fedTokens, tokens: tokens, operations: cacheOperations())
    }

    func step(_ req: ActiveRequest, prefillQuantum: Int) {
      guard !req.cancelled, !req.completed, !req.failed,
            let lm = modelRuntime.languageModel else { return }
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
          if let boundary = req.continuationBoundary, req.cacheSnapshots.continuation == nil {
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
            advanceCoordinator(req, tokens: tail)
            // Decode mutates recurrent cache state irreversibly. Preserve the
            // exact end-of-prompt state in this request and restore it into
            // the same slot at finalize, so even a one-slot worker can reuse
            // it for the next tool-result continuation.
            if req.cacheSnapshots.continuation == nil {
              let snapshotMs = AnchorManager.capturePromptBoundary(
                snapshots: req.cacheSnapshots, promptTokens: req.promptTokens,
                caches: req.caches, fedTokens: req.fedTokens, operations: cacheOperations())
              req.cacheMaterializeMs += snapshotMs
              if snapshotMs > 0 {
                debugLog("captured prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens")
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
          if modelRuntime.eosTokenIds.contains(token) {
            if configuration.debugPrompt {
              debugLog("eos token \(token) (\(modelRuntime.tokenizer?.convertIdToToken(token) ?? "?")) after \(req.generatedCount) tokens; eos set: \(modelRuntime.eosTokenIds)")
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
          modelRuntime.unload()
          emit(id: id, unloaded: true, done: true)
          return false
        }
        guard let model = control.model else {
          emit(id: id, error: "load.model is required")
          return false
        }
        do {
          let modelDirectory = try ModelLoader.validateDirectory(model)
          if modelRuntime.modelIdentifier != model || !modelRuntime.isLoaded {
            try await loadModel(model, directory: modelDirectory)
          }
          emit(id: id, loaded: true, done: true, memory: memorySnapshot(),
            retention: retentionSnapshot(),
            tokenCapabilities: modelRuntime.tokenCapabilities.workerEvent(
              contextOverride: contextOverride))
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
        let needsLoad = modelRuntime.modelIdentifier != candidate.control.model
          || !modelRuntime.isLoaded
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
