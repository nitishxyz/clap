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

    typealias ActiveRequest = MLXActiveRequest

    var active: [ActiveRequest] = []
    var pendingChats: [(id: String?, control: ControlRequest, data: Data, receivedNs: UInt64)] = []
    var controlBacklog: [String] = []
    var allocatorNeedsIdleClear = false
    var nextAdmissionOrder: UInt64 = 0

    func finalize(_ req: ActiveRequest) {
      let result = req.finalize(using: GenerationFinalizer { slotIndex, slot, caches,
        snapshots, promptTokens, fedTokens, sampledTokens, generatedCount, failed in
        CacheExecutor.finalize(coordinator: cacheCoordinator, registry: retainedRegistry,
          slotIndex: slotIndex, slot: slot, caches: &caches, snapshots: snapshots,
          promptTokens: promptTokens, fedTokens: fedTokens, sampledTokens: sampledTokens,
          generatedCount: generatedCount, failed: failed, operations: cacheOperations())
      })
      guard case .completion(let completion)? = result else { return }
      for output in completion.outputs {
        switch output {
        case .token(let token): emit(id: req.id, token: token)
        case .content(let content): emit(id: req.id, content: content)
        }
      }
      let usage = workerUsage(promptTokens: completion.usage.promptTokens,
        completionTokens: completion.usage.completionTokens)
      let facts = completion.timing
      let timing = workerTiming(TimingTelemetryFacts(
        receivedToAdmittedMs: facts.receivedToAdmittedMs,
        templateTokenizeMs: facts.templateTokenizeMs,
        coordinatorPlanMs: facts.coordinatorPlanMs,
        coordinatorApplyMs: facts.coordinatorApplyMs,
        schedulerWaitMs: facts.schedulerWaitMs,
        cacheMaterializeMs: facts.cacheMaterializeMs,
        prefillMs: facts.prefillMs,
        promptTokens: facts.promptTokens,
        reusedTokens: facts.reusedTokens,
        prefillTokens: facts.prefillTokens,
        prefillChunks: facts.prefillChunks,
        firstDecodeMs: facts.firstDecodeMs,
        firstEmitMs: facts.firstEmitMs))
      let cache = completion.cache
      let cacheInfo = WorkerCache(
        hit: cache.reusedTokens > 0,
        reused_tokens: cache.reusedTokens,
        reuse_kind: cache.reuseKind,
        reuse_scope: cache.reuseScope,
        side_request: cache.identity.sideRequest,
        namespace: cache.identity.exportedNamespace,
        donor_slot: cache.decision?.donor,
        target_slot: cache.slotIndex,
        evicted_slots: cache.evictions,
        decision_us: cache.decision?.decisionUs ?? 0,
        planned_reuse_tokens: cache.decision?.plannedReuseTokens ?? cache.reusedTokens,
        realized_reuse_tokens: cache.decision?.realizedReuseTokens ?? cache.reusedTokens,
        fallback: cache.fallback,
        miss_reason: cache.reusedTokens > 0 ? nil : "no_shared_prefix",
        candidates: cache.candidates.map { candidate in
          WorkerCacheCandidate(slot: candidate.slot, generation: candidate.generation,
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
        prompt_token_hash: tokenFingerprint(cache.promptTokens, count: cache.promptTokens.count),
        prompt_token_count: cache.promptTokens.count,
        stable_boundary_token_hash: cache.materializedAnchors.max().map {
          tokenFingerprint(cache.promptTokens, count: $0)
        },
        stable_boundary_token_count: cache.materializedAnchors.max() ?? 0,
        stable_boundary_kind: cache.materializedAnchors.isEmpty ? nil : "prompt",
        automatic_checkpoint_proposed: cache.automaticCheckpointProposed,
        automatic_checkpoint_authorized: cache.anchorPlantAt.filter {
          cache.resolvedBoundaries[$0]?.kind == "automatic_token"
        }.count,
        automatic_checkpoint_materialized: cache.materializedAnchors.filter {
          cache.resolvedBoundaries[$0]?.kind == "automatic_token"
        }.count,
        automatic_checkpoint_deduped: cache.automaticCheckpointDeduped,
        automatic_checkpoint_skipped: max(0, cache.automaticCheckpointProposed
          - cache.automaticCheckpointDeduped - cache.anchorPlantAt.filter {
            cache.resolvedBoundaries[$0]?.kind == "automatic_token"
          }.count),
        stable_boundaries: cache.boundaryTelemetry.map { boundary in
          WorkerCacheBoundary(
            token_hash: boundary.tokenCount.map { tokenFingerprint(cache.promptTokens, count: $0) },
            token_count: boundary.tokenCount, kind: boundary.kind, label: boundary.label,
            requested: boundary.requested, status: boundary.status,
            skip_reason: boundary.skipReason,
            materialized: boundary.tokenCount.map { cache.materializedAnchors.contains($0) })
        })
      emit(id: req.id, done: true,
        cancelled: completion.status == .cancelled ? true : nil,
        finishReason: completion.finishReason, usage: usage, cache: cacheInfo, timing: timing)
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

    func generationBackend(_ lm: any LanguageModel) -> MLXGenerationBackend {
      mlxGenerationBackend(model: lm,
        appendAndAdvance: { slotIndex, slot, caches, fedTokens, tokens in
          CacheExecutor.appendAndAdvance(coordinator: cacheCoordinator,
            slotIndex: slotIndex, slot: slot, caches: caches,
            fedTokens: &fedTokens, tokens: tokens, operations: cacheOperations())
      }, plantAnchor: { plant, caches, fedTokens, identity, scope, structural in
        guard let coordinator = cacheCoordinator else {
          return AnchorResult(materialized: false, evictedVictims: false, materializeMs: 0)
        }
        let result = AnchorManager.materialize(coordinator: coordinator,
          registry: &retainedRegistry, hardCeiling: retentionConfig.hardCeiling,
          boundary: fedTokens, sourceCaches: caches, sourceFedTokens: fedTokens,
          identity: identity, scope: scope, structural: structural,
          useCounter: &kvUseCounter, operations: cacheOperations())
        if result.evictedVictims {
          lastEvictionReason = retentionConfig.physicalByteBudget > 0
            ? "byte_pressure" : "retained_capacity"
        }
        if result.materialized {
          debugLog("planted prefix anchor: \(plant) tokens (exact-state snapshot for non-rewindable caches)")
        }
        return result
      }, captureContinuation: { snapshots, boundary, caches, fedTokens in
        AnchorManager.captureContinuation(snapshots: snapshots, boundary: boundary,
          caches: caches, fedTokens: fedTokens, operations: cacheOperations())
      }, capturePromptBoundary: { snapshots, promptTokens, caches, fedTokens in
        let duration = AnchorManager.capturePromptBoundary(snapshots: snapshots,
          promptTokens: promptTokens, caches: caches, fedTokens: fedTokens,
          operations: cacheOperations())
        if duration > 0 {
          debugLog("captured prompt-boundary anchor: \(promptTokens.count) tokens")
        }
        return duration
      }, now: {
        DispatchTime.now().uptimeNanoseconds
      })
    }

    func step(_ req: ActiveRequest, prefillQuantum: Int) {
      guard let lm = modelRuntime.languageModel else { return }
      let events = GenerationStepper.step(req, prefillQuantum: prefillQuantum,
        decodeLimit: decodeStepsPerPass, eosTokenIds: modelRuntime.eosTokenIds,
        backend: generationBackend(lm))
      for event in events {
        switch event {
        case .prefill(let done, let total):
          emit(id: req.id, prefill: WorkerPrefill(done: done, total: total))
        case .token(let token):
          emit(id: req.id, token: token)
        case .content(let content):
          emit(id: req.id, content: content)
        case .error(let error):
          emit(id: req.id, error: error)
        }
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
