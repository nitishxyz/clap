import ClapCacheBridge
import ClapMLXCache
import ClapMLXGeneration
import ClapMLXModel
import ClapMLXWorkerCore
import ClapCachePolicy
import Foundation
import MLXLMCommon


enum RequestPreparationResult {
  case admitted(MLXActiveRequest)
  case backpressured
  case rejected
}

extension WorkerState {
  func prepareRequest(id: String?, control: ControlRequest, data: Data, receivedNs: UInt64,
                      admissionOrder: UInt64) async -> RequestPreparationResult {
    do {
      let admittedNs = DispatchTime.now().uptimeNanoseconds
      let receivedToAdmittedMs = Double(admittedNs - receivedNs) / 1_000_000
      let templateStartNs = admittedNs
      if control.structured_output?.strength == "required" {
        emit(id: id,
          error: "MLX supports structured output only as best_effort post-validation",
          code: "structured_output_capability_required")
        return .rejected
      }
      // Canonical v1 generate envelopes intentionally omit the model because
      // the resident worker was already bound by the preceding load command.
      // Legacy chat envelopes still carry it, so accept either source.
      guard let model = resolveGenerateModel(requestModel: control.model,
        residentModel: modelRuntime.modelIdentifier) else {
        emit(id: id, error: "chat.model is required")
        return .rejected
      }
      let modelDirectory = try ModelLoader.validateDirectory(model)
      if modelRuntime.modelIdentifier != model || !modelRuntime.isLoaded {
        try await loadModel(model, directory: modelDirectory)
      }
      guard let lm = modelRuntime.languageModel, let tok = modelRuntime.tokenizer else {
        emit(id: id, error: "model is not loaded")
        return .rejected
      }
      var toolSpecs: [ToolSpec]? = nil
      if let envelope = try? JSONDecoder().decode(ToolsEnvelope.self, from: data),
         let rawTools = envelope.tools, !rawTools.isEmpty {
        toolSpecs = rawTools.compactMap { $0.anyValue as? [String: any Sendable] }
        guard toolSpecs?.count == rawTools.count else {
          emit(id: id, error: "one or more caller-provided tools could not be represented for the chat template")
          return .rejected
        }
      }
      let descriptors = (control.cache?.boundaries ?? []).map {
        PromptBoundaryDescriptor(kind: $0.kind, throughMessage: $0.through_message,
          label: $0.label)
      }
      var messages = promptMessages(control.messages ?? [])
      if let structuredOutput = control.structured_output {
        messages.insert(PromptMessage(role: "system",
          content: structuredOutputInstruction(structuredOutput)), at: 0)
      }
      let prompt: PreparedPrompt
      do {
        prompt = try PromptRenderer.render(messages: messages,
          tools: toolSpecs, boundaries: descriptors,
          modelDirectory: modelRuntime.directory ?? modelDirectory,
          tokenizer: promptTokenizerAdapter(tok)) { message in
            if message.hasPrefix("required ") || message.hasPrefix("failed with ") {
              debugLog("chat template for \(self.modelRuntime.modelIdentifier ?? model) \(message)")
            } else { debugLog(message) }
          }
      } catch PromptRendererError.noMessages {
        emit(id: id, error: "chat request contains no messages")
        return .rejected
      }
      let promptTokens = prompt.tokens
      if configuration.debugPrompt {
        debugLog("prompt (\(promptTokens.count) tokens): \(tok.decode(tokenIds: promptTokens, skipSpecialTokens: false))")
      }
      let templateTokenizeMs = Double(
        DispatchTime.now().uptimeNanoseconds - templateStartNs) / 1_000_000
      let maxTokens: Int
      switch modelRuntime.tokenCapabilities.resolveOutputTokens(
        promptTokens: promptTokens.count, requestedMaxTokens: control.max_tokens) {
      case .success(let resolved): maxTokens = resolved
      case .failure(let error):
        emit(id: id, error: error.message, code: error.code)
        return .rejected
      }
      let parameters = GenerateParameters(maxTokens: maxTokens, kvBits: kvBits,
        temperature: Float(control.temperature ?? 0.7), topP: Float(control.top_p ?? 1),
        topK: control.top_k ?? 0, minP: Float(control.min_p ?? 0),
        repetitionPenalty: control.repetition_penalty.map(Float.init),
        presencePenalty: control.presence_penalty.map(Float.init),
        frequencyPenalty: control.frequency_penalty.map(Float.init), seed: control.seed)
      guard let opaqueIdentity = control.cache_identity else {
        emit(id: id, error: "cache_identity is required", code: "cache_identity_required")
        return .rejected
      }
      let identity: CacheIdentity
      do {
        identity = try CacheIdentity(input: opaqueIdentity,
          expected: PhysicalCacheDescriptor(backend: "mlx",
            contextAllocation: contextOverride,
            kvFormat: kvBits == 8 ? "q8_0" : (kvBits == 4 ? "q4_0" : "f16"),
            unifiedKV: false, layoutVersion: 1))
      } catch {
        emit(id: id, error: String(describing: error), code: "invalid_cache_identity")
        return .rejected
      }
      let physicalIdentity = PhysicalCacheIdentity(fingerprint: identity.fingerprint)
      let proposals = configuration.checkpoints.offsets(promptTokens: promptTokens.count)
      let deduped = proposals.filter { checkpoint in
        kvSlots.contains { slot in
          slot.isAnchor && slot.cacheIdentity?.isCompatible(with: physicalIdentity) == true &&
            slot.tokens.count == checkpoint &&
            slot.tokens.elementsEqual(promptTokens.prefix(checkpoint))
        }
      }.count
      let stableBoundaries = prompt.stableBoundaries
      let scopes = Dictionary(uniqueKeysWithValues:
        stableBoundaries.map { ($0, identity.scope) })
      var resolved = prompt.resolvedBoundaries.mapValues { boundary in
        BoundaryInfo(tokenCount: boundary.tokenCount, kind: boundary.kind,
          label: boundary.label, requested: boundary.requested, status: boundary.status,
          skipReason: boundary.skipReason)
      }
      let telemetry = prompt.structuralBoundaries.map { boundary in
        BoundaryInfo(tokenCount: boundary.tokenCount, kind: boundary.kind,
          label: boundary.label, requested: boundary.requested, status: boundary.status,
          skipReason: boundary.skipReason)
      }
      guard let coordinator = cacheCoordinator else { throw CacheCoordinatorError.unavailable }
      if sessionCap > 0 && promptTokens.count + maxTokens > sessionCap {
        emit(id: id, error: "prompt exceeds the per-session context cap; prompt tokens=\(promptTokens.count), max_session_ctx=\(sessionCap), reserved output tokens=\(maxTokens). Reduce the prompt/tool history or raise max_session_ctx / CLAP_MLX_MAX_SESSION_CTX.", code: "context_length_exceeded")
        return .rejected
      }
      let admission: CacheAdmission<KVCache>
      do {
        admission = try CacheExecutor.admit(coordinator: coordinator,
          registry: &retainedRegistry, hardCeiling: retentionConfig.hardCeiling,
          promptTokens: promptTokens, identity: identity,
          physicalIdentity: physicalIdentity, stableBoundaries: stableBoundaries,
          outputReserve: maxTokens, kvQuantized: kvBits != nil,
          useCounter: &kvUseCounter, operations: cacheOperations(create: {
            lm.newCache(parameters: parameters)
          }))
      } catch CacheCoordinatorError.status(_, let status)
          where status == CC_NO_CAPACITY || status == CC_SLOT_BUSY {
        debugLog("cache admission deferred: coordinator status=\(status)")
        return .backpressured
      } catch {
        debugLog("cache coordinator plan failed closed: \(error)")
        throw error
      }
      if admission.evictedVictims {
        lastEvictionReason = retentionConfig.physicalByteBudget > 0
          ? "byte_pressure" : "retained_capacity"
      }
      for checkpoint in admission.anchorBoundaries where resolved[checkpoint] == nil {
        resolved[checkpoint] = .init(tokenCount: checkpoint, kind: "automatic_token",
          label: nil, requested: false, status: "authorized", skipReason: nil)
      }
      let prepared = PreparedRequest(id: id, admissionOrder: admissionOrder,
        admittedNs: admittedNs, receivedToAdmittedMs: receivedToAdmittedMs,
        templateTokenizeMs: templateTokenizeMs,
        coordinatorPlanMs: admission.coordinatorPlanMs,
        coordinatorApplyMs: admission.coordinatorApplyMs,
        cacheMaterializeMs: admission.cacheMaterializeMs,
        streaming: control.stream ?? true, maxTokens: maxTokens,
        promptTokens: promptTokens, reusedTokens: admission.reusedTokens,
        reuseKind: admission.reuseKind, reuseScope: admission.reuseScope,
        cacheIdentity: identity, cacheDecision: admission.decision,
        cacheCandidates: admission.candidates, cacheEvictions: admission.evictions,
        cacheFallback: nil, parameters: parameters, stops: control.stop?.values ?? [],
        anchorPlantAt: admission.anchorBoundaries, anchorPlantScopes: scopes,
        resolvedBoundaries: resolved,
        boundaryTelemetry: telemetry + resolved.values.filter { !$0.requested }
          .sorted { ($0.tokenCount ?? 0) < ($1.tokenCount ?? 0) },
        automaticCheckpointProposed: proposals.count,
        automaticCheckpointDeduped: deduped)
      return .admitted(MLXActiveRequest(prepared: prepared,
        cache: GenerationCacheContext(slotIndex: admission.slotIndex,
          slot: admission.slot, caches: admission.caches),
        fedTokens: admission.fedTokens, suffix: admission.suffix,
        detokenizer: NaiveStreamingDetokenizer(tokenizer: tok)))
    } catch {
      emit(id: id, error: String(describing: error))
      return .rejected
    }
  }

private func structuredOutputInstruction(_ contract: StructuredOutputRequest) -> String {
  if contract.kind == "json_object" {
    return "Return only one valid JSON object. Do not include markdown or explanatory text."
  }
  let schema: String
  if let value = contract.schema,
     let data = try? JSONSerialization.data(withJSONObject: value.foundationValue,
       options: [.sortedKeys]),
     let encoded = String(data: data, encoding: .utf8) {
    schema = encoded
  } else { schema = "{}" }
  return "Return only JSON matching this schema. Do not include markdown or explanatory text. Schema: \(schema)"
}
}
