import ClapMLXCache
import ClapMLXGeneration
import Foundation
import MLXLMCommon

extension WorkerState {
  func finalize(_ request: MLXActiveRequest) {
    let result = request.finalize(using: GenerationFinalizer { slotIndex, slot, caches,
      snapshots, promptTokens, fedTokens, sampledTokens, generatedCount, failed in
      CacheExecutor.finalize(coordinator: self.cacheCoordinator,
        registry: self.retainedRegistry, slotIndex: slotIndex, slot: slot,
        caches: &caches, snapshots: snapshots, promptTokens: promptTokens,
        fedTokens: fedTokens, sampledTokens: sampledTokens,
        generatedCount: generatedCount, failed: failed, operations: self.cacheOperations())
    })
    guard case .completion(let completion)? = result else { return }
    for output in completion.outputs {
      switch output {
      case .token(let token): emit(id: request.id, token: token)
      case .content(let content): emit(id: request.id, content: content)
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
      cacheMaterializeMs: facts.cacheMaterializeMs, prefillMs: facts.prefillMs,
      promptTokens: facts.promptTokens, reusedTokens: facts.reusedTokens,
      prefillTokens: facts.prefillTokens, prefillChunks: facts.prefillChunks,
      firstDecodeMs: facts.firstDecodeMs, firstEmitMs: facts.firstEmitMs))
    let cache = completion.cache
    let cacheInfo = WorkerCache(hit: cache.reusedTokens > 0,
      reused_tokens: cache.reusedTokens, reuse_kind: cache.reuseKind,
      reuse_scope: cache.reuseScope, side_request: cache.identity.sideRequest,
      namespace: cache.identity.exportedNamespace, donor_slot: cache.decision?.donor,
      target_slot: cache.slotIndex, target_generation: cache.decision?.targetGeneration,
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
          busy_eligible: candidate.busyEligible, lease_eligible: candidate.leaseEligible,
          materialized: candidate.materialized, trim_eligible: candidate.trimEligible,
          copy_eligible: candidate.copyEligible, eligible: candidate.eligible,
          selected: candidate.selected,
          rejection: cacheCandidateRejection(candidate.rejection))
      },
      prompt_token_hash: tokenFingerprint(cache.promptTokens, count: cache.promptTokens.count,
        namespace: cache.identity.fingerprint),
      prompt_token_count: cache.promptTokens.count,
      stable_boundary_token_hash: cache.materializedAnchors.max().map {
        tokenFingerprint(cache.promptTokens, count: $0, namespace: cache.identity.fingerprint)
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
          token_hash: boundary.tokenCount.map {
            tokenFingerprint(cache.promptTokens, count: $0,
              namespace: cache.identity.fingerprint)
          }, token_count: boundary.tokenCount, kind: boundary.kind, label: boundary.label,
          requested: boundary.requested, status: boundary.status,
          skip_reason: boundary.skipReason,
          materialized: boundary.tokenCount.map { cache.materializedAnchors.contains($0) })
      })
    emit(id: request.id, done: true,
      cancelled: completion.status == .cancelled ? true : nil,
      finishReason: completion.finishReason, usage: usage, cache: cacheInfo, timing: timing)
  }

  func generationBackend(_ model: any LanguageModel) -> MLXGenerationBackend {
    mlxGenerationBackend(model: model,
      appendAndAdvance: { slotIndex, slot, caches, fedTokens, tokens in
        CacheExecutor.appendAndAdvance(coordinator: self.cacheCoordinator,
          slotIndex: slotIndex, slot: slot, caches: caches,
          fedTokens: &fedTokens, tokens: tokens, operations: self.cacheOperations())
      }, plantAnchor: { plant, caches, fedTokens, identity, scope, structural in
        guard let coordinator = self.cacheCoordinator else {
          return AnchorResult(materialized: false, evictedVictims: false, materializeMs: 0)
        }
        let result = AnchorManager.materialize(coordinator: coordinator,
          registry: &self.retainedRegistry, hardCeiling: self.retentionConfig.hardCeiling,
          boundary: fedTokens, sourceCaches: caches, sourceFedTokens: fedTokens,
          identity: identity, scope: scope, structural: structural,
          useCounter: &self.kvUseCounter, operations: self.cacheOperations())
        if result.evictedVictims {
          self.lastEvictionReason = self.retentionConfig.physicalByteBudget > 0
            ? "byte_pressure" : "retained_capacity"
        }
        if result.materialized {
          debugLog("planted prefix anchor: \(plant) tokens (exact-state snapshot for non-rewindable caches)")
        }
        return result
      }, captureContinuation: { snapshots, boundary, caches, fedTokens in
        AnchorManager.captureContinuation(snapshots: snapshots, boundary: boundary,
          caches: caches, fedTokens: fedTokens, operations: self.cacheOperations())
      }, capturePromptBoundary: { snapshots, promptTokens, caches, fedTokens in
        let duration = AnchorManager.capturePromptBoundary(snapshots: snapshots,
          promptTokens: promptTokens, caches: caches, fedTokens: fedTokens,
          operations: self.cacheOperations())
        if duration > 0 {
          debugLog("captured prompt-boundary anchor: \(promptTokens.count) tokens")
        }
        return duration
      }, now: { DispatchTime.now().uptimeNanoseconds })
  }

  func step(_ request: MLXActiveRequest, prefillQuantum: Int, decodeLimit: Int) {
    guard let model = modelRuntime.languageModel else { return }
    let events = GenerationStepper.step(request, prefillQuantum: prefillQuantum,
      decodeLimit: decodeLimit, eosTokenIds: modelRuntime.eosTokenIds,
      backend: generationBackend(model))
    for event in events {
      switch event {
      case .prefill(let done, let total):
        emit(id: request.id, prefill: WorkerPrefill(done: done, total: total))
      case .token(let token): emit(id: request.id, token: token)
      case .content(let content): emit(id: request.id, content: content)
      case .error(let error): emit(id: request.id, error: error)
      }
    }
  }
}
