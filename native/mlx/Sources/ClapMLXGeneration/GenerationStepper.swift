import ClapCachePolicy

public enum GenerationStepper {
  public static func step<Cache, Iterator, Detokenizer, Parameters>(
    _ request: ActiveRequest<Cache, Iterator, Detokenizer, Parameters>,
    prefillQuantum: Int, decodeLimit: Int,
    eosTokenIds: Set<Int>,
    backend: GenerationBackend<Cache, Iterator, Detokenizer, Parameters>
  ) -> [GenerationEvent] {
    guard request.status == .active else { return [] }
    let started = backend.now()
    request.schedulerWaitMs += Double(started - request.lastStepFinishedNs) / 1_000_000
    defer { request.lastStepFinishedNs = backend.now() }
    do {
      if request.iterator == nil {
        return try prefill(request, quantum: prefillQuantum, backend: backend)
      }
      return try decode(request, limit: decodeLimit, eosTokenIds: eosTokenIds, backend: backend)
    } catch {
      request.failed = true
      return [.error(String(describing: error))]
    }
  }

  private static func prefill<Cache, Iterator, Detokenizer, Parameters>(
    _ request: ActiveRequest<Cache, Iterator, Detokenizer, Parameters>, quantum: Int,
    backend: GenerationBackend<Cache, Iterator, Detokenizer, Parameters>
  ) throws -> [GenerationEvent] {
    var end = min(request.pos + max(1, quantum), request.suffix.count)
    if let plant = request.anchorPlantAt.first(where: {
      !request.anchorPlanted.contains($0) && $0 > request.reusedTokens + request.pos
    }) {
      let relative = plant - request.reusedTokens
      if request.pos < relative && relative < end { end = relative }
    }
    if let boundary = request.continuationBoundary,
       request.cacheSnapshots.continuation == nil {
      let relative = boundary - request.reusedTokens
      if request.pos < relative && relative < end { end = relative }
    }
    if end < request.suffix.count {
      let chunk = Array(request.suffix[request.pos..<end])
      let before = backend.now()
      _ = try backend.prefill(chunk, &request.caches, request.parameters)
      request.prefillMs += Double(backend.now() - before) / 1_000_000
      request.prefillTokens += chunk.count
      request.prefillChunks += 1
      request.pos = end
      backend.appendAndAdvance(request.slotIndex, request.slot, request.caches,
        &request.fedTokens, chunk)
      plantAnchors(request, backend: backend)
      captureContinuation(request, backend: backend)
      return [.prefill(done: request.reusedTokens + request.pos,
        total: request.promptTokens.count)]
    }
    plantAnchors(request, backend: backend)
    captureContinuation(request, backend: backend)
    let tail = Array(request.suffix.dropFirst(request.pos))
    let before = backend.now()
    request.iterator = try backend.prefill(tail, &request.caches, request.parameters)
    request.prefillMs += Double(backend.now() - before) / 1_000_000
    request.prefillTokens += tail.count
    request.prefillChunks += 1
    request.pos = request.suffix.count
    backend.appendAndAdvance(request.slotIndex, request.slot, request.caches,
      &request.fedTokens, tail)
    if request.cacheSnapshots.continuation == nil {
      request.cacheMaterializeMs += backend.capturePromptBoundary(request.cacheSnapshots,
        request.promptTokens, request.caches, request.fedTokens)
    }
    return []
  }

  private static func plantAnchors<Cache, Iterator, Detokenizer, Parameters>(
    _ request: ActiveRequest<Cache, Iterator, Detokenizer, Parameters>,
    backend: GenerationBackend<Cache, Iterator, Detokenizer, Parameters>
  ) {
    for plant in request.anchorPlantAt
      where plant == request.fedTokens.count && !request.anchorPlanted.contains(plant) {
      request.anchorPlanted.insert(plant)
      let info = request.resolvedBoundaries[plant]
      let structural = info.map { $0.kind != "prompt" && $0.kind != "automatic_token" } ?? false
      let result = backend.plantAnchor(plant, request.caches, request.fedTokens,
        request.cacheIdentity, request.anchorPlantScopes[plant] ?? request.cacheIdentity.scope,
        structural)
      request.cacheMaterializeMs += result.materializeMs
      if result.materialized { request.materializedAnchors.insert(plant) }
    }
  }

  private static func captureContinuation<Cache, Iterator, Detokenizer, Parameters>(
    _ request: ActiveRequest<Cache, Iterator, Detokenizer, Parameters>,
    backend: GenerationBackend<Cache, Iterator, Detokenizer, Parameters>
  ) {
    guard let boundary = request.continuationBoundary,
          boundary - request.reusedTokens == request.pos else { return }
    request.cacheMaterializeMs += backend.captureContinuation(request.cacheSnapshots,
      boundary, request.caches, request.fedTokens)
  }

  private static func decode<Cache, Iterator, Detokenizer, Parameters>(
    _ request: ActiveRequest<Cache, Iterator, Detokenizer, Parameters>, limit: Int,
    eosTokenIds: Set<Int>,
    backend: GenerationBackend<Cache, Iterator, Detokenizer, Parameters>
  ) throws -> [GenerationEvent] {
    guard var iterator = request.iterator else { return [] }
    var events: [GenerationEvent] = []
    var steps = 0
    while steps < max(0, limit) {
      let firstStarted = request.generatedCount == 0 ? backend.now() : 0
      guard let token = try backend.nextToken(&iterator) else {
        request.completed = true
        break
      }
      if firstStarted != 0 && request.firstDecodeMs == 0 {
        request.firstDecodeMs = Double(backend.now() - firstStarted) / 1_000_000
      }
      steps += 1
      request.sampledTokens.append(token)
      if eosTokenIds.contains(token) {
        request.finishReason = "stop"
        request.completed = true
        break
      }
      request.generatedCount += 1
      backend.appendToken(&request.detokenizer, token)
      if let chunk = backend.nextText(&request.detokenizer), !chunk.isEmpty {
        request.collected += chunk
        if !request.stops.isEmpty {
          let scan = StopSequencePolicy.scan(collected: request.collected,
            appendedCount: chunk.count, stops: request.stops,
            emittedCount: request.emitted, holdback: request.holdback)
          if let offset = scan.matchOffset {
            request.collected = String(request.collected.prefix(offset))
            if request.streaming && request.emitted < request.collected.count {
              events.append(.token(String(request.collected.dropFirst(request.emitted))))
              request.emitted = request.collected.count
            }
            request.finishReason = "stop"
            request.completed = true
            break
          }
        }
        if request.streaming {
          if request.stops.isEmpty {
            markFirstEmit(request, now: backend.now())
            events.append(.token(chunk))
            request.emitted = request.collected.count
          } else {
            let safe = StopSequencePolicy.scan(collected: request.collected,
              appendedCount: chunk.count, stops: request.stops,
              emittedCount: request.emitted, holdback: request.holdback).safeCount
            if safe > request.emitted {
              markFirstEmit(request, now: backend.now())
              let start = request.collected.index(request.collected.startIndex,
                offsetBy: request.emitted)
              let end = request.collected.index(request.collected.startIndex, offsetBy: safe)
              events.append(.token(String(request.collected[start..<end])))
              request.emitted = safe
            }
          }
        }
      }
      if request.generatedCount >= request.maxTokens {
        request.finishReason = "length"
        request.completed = true
        break
      }
    }
    request.iterator = iterator
    return events
  }

  private static func markFirstEmit<Cache, Iterator, Detokenizer, Parameters>(
    _ request: ActiveRequest<Cache, Iterator, Detokenizer, Parameters>, now: UInt64
  ) {
    if request.firstEmitMs == 0 {
      request.firstEmitMs = Double(now - request.admittedNs) / 1_000_000
    }
  }
}
