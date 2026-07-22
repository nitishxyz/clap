import Foundation
import Darwin
import ClapCacheBridge
import ClapCachePolicy
import ClapMLXCache
import ClapMLXGeneration
import ClapMLXModel
import ClapMLXWorkerCore
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
    let state = WorkerState()
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
    let schedulerCore = WorkerScheduler<ControlRequest, ActiveRequest>()

    // Returns true when the worker should shut down.
    func handleLine(_ line: String) async -> Bool {
      guard !line.isEmpty, let data = line.data(using: .utf8),
            let control = try? JSONDecoder().decode(ControlRequest.self, from: data) else { return false }
      let id = control.id
      let type = control.type ?? "chat"

      if type == "shutdown" {
        for req in active {
          req.cancelled = true
          state.finalize(req)
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
        state.updateMaxActive(requested, control: control)
        emit(id: id, done: true, retention: state.retentionSnapshot(queued: pendingChats.count))
        return false
      }

      if type == "unload" || type == "load" {
        // Model mutations wait until in-flight requests drain.
        if !active.isEmpty || !pendingChats.isEmpty {
          controlBacklog.append(line)
          return false
        }
        if type == "unload" {
          state.invalidateKVCache()
          state.modelRuntime.unload()
          emit(id: id, unloaded: true, done: true)
          return false
        }
        guard let model = control.model else {
          emit(id: id, error: "load.model is required")
          return false
        }
        do {
          let modelDirectory = try ModelLoader.validateDirectory(model)
          if state.modelRuntime.modelIdentifier != model || !state.modelRuntime.isLoaded {
            try await state.loadModel(model, directory: modelDirectory)
          }
          emit(id: id, loaded: true, done: true, memory: memorySnapshot(),
            retention: state.retentionSnapshot(),
            tokenCapabilities: state.modelRuntime.tokenCapabilities.workerEvent(
              contextOverride: state.contextOverride))
        } catch {
          emit(id: id, error: String(describing: error))
        }
        return false
      }

      pendingChats.append((id: id, control: control, data: data,
        receivedNs: DispatchTime.now().uptimeNanoseconds))
      emit(retention: state.retentionSnapshot(queued: pendingChats.count))
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
      while active.count < state.maxActive, !pendingChats.isEmpty {
        if state.retainedRegistry.count >= state.retentionConfig.hardCeiling &&
           state.kvSlots.allSatisfy(\.busy) { break }
        let candidate = pendingChats[0]
        let needsLoad = state.modelRuntime.modelIdentifier != candidate.control.model
          || !state.modelRuntime.isLoaded
        if needsLoad && !active.isEmpty { break }
        pendingChats.removeFirst()
        emit(id: candidate.id, started: true)
        let admissionOrder = schedulerCore.reserveAdmissionOrder()
        switch await state.prepareRequest(id: candidate.id, control: candidate.control,
          data: candidate.data, receivedNs: candidate.receivedNs,
          admissionOrder: admissionOrder) {
        case .admitted(let request):
          active.append(request)
          state.allocatorNeedsIdleClear = true
          emit(retention: state.retentionSnapshot(queued: pendingChats.count))
        case .backpressured:
          pendingChats.insert(candidate, at: 0)
          break
        case .rejected:
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
            state.step(request, prefillQuantum: turn.prefillQuantum,
              decodeLimit: decodeStepsPerPass)
          }
        }
      }
      for request in active where request.completed || request.cancelled || request.failed {
        state.finalize(request)
      }
      active.removeAll { $0.completed || $0.cancelled || $0.failed }
      state.clearAllocatorIfIdle(activeEmpty: active.isEmpty,
        pendingEmpty: pendingChats.isEmpty)
      if !pendingChats.isEmpty {
        try? await Task.sleep(nanoseconds: 10_000_000)
      }
    }
}

await main()
