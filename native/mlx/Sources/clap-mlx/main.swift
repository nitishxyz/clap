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

    var controlBacklog: [String] = []
    let scheduler = ExecutableWorkerScheduler()

    func emitSchedulingEvents(_ events: [WorkerSchedulingEvent]) {
      for event in events {
        switch event {
        case .started(let id): emit(id: id, started: true)
        case .pendingCancelled(let id):
          emit(id: id, done: true, cancelled: true, finishReason: "cancel",
            usage: nil, cache: nil)
        case .queueChanged(let count):
          emit(retention: state.retentionSnapshot(queued: count))
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
        emitSchedulingEvents(scheduler.shutdown(using: state))
        emit(id: id, done: true)
        return true
      }

      if type == "cancel" {
        emitSchedulingEvents(scheduler.cancel(target: control.id))
        return false
      }

      if type == "set_max_active" {
        guard let requested = control.max_active, requested > 0 else {
          emit(id: id, error: "set_max_active.max_active must be positive")
          return false
        }
        state.updateMaxActive(requested, control: control)
        emit(id: id, done: true,
          retention: state.retentionSnapshot(queued: scheduler.queuedCount))
        return false
      }

      if type == "unload" || type == "load" {
        // Model mutations wait until in-flight requests drain.
        if !scheduler.isEmpty {
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

      emitSchedulingEvents(scheduler.enqueue(id: id, control: control, data: data,
        receivedNs: DispatchTime.now().uptimeNanoseconds))
      return false
    }

    mainLoop: while true {
      // Idle: block on input (or drain deferred control work) instead of
      // spinning; busy: poll without blocking so generation keeps stepping.
      if scheduler.isEmpty {
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

      emitSchedulingEvents(await scheduler.admit(using: state))
      scheduler.runRound(using: state, decodeLimit: decodeStepsPerPass)
      state.clearAllocatorIfIdle(activeEmpty: scheduler.activeIsEmpty,
        pendingEmpty: scheduler.pendingIsEmpty)
      if !scheduler.pendingIsEmpty {
        try? await Task.sleep(nanoseconds: 10_000_000)
      }
    }
}

await main()
