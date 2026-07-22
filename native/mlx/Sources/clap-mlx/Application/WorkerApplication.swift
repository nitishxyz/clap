import ClapMLXModel
import Foundation

func debugLog(_ message: String) {
  FileHandle.standardError.write(Data("[clap-mlx] \(message)\n".utf8))
}

final class WorkerApplication {
  private let state: WorkerState
  private let scheduler: ExecutableWorkerScheduler
  private let buffer: LineBuffer
  private var backlog = CommandBacklog()
  private let decodeStepsPerPass = 6

  init(state: WorkerState = WorkerState(),
       scheduler: ExecutableWorkerScheduler = ExecutableWorkerScheduler(),
       buffer: LineBuffer = LineBuffer()) {
    self.state = state
    self.scheduler = scheduler
    self.buffer = buffer
  }

  func run() async {
    let readerTask = Task.detached { [buffer] in
      do {
        for try await line in FileHandle.standardInput.bytes.lines {
          await buffer.push(String(line))
        }
      } catch {}
      await buffer.finish()
    }
    defer { readerTask.cancel() }

    mainLoop: while true {
      // Idle workers block for input and service deferred model commands FIFO.
      if scheduler.isEmpty {
        if let line = backlog.removeFirst() {
          if await dispatch(line) { break mainLoop }
          continue mainLoop
        }
        guard let line = await buffer.next() else { break mainLoop }
        if await dispatch(line) { break mainLoop }
      }
      while let line = await buffer.poll() {
        if await dispatch(line) { break mainLoop }
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

  // Returns true when the worker should shut down.
  private func dispatch(_ line: String) async -> Bool {
    guard !line.isEmpty, let data = line.data(using: .utf8),
          let control = try? JSONDecoder().decode(ControlRequest.self, from: data) else {
      return false
    }
    let id = control.id
    let type = control.type ?? "chat"

    if type == "shutdown" {
      emitSchedulingEvents(scheduler.shutdown(using: state))
      emit(id: id, done: true)
      return true
    }
    if type == "cancel" {
      emitSchedulingEvents(scheduler.cancel(target: id))
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
    if type == "load" || type == "unload" {
      if !scheduler.isEmpty {
        backlog.append(line)
        return false
      }
      if type == "unload" {
        state.invalidateKVCache()
        state.modelRuntime.unload()
        emit(id: id, unloaded: true, done: true)
        return false
      }
      await load(id: id, model: control.model)
      return false
    }

    emitSchedulingEvents(scheduler.enqueue(id: id, control: control, data: data,
      receivedNs: DispatchTime.now().uptimeNanoseconds))
    return false
  }

  private func load(id: String?, model: String?) async {
    guard let model else {
      emit(id: id, error: "load.model is required")
      return
    }
    do {
      let directory = try ModelLoader.validateDirectory(model)
      if state.modelRuntime.modelIdentifier != model || !state.modelRuntime.isLoaded {
        try await state.loadModel(model, directory: directory)
      }
      emit(id: id, loaded: true, done: true, memory: memorySnapshot(),
        retention: state.retentionSnapshot(),
        tokenCapabilities: state.modelRuntime.tokenCapabilities.workerEvent(
          contextOverride: state.contextOverride))
    } catch {
      emit(id: id, error: String(describing: error))
    }
  }

  private func emitSchedulingEvents(_ events: [WorkerSchedulingEvent]) {
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
}
