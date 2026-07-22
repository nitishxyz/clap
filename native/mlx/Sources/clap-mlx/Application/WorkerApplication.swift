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
    emitV1Ready()
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
        if let command = backlog.removeFirst() {
          if await dispatch(command.line, v1Accepted: command.v1RequestID != nil) { break mainLoop }
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
  private func dispatch(_ line: String, v1Accepted: Bool = false) async -> Bool {
    guard !line.isEmpty else { return false }
    let data: Data
    let control: ControlRequest
    let commandID: String?
    let cancellationTarget: String?
    let type: String
    let request: V1Request
    do { request = try decodeV1Request(line) }
    catch let error as V1DecodeError { failV1Decode(error); return false }
    catch { return false }
    guard v1Accepted || acceptV1(request.requestID) else { return false }
    data = request.controlData; control = request.control
    commandID = request.requestID; cancellationTarget = request.targetRequestID
    type = request.type
    let id = commandID

    if type == "shutdown" {
      emitSchedulingEvents(scheduler.shutdown(using: state))
      for command in backlog.removeAll() {
        if let deferredID = command.v1RequestID {
          completeV1Command(deferredID, result: ["kind": "cancelled"])
        }
      }
      if let id { completeV1Command(id, result: ["kind": "shutdown"]) }
      return true
    }
    if type == "cancel" {
      emitSchedulingEvents(scheduler.cancel(target: cancellationTarget))
      if let deferred = backlog.remove(requestID: cancellationTarget)?.v1RequestID {
        completeV1Command(deferred, result: ["kind": "cancelled"])
      }
      if let id { completeV1Command(id, result: ["kind": "cancelled"]) }
      return false
    }
    if type == "set_max_active" {
      guard let requested = control.max_active, requested > 0 else {
        emit(id: id, error: "set_max_active.max_active must be positive")
        return false
      }
      state.updateMaxActive(requested, control: control)
      if let id {
        completeV1Command(id, result: ["kind": "max_active_updated",
          "max_active": state.maxActive])
      }
      return false
    }
    if type == "load" || type == "unload" {
      if !scheduler.isEmpty {
        backlog.append(line, v1RequestID: id)
        return false
      }
      if type == "unload" {
        state.invalidateKVCache()
        state.modelRuntime.unload()
        if let id { completeV1Command(id, result: ["kind": "unloaded"]) }
        return false
      }
      await load(id: id, model: control.model)
      return false
    }

    emitSchedulingEvents(scheduler.enqueue(id: id,
      control: control, data: data,
      receivedNs: DispatchTime.now().uptimeNanoseconds))
    return false
  }

  private func load(id: String?, model: String?) async {
    guard let model else {
      if let id { failV1Command(id, message: "load.model is required") }
      return
    }
    do {
      let directory = try ModelLoader.validateDirectory(model)
      if state.modelRuntime.modelIdentifier != model || !state.modelRuntime.isLoaded {
        try await state.loadModel(model, directory: directory)
      }
      let capabilities = state.modelRuntime.tokenCapabilities.workerEvent(
        contextOverride: state.contextOverride)
      if let id {
        completeV1Command(id, result: ["kind": "loaded", "model": model,
          "token_capabilities": v1JSONObject(capabilities)])
      }
    } catch {
      if let id { failV1Command(id, message: String(describing: error)) }
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
