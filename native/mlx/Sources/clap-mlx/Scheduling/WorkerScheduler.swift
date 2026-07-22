import ClapMLXWorkerCore
import Foundation

final class ExecutableWorkerScheduler {
  private let core = ClapMLXWorkerCore.WorkerScheduler<QueuedChat, MLXActiveRequest>()

  var isEmpty: Bool { core.isEmpty }
  var activeIsEmpty: Bool { core.active.isEmpty }
  var pendingIsEmpty: Bool { core.pending.isEmpty }
  var queuedCount: Int { core.pending.count }
  var hasWork: Bool { !core.isEmpty }

  func enqueue(id: String?, control: ControlRequest, data: Data,
               receivedNs: UInt64) -> [WorkerSchedulingEvent] {
    let chat = QueuedChat(id: id, control: control, data: data, receivedNs: receivedNs)
    core.enqueue(PendingRequest(id: id, model: control.model, receivedNs: receivedNs,
      payload: chat))
    return [.queueChanged(core.pending.count)]
  }

  func cancel(target: String?) -> [WorkerSchedulingEvent] {
    let result = core.cancel(target: target, activeView: Self.view) { $0.cancelled = true }
    return result.pending.map { WorkerSchedulingEvent.pendingCancelled($0.id) }
  }

  func shutdown(using state: WorkerState) -> [WorkerSchedulingEvent] {
    let active = core.removeAllActive()
    for request in active {
      request.cancelled = true
      state.finalize(request)
    }
    let pending = core.removeAllPending()
    return pending.map { .pendingCancelled($0.id) }
  }

  func admit(using state: WorkerState) async -> [WorkerSchedulingEvent] {
    var events: [WorkerSchedulingEvent] = []
    while true {
      let saturated = state.retainedRegistry.count >= state.retentionConfig.hardCeiling &&
        state.kvSlots.allSatisfy(\.busy)
      switch core.decideAdmission(maxActive: state.maxActive,
        loadedModel: state.modelRuntime.modelIdentifier,
        modelLoaded: state.modelRuntime.isLoaded, cacheSaturated: saturated) {
      case .empty, .blocked:
        return events
      case .candidate(let candidate, let admissionOrder):
        let chat = candidate.payload
        events.append(.started(chat.id))
        switch await state.prepareRequest(id: chat.id, control: chat.control,
          data: chat.data, receivedNs: chat.receivedNs,
          admissionOrder: admissionOrder) {
        case .admitted(let request):
          core.appendActive(request)
          state.allocatorNeedsIdleClear = true
          events.append(.queueChanged(core.pending.count))
        case .backpressured:
          core.restoreFront(candidate)
          return events
        case .rejected:
          continue
        }
      }
    }
  }

  func runRound(using state: WorkerState, decodeLimit: Int) {
    for turn in core.latencyRound(view: Self.view) {
      let request = turn.request
      for _ in 0..<turn.turns where !Self.view(request).terminal {
        state.step(request, prefillQuantum: turn.prefillQuantum, decodeLimit: decodeLimit)
      }
    }
    for request in core.active where Self.view(request).terminal {
      state.finalize(request)
    }
    _ = core.removeTerminal(view: Self.view)
  }

  private static func view(_ request: MLXActiveRequest) -> ActiveRequestView {
    ActiveRequestView(id: request.id, admissionOrder: request.admissionOrder,
      residualPrefillTokens: max(0, request.suffix.count - request.pos),
      decoding: request.iterator != nil, emittedFirstToken: request.emitted > 0,
      terminal: request.completed || request.cancelled || request.failed,
      cancelled: request.cancelled)
  }
}
