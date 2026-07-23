import Testing
import ClapCachePolicy
@testable import ClapMLXWorkerCore

@Suite("Protocol-independent worker scheduler")
struct WorkerSchedulerTests {
  @Test("pending requests remain FIFO with monotonic admission order")
  func fifoAdmission() {
    let scheduler = WorkerScheduler<String, TestActive>()
    scheduler.enqueue(PendingRequest(id: "a", model: "m", receivedNs: 1, payload: "a"))
    scheduler.enqueue(PendingRequest(id: "b", model: "m", receivedNs: 2, payload: "b"))
    guard case .candidate(let first, let firstOrder) = scheduler.decideAdmission(
      maxActive: 2, loadedModel: "m", modelLoaded: true, cacheSaturated: false) else {
      Issue.record("expected first candidate"); return
    }
    guard case .candidate(let second, let secondOrder) = scheduler.decideAdmission(
      maxActive: 2, loadedModel: "m", modelLoaded: true, cacheSaturated: false) else {
      Issue.record("expected second candidate"); return
    }
    #expect(first.payload == "a")
    #expect(second.payload == "b")
    #expect(firstOrder == 1)
    #expect(secondOrder == 2)
  }

  @Test("admission reports capacity model switch and cache saturation")
  func admissionBlocks() {
    let scheduler = WorkerScheduler<String, TestActive>()
    scheduler.enqueue(PendingRequest(id: "a", model: "new", receivedNs: 1, payload: "a"))
    scheduler.appendActive(TestActive(id: "running", order: 1))
    assertBlocked(scheduler.decideAdmission(maxActive: 1, loadedModel: "old",
      modelLoaded: true, cacheSaturated: false), .capacity)
    assertBlocked(scheduler.decideAdmission(maxActive: 2, loadedModel: "old",
      modelLoaded: true, cacheSaturated: false), .modelSwitch)
    assertBlocked(scheduler.decideAdmission(maxActive: 2, loadedModel: "new",
      modelLoaded: true, cacheSaturated: true), .cacheSaturation)
  }

  @Test("front restoration preserves candidate and does not spend another order")
  func restoreFront() {
    let scheduler = WorkerScheduler<String, TestActive>()
    let first = PendingRequest(id: "a", model: "m", receivedNs: 1, payload: "a")
    scheduler.enqueue(first)
    scheduler.enqueue(PendingRequest(id: "b", model: "m", receivedNs: 2, payload: "b"))
    guard case .candidate(let candidate, let order) = scheduler.decideAdmission(
      maxActive: 1, loadedModel: "m", modelLoaded: true, cacheSaturated: false) else {
      Issue.record("expected candidate"); return
    }
    scheduler.restoreFront(candidate)
    #expect(scheduler.pending.map(\.id) == ["a", "b"])
    #expect(order == 1)
  }

  @Test("nil and empty cancellation match all pending and active requests")
  func cancelAll() {
    for target in [nil, ""] as [String?] {
      let scheduler = populated()
      let result = scheduler.cancel(target: target, activeView: view) { $0.cancelled = true }
      #expect(result.pending.count == 2)
      #expect(result.active.count == 2)
      #expect(scheduler.pending.isEmpty)
      #expect(scheduler.active.filter { !$0.cancelled }.isEmpty)
    }
  }

  @Test("target cancellation removes matching pending and marks matching active")
  func targetedCancellation() {
    let scheduler = populated()
    let result = scheduler.cancel(target: "b", activeView: view) { $0.cancelled = true }
    #expect(result.pending.map(\.id) == ["b"])
    #expect(result.active.map(\.id) == ["b"])
    #expect(scheduler.pending.map(\.id) == ["a"])
    #expect(scheduler.active.first(where: { $0.id == "b" })?.cancelled == true)
  }

  @Test("latency rounds advance all requests before weighted extras")
  func latencyRound() {
    let scheduler = WorkerScheduler<String, TestActive>()
    scheduler.appendActive(TestActive(id: "long", order: 1, residual: 800))
    scheduler.appendActive(TestActive(id: "near", order: 2, residual: 200))
    scheduler.appendActive(TestActive(id: "decode", order: 3, residual: 0,
      decoding: true, emitted: true))
    let round = scheduler.latencyRound(view: view)
    #expect(round.prefix(3).map { $0.request.id } == ["near", "decode", "long"])
    #expect(round.prefix(3).map(\.prefillQuantum) == [96, 96, 96])
    #expect(round.allSatisfy { $0.turns == 1 })
  }

  @Test("sustained interactive work cannot starve normal or background")
  func priorityStarvationBound() {
    let scheduler = WorkerScheduler<String, TestActive>()
    for index in 0..<8 {
      scheduler.enqueue(PendingRequest(id: "i\(index)", model: "m", receivedNs: UInt64(index),
        payload: "i\(index)", priority: .interactive))
    }
    scheduler.enqueue(PendingRequest(id: "n", model: "m", receivedNs: 20, payload: "n", priority: .normal))
    scheduler.enqueue(PendingRequest(id: "b", model: "m", receivedNs: 21, payload: "b", priority: .background))
    var admitted: [String] = []
    for _ in 0..<7 {
      guard case .candidate(let candidate, _) = scheduler.decideAdmission(maxActive: 20,
        loadedModel: "m", modelLoaded: true, cacheSaturated: false) else { break }
      admitted.append(candidate.payload)
    }
    #expect(admitted == ["i0", "i1", "i2", "i3", "n", "b", "i4"])
    #expect(scheduler.pending.filter { $0.priority == .interactive }.map(\.payload) == ["i5", "i6", "i7"])
  }

  @Test("priority rounds preserve FIFO within class and never cancel active work")
  func priorityRoundSafety() {
    let scheduler = WorkerScheduler<String, TestActive>()
    scheduler.appendActive(TestActive(id: "i1", order: 1, priority: .interactive))
    scheduler.appendActive(TestActive(id: "i2", order: 2, priority: .interactive))
    scheduler.appendActive(TestActive(id: "n", order: 3, priority: .normal))
    scheduler.appendActive(TestActive(id: "b", order: 4, priority: .background))
    let round = scheduler.latencyRound(view: view)
    #expect(round.prefix(4).map { $0.request.id } == ["i1", "i2", "n", "b"])
    #expect(round.prefix(4).map(\.prefillQuantum) == [192, 192, 96, 48])
    #expect(scheduler.active.allSatisfy { !$0.cancelled && !$0.terminal })
  }

  @Test("terminal removal returns terminal requests and keeps runnable order")
  func terminalRemoval() {
    let scheduler = populated()
    scheduler.active[0].terminal = true
    let removed = scheduler.removeTerminal(view: view)
    #expect(removed.map(\.id) == ["a"])
    #expect(scheduler.active.map(\.id) == ["b"])
  }

  private func populated() -> WorkerScheduler<String, TestActive> {
    let scheduler = WorkerScheduler<String, TestActive>()
    scheduler.enqueue(PendingRequest(id: "a", model: "m", receivedNs: 1, payload: "a"))
    scheduler.enqueue(PendingRequest(id: "b", model: "m", receivedNs: 2, payload: "b"))
    scheduler.appendActive(TestActive(id: "a", order: 1))
    scheduler.appendActive(TestActive(id: "b", order: 2))
    return scheduler
  }

  private func assertBlocked(_ decision: AdmissionDecision<String>,
                             _ expected: AdmissionBlockReason) {
    guard case .blocked(let reason) = decision else {
      Issue.record("expected blocked decision"); return
    }
    #expect(reason == expected)
  }
}

private final class TestActive {
  let id: String?
  let order: UInt64
  let residual: Int
  let decoding: Bool
  let emitted: Bool
  let priority: SchedulingPriority
  var terminal = false
  var cancelled = false

  init(id: String?, order: UInt64, residual: Int = 0, decoding: Bool = false,
       emitted: Bool = false, priority: SchedulingPriority = .normal) {
    self.id = id
    self.order = order
    self.residual = residual
    self.decoding = decoding
    self.emitted = emitted
    self.priority = priority
  }
}

private func view(_ request: TestActive) -> ActiveRequestView {
  ActiveRequestView(id: request.id, admissionOrder: request.order,
    residualPrefillTokens: request.residual, decoding: request.decoding,
    emittedFirstToken: request.emitted, terminal: request.terminal || request.cancelled,
    cancelled: request.cancelled, priority: request.priority)
}
