import ClapCachePolicy

public final class WorkerScheduler<Payload, Active> {
  public private(set) var pending: [PendingRequest<Payload>] = []
  public private(set) var active: [Active] = []
  public private(set) var nextAdmissionOrder: UInt64 = 0

  public init() {}

  public var isEmpty: Bool { pending.isEmpty && active.isEmpty }

  public func enqueue(_ request: PendingRequest<Payload>) { pending.append(request) }
  public func restoreFront(_ request: PendingRequest<Payload>) { pending.insert(request, at: 0) }
  public func appendActive(_ request: Active) { active.append(request) }
  public func reserveAdmissionOrder() -> UInt64 {
    nextAdmissionOrder &+= 1
    return nextAdmissionOrder
  }

  public func decideAdmission(maxActive: Int, loadedModel: String?, modelLoaded: Bool,
                              cacheSaturated: Bool) -> AdmissionDecision<Payload> {
    guard !pending.isEmpty else { return .empty }
    guard active.count < maxActive else { return .blocked(.capacity) }
    guard !cacheSaturated else { return .blocked(.cacheSaturation) }
    let candidate = pending[0]
    let needsLoad = loadedModel != candidate.model || !modelLoaded
    guard !needsLoad || active.isEmpty else { return .blocked(.modelSwitch) }
    pending.removeFirst()
    return .candidate(candidate, admissionOrder: reserveAdmissionOrder())
  }

  public func cancel(target: String?, activeView: (Active) -> ActiveRequestView,
                     cancelActive: (Active) -> Void) -> CancellationResult<Payload, Active> {
    var cancelledPending: [PendingRequest<Payload>] = []
    pending.removeAll { request in
      let match = Self.cancelMatches(target: target, requestID: request.id)
      if match { cancelledPending.append(request) }
      return match
    }
    var cancelledActive: [Active] = []
    for request in active where Self.cancelMatches(
      target: target, requestID: activeView(request).id) {
      cancelActive(request)
      cancelledActive.append(request)
    }
    return CancellationResult(pending: cancelledPending, active: cancelledActive)
  }

  public func latencyRound(view: (Active) -> ActiveRequestView) -> [SchedulerTurn<Active>] {
    let byOrder = Dictionary(uniqueKeysWithValues: active.map {
      (view($0).admissionOrder, $0)
    })
    return LatencyScheduler.round(active.map { request in
      let facts = view(request)
      return LatencySchedulerRequest(id: String(facts.admissionOrder),
        admissionOrder: facts.admissionOrder,
        residualPrefillTokens: facts.residualPrefillTokens,
        decoding: facts.decoding, emittedFirstToken: facts.emittedFirstToken,
        cancelled: facts.cancelled)
    }).compactMap { step in
      guard let order = UInt64(step.id), let request = byOrder[order] else { return nil }
      return SchedulerTurn(request: request, prefillQuantum: step.prefillQuantum,
        turns: step.turns)
    }
  }

  @discardableResult
  public func removeTerminal(view: (Active) -> ActiveRequestView) -> [Active] {
    var removed: [Active] = []
    active.removeAll { request in
      let terminal = view(request).terminal
      if terminal { removed.append(request) }
      return terminal
    }
    return removed
  }

  public func removeAllPending() -> [PendingRequest<Payload>] {
    defer { pending.removeAll() }
    return pending
  }

  public func removeAllActive() -> [Active] {
    defer { active.removeAll() }
    return active
  }

  public static func cancelMatches(target: String?, requestID: String?) -> Bool {
    target == nil || target?.isEmpty == true || requestID == target
  }
}
