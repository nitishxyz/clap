public enum AdmissionBlockReason: Equatable, Sendable {
  case capacity
  case modelSwitch
  case cacheSaturation
}

public enum AdmissionDecision<Payload> {
  case candidate(PendingRequest<Payload>, admissionOrder: UInt64)
  case blocked(AdmissionBlockReason)
  case empty
}

public struct SchedulerTurn<Active> {
  public let request: Active
  public let prefillQuantum: Int
  public let turns: Int

  public init(request: Active, prefillQuantum: Int, turns: Int) {
    self.request = request
    self.prefillQuantum = prefillQuantum
    self.turns = turns
  }
}

public struct CancellationResult<Payload, Active> {
  public let pending: [PendingRequest<Payload>]
  public let active: [Active]

  public init(pending: [PendingRequest<Payload>], active: [Active]) {
    self.pending = pending
    self.active = active
  }
}
