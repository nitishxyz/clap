public struct PendingRequest<Payload> {
  public let id: String?
  public let model: String?
  public let receivedNs: UInt64
  public let payload: Payload

  public init(id: String?, model: String?, receivedNs: UInt64, payload: Payload) {
    self.id = id
    self.model = model
    self.receivedNs = receivedNs
    self.payload = payload
  }
}

public struct ActiveRequestView: Equatable, Sendable {
  public let id: String?
  public let admissionOrder: UInt64
  public let residualPrefillTokens: Int
  public let decoding: Bool
  public let emittedFirstToken: Bool
  public let terminal: Bool
  public let cancelled: Bool

  public init(id: String?, admissionOrder: UInt64, residualPrefillTokens: Int,
              decoding: Bool, emittedFirstToken: Bool, terminal: Bool,
              cancelled: Bool) {
    self.id = id
    self.admissionOrder = admissionOrder
    self.residualPrefillTokens = residualPrefillTokens
    self.decoding = decoding
    self.emittedFirstToken = emittedFirstToken
    self.terminal = terminal
    self.cancelled = cancelled
  }
}
