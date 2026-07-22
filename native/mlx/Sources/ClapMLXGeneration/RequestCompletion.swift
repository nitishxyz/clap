public enum RequestStatus: Equatable, Sendable {
  case active
  case completed
  case cancelled
  case failed

  public var isTerminal: Bool { self != .active }
}

public struct RequestCompletion: Equatable, Sendable {
  public let status: RequestStatus
  public let finishReason: String
  public let content: String
  public let generatedTokens: Int

  public init(status: RequestStatus, finishReason: String, content: String,
              generatedTokens: Int) {
    self.status = status
    self.finishReason = finishReason
    self.content = content
    self.generatedTokens = generatedTokens
  }
}
