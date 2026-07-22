public struct PromptTokenLimitError: Error, Equatable, Sendable {
  public let message: String
  public let code: String

  public init(message: String, code: String) {
    self.message = message
    self.code = code
  }
}
