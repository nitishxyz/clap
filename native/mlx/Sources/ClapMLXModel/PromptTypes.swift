public struct PromptTokenLimitError: Error, Equatable, Sendable {
  public let message: String
  public let code: String

  public init(message: String, code: String) {
    self.message = message
    self.code = code
  }
}

public typealias PromptToolSpec = [String: any Sendable]

public struct PromptToolCall: Equatable, Sendable {
  public let name: String
  public let arguments: String?

  public init(name: String, arguments: String?) {
    self.name = name
    self.arguments = arguments
  }
}

public struct PromptMessage: Equatable, Sendable {
  public let role: String
  public let content: String?
  public let toolCalls: [PromptToolCall]?

  public init(role: String, content: String?, toolCalls: [PromptToolCall]? = nil) {
    self.role = role
    self.content = content
    self.toolCalls = toolCalls
  }
}

public struct PromptBoundaryDescriptor: Equatable, Sendable {
  public let kind: String
  public let throughMessage: Int?
  public let label: String?

  public init(kind: String, throughMessage: Int?, label: String?) {
    self.kind = kind
    self.throughMessage = throughMessage
    self.label = label
  }
}

public struct ResolvedPromptBoundary: Equatable, Sendable {
  public let tokenCount: Int?
  public let kind: String
  public let label: String?
  public let requested: Bool
  public let status: String
  public let skipReason: String?

  public init(tokenCount: Int?, kind: String, label: String?, requested: Bool,
              status: String, skipReason: String?) {
    self.tokenCount = tokenCount
    self.kind = kind
    self.label = label
    self.requested = requested
    self.status = status
    self.skipReason = skipReason
  }
}

public struct PreparedPrompt: Equatable, Sendable {
  public let tokens: [Int]
  public let stableBoundaries: [Int]
  public let structuralBoundaries: [ResolvedPromptBoundary]
  public let resolvedBoundaries: [Int: ResolvedPromptBoundary]
  public let usedFallback: Bool
}

public enum PromptRendererError: Error, Equatable, Sendable {
  case noMessages
}
