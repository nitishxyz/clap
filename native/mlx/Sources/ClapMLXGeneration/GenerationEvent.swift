public enum GenerationEvent: Equatable, Sendable {
  case prefill(done: Int, total: Int)
  case token(String)
  case content(String)
  case error(String)
  case completed(RequestCompletion)
}
