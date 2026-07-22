struct BackloggedCommand {
  let line: String
  let v1RequestID: String?
}

struct CommandBacklog {
  private var commands: [BackloggedCommand] = []

  var isEmpty: Bool { commands.isEmpty }

  mutating func append(_ line: String, v1RequestID: String?) {
    commands.append(BackloggedCommand(line: line, v1RequestID: v1RequestID))
  }

  mutating func removeFirst() -> BackloggedCommand? {
    guard !commands.isEmpty else { return nil }
    return commands.removeFirst()
  }

  mutating func removeAll() -> [BackloggedCommand] {
    defer { commands.removeAll() }
    return commands
  }

  mutating func remove(requestID: String?) -> BackloggedCommand? {
    guard let requestID,
          let index = commands.firstIndex(where: { $0.v1RequestID == requestID }) else {
      return nil
    }
    return commands.remove(at: index)
  }
}
