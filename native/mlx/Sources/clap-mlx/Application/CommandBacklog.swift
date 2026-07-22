struct CommandBacklog {
  private var commands: [String] = []

  var isEmpty: Bool { commands.isEmpty }

  mutating func append(_ command: String) {
    commands.append(command)
  }

  mutating func removeFirst() -> String? {
    guard !commands.isEmpty else { return nil }
    return commands.removeFirst()
  }
}
