import Foundation

func emit(id: String? = nil, started: Bool? = nil, token: String? = nil, content: String? = nil, loaded: Bool? = nil, unloaded: Bool? = nil, done: Bool? = nil, error: String? = nil, code: String? = nil, cancelled: Bool? = nil, finishReason: String? = nil, usage: WorkerUsage? = nil, cache: WorkerCache? = nil, timing: WorkerTiming? = nil, prefill: WorkerPrefill? = nil, memory: WorkerMemory? = nil, retention: WorkerRetention? = nil, tokenCapabilities: WorkerTokenCapabilities? = nil) {
  let message = WorkerMessage(id: id, started: started, token: token, content: content, loaded: loaded, unloaded: unloaded, done: done, error: error, code: code, cancelled: cancelled, finish_reason: finishReason, usage: usage, cache: cache, timing: timing, prefill: prefill, memory: memory, retention: retention, token_capabilities: tokenCapabilities)
  let data = try! JSONEncoder().encode(message)
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data([0x0a]))
}

// Buffers stdin lines so the main loop can poll for cancel messages while a
// generation is streaming.
actor LineBuffer {
  private var lines: [String] = []
  private var waiter: CheckedContinuation<String?, Never>?
  private var finished = false

  func push(_ line: String) {
    if let waiter {
      self.waiter = nil
      waiter.resume(returning: line)
    } else {
      lines.append(line)
    }
  }

  func finish() {
    finished = true
    if let waiter {
      self.waiter = nil
      waiter.resume(returning: nil)
    }
  }

  func next() async -> String? {
    if !lines.isEmpty { return lines.removeFirst() }
    if finished { return nil }
    return await withCheckedContinuation { waiter = $0 }
  }

  func poll() -> String? {
    lines.isEmpty ? nil : lines.removeFirst()
  }
}

func isCancelMessage(_ line: String, activeId: String?) -> Bool {
  guard let data = line.data(using: .utf8),
        let control = try? JSONDecoder().decode(ControlRequest.self, from: data),
        control.type == "cancel" else { return false }
  guard let target = control.id, !target.isEmpty else { return true }
  return target == activeId
}
