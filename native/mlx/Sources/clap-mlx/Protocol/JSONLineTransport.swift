import Foundation
import ClapMLXWorkerCore

nonisolated(unsafe) private let v1ProtocolWriter = V1ProtocolWriter()

func emitV1Ready() {
  v1ProtocolWriter.ready()
}

@discardableResult
func acceptV1(_ requestID: String) -> Bool {
  v1ProtocolWriter.accepted(requestID)
}

func failV1Decode(_ error: V1DecodeError) {
  if let id = error.requestID {
    _ = v1ProtocolWriter.accepted(id)
    v1ProtocolWriter.failed(id, code: error.code, message: error.description)
  } else {
    v1ProtocolWriter.diagnostic(error.description)
  }
}

func failV1Command(_ requestID: String, code: String = "worker_error", message: String) {
  v1ProtocolWriter.failed(requestID, code: code, message: message)
}

func v1JSONObject<T: Encodable>(_ value: T) -> Any {
  let data = try! JSONEncoder().encode(value)
  return try! JSONSerialization.jsonObject(with: data)
}

func completeV1Command(_ requestID: String, result: [String: Any]) {
  v1ProtocolWriter.completed(requestID, result: result)
}

func emit(id: String? = nil, started: Bool? = nil, token: String? = nil, content: String? = nil, loaded: Bool? = nil, unloaded: Bool? = nil, done: Bool? = nil, error: String? = nil, code: String? = nil, cancelled: Bool? = nil, finishReason: String? = nil, usage: WorkerUsage? = nil, cache: WorkerCache? = nil, timing: WorkerTiming? = nil, prefill: WorkerPrefill? = nil, memory: WorkerMemory? = nil, retention: WorkerRetention? = nil, tokenCapabilities: WorkerTokenCapabilities? = nil) {
  v1ProtocolWriter.map(id: id, started: started, token: token, content: content,
    loaded: loaded, unloaded: unloaded, done: done, error: error, code: code,
    cancelled: cancelled, finishReason: finishReason, usage: usage, cache: cache,
    timing: timing, prefill: prefill, memory: memory, retention: retention,
    tokenCapabilities: tokenCapabilities)
}

private final class V1ProtocolWriter {
  private let tracker = ProtocolTerminalTracker()
  private var generatedContent: [String: String] = [:]

  func ready() {
    write(["protocol": 1, "type": "ready",
      "worker_capabilities": ["backend": "mlx", "streaming": true],
      "model_capabilities": [:]])
  }

  func accepted(_ id: String) -> Bool {
    guard let scope = tracker.accept(id) else { return false }
    write(scoped(scope, type: "accepted")); return true
  }

  func diagnostic(_ message: String) {
    write(["protocol": 1, "type": "diagnostic", "level": "error", "message": message])
  }

  func failed(_ id: String, code: String, message: String,
              retryable: Bool = false, fatal: Bool = false) {
    guard let scope = tracker.terminal(id) else { return }
    let error = V1ProtocolError(code: code, message: message,
      retryable: retryable, fatal: fatal)
    write(scoped(scope, type: "failed", fields: ["error": jsonObject(error)]))
    generatedContent.removeValue(forKey: id)
  }

  func completed(_ id: String, result: [String: Any]) {
    guard let scope = tracker.terminal(id) else { return }
    write(scoped(scope, type: "completed", fields: ["result": result]))
    generatedContent.removeValue(forKey: id)
  }

  func map(id: String?, started: Bool?, token: String?, content: String?, loaded: Bool?,
           unloaded: Bool?, done: Bool?, error: String?, code: String?, cancelled: Bool?,
           finishReason: String?, usage: WorkerUsage?, cache: WorkerCache?, timing: WorkerTiming?,
           prefill: WorkerPrefill?, memory: WorkerMemory?, retention: WorkerRetention?,
           tokenCapabilities: WorkerTokenCapabilities?) {
    guard let id, !id.isEmpty else {
      var telemetry: [String: Any] = [:]
      if let memory { telemetry["memory"] = jsonObject(memory) }
      if let retention { telemetry["retention"] = jsonObject(retention) }
      if !telemetry.isEmpty {
        write(["protocol": 1, "type": "telemetry", "telemetry": telemetry])
      }
      return
    }
    if let error {
      failed(id, code: code ?? "worker_error", message: error,
        retryable: code == "generation_failed")
      return
    }
    if started == true { event(id, type: "started") }
    if let token {
      generatedContent[id, default: ""] += token
      event(id, type: "token", fields: ["text": token])
    }
    if let content {
      generatedContent[id, default: ""] += content
      event(id, type: "content", fields: ["content": content])
    }
    if let prefill {
      event(id, type: "prefill_progress", fields: ["completed": prefill.done,
        "total": prefill.total])
    }
    guard done == true else { return }
    var result: [String: Any] = [:]
    if loaded == true { result["kind"] = "loaded" }
    else if unloaded == true { result["kind"] = "unloaded" }
    else if cancelled == true {
      result["kind"] = "cancelled"
      result["finish_reason"] = "cancel"
    } else {
      result["kind"] = "generated"
      result["content"] = generatedContent[id] ?? ""
      if let finishReason { result["finish_reason"] = finishReason }
      if let usage { result["usage"] = jsonObject(usage) }
      if let cache { result["cache"] = jsonObject(cache) }
      if let timing { result["timing"] = jsonObject(timing) }
    }
    if let tokenCapabilities { result["token_capabilities"] = jsonObject(tokenCapabilities) }
    completed(id, result: result)
  }

  private func event(_ id: String, type: String, fields: [String: Any] = [:]) {
    guard let scope = tracker.event(id) else { return }
    write(scoped(scope, type: type, fields: fields))
  }

  private func scoped(_ scope: ProtocolSequence, type: String,
                      fields: [String: Any] = [:]) -> [String: Any] {
    var value = fields
    value["protocol"] = 1; value["type"] = type
    value["request_id"] = scope.requestID; value["sequence"] = scope.sequence
    return value
  }

  private func write(_ value: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
  }

  private func jsonObject<T: Encodable>(_ value: T) -> Any {
    let data = try! JSONEncoder().encode(value)
    return try! JSONSerialization.jsonObject(with: data)
  }
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
