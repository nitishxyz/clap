import Foundation
import Darwin
import ClapCacheBridge
import ClapCachePolicy
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

private func startupAvailableMemoryBytes() -> UInt64? {
  var statistics = vm_statistics64()
  var count = mach_msg_type_number_t(
    MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
  let status = withUnsafeMutablePointer(to: &statistics) { pointer in
    pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
    }
  }
  var pageSize: vm_size_t = 0
  guard status == KERN_SUCCESS,
        host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return nil }
  let pages = UInt64(statistics.free_count) + UInt64(statistics.inactive_count) +
    UInt64(statistics.purgeable_count)
  let bytes = pages.multipliedReportingOverflow(by: UInt64(pageSize))
  return bytes.overflow ? nil : bytes.partialValue
}

struct ChatMessage: Decodable {
  let role: String
  let content: String?
  let tool_calls: [IncomingToolCall]?
}

struct IncomingToolCall: Decodable {
  struct Function: Decodable {
    let name: String
    let arguments: String?
  }
  let function: Function
}

struct WorkerUsage: Encodable {
  let prompt_tokens: Int
  let completion_tokens: Int
}

struct WorkerCacheCandidate: Encodable {
  let slot: Int
  let generation: UInt64
  let state: String
  let shared_prefix_tokens: Int
  let namespace_compatible: Bool
  let model_compatible: Bool
  let session_compatible: Bool
  let generation_compatible: Bool
  let busy_eligible: Bool
  let lease_eligible: Bool
  let materialized: Bool
  let trim_eligible: Bool
  let copy_eligible: Bool
  let eligible: Bool
  let selected: Bool
  let rejection: String?
}

struct WorkerCache: Encodable {
  let hit: Bool
  let reused_tokens: Int
  let reuse_kind: String?
  let reuse_scope: String?
  let side_request: Bool
  let namespace: String?
  let donor_slot: Int?
  let target_slot: Int
  let evicted_slots: [Int]
  let decision_us: UInt64
  let planned_reuse_tokens: Int
  let realized_reuse_tokens: Int
  let fallback: String?
  let miss_reason: String?
  let candidates: [WorkerCacheCandidate]
  let prompt_token_hash: String
  let prompt_token_count: Int
  let stable_boundary_token_hash: String?
  let stable_boundary_token_count: Int
  let stable_boundary_kind: String?
}

func tokenFingerprint(_ tokens: [Int], count: Int) -> String {
  let limit = min(max(count, 0), tokens.count)
  var bytes = Array((cacheTelemetryKey + "|tokens-v1|\(limit)|").utf8)
  for token in tokens.prefix(limit) {
    var value = UInt32(truncatingIfNeeded: token).littleEndian
    withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
  }
  return (0..<4).map { domain -> String in
    var hash = UInt64(1_469_598_103_934_665_603) ^ UInt64(domain)
    for byte in bytes {
      hash ^= UInt64(byte)
      hash &*= 1_099_511_628_211
    }
    return String(format: "%016llx", hash)
  }.joined()
}

func cacheCandidateState(_ state: UInt32) -> String {
  switch state {
  case UInt32(CC_SLOT_SESSION): return "session"
  case UInt32(CC_SLOT_PROMPT_BOUNDARY): return "prompt_boundary"
  case UInt32(CC_SLOT_ANCHOR): return "anchor"
  default: return "empty"
  }
}

func cacheCandidateRejection(_ rejection: UInt32) -> String? {
  [1: "namespace", 2: "model_domain", 3: "generation", 4: "busy_lease",
   5: "materialization", 6: "session", 7: "nontrim", 8: "capability",
   9: "min_prefix", 10: "capacity", 11: "absent_anchor", 12: "lower_rank"][rejection]
}

struct WorkerPrefill: Encodable {
  let done: Int
  let total: Int
}

struct WorkerMemory: Encodable {
  let active_bytes: Int
  let cache_bytes: Int
  let peak_active_bytes: Int
}

struct WorkerActivePolicyInputs: Encodable {
  let physical_memory_bytes: UInt64
  let startup_available_bytes: UInt64?
  let model_active_bytes: UInt64?
  let retained_budget_bytes: UInt64
  let retained_bytes: UInt64
  let retained_growth_reserve_bytes: UInt64
  let os_reserve_bytes: UInt64
  let usable_runtime_bytes: UInt64
  let per_active_reserve_bytes: UInt64
  let processor_count: Int
  let hybrid_or_recurrent: Bool
}

struct WorkerActivePolicy: Encodable {
  let mode: String
  let selected_max: Int
  let backend_ceiling: Int
  let hardware_ceiling: Int
  let model_ceiling: Int
  let memory_ceiling: Int
  let reason: String
  let inputs: WorkerActivePolicyInputs
}

struct WorkerRetention: Encodable {
  let max_active: Int
  let queued: Int
  let previous_max_active: Int?
  let last_adjustment_reason: String?
  let last_adjustment_at: String?
  let retained_growth_reserve_bytes: UInt64
  let global_resident_memory_bytes: UInt64?
  let pressure_state: String?
  let active_policy: WorkerActivePolicy
  let active: Int
  let retained_total: Int
  let retained_sessions: Int
  let retained_anchors: Int
  let retained_bytes: UInt64
  let session_bytes: UInt64
  let anchor_bytes: UInt64
  let budget_bytes: UInt64
  let high_watermark_bytes: UInt64
  let low_watermark_bytes: UInt64
  let under_pressure: Bool
  let hard_ceiling: Int
  let eviction_reason: String?
  let eviction_count: UInt64
}

struct WorkerTokenCapabilities: Encodable {
  enum CodingKeys: String, CodingKey {
    case model_context_window
    case effective_context_window
    case max_input_tokens
    case max_output_tokens
    case model_context_window_source
    case max_output_tokens_source
    case backend_allocation_cap
    case user_configured_override
  }

  let model_context_window: Int?
  let effective_context_window: Int?
  let max_input_tokens: Int?
  let max_output_tokens: Int?
  let model_context_window_source: String?
  let max_output_tokens_source: String?
  let backend_allocation_cap: Int?
  let user_configured_override: Int?

  func encode(to encoder: any Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(model_context_window, forKey: .model_context_window)
    try container.encode(effective_context_window, forKey: .effective_context_window)
    try container.encode(max_input_tokens, forKey: .max_input_tokens)
    try container.encode(max_output_tokens, forKey: .max_output_tokens)
    try container.encode(model_context_window_source, forKey: .model_context_window_source)
    try container.encode(max_output_tokens_source, forKey: .max_output_tokens_source)
    try container.encode(backend_allocation_cap, forKey: .backend_allocation_cap)
    try container.encode(user_configured_override, forKey: .user_configured_override)
  }
}

struct WorkerMessage: Encodable {
  let id: String?
  let started: Bool?
  let token: String?
  let content: String?
  let loaded: Bool?
  let unloaded: Bool?
  let done: Bool?
  let error: String?
  let code: String?
  let cancelled: Bool?
  let finish_reason: String?
  let usage: WorkerUsage?
  let cache: WorkerCache?
  let prefill: WorkerPrefill?
  let memory: WorkerMemory?
  let retention: WorkerRetention?
  let token_capabilities: WorkerTokenCapabilities?
}

// OpenAI-style stop can be a bare string or an array of strings.
enum StopField: Decodable {
  case none
  case sequences([String])

  init(from decoder: any Swift.Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let single = try? container.decode(String.self) {
      self = .sequences([single])
    } else if let list = try? container.decode([String].self) {
      self = .sequences(list)
    } else {
      self = .none
    }
  }

  var values: [String] {
    if case .sequences(let list) = self { return list.filter { !$0.isEmpty } }
    return []
  }
}

struct ControlRequest: Decodable {
  let id: String?
  let type: String?
  let model: String?
  let max_active: Int?
  let previous_max_active: Int?
  let limiting_reason: String?
  let last_adjustment_reason: String?
  let last_adjustment_at: String?
  let retained_growth_reserve_bytes: UInt64?
  let global_resident_memory_bytes: UInt64?
  let pressure_state: String?
  let messages: [ChatMessage]?
  let stream: Bool?
  let max_tokens: Int?
  let temperature: Double?
  let top_p: Double?
  let top_k: Int?
  let min_p: Double?
  let seed: UInt64?
  let stop: StopField?
  let repetition_penalty: Double?
  let presence_penalty: Double?
  let frequency_penalty: Double?
  let cache: CacheIntent?
}

func memorySnapshot() -> WorkerMemory {
  let snapshot = Memory.snapshot()
  return WorkerMemory(active_bytes: snapshot.activeMemory, cache_bytes: snapshot.cacheMemory, peak_active_bytes: snapshot.peakMemory)
}

func emit(id: String? = nil, started: Bool? = nil, token: String? = nil, content: String? = nil, loaded: Bool? = nil, unloaded: Bool? = nil, done: Bool? = nil, error: String? = nil, code: String? = nil, cancelled: Bool? = nil, finishReason: String? = nil, usage: WorkerUsage? = nil, cache: WorkerCache? = nil, prefill: WorkerPrefill? = nil, memory: WorkerMemory? = nil, retention: WorkerRetention? = nil, tokenCapabilities: WorkerTokenCapabilities? = nil) {
  let message = WorkerMessage(id: id, started: started, token: token, content: content, loaded: loaded, unloaded: unloaded, done: done, error: error, code: code, cancelled: cancelled, finish_reason: finishReason, usage: usage, cache: cache, prefill: prefill, memory: memory, retention: retention, token_capabilities: tokenCapabilities)
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

struct ToolsEnvelope: Decodable {
  let tools: [JSONValue]?
}

struct EmittedToolCall: Encodable {
  let name: String
  let arguments: [String: JSONValue]
}

struct EmittedToolCalls: Encodable {
  let tool_calls: [EmittedToolCall]
}

// Canonical text form for an assistant message that carried structured
// tool_calls so chat templates see the call the model originally made.
func encodeIncomingToolCalls(_ calls: [IncomingToolCall]) -> String? {
  let converted = calls.map { call -> EmittedToolCall in
    var arguments: [String: JSONValue] = [:]
    if let raw = call.function.arguments, let data = raw.data(using: .utf8),
       let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: data) {
      arguments = decoded
    }
    return EmittedToolCall(name: call.function.name, arguments: arguments)
  }
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  guard let data = try? encoder.encode(EmittedToolCalls(tool_calls: converted)) else { return nil }
  return String(data: data, encoding: .utf8)
}

func messageText(_ message: ChatMessage) -> String? {
  if let content = message.content, !content.isEmpty { return content }
  if let calls = message.tool_calls, !calls.isEmpty { return encodeIncomingToolCalls(calls) }
  return nil
}

func sendableValue(_ value: JSONValue) -> any Sendable {
  switch value {
  case .null: return NSNull()
  case .bool(let v): return v
  case .int(let v): return v
  case .double(let v): return v
  case .string(let v): return v
  case .array(let items): return items.map { sendableValue($0) }
  case .object(let entries): return entries.mapValues { sendableValue($0) }
  }
}

// Structured template message for an assistant turn that carried tool_calls:
// letting the chat template render the model's trained tool-call format keeps
// the continuation prompt byte-identical to what the model generated, which
// preserves KV cache extension even for non-rewindable (sliding-window) caches.
func structuredToolCallMessage(_ message: ChatMessage) -> [String: any Sendable]? {
  guard let calls = message.tool_calls, !calls.isEmpty else { return nil }
  let rendered: [[String: any Sendable]] = calls.map { call in
    var arguments: [String: any Sendable] = [:]
    if let raw = call.function.arguments, let data = raw.data(using: .utf8),
       let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: data) {
      arguments = decoded.mapValues { sendableValue($0) }
    }
    let function: [String: any Sendable] = ["name": call.function.name, "arguments": arguments]
    return ["type": "function", "function": function, "name": call.function.name, "arguments": arguments]
  }
  var result: [String: any Sendable] = ["role": "assistant", "tool_calls": rendered]
  if let content = message.content, !content.isEmpty { result["content"] = content }
  else { result["content"] = "" }
  return result
}

// Some model templates expect a flattened tool and apply string filters to
// JSON-Schema `type`. Try the caller's exact OpenAI tool objects first; this
// compatibility view is only a second render attempt. Nullable single-type
// unions retain their semantics through the sibling `nullable` field used by
// those templates. Multi-type unions remain unchanged rather than being
// silently narrowed.
func templateCompatibleSchemaValue(_ value: any Sendable) -> any Sendable {
  if let dict = value as? [String: any Sendable] {
    var result: [String: any Sendable] = [:]
    var nullableUnion = false
    for (key, entry) in dict {
      if key == "type", let list = entry as? [any Sendable] {
        let types = list.compactMap { $0 as? String }
        let nonNull = types.filter { $0 != "null" }
        if types.count == list.count, types.contains("null"), nonNull.count == 1 {
          result[key] = nonNull[0]
          nullableUnion = true
          continue
        }
      }
      result[key] = templateCompatibleSchemaValue(entry)
    }
    if nullableUnion { result["nullable"] = true }
    return result
  }
  if let list = value as? [any Sendable] {
    return list.map { templateCompatibleSchemaValue($0) }
  }
  return value
}

func templateCompatibleToolSpec(_ spec: ToolSpec) -> ToolSpec {
  var result = spec
  if let function = spec["function"] as? [String: any Sendable] {
    for key in ["name", "description", "parameters"] where result[key] == nil {
      result[key] = function[key]
    }
  }
  return templateCompatibleSchemaValue(result) as? ToolSpec ?? result
}

// Compact JSON for tool specs used by the in-worker instruction fallback.
func toolSpecJson(_ specs: [ToolSpec]) -> String {
  let trimmed = specs.map { spec -> [String: any Sendable] in
    let source = spec["function"] as? [String: any Sendable] ?? spec
    var entry: [String: any Sendable] = [:]
    for key in ["name", "description", "parameters"] {
      if let value = source[key] { entry[key] = value }
    }
    return entry
  }
  guard JSONSerialization.isValidJSONObject(trimmed),
        let data = try? JSONSerialization.data(withJSONObject: trimmed, options: [.sortedKeys]),
        let text = String(data: data, encoding: .utf8) else { return "[]" }
  return text
}

func debugLog(_ message: String) {
  FileHandle.standardError.write(Data("[clap-mlx] \(message)\n".utf8))
}

func usesGemma4FallbackPrompt(_ modelDirectory: URL) -> Bool {
  let configURL = modelDirectory.appendingPathComponent("config.json")
  let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
  let jinjaURL = modelDirectory.appendingPathComponent("chat_template.jinja")
  if FileManager.default.fileExists(atPath: jinjaURL.path) { return false }
  guard let config = try? String(contentsOf: configURL, encoding: .utf8) else { return false }
  let tokenizerConfig = (try? String(contentsOf: tokenizerConfigURL, encoding: .utf8)) ?? ""
  return config.contains("\"model_type\": \"gemma4\"") && !tokenizerConfig.contains("\"chat_template\"")
}

func gemma4Prompt(entries: [(role: String, content: String)]) -> String {
  var parts = ["<bos>"]
  for entry in entries {
    let role = entry.role == "assistant" ? "assistant" : "user"
    parts.append("<|turn>\(role)\n\(entry.content)")
  }
  parts.append("<|turn>assistant\n")
  return parts.joined()
}

func validateModelDirectory(_ path: String) throws -> URL {
  let url = URL(fileURLWithPath: path)
  var isDirectory: ObjCBool = false
  guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
    throw WorkerError.invalidModelDirectory("MLX model directory not found: \(path)")
  }
  guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
    throw WorkerError.invalidModelDirectory("MLX model directory is missing config.json: \(path)")
  }
  return url
}

enum WorkerError: Error, CustomStringConvertible {
  case invalidModelDirectory(String)

  var description: String {
    switch self {
    case .invalidModelDirectory(let message): message
    }
  }
}

// ModelContext members (model, tokenizer) are not Sendable but are only used
// from this single-threaded main loop; ChatSession uses the same pattern.
struct UncheckedBox<T>: @unchecked Sendable {
  let value: T
}

func declaredEosTokenIds(_ url: URL) -> Set<Int> {
  guard let data = try? Data(contentsOf: url),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
  var ids: Set<Int> = []
  func collect(_ value: Any?) {
    if let id = value as? Int { ids.insert(id) }
    if let list = value as? [Any] { for item in list { collect(item) } }
  }
  collect(json["eos_token_id"])
  if let textConfig = json["text_config"] as? [String: Any] { collect(textConfig["eos_token_id"]) }
  return ids
}

func main() async {
    guard #available(macOS 14.0, *) else {
      emit(error: "clap-mlx requires macOS 14 or newer on Apple Silicon")
      exit(2)
    }
    #if !arch(arm64)
    emit(error: "clap-mlx requires Apple Silicon arm64")
    exit(2)
    #endif

    var loadedModel: String?
    var loadedDirectory: URL?
    var languageModel: (any LanguageModel)?
    var tokenizer: (any MLXLMCommon.Tokenizer)?
    var eosTokenIds: Set<Int> = []
    var declaredModelContextLength = 0
    var modelContextLength = 0
    var modelMaxOutputTokens = 0
    var modelContextLengthSource: String?
    var modelMaxOutputTokensSource: String?

    let env = ProcessInfo.processInfo.environment
    let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    let availableMemoryAtStartup = startupAvailableMemoryBytes()
    let retentionConfig = RetentionConfiguration.fromEnvironment(env,
      physicalMemoryBytes: physicalMemoryBytes,
      startupAvailableMemoryBytes: availableMemoryAtStartup)
    let retainedGrowthMinimumBytes = UInt64(env["CLAP_MLX_RETAINED_GROWTH_MIN_BYTES"] ?? "")
      ?? 64 * 1_048_576
    let retainedGrowthReservePercent = UInt64(env["CLAP_MLX_RETAINED_GROWTH_RESERVE_PERCENT"] ?? "")
      ?? 10
    var activePolicy = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: physicalMemoryBytes,
      startupAvailableMemoryBytes: availableMemoryAtStartup, modelActiveBytes: nil,
      retainedBudgetBytes: retentionConfig.physicalByteBudget,
      retainedGrowthMinimumBytes: retainedGrowthMinimumBytes,
      retainedGrowthReservePercent: retainedGrowthReservePercent,
      retainedCeiling: retentionConfig.hardCeiling,
      processorCount: ProcessInfo.processInfo.processorCount))
    var maxActive = activePolicy.selectedMax
    // Context window policy (parity with the llama worker): default to the
    // model's trained context; CLAP_MLX_CONTEXT pins it, and
    // CLAP_MLX_MAX_SESSION_CTX caps any single session's share.
    let contextOverride = Int(env["CLAP_MLX_CONTEXT"] ?? "") ?? 0
    let sessionCap = Int(env["CLAP_MLX_MAX_SESSION_CTX"] ?? "") ?? 0
    // KV cache quantization (parity with CLAP_LLAMA_KV_TYPE): q8_0/q4_0/f16.
    let kvBits: Int? = switch env["CLAP_MLX_KV_TYPE"] ?? "f16" {
    case "q8_0": 8
    case "q4_0": 4
    default: nil
    }

    // Physical KV slots mirror Rust coordinator state. The worker executes
    // authorized operations but does not independently choose cache policy.
    final class KVSlot {
      var caches: [KVCache] = []
      var tokens: [Int] = []
      var lastUsed: UInt64 = 0
      var busy = false
      // Anchor slots hold an exact-state snapshot of a shared prefix (e.g.
      // the org-wide system prompt) taken before rotating/sliding-window
      // caches rotate it away. They are only used via copy() (whole-copy
      // branching) and never extended in place.
      var isAnchor = false
      var anchorScope: String? = nil
      // A request-local end-of-prompt snapshot restored after decode. Unlike
      // a dedicated prefix anchor, this remains the session's normal slot.
      var isPromptBoundary = false
      var coordinatorGeneration: UInt64 = 0
      var cacheIdentity: PhysicalCacheIdentity? = nil
    }
    var retainedRegistry = RetainedRegistry<KVSlot>(maxActive: maxActive,
      hardCeiling: retentionConfig.hardCeiling)
    var kvSlots: [KVSlot] {
      retainedRegistry.slotIDs.compactMap { retainedRegistry.entry(for: $0) }
    }
    var kvUseCounter: UInt64 = 0
    var cacheCoordinator: CacheCoordinator?
    var cacheDomain = ""
    var lastEvictionReason: String?
    var previousMaxActive: Int?
    var coordinatedLimitingReason: String?
    var lastAdjustmentReason: String?
    var lastAdjustmentAt: String?
    var coordinatedGrowthReserveBytes: UInt64?
    var globalResidentMemoryBytes: UInt64?
    var pressureState: String?
    var activePolicyModelBytes: UInt64?
    var activePolicyHybrid = false
    func invalidateKVCache() {
      try? cacheCoordinator?.reset()
      cacheCoordinator = nil
      retainedRegistry = RetainedRegistry<KVSlot>(maxActive: maxActive,
        hardCeiling: retentionConfig.hardCeiling)
      kvUseCounter = 0
      lastEvictionReason = nil
    }

    func clearPhysicalSlot(_ slot: KVSlot) {
      slot.caches = []
      slot.tokens = []
      slot.isAnchor = false
      slot.isPromptBoundary = false
      slot.anchorScope = nil
      slot.cacheIdentity = nil
      slot.coordinatorGeneration = 0
    }

    func physicalCacheBytes(_ caches: [KVCache]) -> UInt64 {
      let arrays = caches.flatMap(\.state).map {
        CacheArrayDescriptor(storageIdentity: UInt64(bitPattern:
          Int64(ObjectIdentifier($0).hashValue)),
          elementCount: $0.size, itemSize: $0.itemSize,
          allocatedBytes: $0.nbytes)
      }
      return max(1, PhysicalCacheByteEstimator.estimate(arrays: arrays))
    }

    func retentionSnapshot(queued: Int = 0) -> WorkerRetention? {
      guard let coordinator = cacheCoordinator,
            let telemetry = try? coordinator.retentionTelemetry() else { return nil }
      let retainedBytes = min(telemetry.total_bytes, retentionConfig.physicalByteBudget)
      let remainingBudget = retentionConfig.physicalByteBudget - retainedBytes
      let growthReserve = min(remainingBudget, max(retainedGrowthMinimumBytes,
        remainingBudget / 100 * min(100, retainedGrowthReservePercent)))
      let policy = WorkerActivePolicy(mode: activePolicy.mode,
        selected_max: maxActive, backend_ceiling: activePolicy.backendCeiling,
        hardware_ceiling: activePolicy.hardwareCeiling, model_ceiling: activePolicy.modelCeiling,
        memory_ceiling: activePolicy.memoryCeiling,
        reason: coordinatedLimitingReason ?? activePolicy.reason,
        inputs: WorkerActivePolicyInputs(physical_memory_bytes: physicalMemoryBytes,
          startup_available_bytes: availableMemoryAtStartup,
          model_active_bytes: activePolicyModelBytes,
          retained_budget_bytes: retentionConfig.physicalByteBudget,
          retained_bytes: retainedBytes,
          retained_growth_reserve_bytes: growthReserve,
          os_reserve_bytes: activePolicy.osReserveBytes,
          usable_runtime_bytes: activePolicy.usableRuntimeBytes,
          per_active_reserve_bytes: activePolicy.perActiveReserveBytes,
          processor_count: ProcessInfo.processInfo.processorCount,
          hybrid_or_recurrent: activePolicyHybrid))
      return WorkerRetention(max_active: maxActive, queued: queued,
        previous_max_active: previousMaxActive,
        last_adjustment_reason: lastAdjustmentReason,
        last_adjustment_at: lastAdjustmentAt,
        retained_growth_reserve_bytes: coordinatedGrowthReserveBytes ?? growthReserve,
        global_resident_memory_bytes: globalResidentMemoryBytes,
        pressure_state: pressureState, active_policy: policy,
        active: retainedRegistry.activeCount,
        retained_total: Int(telemetry.total_slots), retained_sessions: Int(telemetry.session_slots),
        retained_anchors: Int(telemetry.anchor_slots), retained_bytes: telemetry.total_bytes,
        session_bytes: telemetry.session_bytes, anchor_bytes: telemetry.anchor_bytes,
        budget_bytes: telemetry.physical_byte_budget,
        high_watermark_bytes: telemetry.high_watermark_bytes,
        low_watermark_bytes: telemetry.low_watermark_bytes,
        under_pressure: telemetry.under_pressure != 0,
        hard_ceiling: retentionConfig.hardCeiling, eviction_reason: lastEvictionReason,
        eviction_count: telemetry.evictions)
    }

    func loadModel(_ model: String, directory: URL) async throws {
      invalidateKVCache()
      let container = try await loadModelContainer(from: directory, using: #huggingFaceTokenizerLoader())
      let box = await container.perform { context in
        UncheckedBox(value: (context.model, context.tokenizer, context.configuration.extraEOSTokens))
      }
      languageModel = box.value.0
      tokenizer = box.value.1
      loadedModel = model
      loadedDirectory = directory
      eosTokenIds = []
      if let eos = box.value.1.eosTokenId { eosTokenIds.insert(eos) }
      for extra in box.value.2 {
        if let id = box.value.1.convertTokenToId(extra) { eosTokenIds.insert(id) }
      }
      // HF checkpoints declare turn-ending tokens (e.g. gemma's <turn|>) in
      // config.json / generation_config.json eos_token_id, not the tokenizer.
      for file in ["generation_config.json", "config.json"] {
        eosTokenIds.formUnion(declaredEosTokenIds(directory.appendingPathComponent(file)))
      }
      let metadata = DeclaredModelMetadata.load(from: directory)
      declaredModelContextLength = metadata.context?.value ?? 0
      modelContextLengthSource = metadata.context?.source
      let knownContextCaps = [declaredModelContextLength, contextOverride, sessionCap].filter { $0 > 0 }
      modelContextLength = knownContextCaps.min() ?? 0
      modelMaxOutputTokens = metadata.maxOutputTokens?.value ?? 0
      modelMaxOutputTokensSource = metadata.maxOutputTokens?.source
      let outputOverride = Int(env["CLAP_MLX_MAX_OUTPUT"] ?? "") ?? 0
      if outputOverride > 0 {
        if modelMaxOutputTokens == 0 || outputOverride < modelMaxOutputTokens {
          modelMaxOutputTokens = outputOverride
          modelMaxOutputTokensSource = "environment:CLAP_MLX_MAX_OUTPUT"
        }
      }
      cacheDomain = "\(model)|mlx|ctx=\(modelContextLength)|kv=\(kvBits.map(String.init) ?? "f16")|layout=1"
      Memory.clearCache()
      let memory = memorySnapshot()
      activePolicyModelBytes = memory.active_bytes > 0 ? UInt64(memory.active_bytes) : nil
      let capabilityText = [metadata.architecture, metadata.modelType]
        .compactMap { $0?.lowercased() }.joined(separator: " ")
      activePolicyHybrid = ["hybrid", "recurrent", "mamba", "deltanet", "ssm"]
        .contains { capabilityText.contains($0) }
      activePolicy = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
        explicitMax: Int(env["CLAP_MAX_ACTIVE"] ?? ""),
        physicalMemoryBytes: physicalMemoryBytes,
        startupAvailableMemoryBytes: availableMemoryAtStartup,
        modelActiveBytes: activePolicyModelBytes,
        retainedBudgetBytes: retentionConfig.physicalByteBudget,
        retainedGrowthMinimumBytes: retainedGrowthMinimumBytes,
        retainedGrowthReservePercent: retainedGrowthReservePercent,
        retainedCeiling: retentionConfig.hardCeiling,
        processorCount: ProcessInfo.processInfo.processorCount,
        isHybridOrRecurrent: activePolicyHybrid))
      maxActive = activePolicy.selectedMax
      retainedRegistry = RetainedRegistry<KVSlot>(maxActive: maxActive,
        hardCeiling: retentionConfig.hardCeiling)
      do {
        cacheCoordinator = try CacheCoordinator(retention: retentionConfig, capacity: Int.max / 4)
        for slotID in 0..<retentionConfig.initialEntries {
          let slot = KVSlot()
          slot.coordinatorGeneration = try cacheCoordinator?.slot(slotID).generation ?? 0
          try retainedRegistry.register(slotID: UInt32(slotID), entry: slot)
        }
      } catch {
        cacheCoordinator = nil
        debugLog("cache coordinator unavailable; cache admission fails closed: \(error)")
      }
      if kvBits != nil { debugLog("kv cache quantization enabled: \(kvBits!)-bit") }
      debugLog("declared metadata: architecture=\(metadata.architecture ?? "unknown") model_type=\(metadata.modelType ?? "unknown") context_source=\(modelContextLengthSource ?? "unknown") sliding_window=\(metadata.slidingWindow?.value.description ?? "unknown") output_source=\(modelMaxOutputTokensSource ?? "unknown")")
      debugLog("context length: \(modelContextLength > 0 ? String(modelContextLength) : "unknown")\(sessionCap > 0 ? ", session cap \(sessionCap)" : "")")
      debugLog("model loaded; eos token ids: \(eosTokenIds.sorted())")
      debugLog("active concurrency: mode=\(activePolicy.mode) selected=\(activePolicy.selectedMax) reason=\(activePolicy.reason) memory_ceiling=\(activePolicy.memoryCeiling)")
      debugLog("mlx memory after load: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
    }

    let buffer = LineBuffer()
    let readerTask = Task.detached {
      do {
        for try await line in FileHandle.standardInput.bytes.lines {
          await buffer.push(String(line))
        }
      } catch {}
      await buffer.finish()
    }
    defer { readerTask.cancel() }
    // ---- Interleaved multi-request scheduler ----------------------------
    // Mirrors the llama.cpp worker's continuous batching at the scheduling
    // level: several requests are active at once, each stepped in round-robin
    // (one prefill chunk OR a few decode tokens per pass), so a long prefill
    // or generation never blocks other sessions' token streams. MLX evaluates
    // sequences one at a time on Metal (no fused multi-sequence batch yet),
    // so aggregate throughput is shared — but head-of-line blocking is gone.

    let prefillChunk = 512
    let decodeStepsPerPass = 6

    final class ActiveRequest {
      let id: String?
      let streaming: Bool
      let maxTokens: Int
      let promptTokens: [Int]
      let reusedTokens: Int
      let reuseKind: String?
      let reuseScope: String?
      let cacheIdentity: CacheIdentity
      let cacheDecision: CacheDecision?
      let cacheCandidates: [CacheCandidateEvaluation]
      let cacheEvictions: [Int]
      let cacheFallback: String?
      let slotIndex: Int
      let slot: KVSlot
      var caches: [KVCache]
      var promptBoundaryCaches: [KVCache]? = nil
      var continuationBoundary: Int? = nil
      var continuationBoundaryCaches: [KVCache]? = nil
      var fedTokens: [Int]
      var suffix: [Int]
      var pos = 0
      var iterator: TokenIterator?
      var detokenizer: NaiveStreamingDetokenizer
      var sampledTokens: [Int] = []
      var collected = ""
      var emitted = 0  // chars of `collected` already streamed (stop holdback)
      var generatedCount = 0
      var finishReason = "stop"
      var cancelled = false
      var completed = false
      var failed = false
      // Absolute prompt index where a prefix anchor should be captured
      // during this request's prefill (set when a donor existed but its
      // caches could not rewind to the shared boundary).
      var anchorPlantAt: Int? = nil
      var anchorPlantScope: String? = nil
      var anchorPlanted = false
      let parameters: GenerateParameters
      let stops: [String]
      let holdback: Int

      init(id: String?, streaming: Bool, maxTokens: Int, promptTokens: [Int], reusedTokens: Int, reuseKind: String?, reuseScope: String?, cacheIdentity: CacheIdentity, cacheDecision: CacheDecision?, cacheCandidates: [CacheCandidateEvaluation], cacheEvictions: [Int], cacheFallback: String?, slotIndex: Int, slot: KVSlot, caches: [KVCache], fedTokens: [Int], suffix: [Int], detokenizer: NaiveStreamingDetokenizer, parameters: GenerateParameters, stops: [String]) {
        self.id = id
        self.streaming = streaming
        self.maxTokens = maxTokens
        self.promptTokens = promptTokens
        self.reusedTokens = reusedTokens
        self.reuseKind = reuseKind
        self.reuseScope = reuseScope
        self.cacheIdentity = cacheIdentity
        self.cacheDecision = cacheDecision
        self.cacheCandidates = cacheCandidates
        self.cacheEvictions = cacheEvictions
        self.cacheFallback = cacheFallback
        self.slotIndex = slotIndex
        self.slot = slot
        self.caches = caches
        self.fedTokens = fedTokens
        self.suffix = suffix
        self.detokenizer = detokenizer
        self.parameters = parameters
        self.stops = stops
        self.holdback = stops.map { $0.count }.max().map { $0 - 1 } ?? 0
      }
    }

    var active: [ActiveRequest] = []
    var pendingChats: [(id: String?, control: ControlRequest, data: Data)] = []
    var controlBacklog: [String] = []
    var allocatorNeedsIdleClear = false

    // Recurrent caches such as Mamba keep state but intentionally report an
    // offset of zero. Hybrid models still have attention caches whose maximum
    // offset tracks the sequence length; pure recurrent models use the token
    // bookkeeping fallback supplied by the caller.
    func cacheSequenceLength(_ caches: [KVCache], fallback: Int) -> Int {
      let offset = caches.map(\.offset).max() ?? 0
      return offset > 0 ? offset : fallback
    }

    func ensureAdmissionSlot() throws {
      guard !kvSlots.contains(where: { !$0.busy && $0.caches.isEmpty }) else { return }
      guard retainedRegistry.count < retentionConfig.hardCeiling,
            let coordinator = cacheCoordinator else { return }
      let registered = try coordinator.registerSlot()
      let slot = KVSlot()
      slot.coordinatorGeneration = registered.generation
      try retainedRegistry.register(slotID: UInt32(registered.slot), entry: slot)
    }

    func finalize(_ req: ActiveRequest) {
      req.slot.busy = false
      retainedRegistry.release(slotID: UInt32(req.slotIndex))
      if req.failed {
        let generation = req.slot.coordinatorGeneration
        clearPhysicalSlot(req.slot)
        if let coordinator = cacheCoordinator, generation != 0 {
          req.slot.coordinatorGeneration = (try? coordinator.invalidate(
            slot: req.slotIndex, generation: generation)) ?? 0
        }
        return
      }
      if let continuationBoundary = req.continuationBoundary,
         let continuationCaches = req.continuationBoundaryCaches {
        req.slot.caches = continuationCaches
        req.slot.tokens = Array(req.promptTokens.prefix(continuationBoundary))
        req.slot.isPromptBoundary = true
        req.slot.anchorScope = "conversation"
        debugLog("restored rolling conversation anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(continuationBoundary) tokens; discarded \(req.promptTokens.count - continuationBoundary) prompt suffix tokens and \(req.generatedCount) decoded tokens")
      } else if let promptBoundary = req.promptBoundaryCaches {
        req.slot.caches = promptBoundary
        req.slot.tokens = req.promptTokens
        req.slot.isPromptBoundary = true
        req.slot.anchorScope = "conversation"
        debugLog("restored prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens; discarded \(req.generatedCount) decoded tokens")
      } else {
        // The cache offset is authoritative for what is resident; any mismatch
        // here corrupts the next request's prefix trim.
        let full = req.fedTokens + req.sampledTokens
        let cacheLength = cacheSequenceLength(req.caches, fallback: full.count)
        req.slot.tokens = Array(full.prefix(min(cacheLength, full.count)))
        req.slot.isPromptBoundary = false
        req.slot.anchorScope = nil
      }
      if let coordinator = cacheCoordinator, req.slot.coordinatorGeneration != 0 {
        do {
          req.slot.coordinatorGeneration = try coordinator.confirm(
            slot: req.slotIndex, generation: req.slot.coordinatorGeneration,
            tokens: req.slot.tokens,
            state: req.slot.isPromptBoundary ? UInt32(CC_SLOT_PROMPT_BOUNDARY) : UInt32(CC_SLOT_SESSION),
            busy: true, physicalBytes: physicalCacheBytes(req.slot.caches))
          try coordinator.setBusy(slot: req.slotIndex,
                                  generation: req.slot.coordinatorGeneration, busy: false)
        } catch {
          let generation = req.slot.coordinatorGeneration
          if generation != 0 {
            _ = try? coordinator.invalidate(slot: req.slotIndex, generation: generation)
          }
          req.slot.coordinatorGeneration = 0
          debugLog("cache finalize metadata failed: \(error)")
        }
      }
      // Flush any text held back for stop-sequence matching.
      if req.streaming && !req.cancelled && req.emitted < req.collected.count {
        let tail = String(req.collected.dropFirst(req.emitted))
        if !tail.isEmpty { emit(id: req.id, token: tail) }
        req.emitted = req.collected.count
      }
      if !req.streaming && !req.collected.isEmpty && !req.cancelled {
        emit(id: req.id, content: req.collected)
      }
      let usage = WorkerUsage(prompt_tokens: req.promptTokens.count, completion_tokens: req.generatedCount)
      let cacheInfo = WorkerCache(
        hit: req.reusedTokens > 0,
        reused_tokens: req.reusedTokens,
        reuse_kind: req.reuseKind,
        reuse_scope: req.reuseScope,
        side_request: req.cacheIdentity.sideRequest,
        namespace: req.cacheIdentity.exportedNamespace,
        donor_slot: req.cacheDecision?.donor,
        target_slot: req.slotIndex,
        evicted_slots: req.cacheEvictions,
        decision_us: req.cacheDecision?.decisionUs ?? 0,
        planned_reuse_tokens: req.cacheDecision?.plannedReuseTokens ?? req.reusedTokens,
        realized_reuse_tokens: req.cacheDecision?.realizedReuseTokens ?? req.reusedTokens,
        fallback: req.cacheFallback,
        miss_reason: req.reusedTokens > 0 ? nil : "no_shared_prefix",
        candidates: req.cacheCandidates.map { candidate in
          WorkerCacheCandidate(
            slot: candidate.slot, generation: candidate.generation,
            state: cacheCandidateState(candidate.state),
            shared_prefix_tokens: candidate.sharedPrefixTokens,
            namespace_compatible: candidate.namespaceCompatible,
            model_compatible: candidate.modelCompatible,
            session_compatible: candidate.sessionCompatible,
            generation_compatible: candidate.generationCompatible,
            busy_eligible: candidate.busyEligible,
            lease_eligible: candidate.leaseEligible,
            materialized: candidate.materialized,
            trim_eligible: candidate.trimEligible,
            copy_eligible: candidate.copyEligible,
            eligible: candidate.eligible, selected: candidate.selected,
            rejection: cacheCandidateRejection(candidate.rejection))
        },
        prompt_token_hash: tokenFingerprint(req.promptTokens, count: req.promptTokens.count),
        prompt_token_count: req.promptTokens.count,
        stable_boundary_token_hash: req.anchorPlantAt.map {
          tokenFingerprint(req.promptTokens, count: $0)
        },
        stable_boundary_token_count: req.anchorPlantAt ?? 0,
        stable_boundary_kind: req.anchorPlantAt == nil ? nil : "prompt"
      )
      emit(id: req.id, done: true, cancelled: req.cancelled ? true : nil, finishReason: req.cancelled ? "cancel" : req.finishReason, usage: usage, cache: cacheInfo)
    }

    func prepareRequest(id: String?, control: ControlRequest, data: Data) async -> ActiveRequest? {
      do {
        guard let model = control.model else {
          emit(id: id, error: "chat.model is required")
          return nil
        }
        let modelDirectory = try validateModelDirectory(model)
        if loadedModel != model || languageModel == nil {
          try await loadModel(model, directory: modelDirectory)
        }
        guard let lm = languageModel, let tok = tokenizer else {
          emit(id: id, error: "model is not loaded")
          return nil
        }
        let requestedMaxTokens = control.max_tokens
        let temperature = control.temperature ?? 0.7
        // Full sampling parity with the llama worker: top_p/top_k/min_p,
        // seed, repetition/presence/frequency penalties, and opt-in KV cache
        // quantization (CLAP_MLX_KV_TYPE=q8_0|q4_0).
        var toolSpecs: [ToolSpec]? = nil
        if let envelope = try? JSONDecoder().decode(ToolsEnvelope.self, from: data),
           let rawTools = envelope.tools, !rawTools.isEmpty {
          toolSpecs = rawTools.compactMap { raw -> ToolSpec? in
            raw.anyValue as? [String: any Sendable]
          }
          guard toolSpecs?.count == rawTools.count else {
            emit(id: id, error: "one or more caller-provided tools could not be represented for the chat template")
            return nil
          }
        }
        let entries = (control.messages ?? []).compactMap { message -> (role: String, content: String)? in
          guard let content = messageText(message) else { return nil }
          if message.role == "tool" {
            return ("user", "Tool result:\n\(content)")
          }
          return (message.role, content)
        }
        guard !entries.isEmpty else {
          emit(id: id, error: "chat request contains no messages")
          return nil
        }
        let structuredMessages: [[String: any Sendable]] = (control.messages ?? []).compactMap { message in
          if message.role == "assistant", let structured = structuredToolCallMessage(message) {
            return structured
          }
          guard let content = messageText(message) else { return nil }
          if message.role == "tool" {
            return ["role": "user", "content": "Tool result:\n\(content)"]
          }
          return ["role": message.role, "content": content]
        }

        let usesFallback = usesGemma4FallbackPrompt(loadedDirectory ?? modelDirectory)
        var promptTokens: [Int]
        if usesFallback {
          promptTokens = tok.encode(text: gemma4Prompt(entries: entries), addSpecialTokens: false)
        } else {
          do {
            promptTokens = try tok.applyChatTemplate(messages: structuredMessages, tools: toolSpecs)
          } catch {
            if toolSpecs != nil {
              let compatible = (toolSpecs ?? []).map { templateCompatibleToolSpec($0) }
              if let tokens = try? tok.applyChatTemplate(messages: structuredMessages, tools: compatible) {
                debugLog("chat template for \(loadedModel ?? model) required a nullable-preserving compatibility view of all \(compatible.count) caller-provided tools")
                promptTokens = tokens
              } else {
                debugLog("chat template for \(loadedModel ?? model) failed with all \(toolSpecs?.count ?? 0) caller-provided tools (\(error)); retrying with JSON tool instructions")
                let toolJson = toolSpecJson(toolSpecs ?? [])
                let instructions = [
                  "You may call tools by responding with JSON only.",
                  "Use this exact shape when calling tools:",
                  "{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}",
                  "Do not include natural language when calling tools.",
                  "Available tools: \(toolJson)",
                ].joined(separator: "\n")
                var patched = entries
                if let index = patched.firstIndex(where: { $0.role == "system" }) {
                  patched[index] = ("system", instructions + "\n\n" + patched[index].content)
                } else {
                  patched.insert(("system", instructions), at: 0)
                }
                let retryMessages: [[String: any Sendable]] = patched.map { ["role": $0.role, "content": $0.content] }
                if let tokens = try? tok.applyChatTemplate(messages: retryMessages, tools: nil) {
                  promptTokens = tokens
                } else {
                  debugLog("template retry without tools failed; using plain transcript")
                  let transcript = patched.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
                  promptTokens = tok.encode(text: transcript)
                }
              }
            } else {
              debugLog("chat template failed (\(error)); using plain transcript")
              let transcript = entries.map { "\($0.role): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
              promptTokens = tok.encode(text: transcript)
            }
          }
        }
        if ProcessInfo.processInfo.environment["CLAP_MLX_DEBUG_PROMPT"] != nil {
          debugLog("prompt (\(promptTokens.count) tokens): \(tok.decode(tokenIds: promptTokens, skipSpecialTokens: false))")
        }

        // Admission control (parity with the llama worker): reject oversized
        // prompts before any prefill with a structured code the server maps
        // to an OpenAI-style 400.
        if modelContextLength > 0 && promptTokens.count >= modelContextLength {
          emit(id: id, error: "prompt is too long for the loaded model; prompt_tokens=\(promptTokens.count), max_input_tokens=\(modelContextLength - 1), effective_context_window=\(modelContextLength).", code: "context_length_exceeded")
          return nil
        }
        if let requestedMaxTokens, modelMaxOutputTokens > 0, requestedMaxTokens > modelMaxOutputTokens {
          emit(id: id, error: "requested max_tokens=\(requestedMaxTokens) exceeds the loaded model maximum output tokens=\(modelMaxOutputTokens).", code: "max_output_tokens_exceeded")
          return nil
        }
        if requestedMaxTokens == nil && modelContextLength == 0 && modelMaxOutputTokens == 0 {
          emit(id: id, error: "max_tokens is required because this model does not declare token limits.", code: "token_capability_unknown")
          return nil
        }
        let availableOutput = modelContextLength > 0 ? modelContextLength - promptTokens.count : modelMaxOutputTokens
        let maxTokens = requestedMaxTokens
          ?? (modelMaxOutputTokens > 0 ? min(modelMaxOutputTokens, availableOutput) : availableOutput)
        if modelContextLength > 0 && promptTokens.count + maxTokens > modelContextLength {
          emit(id: id, error: "prompt plus requested output exceeds the loaded model context; prompt_tokens=\(promptTokens.count), requested_output_tokens=\(maxTokens), effective_context_window=\(modelContextLength).", code: "context_length_exceeded")
          return nil
        }
        let generateParameters = GenerateParameters(
          maxTokens: maxTokens,
          kvBits: kvBits,
          temperature: Float(temperature),
          topP: Float(control.top_p ?? 1.0),
          topK: control.top_k ?? 0,
          minP: Float(control.min_p ?? 0.0),
          repetitionPenalty: control.repetition_penalty.map { Float($0) },
          presencePenalty: control.presence_penalty.map { Float($0) },
          frequencyPenalty: control.frequency_penalty.map { Float($0) },
          seed: control.seed
        )
        let outputReserve = maxTokens

        let cacheIdentity = CacheIdentity(domain: cacheDomain, requestId: id, intent: control.cache)
        let physicalIdentity = PhysicalCacheIdentity(fingerprint: cacheIdentity.fingerprint)
        var stableBoundaries: [Int] = []
        if !usesFallback {
          let leadingSystemCount = structuredMessages.prefix { ($0["role"] as? String) == "system" }.count
          let systemPrefixCounts = leadingSystemCount > 0 ? Array(1...leadingSystemCount) : []
          let prefixCounts = (toolSpecs?.isEmpty == false ? [0] : []) + systemPrefixCounts
          for count in prefixCounts {
            let prefixMessages = Array(structuredMessages.prefix(count))
            if let rendered = try? tok.applyChatTemplate(messages: prefixMessages, tools: toolSpecs) {
              let boundary = zip(rendered, promptTokens).prefix { $0.0 == $0.1 }.count
              if boundary >= 16 && boundary < promptTokens.count {
                stableBoundaries.append(boundary)
              }
            }
          }
          stableBoundaries = Array(Set(stableBoundaries)).sorted()
        }
        var coordinatorPlan: CachePlan? = nil
        var cacheFallback: String? = cacheCoordinator == nil ? "coordinator_unavailable_no_cache" : nil
        if let coordinator = cacheCoordinator {
          try ensureAdmissionSlot()
          let slotMaterializations = kvSlots.enumerated().map { index, slot in
            let logical = try? coordinator.slot(index)
            let physical = PhysicalSlotRecord(identity: slot.cacheIdentity, tokens: slot.tokens,
              generation: slot.coordinatorGeneration, hasCaches: !slot.caches.isEmpty,
              isAnchor: slot.isAnchor)
            let generationMatches = logical.map { $0.generation == slot.coordinatorGeneration } ?? false
            let residentMatches = logical.map { Int($0.resident_len) == slot.tokens.count } ?? false
            let stateMatches = logical.map {
              (slot.isAnchor && $0.state == UInt32(CC_SLOT_ANCHOR)) ||
                (!slot.isAnchor && $0.state != UInt32(CC_SLOT_ANCHOR))
            } ?? false
            let identityMatches = slot.cacheIdentity?.isCompatible(with: physicalIdentity) ?? false
            let materialized = logical.map {
              physical.isMaterialized(for: physicalIdentity, logicalGeneration: $0.generation,
                logicalResidentLength: Int($0.resident_len), logicalState: $0.state,
                anchorState: UInt32(CC_SLOT_ANCHOR))
            } ?? false
            if !slot.caches.isEmpty && !materialized {
              var rejected: [String] = []
              if !generationMatches { rejected.append("generation") }
              if !residentMatches { rejected.append("resident_length") }
              if !stateMatches { rejected.append("state") }
              if !identityMatches { rejected.append("namespace_identity") }
              debugLog("cache donor rejected: slot=\(index) reasons=\(rejected.joined(separator: ",")) physical_generation=\(slot.coordinatorGeneration) logical_generation=\(logical?.generation ?? 0) physical_tokens=\(slot.tokens.count) logical_resident=\(logical?.resident_len ?? 0) logical_state=\(logical?.state ?? 0) anchor=\(slot.isAnchor)")
            }
            return CacheSlotMaterialization(
              materialized: materialized,
              writable: !slot.busy,
              partialSuffixTrim: materialized && slot.caches.allSatisfy(\.isTrimmable),
              copyable: materialized
            )
          }
          var capabilities = UInt64(CC_CAP_WHOLE_STATE_COPY) |
            UInt64(CC_CAP_PARTIAL_SUFFIX_TRIM) | UInt64(CC_CAP_PARTIAL_PREFIX_BRANCH) |
            UInt64(CC_CAP_SAFE_BUSY_DONOR) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT) |
            UInt64(CC_CAP_RELIABLE_RESIDENT_LENGTH) |
            UInt64(CC_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS)
          if slotMaterializations.contains(where: { $0.materialized && !$0.partialSuffixTrim }) {
            capabilities |= UInt64(CC_CAP_SLIDING_WINDOW) | UInt64(CC_CAP_RECURRENT_OR_HYBRID)
          }
          if kvBits != nil { capabilities |= UInt64(CC_CAP_KV_QUANTIZED) }
          do {
            coordinatorPlan = try coordinator.plan(tokens: promptTokens, identity: cacheIdentity,
              capabilities: capabilities, slots: slotMaterializations,
              stableBoundaries: stableBoundaries, outputReserve: outputReserve)
            if let view = coordinatorPlan?.view {
              debugLog("cache coordinator plan: operation=\(view.operation) reuse=\(view.reuseTokens) donor=\(view.donor.map(String.init) ?? "none") target=\(view.target)")
            }
          } catch {
            cacheFallback = "coordinator_plan_failed_closed"
            debugLog("cache coordinator plan failed closed: \(error)")
            throw error
          }
          guard coordinatorPlan != nil else { throw CacheCoordinatorError.unavailable }
        }
        guard coordinatorPlan != nil else { throw CacheCoordinatorError.unavailable }
        if sessionCap > 0 && promptTokens.count + outputReserve > sessionCap {
          emit(id: id, error: "prompt exceeds the per-session context cap; prompt tokens=\(promptTokens.count), max_session_ctx=\(sessionCap), reserved output tokens=\(outputReserve). Reduce the prompt/tool history or raise max_session_ctx / CLAP_MLX_MAX_SESSION_CTX.", code: "context_length_exceeded")
          return nil
        }

        // Rust is the sole cache policy authority. These variables describe
        // only the coordinator-selected physical operation.
        var bestPrefix = 0
        let anchorCandidate = coordinatorPlan?.view.anchorTokens ?? 0
        var branchDonor: KVSlot? = nil
        var branchPrefix = bestPrefix
        if let planned = coordinatorPlan {
          let view = planned.view
          guard view.target < kvSlots.count,
                view.donor == nil || view.donor! < kvSlots.count else {
            try planned.abort()
            throw CacheCoordinatorError.unavailable
          }
          let targetSlot = kvSlots[view.target]
          if view.operation == UInt32(CC_OPERATION_CONTINUE) {
            let trimNeeded = targetSlot.tokens.count - view.reuseTokens
            guard view.donor == view.target, !targetSlot.caches.isEmpty, trimNeeded >= 0,
                  trimNeeded == 0 || targetSlot.caches.allSatisfy(\.isTrimmable) else {
              try planned.abort()
              throw CacheCoordinatorError.unavailable
            }
          } else if view.operation == UInt32(CC_OPERATION_BRANCH) ||
                    view.operation == UInt32(CC_OPERATION_RESTORE) {
            guard let donorIndex = view.donor, donorIndex != view.target else {
              try planned.abort()
              throw CacheCoordinatorError.unavailable
            }
            let donorSlot = kvSlots[donorIndex]
            let donorOffset = cacheSequenceLength(donorSlot.caches, fallback: donorSlot.tokens.count)
            let trimNeeded = donorOffset - view.reuseTokens
            guard !donorSlot.caches.isEmpty, trimNeeded >= 0,
                  trimNeeded == 0 || donorSlot.caches.allSatisfy(\.isTrimmable) else {
              try planned.abort()
              throw CacheCoordinatorError.unavailable
            }
          }
          if view.operation == UInt32(CC_OPERATION_CONTINUE) {
            bestPrefix = view.reuseTokens
            branchDonor = nil
          } else if view.operation == UInt32(CC_OPERATION_BRANCH) ||
                    view.operation == UInt32(CC_OPERATION_RESTORE) {
            bestPrefix = 0
            branchDonor = view.donor.map { kvSlots[$0] }
            branchPrefix = view.reuseTokens
          } else {
            bestPrefix = 0
            branchDonor = nil
            branchPrefix = 0
          }
        }
        guard let planned = coordinatorPlan else { throw CacheCoordinatorError.unavailable }
        let slot = kvSlots[planned.view.target]
        if planned.view.operation != UInt32(CC_OPERATION_CONTINUE) {
          clearPhysicalSlot(slot)
        }
        kvUseCounter += 1
        slot.lastUsed = kvUseCounter
        slot.busy = true
        var prefix = bestPrefix
        if prefix == promptTokens.count { prefix -= 1 }  // always feed at least one token for logits

        var caches: [KVCache]
        var fedTokens: [Int]
        var suffix: [Int]
        var reusedTokens = 0
        let reuseKind = normalizedCacheReuseKind(operation: planned.view.operation)
        var reuseScope: String? = nil
        var branched = false
        if let donor = branchDonor {
          var sharedPrefix = branchPrefix
          if sharedPrefix == promptTokens.count { sharedPrefix -= 1 }
          let cloned = donor.caches.map { $0.copy() }
          let cloneOffset = cacheSequenceLength(cloned, fallback: donor.tokens.count)
          let trimNeeded = cloneOffset - sharedPrefix
          if trimNeeded > 0 {
            for cache in cloned { cache.trim(trimNeeded) }
          }
          caches = cloned
          fedTokens = Array(promptTokens.prefix(sharedPrefix))
          suffix = Array(promptTokens.dropFirst(sharedPrefix))
          reusedTokens = sharedPrefix
          reuseScope = donor.anchorScope
          prefix = sharedPrefix
          branched = true
          debugLog("kv prefix branch: cloned \(sharedPrefix)/\(promptTokens.count) shared tokens from \(donor.isAnchor ? "an anchor" : "another slot"), prefilling \(suffix.count)")
        } else {
          caches = []
          fedTokens = []
          suffix = promptTokens
        }
        let trimmable = !slot.caches.isEmpty && slot.caches.allSatisfy { $0.isTrimmable }
        let trimNeeded = slot.tokens.count - prefix
        if branched {
          // cache assignment handled above
        } else if prefix > 0 {
          if trimNeeded > 0 {
            for cache in slot.caches { cache.trim(trimNeeded) }
          }
          caches = slot.caches
          fedTokens = Array(promptTokens.prefix(prefix))
          suffix = Array(promptTokens.dropFirst(prefix))
          reusedTokens = prefix
          reuseScope = slot.anchorScope
          debugLog("kv prefix reuse (slot \(kvSlots.firstIndex(where: { $0 === slot }) ?? -1)): \(prefix)/\(promptTokens.count) tokens cached, prefilling \(suffix.count)")
        } else {
          caches = lm.newCache(parameters: generateParameters)
          fedTokens = []
          suffix = promptTokens
          if !slot.tokens.isEmpty {
            let reason = trimmable || slot.caches.isEmpty
              ? "no usable prefix"
              : "recurrent cache cannot rewind (matched \(prefix), needs trim of \(trimNeeded))"
            debugLog("kv cache miss: \(reason) (cached \(slot.tokens.count), prompt \(promptTokens.count))")
          }
        }
        slot.caches = caches
        slot.tokens = fedTokens
        slot.isPromptBoundary = false
        slot.anchorScope = nil
        slot.cacheIdentity = physicalIdentity

        var cacheDecision: CacheDecision? = nil
        let slotIndex = planned.view.target
        try retainedRegistry.activate(slotID: UInt32(slotIndex))
        if let planned = coordinatorPlan {
          do {
            let victims = planned.view.evictions.filter { $0 != slotIndex }.map(UInt32.init)
            try retainedRegistry.validateEvictions(victims)
            cacheDecision = try planned.commit(residentTokens: reusedTokens,
                                               state: UInt32(CC_SLOT_SESSION),
                                               physicalBytes: physicalCacheBytes(caches))
            slot.coordinatorGeneration = try cacheCoordinator?.slot(
              slotIndex).generation ?? 0
            retainedRegistry.reconcileEvictions(victims) { _, victim in
              clearPhysicalSlot(victim)
            }
            if !victims.isEmpty {
              lastEvictionReason = retentionConfig.physicalByteBudget > 0
                ? "byte_pressure" : "retained_capacity"
            }
            reusedTokens = cacheDecision?.realizedReuseTokens ?? reusedTokens
            reuseScope = cacheScopeName(cacheDecision?.scope ?? cacheIdentity.scope)
          } catch {
            retainedRegistry.release(slotID: UInt32(slotIndex))
            clearPhysicalSlot(slot)
            throw error
          }
        }

        let request = ActiveRequest(
          id: id,
          streaming: control.stream ?? true,
          maxTokens: maxTokens,
          promptTokens: promptTokens,
          reusedTokens: reusedTokens,
          reuseKind: reuseKind,
          reuseScope: reuseScope,
          cacheIdentity: cacheIdentity,
          cacheDecision: cacheDecision,
          cacheCandidates: coordinatorPlan?.candidates ?? [],
          cacheEvictions: coordinatorPlan?.view.evictions ?? [],
          cacheFallback: cacheFallback,
          slotIndex: slotIndex,
          slot: slot,
          caches: caches,
          fedTokens: fedTokens,
          suffix: suffix,
          detokenizer: NaiveStreamingDetokenizer(tokenizer: tok),
          parameters: generateParameters,
          stops: control.stop?.values ?? []
        )
        // Plant only the exact shared boundary selected by Rust.
        if reusedTokens == 0, anchorCandidate >= 16, anchorCandidate < promptTokens.count {
          request.anchorPlantAt = anchorCandidate
          request.anchorPlantScope = "shared"
        }
        return request
      } catch {
        emit(id: id, error: String(describing: error))
        return nil
      }
    }

    @discardableResult
    func snapshotAnchor(_ req: ActiveRequest, at plant: Int, reason: String) -> Bool {
      let boundary = Array(req.promptTokens.prefix(plant))
      let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
      guard req.fedTokens == boundary, offset == plant else {
        debugLog("\(reason) anchor skipped: fed=\(req.fedTokens.count) offset=\(offset) plant=\(plant)")
        return false
      }
      // Rust owns anchor deduplication, target choice, and eviction policy.
      guard let coordinator = cacheCoordinator else { return false }
      let anchor: KVSlot
      let anchorPlan: CachePlan
      do {
        try ensureAdmissionSlot()
        let slotMaterializations = kvSlots.map {
          CacheSlotMaterialization(
            materialized: false,
            writable: !$0.busy,
            partialSuffixTrim: false,
            copyable: false
          )
        }
        anchorPlan = try coordinator.plan(tokens: boundary, identity: req.cacheIdentity,
          capabilities: UInt64(CC_CAP_WHOLE_STATE_COPY) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT),
          slots: slotMaterializations,
          outputReserve: 0, state: UInt32(CC_SLOT_ANCHOR))
        guard anchorPlan.view.target < kvSlots.count,
              !kvSlots[anchorPlan.view.target].busy else {
          try? anchorPlan.abort()
          debugLog("\(reason) coordinated anchor skipped: target unavailable")
          return false
        }
        anchor = kvSlots[anchorPlan.view.target]
      } catch {
        debugLog("\(reason) coordinated anchor skipped: \(error)")
        return false
      }
      if anchorPlan.view.operation == UInt32(CC_OPERATION_NOOP) {
        do {
          _ = try anchorPlan.commit(residentTokens: plant, state: UInt32(CC_SLOT_ANCHOR),
            physicalBytes: physicalCacheBytes(anchor.caches))
          debugLog("coordinated \(reason) anchor already exists: slot=\(anchorPlan.view.target)")
          return true
        } catch {
          debugLog("\(reason) coordinated anchor no-op failed: \(error)")
          return false
        }
      }
      anchor.isAnchor = true
      anchor.anchorScope = req.anchorPlantScope
      anchor.caches = req.caches.map { $0.copy() }
      anchor.tokens = boundary
      anchor.cacheIdentity = PhysicalCacheIdentity(fingerprint: req.cacheIdentity.fingerprint)
      do {
        let index = anchorPlan.view.target
        let victims = anchorPlan.view.evictions.filter { $0 != index }.map(UInt32.init)
        try retainedRegistry.validateEvictions(victims)
        _ = try anchorPlan.commit(residentTokens: plant, state: UInt32(CC_SLOT_ANCHOR),
          physicalBytes: physicalCacheBytes(anchor.caches))
        let logical = try coordinator.slot(index)
        anchor.coordinatorGeneration = logical.generation
        retainedRegistry.reconcileEvictions(victims) { _, victim in
          clearPhysicalSlot(victim)
        }
        if !victims.isEmpty {
          lastEvictionReason = retentionConfig.physicalByteBudget > 0
            ? "byte_pressure" : "retained_capacity"
        }
        let fingerprint = req.cacheIdentity.fingerprint.map { String(format: "%02x", $0) }.joined()
        let flags = CacheSlotMaterialization(materialized: !anchor.caches.isEmpty,
          writable: !anchor.busy, partialSuffixTrim: anchor.caches.allSatisfy(\.isTrimmable),
          copyable: !anchor.caches.isEmpty).flags
        debugLog("coordinated \(reason) anchor committed: slot=\(index) logical_state=\(logical.state) logical_generation=\(logical.generation) logical_resident=\(logical.resident_len) namespace=\(fingerprint) physical_generation=\(anchor.coordinatorGeneration) physical_tokens=\(anchor.tokens.count) physical_identity=\(anchor.cacheIdentity != nil) flags=\(flags)")
      } catch {
        clearPhysicalSlot(anchor)
        debugLog("\(reason) coordinated anchor commit failed: \(error)")
        return false
      }
      kvUseCounter += 1
      anchor.lastUsed = kvUseCounter
      debugLog("planted \(reason) anchor: \(plant) tokens (exact-state snapshot for non-rewindable caches)")
      return true
    }

    func plantAnchor(_ req: ActiveRequest) {
      guard !req.anchorPlanted, let plant = req.anchorPlantAt else { return }
      req.anchorPlanted = true
      _ = snapshotAnchor(req, at: plant, reason: "prefix")
    }

    func captureContinuationBoundary(_ req: ActiveRequest) {
      guard req.continuationBoundaryCaches == nil,
            let boundary = req.continuationBoundary,
            boundary == req.fedTokens.count,
            req.caches.contains(where: { !$0.isTrimmable }) else { return }
      let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
      guard offset == boundary else {
        debugLog("rolling conversation anchor skipped: fed=\(req.fedTokens.count) offset=\(offset) boundary=\(boundary)")
        return
      }
      req.continuationBoundaryCaches = req.caches.map { $0.copy() }
      debugLog("captured rolling conversation anchor: \(boundary) tokens")
    }

    func advanceCoordinator(_ req: ActiveRequest, tokens: [Int]) {
      guard !tokens.isEmpty, let coordinator = cacheCoordinator,
            req.slot.coordinatorGeneration != 0 else { return }
      do {
        req.slot.coordinatorGeneration = try coordinator.advance(
          slot: req.slotIndex, generation: req.slot.coordinatorGeneration,
          tokens: tokens, state: UInt32(CC_SLOT_SESSION), busy: true,
          physicalBytes: physicalCacheBytes(req.caches))
      } catch {
        let generation = req.slot.coordinatorGeneration
        if generation != 0 {
          _ = try? coordinator.invalidate(slot: req.slotIndex, generation: generation)
        }
        req.slot.coordinatorGeneration = 0
        debugLog("cache metadata advance reconciled after error: \(error)")
      }
    }

    func step(_ req: ActiveRequest) {
      guard !req.cancelled, !req.completed, !req.failed, let lm = languageModel else { return }
      do {
        if req.iterator == nil {
          // Split chunk boundaries at the anchor plant point so the cache
          // state exactly at the shared boundary exists for snapshotting.
          var chunkEnd = min(req.pos + prefillChunk, req.suffix.count)
          if let plant = req.anchorPlantAt, !req.anchorPlanted {
            let rel = plant - req.reusedTokens
            if req.pos < rel && rel < chunkEnd { chunkEnd = rel }
          }
          if let boundary = req.continuationBoundary, req.continuationBoundaryCaches == nil {
            let rel = boundary - req.reusedTokens
            if req.pos < rel && rel < chunkEnd { chunkEnd = rel }
          }
          if chunkEnd < req.suffix.count {
            // Creating the iterator prefills the chunk into the shared cache;
            // the sampled token is discarded.
            let chunk = Array(req.suffix[req.pos ..< chunkEnd])
            _ = try TokenIterator(input: LMInput(tokens: MLXArray(chunk)), model: lm, cache: req.caches, parameters: req.parameters)
            req.pos = chunkEnd
            req.fedTokens.append(contentsOf: chunk)
            req.slot.tokens = req.fedTokens
            advanceCoordinator(req, tokens: chunk)
            emit(id: req.id, prefill: WorkerPrefill(done: req.reusedTokens + req.pos, total: req.promptTokens.count))
            if let plant = req.anchorPlantAt, !req.anchorPlanted, plant - req.reusedTokens == req.pos {
              plantAnchor(req)
            }
            if let boundary = req.continuationBoundary, boundary - req.reusedTokens == req.pos {
              captureContinuationBoundary(req)
            }
          } else {
            if let plant = req.anchorPlantAt, !req.anchorPlanted, plant - req.reusedTokens == req.pos {
              plantAnchor(req)
            }
            if let boundary = req.continuationBoundary, boundary - req.reusedTokens == req.pos {
              captureContinuationBoundary(req)
            }
            let tail = Array(req.suffix.dropFirst(req.pos))
            req.iterator = try TokenIterator(input: LMInput(tokens: MLXArray(tail)), model: lm, cache: req.caches, parameters: req.parameters)
            req.pos = req.suffix.count
            req.fedTokens.append(contentsOf: tail)
            req.slot.tokens = req.fedTokens
            advanceCoordinator(req, tokens: tail)
            // Decode mutates recurrent cache state irreversibly. Preserve the
            // exact end-of-prompt state in this request and restore it into
            // the same slot at finalize, so even a one-slot worker can reuse
            // it for the next tool-result continuation.
            if req.continuationBoundaryCaches == nil && req.caches.contains(where: { !$0.isTrimmable }) {
              let offset = cacheSequenceLength(req.caches, fallback: req.fedTokens.count)
              if offset == req.promptTokens.count {
                req.promptBoundaryCaches = req.caches.map { $0.copy() }
                debugLog("captured prompt-boundary anchor in slot \(kvSlots.firstIndex(where: { $0 === req.slot }) ?? -1): \(req.promptTokens.count) tokens")
              } else {
                debugLog("prompt-boundary anchor skipped: cache offset \(offset), prompt \(req.promptTokens.count)")
              }
            }
          }
          return
        }
        guard var it = req.iterator else { return }
        var steps = 0
        while steps < decodeStepsPerPass {
          guard let token = it.next() else {
            req.completed = true
            break
          }
          steps += 1
          req.sampledTokens.append(token)
          if eosTokenIds.contains(token) {
            if ProcessInfo.processInfo.environment["CLAP_MLX_DEBUG_PROMPT"] != nil {
              debugLog("eos token \(token) (\(tokenizer?.convertIdToToken(token) ?? "?")) after \(req.generatedCount) tokens; eos set: \(eosTokenIds)")
            }
            req.finishReason = "stop"
            req.completed = true
            break
          }
          req.generatedCount += 1
          req.detokenizer.append(token: token)
          if let chunk = req.detokenizer.next(), !chunk.isEmpty {
            req.collected += chunk
            // Stop sequences: scan collected text; on match truncate and
            // finish. While streaming, hold back enough of the tail that a
            // stop split across tokens is never emitted.
            if !req.stops.isEmpty {
              // Only the tail can contain a new match: the window covers the
              // fresh chunk plus a full stop length of earlier context.
              let windowStart = req.collected.index(
                req.collected.endIndex,
                offsetBy: -min(req.collected.count, chunk.count + req.holdback),
              )
              var earliest: Range<String.Index>? = nil
              for stop in req.stops {
                if let found = req.collected.range(of: stop, range: windowStart..<req.collected.endIndex),
                   earliest == nil || found.lowerBound < earliest!.lowerBound {
                  earliest = found
                }
              }
              if let match = earliest {
                req.collected = String(req.collected[..<match.lowerBound])
                if req.streaming && req.emitted < req.collected.count {
                  emit(id: req.id, token: String(req.collected.dropFirst(req.emitted)))
                  req.emitted = req.collected.count
                }
                req.finishReason = "stop"
                req.completed = true
                break
              }
            }
            if req.streaming {
              if req.stops.isEmpty {
                emit(id: req.id, token: chunk)
                req.emitted = req.collected.count
              } else {
                let safe = max(req.emitted, req.collected.count - req.holdback)
                if safe > req.emitted {
                  let start = req.collected.index(req.collected.startIndex, offsetBy: req.emitted)
                  let end = req.collected.index(req.collected.startIndex, offsetBy: safe)
                  emit(id: req.id, token: String(req.collected[start..<end]))
                  req.emitted = safe
                }
              }
            }
          }
          if req.generatedCount >= req.maxTokens {
            req.finishReason = "length"
            req.completed = true
            break
          }
        }
        req.iterator = it
      } catch {
        emit(id: req.id, error: String(describing: error))
        req.failed = true
      }
    }

    // Returns true when the worker should shut down.
    func handleLine(_ line: String) async -> Bool {
      guard !line.isEmpty, let data = line.data(using: .utf8),
            let control = try? JSONDecoder().decode(ControlRequest.self, from: data) else { return false }
      let id = control.id
      let type = control.type ?? "chat"

      if type == "shutdown" {
        for req in active {
          req.cancelled = true
          finalize(req)
        }
        active.removeAll()
        for pending in pendingChats {
          emit(id: pending.id, done: true, cancelled: true, finishReason: "cancel", usage: nil, cache: nil)
        }
        pendingChats.removeAll()
        emit(id: id, done: true)
        return true
      }

      if type == "cancel" {
        let target = control.id
        let matchesAll = target == nil || target?.isEmpty == true
        for req in active where matchesAll || req.id == target {
          req.cancelled = true
        }
        var remaining: [(id: String?, control: ControlRequest, data: Data)] = []
        for pending in pendingChats {
          if matchesAll || pending.id == target {
            emit(id: pending.id, done: true, cancelled: true, finishReason: "cancel", usage: nil, cache: nil)
          } else {
            remaining.append(pending)
          }
        }
        pendingChats = remaining
        return false
      }

      if type == "set_max_active" {
        guard let requested = control.max_active, requested > 0 else {
          emit(id: id, error: "set_max_active.max_active must be positive")
          return false
        }
        let oldMaxActive = maxActive
        maxActive = min(requested, activePolicy.backendCeiling,
          activePolicy.hardwareCeiling, activePolicy.modelCeiling,
          max(1, retentionConfig.hardCeiling))
        previousMaxActive = control.previous_max_active ?? oldMaxActive
        coordinatedLimitingReason = control.limiting_reason
        lastAdjustmentReason = control.last_adjustment_reason
        lastAdjustmentAt = control.last_adjustment_at
        coordinatedGrowthReserveBytes = control.retained_growth_reserve_bytes
        globalResidentMemoryBytes = control.global_resident_memory_bytes
        pressureState = control.pressure_state
        retainedRegistry.updateMaxActive(maxActive)
        emit(id: id, done: true, retention: retentionSnapshot(queued: pendingChats.count))
        return false
      }

      if type == "unload" || type == "load" {
        // Model mutations wait until in-flight requests drain.
        if !active.isEmpty || !pendingChats.isEmpty {
          controlBacklog.append(line)
          return false
        }
        if type == "unload" {
          invalidateKVCache()
          languageModel = nil
          tokenizer = nil
          loadedDirectory = nil
          loadedModel = nil
          emit(id: id, unloaded: true, done: true)
          return false
        }
        guard let model = control.model else {
          emit(id: id, error: "load.model is required")
          return false
        }
        do {
          let modelDirectory = try validateModelDirectory(model)
          if loadedModel != model || languageModel == nil {
            try await loadModel(model, directory: modelDirectory)
          }
          emit(id: id, loaded: true, done: true, memory: memorySnapshot(),
            retention: retentionSnapshot(), tokenCapabilities: WorkerTokenCapabilities(
            model_context_window: declaredModelContextLength > 0 ? declaredModelContextLength : nil,
            effective_context_window: modelContextLength > 0 ? modelContextLength : nil,
            max_input_tokens: modelContextLength > 0 ? modelContextLength - 1 : nil,
            max_output_tokens: modelMaxOutputTokens > 0 ? modelMaxOutputTokens : nil,
            model_context_window_source: modelContextLengthSource,
            max_output_tokens_source: modelMaxOutputTokensSource,
            backend_allocation_cap: contextOverride > 0 ? contextOverride : (declaredModelContextLength > 0 ? declaredModelContextLength : nil),
            user_configured_override: contextOverride > 0 ? contextOverride : nil
          ))
        } catch {
          emit(id: id, error: String(describing: error))
        }
        return false
      }

      pendingChats.append((id: id, control: control, data: data))
      emit(retention: retentionSnapshot(queued: pendingChats.count))
      return false
    }

    mainLoop: while true {
      // Idle: block on input (or drain deferred control work) instead of
      // spinning; busy: poll without blocking so generation keeps stepping.
      if active.isEmpty && pendingChats.isEmpty {
        if !controlBacklog.isEmpty {
          let line = controlBacklog.removeFirst()
          if await handleLine(line) { break mainLoop }
          continue mainLoop
        }
        guard let line = await buffer.next() else { break mainLoop }
        if await handleLine(line) { break mainLoop }
      }
      while let line = await buffer.poll() {
        if await handleLine(line) { break mainLoop }
      }

      // Admit pending chats up to the parallel limit. Requests for a
      // different model wait until the current model's requests drain.
      while active.count < maxActive, !pendingChats.isEmpty {
        if retainedRegistry.count >= retentionConfig.hardCeiling && kvSlots.allSatisfy(\.busy) { break }
        let candidate = pendingChats[0]
        let needsLoad = loadedModel != candidate.control.model || languageModel == nil
        if needsLoad && !active.isEmpty { break }
        pendingChats.removeFirst()
        emit(id: candidate.id, started: true)
        if let request = await prepareRequest(id: candidate.id, control: candidate.control, data: candidate.data) {
          active.append(request)
          allocatorNeedsIdleClear = true
          emit(retention: retentionSnapshot(queued: pendingChats.count))
        }
      }

      // Round-robin: one prefill chunk or a few decode tokens per request.
      for request in active {
        step(request)
      }
      for request in active where request.completed || request.cancelled || request.failed {
        finalize(request)
      }
      active.removeAll { $0.completed || $0.cancelled || $0.failed }
      if active.isEmpty && pendingChats.isEmpty && allocatorNeedsIdleClear {
        Memory.clearCache()
        let memory = memorySnapshot()
        debugLog("mlx memory after idle clear: active=\(memory.active_bytes) cache=\(memory.cache_bytes) peak=\(memory.peak_active_bytes)")
        emit(memory: memory, retention: retentionSnapshot())
        allocatorNeedsIdleClear = false
      }
    }
}

await main()
