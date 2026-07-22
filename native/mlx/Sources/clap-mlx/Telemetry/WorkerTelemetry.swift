import ClapCacheBridge
import ClapMLXCache
import Foundation

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

func workerUsage(promptTokens: Int, completionTokens: Int) -> WorkerUsage {
  WorkerUsage(prompt_tokens: promptTokens, completion_tokens: completionTokens)
}
