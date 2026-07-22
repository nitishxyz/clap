import ClapMLXCache
import Foundation

func testCacheIdentity() -> CacheIdentity {
  let namespace = String(repeating: "40", count: 32)
  let session = String(repeating: "f2", count: 32)
  let value: [String: Any] = [
    "version": 1, "generation": "sec_test-generation",
    "tenant_root": String(repeating: "6f", count: 32),
    "project_fingerprint": String(repeating: "f1", count: 32),
    "harness_fingerprint": String(repeating: "94", count: 32),
    "agent_fingerprint": String(repeating: "d8", count: 32),
    "session_fingerprint": session, "scope": "session",
    "scope_fingerprint": session, "namespace_fingerprint": namespace,
    "namespace_id": String(UInt64(namespace.prefix(16), radix: 16)!),
    "priority": "interactive", "side_request": false,
    "display": ["namespace": "workspace"],
    "physical": ["fingerprint": String(repeating: "5b", count: 32),
      "backend": "mlx", "resolved_revision": "sha256:test",
      "model_artifact_fingerprint": String(repeating: "dd", count: 32),
      "tokenizer_fingerprint": String(repeating: "ee", count: 32),
      "context_allocation": 8192, "kv_format": "q8_0", "unified_kv": false,
      "layout_version": 1],
  ]
  let data = try! JSONSerialization.data(withJSONObject: value)
  let input = try! JSONDecoder().decode(OpaqueCacheIdentityInput.self, from: data)
  return try! CacheIdentity(input: input, expected: PhysicalCacheDescriptor(backend: "mlx",
    contextAllocation: 8192, kvFormat: "q8_0", unifiedKV: false, layoutVersion: 1))
}
