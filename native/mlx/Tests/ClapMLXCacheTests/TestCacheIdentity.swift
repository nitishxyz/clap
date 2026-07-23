import ClapMLXCache
import Foundation

func sharedCacheIdentityFixture() throws -> [String: Any] {
  var root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
  while !FileManager.default.fileExists(atPath: root.appendingPathComponent("package.json").path) {
    let parent = root.deletingLastPathComponent()
    guard parent.path != root.path else { throw CocoaError(.fileNoSuchFile) }
    root = parent
  }
  let url = root.appendingPathComponent(
    "packages/worker-protocol/fixtures/v1/cache-identity-vector.json")
  return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
}

let testPhysicalCacheDescriptor = PhysicalCacheDescriptor(backend: "mlx",
  contextAllocation: 8192, kvFormat: "q8_0", unifiedKV: false, layoutVersion: 1)

func testOpaqueIdentityValue(namespaceFingerprint: String =
    "40d51ffb8e7b5c35e3a4519b9ce017396cc5c8d6c89a2c7652d087742b216329",
    namespaceID: String = "4671675353754459189",
    sessionFingerprint: String =
      "f269f6ff4a9cae43ffb8a8a7510e1c6a63669634fb08deca7bfc6dd304959b41",
    displayNamespace: String = "workspace", backend: String = "mlx") -> [String: Any] {
  [
    "version": 1, "generation": "sec_123e4567-e89b-42d3-a456-426614174000",
    "tenant_root": "6f64c827dbe30bac5d1d2b7e37c70ba23e012c793624af8d64222adac5112f44",
    "project_fingerprint": "f1482819163f613fa61f18a23b2aa510319dd4d8db62934fca3a1d7475b47c3b",
    "harness_fingerprint": "94bdbacbdf215f56a6e55920c34a287947d6b21bb1194208205e2b83eb302027",
    "agent_fingerprint": "d844f6c06cc42e03137243f37db076fcb86b602a409118a62001f67604d793e2",
    "session_fingerprint": sessionFingerprint,
    "scope": "session",
    "scope_fingerprint": sessionFingerprint,
    "namespace_fingerprint": namespaceFingerprint, "namespace_id": namespaceID,
    "priority": "normal", "side_request": true,
    "display": ["namespace": displayNamespace, "project": "payments",
      "harness": "coding-v4", "agent": "reviewer", "session": "session-99"],
    "physical": ["fingerprint": "5befd882f13033f62c62fd673e60b7758564412e14c6f18cbb799b541847ecb5",
      "backend": backend, "resolved_revision": "sha256:abc123",
      "model_artifact_fingerprint": String(repeating: "d", count: 64),
      "tokenizer_fingerprint": String(repeating: "e", count: 64),
      "context_allocation": 8192, "kv_format": "q8_0", "unified_kv": false,
      "layout_version": backend == "llama" ? 3 : 1],
  ]
}

func decodeTestOpaqueIdentity(_ value: [String: Any]) throws -> OpaqueCacheIdentityInput {
  try JSONDecoder().decode(OpaqueCacheIdentityInput.self,
    from: JSONSerialization.data(withJSONObject: value))
}

func testCacheIdentity(namespaceByte: String = "40", sessionByte: String = "f2",
                       displayNamespace: String = "workspace") -> CacheIdentity {
  let namespace = String(repeating: namespaceByte, count: 32)
  let namespaceID = String(UInt64(namespace.prefix(16), radix: 16)!)
  let input = try! decodeTestOpaqueIdentity(testOpaqueIdentityValue(
    namespaceFingerprint: namespace, namespaceID: namespaceID,
    sessionFingerprint: String(repeating: sessionByte, count: 32),
    displayNamespace: displayNamespace))
  return try! CacheIdentity(input: input, expected: testPhysicalCacheDescriptor)
}
