import ClapCacheBridge
import Foundation
import Testing
@testable import ClapMLXCache

@Suite("Opaque cache identity")
struct CacheIdentityTests {
  private let llamaPhysical = PhysicalCacheDescriptor(backend: "llama",
    contextAllocation: 8192, kvFormat: "q8_0", unifiedKV: false, layoutVersion: 3)

  @Test("shared vector reduces exact digests and enums")
  func sharedVector() throws {
    let input = try decodeTestOpaqueIdentity(testOpaqueIdentityValue(backend: "llama"))
    let identity = try CacheIdentity(input: input, expected: llamaPhysical)
    #expect(identity.tenant == 0x6f64c827dbe30bac)
    #expect(identity.project == 0xf1482819163f613f)
    #expect(identity.harness == 0x94bdbacbdf215f56)
    #expect(identity.agent == 0xd844f6c06cc42e03)
    #expect(identity.session == 0xf269f6ff4a9cae43)
    #expect(identity.scope == UInt32(CC_SCOPE_SESSION))
    #expect(identity.priority == UInt32(CC_PRIORITY_BACKGROUND))
    #expect(identity.sideRequest)
    #expect(identity.fingerprint.first == 0x40)
    #expect(identity.fingerprint.last == 0x29)
    #expect(identity.exportedNamespace == "workspace")
  }

  @Test("display relabeling cannot change cache authority")
  func relabel() throws {
    let first = try CacheIdentity(input: decodeTestOpaqueIdentity(
      testOpaqueIdentityValue(backend: "llama")), expected: llamaPhysical)
    let second = try CacheIdentity(input: decodeTestOpaqueIdentity(
      testOpaqueIdentityValue(displayNamespace: "untrusted-other", backend: "llama")),
      expected: llamaPhysical)
    #expect(first.fingerprint == second.fingerprint)
    #expect(first.tenant == second.tenant)
    #expect(first.session == second.session)
    #expect(second.exportedNamespace == "untrusted-other")
  }

  @Test("missing unknown and malformed identity fields are rejected")
  func malformed() {
    var missing = testOpaqueIdentityValue(backend: "llama")
    missing.removeValue(forKey: "tenant_root")
    #expect(throws: (any Error).self) { try decodeTestOpaqueIdentity(missing) }
    var unknown = testOpaqueIdentityValue(backend: "llama")
    unknown["tenant"] = "caller-label"
    #expect(throws: (any Error).self) { try decodeTestOpaqueIdentity(unknown) }
    var uppercase = testOpaqueIdentityValue(backend: "llama")
    uppercase["tenant_root"] = String(repeating: "A", count: 64)
    #expect(throws: CacheIdentityError.self) {
      try CacheIdentity(input: decodeTestOpaqueIdentity(uppercase), expected: llamaPhysical)
    }
  }

  @Test("generation namespace and scope integrity are enforced")
  func integrity() throws {
    var generation = testOpaqueIdentityValue(backend: "llama")
    generation["generation"] = ""
    #expect(throws: CacheIdentityError.self) {
      try CacheIdentity(input: decodeTestOpaqueIdentity(generation), expected: llamaPhysical)
    }
    var namespace = testOpaqueIdentityValue(backend: "llama")
    namespace["namespace_id"] = "1"
    #expect(throws: CacheIdentityError.self) {
      try CacheIdentity(input: decodeTestOpaqueIdentity(namespace), expected: llamaPhysical)
    }
    var scope = testOpaqueIdentityValue(backend: "llama")
    scope["scope"] = "project"
    #expect(throws: CacheIdentityError.self) {
      try CacheIdentity(input: decodeTestOpaqueIdentity(scope), expected: llamaPhysical)
    }
  }

  @Test("physical runtime traits must match")
  func physicalMismatch() throws {
    let input = try decodeTestOpaqueIdentity(testOpaqueIdentityValue(backend: "llama"))
    #expect(throws: CacheIdentityError.self) {
      try CacheIdentity(input: input, expected: PhysicalCacheDescriptor(backend: "mlx",
        contextAllocation: 8192, kvFormat: "q8_0", unifiedKV: false, layoutVersion: 3))
    }
    #expect(throws: CacheIdentityError.self) {
      try CacheIdentity(input: input, expected: PhysicalCacheDescriptor(backend: "llama",
        contextAllocation: 4096, kvFormat: "q8_0", unifiedKV: false, layoutVersion: 3))
    }
  }
}
