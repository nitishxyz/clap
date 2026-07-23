import ClapCacheBridge
import Foundation

public struct OpaquePhysicalCacheIdentityInput: Decodable, Equatable, Sendable {
  public let fingerprint: String
  public let backend: String
  public let resolvedRevision: String
  public let modelArtifactFingerprint: String
  public let tokenizerFingerprint: String
  public let contextAllocation: Int
  public let kvFormat: String
  public let unifiedKV: Bool
  public let layoutVersion: Int

  enum CodingKeys: String, CodingKey, CaseIterable {
    case fingerprint, backend
    case resolvedRevision = "resolved_revision"
    case modelArtifactFingerprint = "model_artifact_fingerprint"
    case tokenizerFingerprint = "tokenizer_fingerprint"
    case contextAllocation = "context_allocation"
    case kvFormat = "kv_format"
    case unifiedKV = "unified_kv"
    case layoutVersion = "layout_version"
  }

  public init(from decoder: any Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    try rejectUnknownKeys(decoder, allowed: Set(CodingKeys.allCases.map(\.stringValue)),
      name: "cache_identity.physical")
    fingerprint = try values.decode(String.self, forKey: .fingerprint)
    backend = try values.decode(String.self, forKey: .backend)
    resolvedRevision = try values.decode(String.self, forKey: .resolvedRevision)
    modelArtifactFingerprint = try values.decode(String.self, forKey: .modelArtifactFingerprint)
    tokenizerFingerprint = try values.decode(String.self, forKey: .tokenizerFingerprint)
    contextAllocation = try values.decode(Int.self, forKey: .contextAllocation)
    kvFormat = try values.decode(String.self, forKey: .kvFormat)
    unifiedKV = try values.decode(Bool.self, forKey: .unifiedKV)
    layoutVersion = try values.decode(Int.self, forKey: .layoutVersion)
  }
}

public struct OpaqueCacheIdentityDisplay: Decodable, Equatable, Sendable {
  public let namespace: String?
  public let project: String?
  public let harness: String?
  public let agent: String?
  public let session: String?

  enum CodingKeys: String, CodingKey, CaseIterable {
    case namespace, project, harness, agent, session
  }

  public init(from decoder: any Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    try rejectUnknownKeys(decoder, allowed: Set(CodingKeys.allCases.map(\.stringValue)),
      name: "cache_identity.display")
    namespace = try values.decodeIfPresent(String.self, forKey: .namespace)
    project = try values.decodeIfPresent(String.self, forKey: .project)
    harness = try values.decodeIfPresent(String.self, forKey: .harness)
    agent = try values.decodeIfPresent(String.self, forKey: .agent)
    session = try values.decodeIfPresent(String.self, forKey: .session)
  }
}

public struct OpaqueCacheIdentityInput: Decodable, Equatable, Sendable {
  public let version: Int
  public let generation: String
  public let tenantRoot: String
  public let projectFingerprint: String?
  public let harnessFingerprint: String?
  public let agentFingerprint: String?
  public let sessionFingerprint: String?
  public let scope: String
  public let scopeFingerprint: String
  public let namespaceFingerprint: String
  public let namespaceID: String
  public let priority: String
  public let sideRequest: Bool
  public let display: OpaqueCacheIdentityDisplay
  public let physical: OpaquePhysicalCacheIdentityInput

  enum CodingKeys: String, CodingKey, CaseIterable {
    case version, generation
    case tenantRoot = "tenant_root"
    case projectFingerprint = "project_fingerprint"
    case harnessFingerprint = "harness_fingerprint"
    case agentFingerprint = "agent_fingerprint"
    case sessionFingerprint = "session_fingerprint"
    case scope
    case scopeFingerprint = "scope_fingerprint"
    case namespaceFingerprint = "namespace_fingerprint"
    case namespaceID = "namespace_id"
    case priority
    case sideRequest = "side_request"
    case display, physical
  }

  public init(from decoder: any Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    try rejectUnknownKeys(decoder, allowed: Set(CodingKeys.allCases.map(\.stringValue)),
      name: "cache_identity")
    version = try values.decode(Int.self, forKey: .version)
    generation = try values.decode(String.self, forKey: .generation)
    tenantRoot = try values.decode(String.self, forKey: .tenantRoot)
    projectFingerprint = try values.decodeIfPresent(String.self, forKey: .projectFingerprint)
    harnessFingerprint = try values.decodeIfPresent(String.self, forKey: .harnessFingerprint)
    agentFingerprint = try values.decodeIfPresent(String.self, forKey: .agentFingerprint)
    sessionFingerprint = try values.decodeIfPresent(String.self, forKey: .sessionFingerprint)
    scope = try values.decode(String.self, forKey: .scope)
    scopeFingerprint = try values.decode(String.self, forKey: .scopeFingerprint)
    namespaceFingerprint = try values.decode(String.self, forKey: .namespaceFingerprint)
    namespaceID = try values.decode(String.self, forKey: .namespaceID)
    priority = try values.decodeIfPresent(String.self, forKey: .priority) ?? "normal"
    sideRequest = try values.decode(Bool.self, forKey: .sideRequest)
    display = try values.decode(OpaqueCacheIdentityDisplay.self, forKey: .display)
    physical = try values.decode(OpaquePhysicalCacheIdentityInput.self, forKey: .physical)
  }
}

public struct PhysicalCacheDescriptor: Equatable, Sendable {
  public let backend: String
  public let contextAllocation: Int
  public let kvFormat: String
  public let unifiedKV: Bool
  public let layoutVersion: Int

  public init(backend: String, contextAllocation: Int, kvFormat: String,
              unifiedKV: Bool, layoutVersion: Int) {
    self.backend = backend
    self.contextAllocation = contextAllocation
    self.kvFormat = kvFormat
    self.unifiedKV = unifiedKV
    self.layoutVersion = layoutVersion
  }
}

public enum CacheIdentityError: Error, CustomStringConvertible, Equatable {
  case invalid(String)
  public var description: String {
    switch self { case .invalid(let message): return message }
  }
}

public struct CacheIdentity {
  public let fingerprint: [UInt8]
  public let tenant: UInt64
  public let project: UInt64
  public let harness: UInt64
  public let agent: UInt64
  public let session: UInt64
  public let scope: UInt32
  public let priority: UInt32
  public let sideRequest: Bool
  public let exportedNamespace: String?
  public let generation: String
  public let physicalFingerprint: String

  public init(input: OpaqueCacheIdentityInput, expected: PhysicalCacheDescriptor) throws {
    guard input.version == 1 else { throw invalid("cache_identity.version must be 1") }
    try validateString(input.generation, key: "generation", maximum: 64)
    let tenantBytes = try digest(input.tenantRoot, key: "tenant_root")
    let namespaceBytes = try digest(input.namespaceFingerprint, key: "namespace_fingerprint")
    let scopeBytes = try digest(input.scopeFingerprint, key: "scope_fingerprint")
    let projectBytes = try input.projectFingerprint.map { try digest($0, key: "project_fingerprint") }
    let harnessBytes = try input.harnessFingerprint.map { try digest($0, key: "harness_fingerprint") }
    let agentBytes = try input.agentFingerprint.map { try digest($0, key: "agent_fingerprint") }
    let sessionBytes = try input.sessionFingerprint.map { try digest($0, key: "session_fingerprint") }
    guard let namespaceID = UInt64(input.namespaceID), input.namespaceID.first != "0",
          String(namespaceID) == input.namespaceID, namespaceID != 0 else {
      throw invalid("cache_identity.namespace_id must be a nonzero decimal u64")
    }
    guard namespaceID == reduce(namespaceBytes) else {
      throw invalid("cache_identity.namespace_id does not match namespace_fingerprint")
    }
    let selectedScope: ([UInt8]?, UInt32) = switch input.scope {
    case "tenant": (tenantBytes, UInt32(CC_SCOPE_TENANT))
    case "project": (projectBytes, UInt32(CC_SCOPE_PROJECT))
    case "harness": (harnessBytes, UInt32(CC_SCOPE_HARNESS))
    case "agent": (agentBytes, UInt32(CC_SCOPE_AGENT))
    case "session": (sessionBytes, UInt32(CC_SCOPE_SESSION))
    default: throw invalid("cache_identity.scope is invalid")
    }
    guard selectedScope.0 == scopeBytes else {
      throw invalid("\(input.scope) scope fingerprint mismatch")
    }
    let selectedPriority: UInt32 = switch input.priority {
    case "interactive": UInt32(CC_PRIORITY_INTERACTIVE)
    case "normal": UInt32(CC_PRIORITY_NORMAL)
    case "background": UInt32(CC_PRIORITY_BACKGROUND)
    default: throw invalid("cache_identity.priority is invalid")
    }
    for (key, label) in [("namespace", input.display.namespace),
                         ("project", input.display.project), ("harness", input.display.harness),
                         ("agent", input.display.agent), ("session", input.display.session)] {
      if let label { try validateString(label, key: "display.\(key)", maximum: 128) }
    }
    _ = try digest(input.physical.fingerprint, key: "physical.fingerprint")
    _ = try digest(input.physical.modelArtifactFingerprint,
      key: "physical.model_artifact_fingerprint")
    _ = try digest(input.physical.tokenizerFingerprint,
      key: "physical.tokenizer_fingerprint")
    try validateString(input.physical.resolvedRevision,
      key: "physical.resolved_revision", maximum: 256)
    try validateString(input.physical.kvFormat, key: "physical.kv_format", maximum: 64)
    guard input.physical.backend == expected.backend,
          input.physical.contextAllocation == expected.contextAllocation,
          input.physical.kvFormat == expected.kvFormat,
          input.physical.unifiedKV == expected.unifiedKV,
          input.physical.layoutVersion == expected.layoutVersion else {
      throw invalid("cache_identity physical model domain does not match loaded runtime")
    }
    fingerprint = namespaceBytes
    tenant = reduce(tenantBytes)
    project = projectBytes.map(reduce) ?? 0
    harness = harnessBytes.map(reduce) ?? 0
    agent = agentBytes.map(reduce) ?? 0
    session = sessionBytes.map(reduce) ?? 0
    scope = selectedScope.1
    priority = selectedPriority
    sideRequest = input.sideRequest
    exportedNamespace = input.display.namespace
    generation = input.generation
    physicalFingerprint = input.physical.fingerprint
  }
}

private struct AnyCodingKey: CodingKey {
  let stringValue: String
  let intValue: Int? = nil
  init?(stringValue: String) { self.stringValue = stringValue }
  init?(intValue: Int) { return nil }
}

private func rejectUnknownKeys(_ decoder: any Decoder, allowed: Set<String>,
                               name: String) throws {
  let keys = try decoder.container(keyedBy: AnyCodingKey.self).allKeys.map(\.stringValue)
  if let unknown = keys.first(where: { !allowed.contains($0) }) {
    throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
      debugDescription: "\(name) contains unknown field: \(unknown)"))
  }
}

private func validateString(_ value: String, key: String, maximum: Int) throws {
  guard !value.isEmpty, value.utf8.count <= maximum else {
    throw invalid("cache_identity.\(key) has invalid length")
  }
}

private func digest(_ value: String, key: String) throws -> [UInt8] {
  guard value.utf8.count == 64 else {
    throw invalid("cache_identity.\(key) must be 64 lowercase hex characters")
  }
  let characters = Array(value.utf8)
  var bytes: [UInt8] = []
  bytes.reserveCapacity(32)
  for index in stride(from: 0, to: 64, by: 2) {
    guard let high = nibble(characters[index]), let low = nibble(characters[index + 1]) else {
      throw invalid("cache identity fingerprints must use lowercase hexadecimal")
    }
    bytes.append((high << 4) | low)
  }
  return bytes
}

private func nibble(_ character: UInt8) -> UInt8? {
  switch character {
  case 48...57: character - 48
  case 97...102: character - 97 + 10
  default: nil
  }
}

private func reduce(_ bytes: [UInt8]) -> UInt64 {
  let value = bytes.prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
  return value == 0 ? 1 : value
}

private func invalid(_ message: String) -> CacheIdentityError {
  .invalid(message)
}
