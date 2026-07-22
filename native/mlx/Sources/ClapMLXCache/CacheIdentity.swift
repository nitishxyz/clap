import ClapCacheBridge
import Foundation

public let cacheTelemetryKey = ProcessInfo.processInfo.environment["CLAP_TELEMETRY_HMAC_KEY"]
  ?? UUID().uuidString + UUID().uuidString

public struct CacheIdentityInput: Equatable, Sendable {
  public let namespace: String?
  public let tenant: String?
  public let project: String?
  public let harness: String?
  public let agent: String?
  public let session: String?
  public let priority: String?
  public let sideRequest: Bool

  public init(namespace: String?, tenant: String?, project: String?, harness: String?,
              agent: String?, session: String?, priority: String?, sideRequest: Bool) {
    self.namespace = namespace
    self.tenant = tenant
    self.project = project
    self.harness = harness
    self.agent = agent
    self.session = session
    self.priority = priority
    self.sideRequest = sideRequest
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

  public init(domain: String, input: CacheIdentityInput, telemetryKey: String = cacheTelemetryKey) {
    let isolation = input.tenant ?? input.namespace ?? "local"
    let keyed = telemetryKey + "|"
    var bytes = [UInt8](repeating: 0, count: 32)
    (keyed + domain + "|tenant=" + isolation).withCString { cc_fingerprint_string($0, &bytes) }
    func hash(_ value: String?) -> UInt64 {
      (keyed + (value ?? "")).withCString { cc_hash_string($0) }
    }
    fingerprint = bytes
    tenant = hash(isolation)
    project = hash(input.project)
    harness = hash(input.harness)
    agent = hash(input.agent)
    session = input.session?.isEmpty == false ? hash(input.session) : 0
    if input.session?.isEmpty == false { scope = UInt32(CC_SCOPE_SESSION) }
    else if input.agent?.isEmpty == false { scope = UInt32(CC_SCOPE_AGENT) }
    else if input.project?.isEmpty == false { scope = UInt32(CC_SCOPE_PROJECT) }
    else if input.harness?.isEmpty == false { scope = UInt32(CC_SCOPE_HARNESS) }
    else { scope = UInt32(CC_SCOPE_TENANT) }
    priority = input.priority == "background"
      ? UInt32(CC_PRIORITY_BACKGROUND) : UInt32(CC_PRIORITY_INTERACTIVE)
    sideRequest = input.sideRequest
    exportedNamespace = input.namespace ?? input.tenant
  }
}
