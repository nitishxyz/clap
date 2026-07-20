public struct PhysicalCacheIdentity: Equatable, Sendable {
  public let fingerprint: [UInt8]

  public init(fingerprint: [UInt8]) {
    self.fingerprint = fingerprint
  }

  public func isCompatible(with other: PhysicalCacheIdentity) -> Bool {
    self == other
  }
}

public struct PhysicalCacheRecord: Equatable, Sendable {
  public let identity: PhysicalCacheIdentity
  public let tokens: [Int]

  public init(identity: PhysicalCacheIdentity, tokens: [Int]) {
    self.identity = identity
    self.tokens = tokens
  }

  public func exactAnchorMatch(identity: PhysicalCacheIdentity, tokens: [Int]) -> Bool {
    self.identity == identity && self.tokens == tokens
  }

  public func commonPrefix(identity: PhysicalCacheIdentity, tokens: [Int]) -> Int {
    guard self.identity == identity else { return 0 }
    return zip(self.tokens, tokens).prefix { pair in pair.0 == pair.1 }.count
  }
}

public struct PhysicalSlotRecord: Equatable, Sendable {
  public let identity: PhysicalCacheIdentity?
  public let tokens: [Int]
  public let generation: UInt64
  public let hasCaches: Bool
  public let isAnchor: Bool

  public init(identity: PhysicalCacheIdentity?, tokens: [Int], generation: UInt64,
              hasCaches: Bool, isAnchor: Bool) {
    self.identity = identity
    self.tokens = tokens
    self.generation = generation
    self.hasCaches = hasCaches
    self.isAnchor = isAnchor
  }

  public func isMaterialized(for identity: PhysicalCacheIdentity, logicalGeneration: UInt64,
                             logicalResidentLength: Int, logicalState: UInt32,
                             anchorState: UInt32) -> Bool {
    hasCaches && self.identity == identity && generation == logicalGeneration &&
      tokens.count == logicalResidentLength && (isAnchor == (logicalState == anchorState))
  }
}
