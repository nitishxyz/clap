import ClapCachePolicy

public final class CacheSlot<Cache> {
  public var caches: [Cache]
  public var tokens: [Int]
  public var lastUsed: UInt64
  public var busy: Bool
  public var isAnchor: Bool
  public var anchorScope: String?
  public var isPromptBoundary: Bool
  public var coordinatorGeneration: UInt64
  public var cacheIdentity: PhysicalCacheIdentity?

  public init(caches: [Cache] = [], tokens: [Int] = [], lastUsed: UInt64 = 0,
              busy: Bool = false, isAnchor: Bool = false, anchorScope: String? = nil,
              isPromptBoundary: Bool = false, coordinatorGeneration: UInt64 = 0,
              cacheIdentity: PhysicalCacheIdentity? = nil) {
    self.caches = caches
    self.tokens = tokens
    self.lastUsed = lastUsed
    self.busy = busy
    self.isAnchor = isAnchor
    self.anchorScope = anchorScope
    self.isPromptBoundary = isPromptBoundary
    self.coordinatorGeneration = coordinatorGeneration
    self.cacheIdentity = cacheIdentity
  }

  public func clear() {
    caches = []
    tokens = []
    isAnchor = false
    isPromptBoundary = false
    anchorScope = nil
    cacheIdentity = nil
    coordinatorGeneration = 0
  }
}
