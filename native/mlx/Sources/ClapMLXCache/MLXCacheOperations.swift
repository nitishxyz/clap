import ClapCachePolicy
import MLXLMCommon

public func mlxCacheOperations(create: @escaping () throws -> [KVCache] = { [] },
                               log: @escaping (String) -> Void = { _ in })
  -> CacheOperations<KVCache> {
  CacheOperations(isTrimmable: { $0.isTrimmable }, copy: { $0.copy() },
    trim: { $0.trim($1) }, sequenceLength: { caches, fallback in
      let offset = caches.map(\.offset).max() ?? 0
      return offset > 0 ? offset : fallback
    }, create: create, physicalBytes: mlxPhysicalCacheBytes,
    prepareForRetention: freezeShareableMLXCaches, log: log)
}
