import ClapCachePolicy
import MLXLMCommon

public func mlxCacheOperations(create: @escaping () throws -> [KVCache] = { [] },
                               log: @escaping (String) -> Void = { _ in })
  -> CacheOperations<KVCache> {
  CacheOperations(isTrimmable: { $0.isTrimmable }, copy: { $0.copy() },
    trim: { $0.trim($1) }, sequenceLength: { caches, fallback in
      let offset = caches.map(\.offset).max() ?? 0
      return offset > 0 ? offset : fallback
    }, create: create, physicalBytes: { caches in
      let arrays = caches.flatMap(\.state).map {
        CacheArrayDescriptor(storageIdentity: UInt64(bitPattern:
          Int64(ObjectIdentifier($0).hashValue)),
          elementCount: $0.size, itemSize: $0.itemSize,
          allocatedBytes: $0.nbytes)
      }
      return max(1, PhysicalCacheByteEstimator.estimate(arrays: arrays))
    }, log: log)
}
