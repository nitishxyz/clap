import MLX
import MLXLMCommon
import Testing
@testable import ClapMLXCache

@Suite("MLX shared physical cache")
struct SharedKVCacheTests {
  @Test("copy-on-write branches retain shared immutable prefixes after append")
  func copyOnWriteIsolation() throws {
    let source = SharedKVCache(step: 8)
    let keys = MLXArray.zeros([1, 1, 2, 4], dtype: .float16)
    let values = MLXArray.zeros([1, 1, 2, 4], dtype: .float16)
    _ = source.update(keys: keys, values: values)
    let branch = try #require(source.copy() as? SharedKVCache)

    let before = mlxPhysicalCacheUsage([
      CacheSlot<KVCache>(caches: [source]),
      CacheSlot<KVCache>(caches: [branch], isAnchor: true),
    ])
    #expect(before.storageObjects == 1)
    #expect(before.sharedStorageObjects == 1)
    #expect(before.sharedBytes == before.uniqueBytes)
    #expect(before.referencedBytes == before.uniqueBytes * 2)

    _ = branch.update(
      keys: MLXArray.zeros([1, 1, 1, 4], dtype: .float16),
      values: MLXArray.zeros([1, 1, 1, 4], dtype: .float16))
    #expect(source.offset == 2)
    #expect(branch.offset == 3)

    let after = mlxPhysicalCacheUsage([
      CacheSlot<KVCache>(caches: [source]),
      CacheSlot<KVCache>(caches: [branch], isAnchor: true),
    ])
    #expect(after.storageObjects == 2)
    #expect(after.sharedStorageObjects == 1)
    #expect(after.sharedBytes == before.uniqueBytes)
    #expect(after.referencedBytes > after.uniqueBytes)
  }

  @Test("ordinary attention freezes into shared storage")
  func freezesOrdinaryAttention() throws {
    let ordinary = KVCacheSimple()
    _ = ordinary.update(
      keys: MLXArray.zeros([1, 1, 3, 4], dtype: .float16),
      values: MLXArray.zeros([1, 1, 3, 4], dtype: .float16))
    let frozen = freezeShareableMLXCaches([ordinary])
    let shared = try #require(frozen.first as? SharedKVCache)
    #expect(shared.offset == 3)
    #expect(shared.isTrimmable)
    #expect(mlxCachesSupportPhysicalSharing(frozen))
    #expect(mlxPhysicalCacheBytes(frozen) > UInt64(3 * 4 * 2 * 2))
  }

  @Test("rotating, chunked, recurrent-like, and quantized caches stay on safe fallback")
  func unsupportedClassesRemainUnshared() {
    let unsupported: [KVCache] = [
      RotatingKVCache(maxSize: 16),
      ChunkedKVCache(chunkSize: 8),
      QuantizedKVCache(groupSize: 64, bits: 4),
      MambaCache(),
    ]
    let frozen = freezeShareableMLXCaches(unsupported)
    #expect(!mlxCachesSupportPhysicalSharing(frozen))
    #expect(frozen.allSatisfy { !($0 is SharedKVCache) })
  }
}
