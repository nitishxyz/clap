import MLX
import MLXLMCommon

private final class SharedKVStorage {
  let keys: MLXArray
  let values: MLXArray

  init(keys: MLXArray, values: MLXArray) {
    self.keys = keys
    self.values = values
  }

  var allocatedBytes: UInt64 {
    UInt64(max(0, keys.nbytes)) + UInt64(max(0, values.nbytes))
  }
}

private struct SharedKVSegment {
  let storage: SharedKVStorage
  var length: Int
}

public final class SharedKVCache: KVCache {
  private var segments: [SharedKVSegment] = []
  private var tail: SharedKVStorage?
  private var tailLength = 0
  public var offset: Int = 0
  public var maxSize: Int? { nil }
  public var step: Int

  public init(step: Int = 256) {
    self.step = step
  }

  private init(segments: [SharedKVSegment], offset: Int, step: Int) {
    self.segments = segments
    self.offset = offset
    self.step = step
  }

  public func innerState() -> [MLXArray] {
    let storages = segments.map(\.storage) + (tail.map { [$0] } ?? [])
    return storages.flatMap { [$0.keys, $0.values] }
  }

  public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
    let incoming = keys.dim(2)
    precondition(incoming > 0, "SharedKVCache update requires at least one token")
    prepareTail(keys: keys, values: values, incoming: incoming)
    let start = tailLength
    tailLength += incoming
    offset += incoming
    tail!.keys[.ellipsis, start..<tailLength, 0...] = keys
    tail!.values[.ellipsis, start..<tailLength, 0...] = values
    return materializedState()
  }

  public var state: [MLXArray] {
    get {
      guard offset > 0 else { return [] }
      let materialized = materializedState()
      return [materialized.0, materialized.1]
    }
    set {
      precondition(newValue.count == 2, "SharedKVCache state must have keys and values")
      segments = []
      tail = SharedKVStorage(keys: newValue[0], values: newValue[1])
      tailLength = newValue[0].dim(2)
      offset = tailLength
    }
  }

  public var metaState: [String] {
    get { [String(step), String(offset)] }
    set {
      precondition(newValue.count == 2, "SharedKVCache metadata must contain step and offset")
      guard let step = Int(newValue[0]), let offset = Int(newValue[1]), step > 0, offset >= 0 else {
        preconditionFailure("SharedKVCache metadata is invalid")
      }
      let capacity = segments.reduce(0) { $0 + $1.length } + (tail?.keys.dim(2) ?? 0)
      precondition(offset <= capacity, "SharedKVCache offset exceeds physical state")
      self.step = step
      self.offset = offset
      if segments.isEmpty {
        tailLength = offset
      }
    }
  }

  public var isTrimmable: Bool { true }

  @discardableResult
  public func trim(_ n: Int) -> Int {
    var remaining = min(offset, max(0, n))
    let trimmed = remaining
    if remaining <= tailLength {
      tailLength -= remaining
      offset -= remaining
      return trimmed
    }
    remaining -= tailLength
    tail = nil
    tailLength = 0
    while remaining > 0, !segments.isEmpty {
      let index = segments.count - 1
      let amount = min(remaining, segments[index].length)
      segments[index].length -= amount
      remaining -= amount
      if segments[index].length == 0 { segments.removeLast() }
    }
    offset -= trimmed
    return trimmed
  }

  public func copy() -> any KVCache {
    var frozen = segments
    if let tail, tailLength > 0 {
      frozen.append(SharedKVSegment(storage: tail, length: tailLength))
    }
    return SharedKVCache(segments: frozen, offset: offset, step: step)
  }

  public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
    -> MLXFast.ScaledDotProductAttentionMaskMode {
    if n == 1 { return .none }
    if returnArray || (windowSize != nil && n > windowSize!) {
      return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
    }
    return .causal
  }

  public func prepare(lengths: [Int]?) {}
  public func prepare(lengths: MLXArray?) {}
  public func finalize() {}

  var physicalStorages: [(ObjectIdentifier, UInt64)] {
    var result: [(ObjectIdentifier, UInt64)] = []
    var seen = Set<ObjectIdentifier>()
    for storage in segments.map(\.storage) + (tail.map { [$0] } ?? []) {
      let identity = ObjectIdentifier(storage)
      if seen.insert(identity).inserted {
        result.append((identity, storage.allocatedBytes))
      }
    }
    return result
  }

  private func prepareTail(keys: MLXArray, values: MLXArray, incoming: Int) {
    if let current = tail {
      let hasCapacity = tailLength + incoming <= current.keys.dim(2)
      if hasCapacity && isKnownUniquelyReferenced(&tail) { return }
      if tailLength > 0 {
        segments.append(SharedKVSegment(storage: current, length: tailLength))
      }
      tail = nil
      tailLength = 0
    }
    let capacity = max(step, ((incoming + step - 1) / step) * step)
    tail = SharedKVStorage(
      keys: MLXArray.zeros([keys.dim(0), keys.dim(1), capacity, keys.dim(3)], dtype: keys.dtype),
      values: MLXArray.zeros([values.dim(0), values.dim(1), capacity, values.dim(3)],
        dtype: values.dtype))
  }

  private func materializedState() -> (MLXArray, MLXArray) {
    var keys = segments.map { $0.storage.keys[.ellipsis, ..<$0.length, 0...] }
    var values = segments.map { $0.storage.values[.ellipsis, ..<$0.length, 0...] }
    if let tail, tailLength > 0 {
      keys.append(tail.keys[.ellipsis, ..<tailLength, 0...])
      values.append(tail.values[.ellipsis, ..<tailLength, 0...])
    }
    precondition(!keys.isEmpty, "SharedKVCache has no materialized state")
    if keys.count == 1 { return (keys[0], values[0]) }
    return (concatenated(keys, axis: 2), concatenated(values, axis: 2))
  }
}

public func mlxCachesSupportPhysicalSharing(_ caches: [KVCache]) -> Bool {
  !caches.isEmpty && caches.allSatisfy {
    $0 is SharedKVCache || Swift.type(of: $0) == KVCacheSimple.self
  }
}

public func freezeShareableMLXCaches(_ caches: [KVCache]) -> [KVCache] {
  caches.map { cache in
    guard Swift.type(of: cache) == KVCacheSimple.self,
          let ordinary = cache as? KVCacheSimple else { return cache }
    let shared = SharedKVCache(step: ordinary.step)
    let state = ordinary.innerState()
    if !state.isEmpty {
      shared.state = state
      shared.metaState = [String(ordinary.step), String(ordinary.offset)]
    }
    return shared
  }
}

public struct MLXPhysicalCacheUsage: Equatable {
  public let uniqueBytes: UInt64
  public let referencedBytes: UInt64
  public let sharedBytes: UInt64
  public let sessionBytes: UInt64
  public let anchorBytes: UInt64
  public let storageObjects: Int
  public let sharedStorageObjects: Int
}

public func mlxPhysicalCacheBytes(_ caches: [KVCache]) -> UInt64 {
  var bytesByStorage: [ObjectIdentifier: UInt64] = [:]
  for cache in caches {
    for (identity, bytes) in physicalComponents(cache) {
      bytesByStorage[identity] = max(bytesByStorage[identity] ?? 0, bytes)
    }
  }
  return max(1, bytesByStorage.values.reduce(0, saturatedAdd))
}

public func mlxPhysicalCacheUsage(_ slots: [CacheSlot<KVCache>]) -> MLXPhysicalCacheUsage {
  var bytesByStorage: [ObjectIdentifier: UInt64] = [:]
  var references: [ObjectIdentifier: Int] = [:]
  var sessionStorage = Set<ObjectIdentifier>()
  var anchorStorage = Set<ObjectIdentifier>()
  var referencedBytes: UInt64 = 0

  for slot in slots where !slot.caches.isEmpty {
    var slotStorage = Set<ObjectIdentifier>()
    for cache in slot.caches {
      for (identity, bytes) in physicalComponents(cache) {
        bytesByStorage[identity] = max(bytesByStorage[identity] ?? 0, bytes)
        slotStorage.insert(identity)
      }
    }
    for identity in slotStorage {
      references[identity, default: 0] += 1
      referencedBytes = saturatedAdd(referencedBytes, bytesByStorage[identity] ?? 0)
      if slot.isAnchor { anchorStorage.insert(identity) } else { sessionStorage.insert(identity) }
    }
  }

  let uniqueBytes = bytesByStorage.values.reduce(0, saturatedAdd)
  let shared = references.compactMap { identity, count in
    count > 1 ? bytesByStorage[identity] : nil
  }.reduce(0, saturatedAdd)
  return MLXPhysicalCacheUsage(
    uniqueBytes: uniqueBytes,
    referencedBytes: referencedBytes,
    sharedBytes: shared,
    sessionBytes: sessionStorage.compactMap { bytesByStorage[$0] }.reduce(0, saturatedAdd),
    anchorBytes: anchorStorage.compactMap { bytesByStorage[$0] }.reduce(0, saturatedAdd),
    storageObjects: bytesByStorage.count,
    sharedStorageObjects: references.values.filter { $0 > 1 }.count)
}

private func physicalComponents(_ cache: KVCache) -> [(ObjectIdentifier, UInt64)] {
  if let shared = cache as? SharedKVCache { return shared.physicalStorages }
  return cache.innerState().map { array in
    (ObjectIdentifier(array), UInt64(max(0, array.nbytes)))
  }
}

private func saturatedAdd(_ left: UInt64, _ right: UInt64) -> UInt64 {
  let (value, overflow) = left.addingReportingOverflow(right)
  return overflow ? UInt64.max : value
}
