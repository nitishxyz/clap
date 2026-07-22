import ClapMLXCache

enum PhysicalCacheShape {
  case standard
  case sliding(window: Int)
  case recurrent
  case hybridComponent
}

final class PhysicalCacheModel {
  let id: Int
  var residentLength: Int
  var storedLength: Int
  let shape: PhysicalCacheShape
  private let onRelease: (Int) -> Void

  init(id: Int, residentLength: Int = 0, storedLength: Int? = nil,
       shape: PhysicalCacheShape = .standard,
       onRelease: @escaping (Int) -> Void = { _ in }) {
    self.id = id
    self.residentLength = residentLength
    self.storedLength = storedLength ?? residentLength
    self.shape = shape
    self.onRelease = onRelease
  }

  deinit { onRelease(id) }
}

enum PhysicalCacheOperation: Equatable {
  case create(Int)
  case copy(source: Int, target: Int)
  case trim(Int, by: Int)
  case release(Int)
}

enum PhysicalCacheModelFailure: Error {
  case create
  case copy
  case trim
}

final class PhysicalCacheState {
  private var nextID = 0
  var operations: [PhysicalCacheOperation] = []
  var beforeMutation: ((PhysicalCacheOperation) throws -> Void)?
  var onRelease: ((Int) -> Void)?
  var failure: PhysicalCacheModelFailure?
  var failCopyNumber: Int?
  private var copyCount = 0

  func make(residentLength: Int = 0, storedLength: Int? = nil,
            shape: PhysicalCacheShape = .standard) throws -> PhysicalCacheModel {
    let model = model(residentLength: residentLength, storedLength: storedLength, shape: shape)
    let operation = PhysicalCacheOperation.create(model.id)
    try prepare(operation, matching: .create)
    operations.append(operation)
    return model
  }

  func copy(_ source: PhysicalCacheModel) throws -> PhysicalCacheModel {
    let target = model(residentLength: source.residentLength,
      storedLength: source.storedLength, shape: source.shape)
    let operation = PhysicalCacheOperation.copy(source: source.id, target: target.id)
    copyCount += 1
    if copyCount == failCopyNumber { throw PhysicalCacheModelFailure.copy }
    try prepare(operation, matching: .copy)
    operations.append(operation)
    return target
  }

  func trim(_ model: PhysicalCacheModel, by count: Int) throws {
    let operation = PhysicalCacheOperation.trim(model.id, by: count)
    try prepare(operation, matching: .trim)
    model.residentLength -= count
    model.storedLength = min(model.storedLength, model.residentLength)
    operations.append(operation)
  }

  func cacheOperations() -> CacheOperations<PhysicalCacheModel> {
    CacheOperations(isTrimmable: {
      if case .recurrent = $0.shape { return false }
      return true
    }, copy: copy,
      trim: trim, sequenceLength: { caches, fallback in
        caches.map(\.residentLength).max() ?? fallback
      }, create: { [try self.make()] },
      physicalBytes: { UInt64(max(1, $0.count * 64)) })
  }

  private func prepare(_ operation: PhysicalCacheOperation,
                       matching expected: PhysicalCacheModelFailure) throws {
    try beforeMutation?(operation)
    if failure == expected {
      failure = nil
      throw expected
    }
  }

  private func model(residentLength: Int, storedLength: Int?,
                     shape: PhysicalCacheShape) -> PhysicalCacheModel {
    let id = nextID
    nextID += 1
    return PhysicalCacheModel(id: id, residentLength: residentLength,
      storedLength: storedLength, shape: shape) { [weak self] id in
      self?.operations.append(.release(id))
      self?.onRelease?(id)
    }
  }
}
