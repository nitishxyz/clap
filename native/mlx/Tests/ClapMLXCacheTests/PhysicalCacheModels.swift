import ClapMLXCache

final class PhysicalCacheModel {
  let id: Int
  var residentLength: Int
  private let onRelease: (Int) -> Void

  init(id: Int, residentLength: Int = 0, onRelease: @escaping (Int) -> Void = { _ in }) {
    self.id = id
    self.residentLength = residentLength
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

  func make(residentLength: Int = 0) throws -> PhysicalCacheModel {
    let model = model(residentLength: residentLength)
    let operation = PhysicalCacheOperation.create(model.id)
    try prepare(operation, matching: .create)
    operations.append(operation)
    return model
  }

  func copy(_ source: PhysicalCacheModel) throws -> PhysicalCacheModel {
    let target = model(residentLength: source.residentLength)
    let operation = PhysicalCacheOperation.copy(source: source.id, target: target.id)
    try prepare(operation, matching: .copy)
    operations.append(operation)
    return target
  }

  func trim(_ model: PhysicalCacheModel, by count: Int) throws {
    let operation = PhysicalCacheOperation.trim(model.id, by: count)
    try prepare(operation, matching: .trim)
    model.residentLength -= count
    operations.append(operation)
  }

  func cacheOperations() -> CacheOperations<PhysicalCacheModel> {
    CacheOperations(isTrimmable: { _ in true }, copy: copy,
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

  private func model(residentLength: Int) -> PhysicalCacheModel {
    let id = nextID
    nextID += 1
    return PhysicalCacheModel(id: id, residentLength: residentLength) { [weak self] id in
      self?.operations.append(.release(id))
      self?.onRelease?(id)
    }
  }
}
