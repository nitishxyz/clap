public protocol CacheCoordinatorMutating {
  func confirm(slot: Int, generation: UInt64, tokens: [Int], state: UInt32,
               busy: Bool, physicalBytes: UInt64) throws -> UInt64
  func setBusy(slot: Int, generation: UInt64, busy: Bool) throws
  func invalidate(slot: Int, generation: UInt64) throws -> UInt64
}

public protocol PhysicalCacheMutating {
  associatedtype Snapshot
  func install(_ snapshot: Snapshot, in slot: Int) throws
  func clear(slot: Int)
}

public struct CacheTransaction<Coordinator: CacheCoordinatorMutating,
                               Physical: PhysicalCacheMutating> {
  public let coordinator: Coordinator
  public let physical: Physical

  public init(coordinator: Coordinator, physical: Physical) {
    self.coordinator = coordinator
    self.physical = physical
  }

  @discardableResult
  public func installAndConfirm(_ snapshot: Physical.Snapshot, slot: Int,
                                generation: UInt64, tokens: [Int], state: UInt32,
                                physicalBytes: UInt64) throws -> UInt64 {
    var mutationGeneration = generation
    do {
      try physical.install(snapshot, in: slot)
      let next = try coordinator.confirm(slot: slot, generation: generation, tokens: tokens,
        state: state, busy: true, physicalBytes: physicalBytes)
      mutationGeneration = next
      try coordinator.setBusy(slot: slot, generation: next, busy: false)
      return next
    } catch {
      physical.clear(slot: slot)
      _ = try? coordinator.invalidate(slot: slot, generation: mutationGeneration)
      throw error
    }
  }
}

extension CacheCoordinator: CacheCoordinatorMutating {}
