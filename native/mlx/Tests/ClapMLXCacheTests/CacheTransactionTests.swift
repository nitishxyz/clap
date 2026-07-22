import ClapCacheBridge
import Testing
@testable import ClapMLXCache

@Suite("MLX cache transaction seams")
struct CacheTransactionTests {
  @Test("transaction installs, confirms, and releases busy state")
  func commits() throws {
    let coordinator = FakeCoordinator()
    coordinator.nextGeneration = 9
    let physical = FakePhysical()
    let transaction = CacheTransaction(coordinator: coordinator, physical: physical)
    let generation = try transaction.installAndConfirm([1, 2], slot: 3, generation: 4,
      tokens: [10, 11], state: 2, physicalBytes: 64)
    #expect(generation == 9)
    #expect(physical.installed[3] == [1, 2])
    #expect(coordinator.confirmed == [3])
    #expect(coordinator.busyUpdates.count == 1)
    #expect(coordinator.busyUpdates[0].0 == 9)
    #expect(!coordinator.busyUpdates[0].1)
    #expect(coordinator.invalidated.isEmpty)
  }

  @Test("transaction clears physical state and invalidates on confirm failure")
  func rollsBack() {
    let coordinator = FakeCoordinator()
    coordinator.failure = FakeError.confirm
    let physical = FakePhysical()
    let transaction = CacheTransaction(coordinator: coordinator, physical: physical)
    #expect(throws: FakeError.confirm) {
      try transaction.installAndConfirm([1], slot: 2, generation: 7,
        tokens: [10], state: 1, physicalBytes: 32)
    }
    #expect(physical.cleared == [2])
    #expect(coordinator.invalidated.count == 1)
    #expect(coordinator.invalidated[0].0 == 2)
    #expect(coordinator.invalidated[0].1 == 7)
  }
}

private enum FakeError: Error { case confirm }

private final class FakeCoordinator: CacheCoordinatorMutating {
  var nextGeneration: UInt64 = 1
  var failure: FakeError?
  var confirmed: [Int] = []
  var busyUpdates: [(UInt64, Bool)] = []
  var invalidated: [(Int, UInt64)] = []

  func confirm(slot: Int, generation: UInt64, tokens: [Int], state: UInt32,
               busy: Bool, physicalBytes: UInt64) throws -> UInt64 {
    if let failure { throw failure }
    confirmed.append(slot)
    return nextGeneration
  }

  func setBusy(slot: Int, generation: UInt64, busy: Bool) throws {
    busyUpdates.append((generation, busy))
  }

  func invalidate(slot: Int, generation: UInt64) throws -> UInt64 {
    invalidated.append((slot, generation))
    return generation + 1
  }
}

private final class FakePhysical: PhysicalCacheMutating {
  var installed: [Int: [Int]] = [:]
  var cleared: [Int] = []

  func install(_ snapshot: [Int], in slot: Int) throws { installed[slot] = snapshot }
  func clear(slot: Int) { cleared.append(slot); installed[slot] = nil }
}
