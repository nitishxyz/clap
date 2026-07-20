import ClapCachePolicy
import XCTest

final class RetentionPolicyTests: XCTestCase {
  private final class Entry {
    let identity: Int
    init(_ identity: Int) { self.identity = identity }
  }

  func testRegistryGrowsTo128WithStableSlotIdentity() throws {
    let registry = RetainedRegistry<Entry>(maxActive: 4, hardCeiling: 128)
    var entries: [Entry] = []
    for id in 0..<128 {
      let entry = Entry(id)
      entries.append(entry)
      try registry.register(slotID: UInt32(id), entry: entry)
    }
    XCTAssertEqual(registry.count, 128)
    XCTAssertEqual(registry.slotIDs, (0..<128).map(UInt32.init))
    for id in 0..<128 {
      XCTAssertTrue(registry.entry(for: UInt32(id)) === entries[id])
    }
    XCTAssertThrowsError(try registry.register(slotID: 128, entry: Entry(128))) {
      XCTAssertEqual($0 as? RetainedRegistryError, .hardCeiling)
    }
  }

  func testActiveSetIsSeparateAndCancellationReleasesWithoutRemoving() throws {
    let registry = RetainedRegistry<Entry>(maxActive: 2, hardCeiling: 100)
    for id in 0..<100 { try registry.register(slotID: UInt32(id), entry: Entry(id)) }
    try registry.activate(slotID: 3)
    try registry.activate(slotID: 75)
    XCTAssertEqual(registry.activeCount, 2)
    XCTAssertThrowsError(try registry.activate(slotID: 99)) {
      XCTAssertEqual($0 as? RetainedRegistryError, .maxActive)
    }
    registry.release(slotID: 3)
    XCTAssertEqual(registry.activeCount, 1)
    XCTAssertEqual(registry.count, 100)
    XCTAssertNotNil(registry.entry(for: 3))
  }

  func testActiveLimitChangesInPlaceWithoutDroppingEntriesOrActiveWork() throws {
    let registry = RetainedRegistry<Entry>(maxActive: 3, hardCeiling: 100)
    for id in 0..<10 { try registry.register(slotID: UInt32(id), entry: Entry(id)) }
    try registry.activate(slotID: 1)
    try registry.activate(slotID: 2)
    registry.updateMaxActive(1)
    XCTAssertEqual(registry.activeCount, 2)
    XCTAssertEqual(registry.count, 10)
    XCTAssertThrowsError(try registry.activate(slotID: 3))
    registry.release(slotID: 1)
    registry.release(slotID: 2)
    try registry.activate(slotID: 3)
    registry.updateMaxActive(4)
    try registry.activate(slotID: 4)
    XCTAssertEqual(registry.activeCount, 2)
    XCTAssertEqual(registry.count, 10)
  }

  func testSessionAndAnchorEntriesCanCoexist() throws {
    let registry = RetainedRegistry<Entry>(maxActive: 1, hardCeiling: 3)
    let session = Entry(1)
    let anchor = Entry(2)
    try registry.register(slotID: 7, entry: session)
    try registry.register(slotID: 9, entry: anchor)
    try registry.activate(slotID: 7)
    XCTAssertTrue(registry.entry(for: 7) === session)
    XCTAssertTrue(registry.entry(for: 9) === anchor)
    XCTAssertFalse(registry.isActive(slotID: 9))
  }

  func testPressureVictimsArePhysicallyClearedExactlyOnce() throws {
    let registry = RetainedRegistry<Entry>(maxActive: 1, hardCeiling: 4)
    for id in 0..<4 { try registry.register(slotID: UInt32(id), entry: Entry(id)) }
    var cleared: [UInt32] = []
    try registry.validateEvictions([1, 3])
    registry.reconcileEvictions([1, 3]) { slotID, _ in cleared.append(slotID) }
    XCTAssertEqual(cleared, [1, 3])
    XCTAssertEqual(registry.count, 4)
  }

  func testEstimatorCoversStandardRotatingAndRecurrentArraysAndDeduplicates() {
    let standard = [
      CacheArrayDescriptor(storageIdentity: 1, elementCount: 2_048, itemSize: 2),
      CacheArrayDescriptor(storageIdentity: 2, elementCount: 2_048, itemSize: 2),
    ]
    let rotating = [
      CacheArrayDescriptor(storageIdentity: 3, elementCount: 4_096, itemSize: 2,
        allocatedBytes: 12_288),
      CacheArrayDescriptor(storageIdentity: 3, elementCount: 4_096, itemSize: 2,
        allocatedBytes: 12_288),
    ]
    let recurrentHybrid = [
      CacheArrayDescriptor(storageIdentity: 4, elementCount: 512, itemSize: 4),
      CacheArrayDescriptor(storageIdentity: 5, elementCount: 256, itemSize: 4),
      CacheArrayDescriptor(storageIdentity: 6, elementCount: 1_024, itemSize: 2),
    ]
    XCTAssertEqual(PhysicalCacheByteEstimator.estimate(
      arrays: standard + rotating + recurrentHybrid), 25_600)
  }

  func testRetentionEnvironmentDefaultsAndConfiguredGrowthBudget() {
    let defaultBudget: UInt64 = 2 << 30
    XCTAssertEqual(RetentionConfiguration.fromEnvironment([:], physicalMemoryBytes: 16 << 30),
      RetentionConfiguration(initialEntries: 4, hardCeiling: 256,
        physicalByteBudget: defaultBudget,
        highWatermarkBytes: UInt64(Double(defaultBudget) * 90 / 100),
        lowWatermarkBytes: UInt64(Double(defaultBudget) * 75 / 100)))
    let largeDefaultBudget: UInt64 = 96 << 30
    XCTAssertEqual(RetentionConfiguration.fromEnvironment([:], physicalMemoryBytes: 512 << 30),
      RetentionConfiguration(initialEntries: 4, hardCeiling: 256,
        physicalByteBudget: largeDefaultBudget,
        highWatermarkBytes: UInt64(Double(largeDefaultBudget) * 90 / 100),
        lowWatermarkBytes: UInt64(Double(largeDefaultBudget) * 75 / 100)))
    let clampedBudget: UInt64 = 1 << 30
    XCTAssertEqual(RetentionConfiguration.fromEnvironment([:], physicalMemoryBytes: 64 << 30,
      startupAvailableMemoryBytes: 8 << 30),
      RetentionConfiguration(initialEntries: 4, hardCeiling: 256,
        physicalByteBudget: clampedBudget,
        highWatermarkBytes: UInt64(Double(clampedBudget) * 90 / 100),
        lowWatermarkBytes: UInt64(Double(clampedBudget) * 75 / 100)))
    let configured = RetentionConfiguration.fromEnvironment([
      "CLAP_MLX_RETAINED_INITIAL": "8",
      "CLAP_MLX_RETAINED_MAX": "128",
      "CLAP_MLX_RETAINED_BUDGET_PERCENT": "25",
      "CLAP_MLX_RETAINED_HIGH_PERCENT": "80",
      "CLAP_MLX_RETAINED_LOW_PERCENT": "60",
    ], physicalMemoryBytes: 16 << 30, startupAvailableMemoryBytes: 1)
    XCTAssertEqual(configured.initialEntries, 8)
    XCTAssertEqual(configured.hardCeiling, 128)
    XCTAssertEqual(configured.physicalByteBudget, 4 << 30)
    XCTAssertEqual(configured.highWatermarkBytes, 3_435_973_836)
    XCTAssertEqual(configured.lowWatermarkBytes, 2_576_980_377)
  }
}
