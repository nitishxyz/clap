import XCTest
@testable import ClapCachePolicy

final class ActiveConcurrencyPolicyTests: XCTestCase {
  private let gib: UInt64 = 1_073_741_824

  func testSmallAndLargeMemoryAreBounded() {
    let small = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: 16 * gib, startupAvailableMemoryBytes: 6 * gib,
      modelActiveBytes: 3 * gib, retainedBudgetBytes: 1 * gib,
      retainedCeiling: 64, processorCount: 10))
    XCTAssertEqual(small.selectedMax, 3)
    XCTAssertEqual(small.reason, "memory_ceiling")

    let large = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: 128 * gib, startupAvailableMemoryBytes: 96 * gib,
      modelActiveBytes: 8 * gib, retainedBudgetBytes: 8 * gib,
      retainedCeiling: 64, processorCount: 16))
    XCTAssertEqual(large.selectedMax, 8)
    XCTAssertLessThanOrEqual(large.selectedMax, 16)
  }

  func testExplicitOverrideAndHybridCap() {
    let fixed = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      explicitMax: 6, physicalMemoryBytes: 128 * gib,
      startupAvailableMemoryBytes: 96 * gib, modelActiveBytes: 4 * gib,
      retainedBudgetBytes: 4 * gib, retainedCeiling: 64, processorCount: 16))
    XCTAssertEqual(fixed.mode, "fixed")
    XCTAssertEqual(fixed.selectedMax, 6)

    let hybrid = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      explicitMax: 16, physicalMemoryBytes: 128 * gib,
      startupAvailableMemoryBytes: 96 * gib, modelActiveBytes: 4 * gib,
      retainedBudgetBytes: 4 * gib, retainedCeiling: 64, processorCount: 16,
      isHybridOrRecurrent: true))
    XCTAssertEqual(hybrid.selectedMax, 2)
  }

  func testUnknownMetricsStayConservative() {
    let decision = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: 16 * gib, startupAvailableMemoryBytes: nil,
      modelActiveBytes: nil, retainedBudgetBytes: 1 * gib,
      retainedCeiling: 64, processorCount: 16))
    XCTAssertGreaterThanOrEqual(decision.selectedMax, 1)
    XCTAssertLessThanOrEqual(decision.selectedMax, 2)
  }

  func testOnlyRetainedUseAndBoundedGrowthReserveReduceActiveCapacity() {
    let withoutRetention = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: 64 * gib, startupAvailableMemoryBytes: 24 * gib,
      modelActiveBytes: 8 * gib, retainedBudgetBytes: 0,
      retainedCeiling: 64, processorCount: 16))
    let withRetention = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: 64 * gib, startupAvailableMemoryBytes: 24 * gib,
      modelActiveBytes: 8 * gib, retainedBudgetBytes: 8 * gib,
      retainedBytes: 72 * 1_048_576, recentRetainedGrowthBytes: 16 * 1_048_576,
      retainedCeiling: 64, processorCount: 16))
    XCTAssertEqual(withRetention.retainedBytes, 72 * 1_048_576)
    XCTAssertEqual(withRetention.retainedGrowthReserveBytes,
      (8 * gib - 72 * 1_048_576) / 100 * 10)
    XCTAssertGreaterThan(withRetention.memoryCeiling, 1)
    XCTAssertEqual(withRetention.usableRuntimeBytes + withRetention.retainedBytes
      + withRetention.retainedGrowthReserveBytes, withoutRetention.usableRuntimeBytes)
  }

  func testGrowthReserveIsDeterministicAndCappedByRemainingBudget() {
    let decision = ActiveConcurrencyPolicy.selectMLX(ActiveConcurrencyInputs(
      physicalMemoryBytes: 32 * gib, startupAvailableMemoryBytes: 16 * gib,
      modelActiveBytes: 6 * gib, retainedBudgetBytes: gib,
      retainedBytes: 900 * 1_048_576, recentRetainedGrowthBytes: 512 * 1_048_576,
      retainedGrowthMinimumBytes: 64 * 1_048_576, retainedGrowthReservePercent: 10,
      retainedCeiling: 64, processorCount: 12))
    XCTAssertEqual(decision.retainedGrowthReserveBytes, gib - 900 * 1_048_576)
  }
}
