import XCTest
@testable import ClapCachePolicy

final class CacheTelemetryTests: XCTestCase {
  func testNormalizesReuseKindFromCoordinatorOperation() {
    XCTAssertNil(normalizedCacheReuseKind(operation: 0))
    XCTAssertEqual(normalizedCacheReuseKind(operation: 1), "slot")
    XCTAssertEqual(normalizedCacheReuseKind(operation: 2), "branch")
    XCTAssertEqual(normalizedCacheReuseKind(operation: 3), "anchor")
  }

  func testSessionBranchAndAnchorRestoreRemainDistinct() {
    let sessionDonorOperation: UInt32 = 2
    let committedAnchorOperation: UInt32 = 3

    XCTAssertEqual(normalizedCacheReuseKind(operation: sessionDonorOperation), "branch")
    XCTAssertEqual(normalizedCacheReuseKind(operation: committedAnchorOperation), "anchor")
  }
}
