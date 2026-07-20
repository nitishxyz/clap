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

  func testExactTemplateBoundaryAcceptsExactPrefixAndAuthoritativeEosSuffix() {
    XCTAssertEqual(exactTemplateBoundary(prefix: [1, 2, 3], final: [1, 2, 3, 4], eosToken: 99), 3)
    XCTAssertEqual(exactTemplateBoundary(prefix: [1, 2, 99], final: [1, 2, 4, 5], eosToken: 99), 2)
    XCTAssertEqual(exactTemplateBoundary(prefix: [1, 2, 99, 99], final: [1, 2, 4], eosToken: 99), 2)
  }

  func testExactTemplateBoundaryRejectsContentCutsAndUnknownTerminalSuffixes() {
    XCTAssertNil(exactTemplateBoundary(prefix: [], final: [1, 2], eosToken: 99))
    XCTAssertNil(exactTemplateBoundary(prefix: [1, 7], final: [1, 2, 3], eosToken: 99))
    XCTAssertNil(exactTemplateBoundary(prefix: [1, 2, 99], final: [1, 7, 3], eosToken: 99))
    XCTAssertNil(exactTemplateBoundary(prefix: [1, 2, 99], final: [1, 2, 3], eosToken: nil))
  }
}
