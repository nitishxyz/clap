import ClapCachePolicy
import XCTest

final class PhysicalCacheIdentityTests: XCTestCase {
  func testIdenticalTokensInDifferentNamespacesNeverMatch() {
    let tenantA = PhysicalCacheIdentity(fingerprint: [1, 2, 3])
    let tenantB = PhysicalCacheIdentity(fingerprint: [1, 2, 4])
    let record = PhysicalCacheRecord(identity: tenantA, tokens: [10, 20, 30])

    XCTAssertEqual(record.commonPrefix(identity: tenantB, tokens: [10, 20, 30]), 0)
    XCTAssertFalse(record.exactAnchorMatch(identity: tenantB, tokens: [10, 20, 30]))
  }

  func testSameNamespaceAnchorAndPrefixRemainEligible() {
    let tenant = PhysicalCacheIdentity(fingerprint: [9, 8, 7])
    let record = PhysicalCacheRecord(identity: tenant, tokens: [10, 20, 30])

    XCTAssertEqual(record.commonPrefix(identity: tenant, tokens: [10, 20, 99]), 2)
    XCTAssertTrue(record.exactAnchorMatch(identity: tenant, tokens: [10, 20, 30]))
  }

  func testModelDomainIsPartOfCompatibilityFingerprint() {
    let modelA = PhysicalCacheIdentity(fingerprint: [4, 1])
    let modelB = PhysicalCacheIdentity(fingerprint: [4, 2])

    XCTAssertFalse(modelA.isCompatible(with: modelB))
  }

  func testCommittedAnchorRequiresExactPhysicalIdentityGenerationLengthAndState() {
    let tenant = PhysicalCacheIdentity(fingerprint: [7, 7, 7])
    let anchor = PhysicalSlotRecord(identity: tenant, tokens: [1, 2, 3], generation: 44,
                                    hasCaches: true, isAnchor: true)

    XCTAssertTrue(anchor.isMaterialized(for: tenant, logicalGeneration: 44,
      logicalResidentLength: 3, logicalState: 3, anchorState: 3))
    XCTAssertFalse(anchor.isMaterialized(for: tenant, logicalGeneration: 43,
      logicalResidentLength: 3, logicalState: 3, anchorState: 3))
    XCTAssertFalse(anchor.isMaterialized(for: PhysicalCacheIdentity(fingerprint: [8]),
      logicalGeneration: 44, logicalResidentLength: 3, logicalState: 3, anchorState: 3))
    XCTAssertFalse(anchor.isMaterialized(for: tenant, logicalGeneration: 44,
      logicalResidentLength: 2, logicalState: 3, anchorState: 3))
    XCTAssertFalse(anchor.isMaterialized(for: tenant, logicalGeneration: 44,
      logicalResidentLength: 3, logicalState: 2, anchorState: 3))
  }
}
