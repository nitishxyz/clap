import ClapCacheBridge
import XCTest

final class ClapCacheBridgeTests: XCTestCase {
  func testPlanCommitBranchInvalidateAndReset() throws {
    let manager = try XCTUnwrap(cc_manager_create_with_retention(3, 2, 64, 2, 3, 0, 0, 0))
    defer { cc_manager_destroy(manager) }
    let namespace = [UInt8](repeating: 7, count: 32)
    let firstTokens: [Int32] = [1, 2, 3, 4]
    let stableBoundaries: [UInt64] = [2]
    let first = namespace.withUnsafeBufferPointer { namespace in
      firstTokens.withUnsafeBufferPointer { tokens in
        stableBoundaries.withUnsafeBufferPointer { boundaries in
          cc_manager_plan(manager, tokens.baseAddress, tokens.count, namespace.baseAddress,
            1, 2, 3, 4, 10, UInt32(CC_SCOPE_SESSION), UInt32(CC_PRIORITY_INTERACTIVE),
            0, UInt64(CC_CAP_PARTIAL_PREFIX_BRANCH) | UInt64(CC_CAP_PROMPT_BOUNDARY_SNAPSHOT),
            nil, 0, boundaries.baseAddress, boundaries.count, 0, UInt32(CC_SLOT_SESSION))
        }
      }
    }
    let firstPlan = try XCTUnwrap(first)
    var firstView = cc_plan_view_t()
    XCTAssertEqual(cc_plan_view(firstPlan, &firstView), Int32(CC_OK))
    XCTAssertEqual(firstView.operation, UInt32(CC_OPERATION_FRESH))
    XCTAssertEqual(firstView.anchor_tokens, 2)
    var firstDecision = cc_decision_t()
    XCTAssertEqual(cc_plan_commit(firstPlan, 0, UInt32(CC_SLOT_SESSION), 0, &firstDecision), Int32(CC_OK))
    cc_plan_destroy(firstPlan)

    var firstInfo = cc_slot_info_t()
    XCTAssertEqual(cc_manager_slot(manager, firstView.target_slot, &firstInfo), Int32(CC_OK))
    var generation: UInt64 = 0
    XCTAssertEqual(firstTokens.withUnsafeBufferPointer {
      cc_manager_advance(manager, firstView.target_slot, firstInfo.generation,
        $0.baseAddress, $0.count, UInt32(CC_SLOT_SESSION), 0, 0, &generation)
    }, Int32(CC_OK))
    XCTAssertEqual(cc_manager_set_busy(manager, firstView.target_slot,
      firstInfo.generation, 0), Int32(CC_STALE_PLAN))

    let related: [Int32] = [1, 2, 3, 9]
    var slotCapabilities = [UInt8](repeating: UInt8(CC_SLOT_WRITABLE), count: 3)
    slotCapabilities[Int(firstView.target_slot)] = UInt8(CC_SLOT_MATERIALIZED) |
      UInt8(CC_SLOT_PARTIAL_SUFFIX_TRIM) | UInt8(CC_SLOT_COPY)
    let branch = namespace.withUnsafeBufferPointer { namespace in
      related.withUnsafeBufferPointer { tokens in
        slotCapabilities.withUnsafeBufferPointer { slots in
          cc_manager_plan(manager, tokens.baseAddress, tokens.count, namespace.baseAddress,
            1, 2, 3, 4, 11, UInt32(CC_SCOPE_SESSION), UInt32(CC_PRIORITY_INTERACTIVE),
            0, UInt64(CC_CAP_PARTIAL_PREFIX_BRANCH), slots.baseAddress, slots.count,
            nil, 0, 0, UInt32(CC_SLOT_SESSION))
        }
      }
    }
    let branchPlan = try XCTUnwrap(branch)
    var branchView = cc_plan_view_t()
    XCTAssertEqual(cc_plan_view(branchPlan, &branchView), Int32(CC_OK))
    XCTAssertEqual(branchView.operation, UInt32(CC_OPERATION_BRANCH))
    XCTAssertEqual(branchView.reuse_tokens, 3)
    var branchDecision = cc_decision_t()
    XCTAssertEqual(cc_plan_commit(branchPlan, 3, UInt32(CC_SLOT_SESSION), 0, &branchDecision), Int32(CC_OK))
    cc_plan_destroy(branchPlan)
    XCTAssertEqual(branchDecision.hit, 1)

    var branchInfo = cc_slot_info_t()
    XCTAssertEqual(cc_manager_slot(manager, branchView.target_slot, &branchInfo), Int32(CC_OK))
    var invalidated: UInt64 = 0
    XCTAssertEqual(cc_manager_invalidate(manager, branchView.target_slot,
      branchInfo.generation, &invalidated), Int32(CC_OK))
    XCTAssertGreaterThan(invalidated, branchInfo.generation)

    slotCapabilities = [UInt8](repeating: UInt8(CC_SLOT_WRITABLE), count: 3)
    let excluded = namespace.withUnsafeBufferPointer { namespace in
      related.withUnsafeBufferPointer { tokens in
        slotCapabilities.withUnsafeBufferPointer { slots in
          cc_manager_plan(manager, tokens.baseAddress, tokens.count, namespace.baseAddress,
            1, 2, 3, 4, 12, UInt32(CC_SCOPE_SESSION), UInt32(CC_PRIORITY_INTERACTIVE),
            0, UInt64(CC_CAP_PARTIAL_PREFIX_BRANCH), slots.baseAddress, slots.count,
            nil, 0, 0, UInt32(CC_SLOT_SESSION))
        }
      }
    }
    let excludedPlan = try XCTUnwrap(excluded)
    var excludedView = cc_plan_view_t()
    XCTAssertEqual(cc_plan_view(excludedPlan, &excludedView), Int32(CC_OK))
    XCTAssertEqual(excludedView.operation, UInt32(CC_OPERATION_FRESH))
    XCTAssertEqual(excludedView.has_donor, 0)
    XCTAssertEqual(excludedView.reuse_tokens, 0)
    XCTAssertEqual(cc_plan_abort(excludedPlan), Int32(CC_OK))
    cc_plan_destroy(excludedPlan)

    var epoch: UInt64 = 0
    XCTAssertEqual(cc_manager_reset(manager, &epoch), Int32(CC_OK))
    XCTAssertGreaterThan(epoch, 1)
  }

  func testContextSizedPromptIsIndependentOfRetainedCapacityHint() throws {
    let manager = try XCTUnwrap(cc_manager_create_with_retention(1, 1, 1, 1, 1, 0, 0, 0))
    defer { cc_manager_destroy(manager) }
    let namespace = [UInt8](repeating: 3, count: 32)
    let tooLarge: [Int32] = [1, 2]
    let planned = namespace.withUnsafeBufferPointer { namespace in
      tooLarge.withUnsafeBufferPointer { tokens in
        cc_manager_plan(manager, tokens.baseAddress, tokens.count, namespace.baseAddress,
          1, 0, 0, 0, 1, UInt32(CC_SCOPE_SESSION), UInt32(CC_PRIORITY_INTERACTIVE),
          0, 0, nil, 0, nil, 0, 0, UInt32(CC_SLOT_SESSION))
      }
    }
    let firstPlan = try XCTUnwrap(planned)
    XCTAssertEqual(cc_manager_last_status(manager), Int32(CC_OK))
    XCTAssertEqual(cc_plan_abort(firstPlan), Int32(CC_OK))
    cc_plan_destroy(firstPlan)

    let fits: [Int32] = [1]
    let recovered = namespace.withUnsafeBufferPointer { namespace in
      fits.withUnsafeBufferPointer { tokens in
        cc_manager_plan(manager, tokens.baseAddress, tokens.count, namespace.baseAddress,
          1, 0, 0, 0, 2, UInt32(CC_SCOPE_SESSION), UInt32(CC_PRIORITY_INTERACTIVE),
          0, 0, nil, 0, nil, 0, 0, UInt32(CC_SLOT_SESSION))
      }
    }
    let recoveredPlan = try XCTUnwrap(recovered)
    XCTAssertEqual(cc_plan_abort(recoveredPlan), Int32(CC_OK))
    cc_plan_destroy(recoveredPlan)
  }

  func testDynamicRetentionABIRegistrationTelemetryAndCeiling() throws {
    let manager = try XCTUnwrap(cc_manager_create_with_retention(
      1, 1, 64, 3, 3, 1_000, 800, 500))
    defer { cc_manager_destroy(manager) }

    var slot: UInt32 = 0
    var generation: UInt64 = 0
    XCTAssertEqual(cc_manager_register_slot(manager, &slot, &generation), Int32(CC_OK))
    XCTAssertEqual(slot, 1)
    XCTAssertGreaterThan(generation, 0)
    XCTAssertEqual(cc_manager_register_slot(manager, &slot, &generation), Int32(CC_OK))
    XCTAssertEqual(slot, 2)
    XCTAssertEqual(cc_manager_register_slot(manager, &slot, &generation), Int32(CC_NO_CAPACITY))

    var telemetry = cc_retention_telemetry_t()
    XCTAssertEqual(cc_manager_retention_telemetry(manager, &telemetry), Int32(CC_OK))
    XCTAssertEqual(telemetry.total_slots, 3)
    XCTAssertEqual(telemetry.physical_byte_budget, 1_000)
    XCTAssertEqual(telemetry.high_watermark_bytes, 800)
    XCTAssertEqual(telemetry.low_watermark_bytes, 500)

    XCTAssertEqual(cc_manager_set_anchor_protected(manager, 0, 1, 1),
      Int32(CC_INVALID_ARGUMENT))
  }
}
