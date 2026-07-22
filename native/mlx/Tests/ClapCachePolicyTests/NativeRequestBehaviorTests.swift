import XCTest
@testable import ClapCachePolicy

final class NativeRequestBehaviorTests: XCTestCase {
  func testStopSplitAcrossChunksIsHeldAndThenRemoved() {
    let stops = ["</stop>"]
    let holdback = stops.map(\.count).max()! - 1
    let first = StopSequencePolicy.scan(
      collected: "answer</st", appendedCount: 4, stops: stops,
      emittedCount: 0, holdback: holdback)
    XCTAssertNil(first.matchOffset)
    XCTAssertEqual(first.safeCount, 4)

    let complete = StopSequencePolicy.scan(
      collected: "answer</stop>tail", appendedCount: 7, stops: stops,
      emittedCount: first.safeCount, holdback: holdback)
    XCTAssertEqual(complete.matchOffset, 6)
    XCTAssertEqual(String("answer</stop>tail".prefix(complete.matchOffset!)), "answer")
  }

  func testUnicodeCharacterCountsRemainAttachedToStopOffset() {
    let result = StopSequencePolicy.scan(
      collected: "🙂okEND", appendedCount: 3, stops: ["END"],
      emittedCount: 0, holdback: 2)
    XCTAssertEqual(result.matchOffset, 3)
    XCTAssertEqual(String("🙂okEND".prefix(result.matchOffset!)), "🙂ok")
  }

  func testCancellationAddressesQueuedAndActiveMixedIDs() {
    let activeIDs: [String?] = [nil, "one", "two", "two"]
    let queuedIDs: [String?] = ["queued", "two", nil]
    XCTAssertEqual(activeIDs.indices.filter {
      RequestCancellationPolicy.matches(target: "two", requestID: activeIDs[$0])
    }, [2, 3])
    XCTAssertEqual(queuedIDs.indices.filter {
      RequestCancellationPolicy.matches(target: "two", requestID: queuedIDs[$0])
    }, [1])
    XCTAssertEqual(activeIDs.indices.filter {
      RequestCancellationPolicy.matches(target: nil, requestID: activeIDs[$0])
    }, [0, 1, 2, 3])
    XCTAssertEqual(queuedIDs.indices.filter {
      RequestCancellationPolicy.matches(target: "", requestID: queuedIDs[$0])
    }, [0, 1, 2])
    XCTAssertFalse(RequestCancellationPolicy.matches(target: "one", requestID: nil))
  }
}
