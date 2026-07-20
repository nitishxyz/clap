import XCTest
@testable import ClapCachePolicy

final class LatencySchedulerTests: XCTestCase {
  private func request(_ id: String, _ order: UInt64, residual: Int,
                       decoding: Bool = false, emitted: Bool = false,
                       cancelled: Bool = false) -> LatencySchedulerRequest {
    LatencySchedulerRequest(id: id, admissionOrder: order,
      residualPrefillTokens: residual, decoding: decoding,
      emittedFirstToken: emitted, cancelled: cancelled)
  }

  func testLongPrefillAndTwentyFiveTokenHit() {
    let round = LatencyScheduler.round([
      request("long", 1, residual: 12_877),
      request("hit", 2, residual: 25),
    ])
    XCTAssertEqual(round.map(\.id), ["hit", "long"])
    XCTAssertEqual(round.map(\.prefillQuantum), [96, 96])
    XCTAssertEqual(round.map(\.turns), [3, 1])
  }

  func testLongPrefillAndTwoHundredFiftyFiveTokenHit() {
    let round = LatencyScheduler.round([
      request("long", 1, residual: 12_877),
      request("hit", 2, residual: 255),
    ])
    XCTAssertEqual(round.map(\.id), ["hit", "long"])
    XCTAssertEqual(round.map(\.turns), [5, 1])
  }

  func testMultipleLongPrefillsKeepAdmissionOrder() {
    let round = LatencyScheduler.round([
      request("second", 2, residual: 8_000),
      request("first", 1, residual: 12_000),
      request("third", 3, residual: 6_000),
    ])
    XCTAssertEqual(round.map(\.id), ["first", "second", "third"])
  }

  func testContinuousArrivalsCannotStarveLongRequest() {
    for arrival in 2...100 {
      let round = LatencyScheduler.round([
        request("long", 1, residual: 20_000),
        request("short-\(arrival)", UInt64(arrival), residual: 25),
      ])
      XCTAssertEqual(round.count, 2)
      XCTAssertEqual(round.last?.id, "long")
    }
  }

  func testCancellationRemovesRequestFromRound() {
    let round = LatencyScheduler.round([
      request("cancelled", 1, residual: 25, cancelled: true),
      request("live", 2, residual: 10_000),
    ])
    XCTAssertEqual(round, [.init(id: "live", prefillQuantum: 512, turns: 1)])
  }

  func testDecodeFairnessAndFirstTokenBoost() {
    let round = LatencyScheduler.round([
      request("long", 1, residual: 10_000),
      request("decode", 2, residual: 0, decoding: true, emitted: true),
      request("first-decode", 3, residual: 0, decoding: true),
    ])
    XCTAssertEqual(round.map(\.id), ["first-decode", "decode", "long"])
    XCTAssertEqual(Set(round.map(\.id)).count, 3)
  }
}
