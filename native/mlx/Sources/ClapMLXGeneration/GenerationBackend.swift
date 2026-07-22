import ClapMLXCache

public struct GenerationBackend<Cache, Iterator, Detokenizer, Parameters> {
  public let prefill: ([Int], inout [Cache], Parameters) throws -> Iterator
  public let nextToken: (inout Iterator) throws -> Int?
  public let appendToken: (inout Detokenizer, Int) -> Void
  public let nextText: (inout Detokenizer) -> String?
  public let appendAndAdvance: (Int, CacheSlot<Cache>, [Cache], inout [Int], [Int]) -> Void
  public let plantAnchor: (Int, [Cache], [Int], CacheIdentity, UInt32, Bool) -> AnchorResult
  public let captureContinuation: (CacheSnapshots<Cache>, Int, [Cache], [Int]) -> Double
  public let capturePromptBoundary: (CacheSnapshots<Cache>, [Int], [Cache], [Int]) -> Double
  public let now: () -> UInt64

  public init(prefill: @escaping ([Int], inout [Cache], Parameters) throws -> Iterator,
              nextToken: @escaping (inout Iterator) throws -> Int?,
              appendToken: @escaping (inout Detokenizer, Int) -> Void,
              nextText: @escaping (inout Detokenizer) -> String?,
              appendAndAdvance: @escaping (Int, CacheSlot<Cache>, [Cache], inout [Int], [Int]) -> Void,
              plantAnchor: @escaping (Int, [Cache], [Int], CacheIdentity, UInt32, Bool) -> AnchorResult,
              captureContinuation: @escaping (CacheSnapshots<Cache>, Int, [Cache], [Int]) -> Double,
              capturePromptBoundary: @escaping (CacheSnapshots<Cache>, [Int], [Cache], [Int]) -> Double,
              now: @escaping () -> UInt64) {
    self.prefill = prefill
    self.nextToken = nextToken
    self.appendToken = appendToken
    self.nextText = nextText
    self.appendAndAdvance = appendAndAdvance
    self.plantAnchor = plantAnchor
    self.captureContinuation = captureContinuation
    self.capturePromptBoundary = capturePromptBoundary
    self.now = now
  }
}
