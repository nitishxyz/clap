import ClapMLXCache
import ClapMLXGeneration
import MLX
import MLXLMCommon


typealias MLXActiveRequest = ActiveRequest<KVCache, TokenIterator,
  NaiveStreamingDetokenizer, GenerateParameters>

typealias MLXGenerationBackend = GenerationBackend<KVCache, TokenIterator,
  NaiveStreamingDetokenizer, GenerateParameters>

func mlxGenerationBackend(model: any LanguageModel,
                          appendAndAdvance: @escaping (Int, CacheSlot<KVCache>,
                            [KVCache], inout [Int], [Int]) -> Void,
                          plantAnchor: @escaping (Int, [KVCache], [Int], CacheIdentity,
                            UInt32, Bool) -> AnchorResult,
                          captureContinuation: @escaping (CacheSnapshots<KVCache>, Int,
                            [KVCache], [Int]) -> Double,
                          capturePromptBoundary: @escaping (CacheSnapshots<KVCache>, [Int],
                            [KVCache], [Int]) -> Double,
                          now: @escaping () -> UInt64) -> MLXGenerationBackend {
  GenerationBackend(prefill: { tokens, caches, parameters in
    try TokenIterator(input: LMInput(tokens: MLXArray(tokens)), model: model,
      cache: caches, parameters: parameters)
  }, nextToken: { iterator in
    iterator.next()
  }, appendToken: { detokenizer, token in
    detokenizer.append(token: token)
  }, nextText: { detokenizer in
    detokenizer.next()
  }, appendAndAdvance: appendAndAdvance, plantAnchor: plantAnchor,
  captureContinuation: captureContinuation,
  capturePromptBoundary: capturePromptBoundary, now: now)
}
