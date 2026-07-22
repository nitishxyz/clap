import ClapCachePolicy
import Foundation

public struct ModelTokenCapabilities: Equatable, Sendable {
  public let declaredContextLength: Int
  public let effectiveContextLength: Int
  public let maxOutputTokens: Int
  public let contextLengthSource: String?
  public let maxOutputTokensSource: String?
  public let hybridOrRecurrent: Bool

  public static let empty = ModelTokenCapabilities(
    declaredContextLength: 0, effectiveContextLength: 0, maxOutputTokens: 0,
    contextLengthSource: nil, maxOutputTokensSource: nil, hybridOrRecurrent: false)

  public static func derive(metadata: DeclaredModelMetadata, contextOverride: Int,
                            sessionCap: Int, outputOverride: Int) -> ModelTokenCapabilities {
    let declaredContextLength = metadata.context?.value ?? 0
    let knownContextCaps = [declaredContextLength, contextOverride, sessionCap].filter { $0 > 0 }
    let effectiveContextLength = knownContextCaps.min() ?? 0
    var maxOutputTokens = metadata.maxOutputTokens?.value ?? 0
    var maxOutputTokensSource = metadata.maxOutputTokens?.source
    if outputOverride > 0 {
      if maxOutputTokens == 0 || outputOverride < maxOutputTokens {
        maxOutputTokens = outputOverride
        maxOutputTokensSource = "environment:CLAP_MLX_MAX_OUTPUT"
      }
    }
    let capabilityText = [metadata.architecture, metadata.modelType]
      .compactMap { $0?.lowercased() }.joined(separator: " ")
    let hybridOrRecurrent = ["hybrid", "recurrent", "mamba", "deltanet", "ssm"]
      .contains { capabilityText.contains($0) }
    return ModelTokenCapabilities(declaredContextLength: declaredContextLength,
      effectiveContextLength: effectiveContextLength,
      maxOutputTokens: maxOutputTokens,
      contextLengthSource: metadata.context?.source,
      maxOutputTokensSource: maxOutputTokensSource,
      hybridOrRecurrent: hybridOrRecurrent)
  }

  public func resolveOutputTokens(promptTokens: Int,
                                  requestedMaxTokens: Int?) -> Result<Int, PromptTokenLimitError> {
    if effectiveContextLength > 0 && promptTokens >= effectiveContextLength {
      return .failure(PromptTokenLimitError(
        message: "prompt is too long for the loaded model; prompt_tokens=\(promptTokens), max_input_tokens=\(effectiveContextLength - 1), effective_context_window=\(effectiveContextLength).",
        code: "context_length_exceeded"))
    }
    if let requestedMaxTokens, maxOutputTokens > 0, requestedMaxTokens > maxOutputTokens {
      return .failure(PromptTokenLimitError(
        message: "requested max_tokens=\(requestedMaxTokens) exceeds the loaded model maximum output tokens=\(maxOutputTokens).",
        code: "max_output_tokens_exceeded"))
    }
    if requestedMaxTokens == nil && effectiveContextLength == 0 && maxOutputTokens == 0 {
      return .failure(PromptTokenLimitError(
        message: "max_tokens is required because this model does not declare token limits.",
        code: "token_capability_unknown"))
    }
    let availableOutput = effectiveContextLength > 0
      ? effectiveContextLength - promptTokens : maxOutputTokens
    let resolved = requestedMaxTokens
      ?? (maxOutputTokens > 0 ? min(maxOutputTokens, availableOutput) : availableOutput)
    if effectiveContextLength > 0 && promptTokens + resolved > effectiveContextLength {
      return .failure(PromptTokenLimitError(
        message: "prompt plus requested output exceeds the loaded model context; prompt_tokens=\(promptTokens), requested_output_tokens=\(resolved), effective_context_window=\(effectiveContextLength).",
        code: "context_length_exceeded"))
    }
    return .success(resolved)
  }
}

public func declaredEosTokenIds(_ url: URL) -> Set<Int> {
  guard let data = try? Data(contentsOf: url),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
  var ids: Set<Int> = []
  func collect(_ value: Any?) {
    if let id = value as? Int { ids.insert(id) }
    if let list = value as? [Any] { for item in list { collect(item) } }
  }
  collect(json["eos_token_id"])
  if let textConfig = json["text_config"] as? [String: Any] { collect(textConfig["eos_token_id"]) }
  return ids
}
