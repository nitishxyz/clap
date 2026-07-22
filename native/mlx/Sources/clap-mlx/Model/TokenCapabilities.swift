import ClapCachePolicy
import Foundation

struct ModelTokenCapabilities {
  let declaredContextLength: Int
  let effectiveContextLength: Int
  let maxOutputTokens: Int
  let contextLengthSource: String?
  let maxOutputTokensSource: String?
  let hybridOrRecurrent: Bool

  static let empty = ModelTokenCapabilities(
    declaredContextLength: 0, effectiveContextLength: 0, maxOutputTokens: 0,
    contextLengthSource: nil, maxOutputTokensSource: nil, hybridOrRecurrent: false)

  static func derive(metadata: DeclaredModelMetadata, contextOverride: Int,
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

  func workerEvent(contextOverride: Int) -> WorkerTokenCapabilities {
    WorkerTokenCapabilities(
      model_context_window: declaredContextLength > 0 ? declaredContextLength : nil,
      effective_context_window: effectiveContextLength > 0 ? effectiveContextLength : nil,
      max_input_tokens: effectiveContextLength > 0 ? effectiveContextLength - 1 : nil,
      max_output_tokens: maxOutputTokens > 0 ? maxOutputTokens : nil,
      model_context_window_source: contextLengthSource,
      max_output_tokens_source: maxOutputTokensSource,
      backend_allocation_cap: contextOverride > 0 ? contextOverride
        : (declaredContextLength > 0 ? declaredContextLength : nil),
      user_configured_override: contextOverride > 0 ? contextOverride : nil)
  }
}

func declaredEosTokenIds(_ url: URL) -> Set<Int> {
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
