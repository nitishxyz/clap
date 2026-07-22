import ClapMLXModel

extension ModelTokenCapabilities {
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
