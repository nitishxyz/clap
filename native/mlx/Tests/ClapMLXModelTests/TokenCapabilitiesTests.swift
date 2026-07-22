import Foundation
import Testing
import ClapCachePolicy
@testable import ClapMLXModel

@Suite("Model token capabilities")
struct TokenCapabilitiesTests {
  @Test("derives context and output sources with configured caps")
  func derivesSourcesAndCaps() throws {
    let directory = try fixture(config: """
      {"model_type":"llama","architectures":["LlamaForCausalLM"],"max_position_embeddings":32768}
      """, generation: "{\"max_new_tokens\":2048}")
    defer { try? FileManager.default.removeItem(at: directory) }

    let metadata = DeclaredModelMetadata.load(from: directory)
    let capabilities = ModelTokenCapabilities.derive(metadata: metadata,
      contextOverride: 16384, sessionCap: 8192, outputOverride: 1024)

    #expect(capabilities.declaredContextLength == 32768)
    #expect(capabilities.effectiveContextLength == 8192)
    #expect(capabilities.contextLengthSource == "config.json:max_position_embeddings")
    #expect(capabilities.maxOutputTokens == 1024)
    #expect(capabilities.maxOutputTokensSource == "environment:CLAP_MLX_MAX_OUTPUT")
  }

  @Test("keeps declared output source when override is not tighter")
  func keepsDeclaredOutputSource() throws {
    let directory = try fixture(config: "{\"max_position_embeddings\":4096}",
      generation: "{\"max_new_tokens\":256}")
    defer { try? FileManager.default.removeItem(at: directory) }

    let capabilities = ModelTokenCapabilities.derive(
      metadata: DeclaredModelMetadata.load(from: directory),
      contextOverride: 0, sessionCap: 0, outputOverride: 512)
    #expect(capabilities.maxOutputTokens == 256)
    #expect(capabilities.maxOutputTokensSource == "generation_config.json:max_new_tokens")
  }

  @Test("classifies recurrent and hybrid model metadata")
  func classifiesHybridModels() throws {
    for marker in ["HybridForCausalLM", "recurrent", "Mamba2", "DeltaNet", "custom_ssm"] {
      let directory = try fixture(config: "{\"model_type\":\"\(marker)\",\"architectures\":[\"\(marker)\"]}")
      defer { try? FileManager.default.removeItem(at: directory) }
      let capabilities = ModelTokenCapabilities.derive(
        metadata: DeclaredModelMetadata.load(from: directory),
        contextOverride: 0, sessionCap: 0, outputOverride: 0)
      #expect(capabilities.hybridOrRecurrent)
    }
  }

  @Test("parses scalar, array, and nested declared EOS token IDs")
  func parsesEosTokenIds() throws {
    let directory = try fixture(config: """
      {"eos_token_id":2,"text_config":{"eos_token_id":[3,4]}}
      """)
    defer { try? FileManager.default.removeItem(at: directory) }
    #expect(declaredEosTokenIds(directory.appendingPathComponent("config.json")) == [2, 3, 4])
    #expect(declaredEosTokenIds(directory.appendingPathComponent("missing.json")).isEmpty)
  }

  @Test("preserves prompt and output limit errors and default resolution")
  func resolvesLimits() throws {
    let directory = try fixture(config: "{\"max_position_embeddings\":100}",
      generation: "{\"max_new_tokens\":20}")
    defer { try? FileManager.default.removeItem(at: directory) }
    let capabilities = ModelTokenCapabilities.derive(
      metadata: DeclaredModelMetadata.load(from: directory),
      contextOverride: 0, sessionCap: 0, outputOverride: 0)

    #expect(capabilities.resolveOutputTokens(promptTokens: 80, requestedMaxTokens: nil) == .success(20))
    #expect(capabilities.resolveOutputTokens(promptTokens: 100, requestedMaxTokens: 1) == .failure(
      PromptTokenLimitError(message: "prompt is too long for the loaded model; prompt_tokens=100, max_input_tokens=99, effective_context_window=100.", code: "context_length_exceeded")))
    #expect(capabilities.resolveOutputTokens(promptTokens: 10, requestedMaxTokens: 21) == .failure(
      PromptTokenLimitError(message: "requested max_tokens=21 exceeds the loaded model maximum output tokens=20.", code: "max_output_tokens_exceeded")))
    #expect(capabilities.resolveOutputTokens(promptTokens: 90, requestedMaxTokens: 11) == .failure(
      PromptTokenLimitError(message: "prompt plus requested output exceeds the loaded model context; prompt_tokens=90, requested_output_tokens=11, effective_context_window=100.", code: "context_length_exceeded")))

    #expect(ModelTokenCapabilities.empty.resolveOutputTokens(
      promptTokens: 10, requestedMaxTokens: nil) == .failure(
        PromptTokenLimitError(message: "max_tokens is required because this model does not declare token limits.", code: "token_capability_unknown")))
  }

  private func fixture(config: String, generation: String? = nil) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
    if let generation {
      try Data(generation.utf8).write(to: directory.appendingPathComponent("generation_config.json"))
    }
    return directory
  }
}
