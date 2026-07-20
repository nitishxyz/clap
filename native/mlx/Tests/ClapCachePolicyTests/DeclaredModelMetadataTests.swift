import Foundation
import Testing
@testable import ClapCachePolicy

@Suite("Declared model metadata")
struct DeclaredModelMetadataTests {
  @Test("Gemma 4 uses text context, not sliding or multimodal sibling limits")
  func gemma4() throws {
    let directory = try fixture(
      config: """
      {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
          "model_type": "gemma4_text",
          "max_position_embeddings": 131072,
          "sliding_window": 512,
          "vocab_size": 262144,
          "hidden_size": 2560,
          "num_hidden_layers": 42,
          "head_dim": 256,
          "eos_token_id": 1,
          "pad_token_id": 0
        },
        "vision_config": {
          "model_type": "gemma4_vision",
          "architectures": null,
          "max_position_embeddings": 131072,
          "hidden_size": 768,
          "num_hidden_layers": 16,
          "head_dim": 64
        },
        "audio_config": {
          "model_type": "gemma4_audio",
          "architectures": null,
          "hidden_size": 1024,
          "num_hidden_layers": 12
        },
        "generation_config": {"eos_token_id": [1, 106, 50], "pad_token_id": 0}
      }
      """,
      generation: "{\"eos_token_id\":[1,106,50],\"pad_token_id\":0}",
      tokenizer: "{\"model_max_length\":1000000000000000019884624838656,\"tokenizer_class\":\"GemmaTokenizer\"}"
    )
    defer { try? FileManager.default.removeItem(at: directory) }

    let metadata = DeclaredModelMetadata.load(from: directory)
    #expect(metadata.architecture == "Gemma4ForConditionalGeneration")
    #expect(metadata.modelType == "gemma4")
    #expect(metadata.context == DeclaredInteger(value: 131072, source: "config.json:text_config.max_position_embeddings"))
    #expect(metadata.slidingWindow == DeclaredInteger(value: 512, source: "config.json:text_config.sliding_window"))
    #expect(metadata.maxOutputTokens == nil)
  }

  @Test("flat profiles and generation config use explicit precedence")
  func flatProfile() throws {
    let directory = try fixture(
      config: "{\"model_type\":\"llama\",\"architectures\":[\"LlamaForCausalLM\"],\"max_position_embeddings\":32768,\"max_new_tokens\":1024}",
      generation: "{\"max_new_tokens\":2048,\"max_length\":999999}"
    )
    defer { try? FileManager.default.removeItem(at: directory) }

    let metadata = DeclaredModelMetadata.load(from: directory)
    #expect(metadata.context == DeclaredInteger(value: 32768, source: "config.json:max_position_embeddings"))
    #expect(metadata.maxOutputTokens == DeclaredInteger(value: 2048, source: "generation_config.json:max_new_tokens"))
  }

  @Test("known language containers outrank top-level compatibility fields")
  func nestedProfile() throws {
    let directory = try fixture(config: """
      {
        "model_type":"multimodal",
        "max_position_embeddings":4096,
        "language_config":{"model_type":"nested_text","max_sequence_length":65536,"max_new_tokens":8192},
        "vision_config":{"max_position_embeddings":2048}
      }
      """)
    defer { try? FileManager.default.removeItem(at: directory) }

    let metadata = DeclaredModelMetadata.load(from: directory)
    #expect(metadata.context == DeclaredInteger(value: 65536, source: "config.json:language_config.max_sequence_length"))
    #expect(metadata.maxOutputTokens == DeclaredInteger(value: 8192, source: "config.json:language_config.max_new_tokens"))
  }

  private func fixture(config: String, generation: String? = nil, tokenizer: String? = nil) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
    if let generation { try Data(generation.utf8).write(to: directory.appendingPathComponent("generation_config.json")) }
    if let tokenizer { try Data(tokenizer.utf8).write(to: directory.appendingPathComponent("tokenizer_config.json")) }
    return directory
  }
}
