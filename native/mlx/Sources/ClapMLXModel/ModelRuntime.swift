import ClapCachePolicy
import Foundation
import MLXLLM
import MLXLMCommon

public final class ModelRuntime {
  public private(set) var modelIdentifier: String?
  public private(set) var directory: URL?
  public private(set) var languageModel: (any LanguageModel)?
  public private(set) var tokenizer: (any MLXLMCommon.Tokenizer)?
  public private(set) var eosTokenIds: Set<Int> = []
  public private(set) var metadata: DeclaredModelMetadata?
  public private(set) var tokenCapabilities = ModelTokenCapabilities.empty

  public init() {}

  public var isLoaded: Bool { languageModel != nil }

  public func load(identifier: String, directory: URL, contextOverride: Int,
                   sessionCap: Int, outputOverride: Int) async throws {
    let components = try await ModelLoader.load(from: directory)
    let metadata = DeclaredModelMetadata.load(from: directory)
    let capabilities = ModelTokenCapabilities.derive(metadata: metadata,
      contextOverride: contextOverride, sessionCap: sessionCap,
      outputOverride: outputOverride)
    var eosTokenIds: Set<Int> = []
    if let eos = components.tokenizer.eosTokenId { eosTokenIds.insert(eos) }
    for extra in components.extraEOSTokens {
      if let id = components.tokenizer.convertTokenToId(extra) { eosTokenIds.insert(id) }
    }
    for file in ["generation_config.json", "config.json"] {
      eosTokenIds.formUnion(declaredEosTokenIds(directory.appendingPathComponent(file)))
    }

    languageModel = components.model
    tokenizer = components.tokenizer
    modelIdentifier = identifier
    self.directory = directory
    self.eosTokenIds = eosTokenIds
    self.metadata = metadata
    tokenCapabilities = capabilities
  }

  public func unload() {
    languageModel = nil
    tokenizer = nil
    modelIdentifier = nil
    directory = nil
    eosTokenIds = []
    metadata = nil
    tokenCapabilities = .empty
  }
}
