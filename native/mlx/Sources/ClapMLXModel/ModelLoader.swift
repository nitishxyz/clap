import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public enum ModelLoaderError: Error, CustomStringConvertible, Equatable {
  case invalidModelDirectory(String)

  public var description: String {
    switch self {
    case .invalidModelDirectory(let message): message
    }
  }
}

public enum ModelLoader {
  public static func validateDirectory(_ path: String) throws -> URL {
    let url = URL(fileURLWithPath: path)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
          isDirectory.boolValue else {
      throw ModelLoaderError.invalidModelDirectory("MLX model directory not found: \(path)")
    }
    guard FileManager.default.fileExists(
      atPath: url.appendingPathComponent("config.json").path) else {
      throw ModelLoaderError.invalidModelDirectory(
        "MLX model directory is missing config.json: \(path)")
    }
    return url
  }

  static func load(from directory: URL) async throws -> LoadedModelComponents {
    let container = try await ModelDirectoryCompatibility.withCompatibleDirectory(
      for: directory
    ) { compatibleDirectory in
      try await loadModelContainer(
        from: compatibleDirectory, using: #huggingFaceTokenizerLoader())
    }
    let box = await container.perform { context in
      UncheckedBox(value: (context.model, context.tokenizer,
        context.configuration.extraEOSTokens))
    }
    return LoadedModelComponents(model: box.value.0, tokenizer: box.value.1,
      extraEOSTokens: box.value.2)
  }
}

struct LoadedModelComponents {
  let model: any LanguageModel
  let tokenizer: any MLXLMCommon.Tokenizer
  let extraEOSTokens: Set<String>
}

private struct UncheckedBox<T>: @unchecked Sendable {
  let value: T
}
