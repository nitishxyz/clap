import Foundation

public struct DeclaredInteger: Equatable, Sendable {
  public let value: Int
  public let source: String
}

public struct DeclaredModelMetadata: Equatable, Sendable {
  public let architecture: String?
  public let modelType: String?
  public let context: DeclaredInteger?
  public let slidingWindow: DeclaredInteger?
  public let maxOutputTokens: DeclaredInteger?

  private static let languageContainers = ["text_config", "language_config", "llm_config"]
  private static let contextKeys = ["max_position_embeddings", "max_sequence_length", "seq_length", "n_positions", "n_ctx", "context_length"]

  public static func load(from directory: URL) -> DeclaredModelMetadata {
    let config = json(at: directory.appendingPathComponent("config.json")) ?? [:]
    let generation = json(at: directory.appendingPathComponent("generation_config.json")) ?? [:]
    let tokenizer = json(at: directory.appendingPathComponent("tokenizer_config.json")) ?? [:]
    let languageConfigs = languageContainers.compactMap { key -> (String, [String: Any])? in
      guard let value = config[key] as? [String: Any] else { return nil }
      return (key, value)
    }

    var contextCandidates: [(Any?, String)] = []
    for (container, value) in languageConfigs {
      for key in contextKeys { contextCandidates.append((value[key], "config.json:\(container).\(key)")) }
    }
    for key in contextKeys { contextCandidates.append((config[key], "config.json:\(key)")) }
    contextCandidates.append((tokenizer["model_max_length"], "tokenizer_config.json:model_max_length"))

    var outputCandidates: [(Any?, String)] = [
      (generation["max_new_tokens"], "generation_config.json:max_new_tokens"),
      ((config["generation_config"] as? [String: Any])?["max_new_tokens"], "config.json:generation_config.max_new_tokens"),
      (config["max_output_tokens"], "config.json:max_output_tokens"),
      (config["max_new_tokens"], "config.json:max_new_tokens"),
    ]
    for (container, value) in languageConfigs {
      outputCandidates.append((value["max_output_tokens"], "config.json:\(container).max_output_tokens"))
      outputCandidates.append((value["max_new_tokens"], "config.json:\(container).max_new_tokens"))
    }

    let architecture = firstString(config["architectures"])
      ?? languageConfigs.lazy.compactMap { firstString($0.1["architectures"]) }.first
    let modelType = config["model_type"] as? String
      ?? languageConfigs.lazy.compactMap { $0.1["model_type"] as? String }.first
    let slidingWindow = languageConfigs.lazy.compactMap { container, value in
      positiveInteger(value["sliding_window"], source: "config.json:\(container).sliding_window")
    }.first ?? positiveInteger(config["sliding_window"], source: "config.json:sliding_window")

    return DeclaredModelMetadata(
      architecture: architecture,
      modelType: modelType,
      context: firstPositiveInteger(contextCandidates),
      slidingWindow: slidingWindow,
      maxOutputTokens: firstPositiveInteger(outputCandidates)
    )
  }

  private static func json(at url: URL) -> [String: Any]? {
    guard let data = try? Data(contentsOf: url),
          let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    return value
  }

  private static func firstString(_ value: Any?) -> String? {
    (value as? [Any])?.first(where: { $0 is String }) as? String
  }

  private static func firstPositiveInteger(_ candidates: [(Any?, String)]) -> DeclaredInteger? {
    candidates.lazy.compactMap { positiveInteger($0.0, source: $0.1) }.first
  }

  private static func positiveInteger(_ value: Any?, source: String) -> DeclaredInteger? {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
    let integer = number.int64Value
    guard integer > 0, integer <= Int64(Int.max), number.doubleValue == Double(integer) else { return nil }
    return DeclaredInteger(value: Int(integer), source: source)
  }
}
