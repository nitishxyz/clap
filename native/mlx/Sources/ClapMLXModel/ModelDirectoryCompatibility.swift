import CoreFoundation
import Foundation

enum ModelDirectoryCompatibility {
  static func withCompatibleDirectory<Result>(
    for directory: URL,
    operation: (URL) async throws -> Result
  ) async throws -> Result {
    guard let config = try normalizedConfig(in: directory) else {
      return try await operation(directory)
    }

    let mirror = FileManager.default.temporaryDirectory
      .appendingPathComponent("clap-mlx-model-\(UUID().uuidString)", isDirectory: true)
    try createMirror(of: directory, at: mirror, normalizedConfig: config)
    defer { try? FileManager.default.removeItem(at: mirror) }
    return try await operation(mirror)
  }

  private static func normalizedConfig(in directory: URL) throws -> Data? {
    let configURL = directory.appendingPathComponent("config.json")
    let original = try Data(contentsOf: configURL)
    guard var object = try JSONSerialization.jsonObject(with: original) as? [String: Any],
          object["model_type"] as? String == "granitemoehybrid" else {
      return nil
    }

    var changed = false
    if object["rope_theta"] == nil,
       let ropeParameters = object["rope_parameters"] as? [String: Any],
       let ropeTheta = ropeParameters["rope_theta"], isJSONNumber(ropeTheta) {
      object["rope_theta"] = ropeTheta
      changed = true
    }

    if isAntaresTaggedInfinity(object["time_step_limit"]),
       hasOnlyAttentionLayers(object["layer_types"]) {
      object.removeValue(forKey: "time_step_limit")
      changed = true
    }

    guard changed else { return nil }
    return try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
  }

  private static func isJSONNumber(_ value: Any) -> Bool {
    guard let number = value as? NSNumber else { return false }
    return CFGetTypeID(number) != CFBooleanGetTypeID()
  }

  private static func isAntaresTaggedInfinity(_ value: Any?) -> Bool {
    guard let values = value as? [Any], values.count == 2,
          isJSONNumber(values[0]),
          let tag = values[1] as? [String: Any], tag.count == 1,
          tag["__float__"] as? String == "Infinity" else {
      return false
    }
    return true
  }

  private static func hasOnlyAttentionLayers(_ value: Any?) -> Bool {
    guard let layers = value as? [Any], !layers.isEmpty else { return false }
    return layers.allSatisfy { ($0 as? String) == "attention" }
  }

  private static func createMirror(of source: URL, at mirror: URL,
                                   normalizedConfig: Data) throws {
    let fileManager = FileManager.default
    try fileManager.createDirectory(at: mirror, withIntermediateDirectories: false)
    do {
      try mirrorContents(of: source, at: mirror, relativePath: "")
      try normalizedConfig.write(
        to: mirror.appendingPathComponent("config.json"), options: .atomic)
    } catch {
      try? fileManager.removeItem(at: mirror)
      throw error
    }
  }

  private static func mirrorContents(of source: URL, at mirror: URL,
                                     relativePath: String) throws {
    let fileManager = FileManager.default
    let sourceDirectory = relativePath.isEmpty
      ? source : source.appendingPathComponent(relativePath, isDirectory: true)
    let entries = try fileManager.contentsOfDirectory(
      at: sourceDirectory,
      includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])

    for entry in entries {
      let relativeEntry = relativePath.isEmpty
        ? entry.lastPathComponent
        : relativePath + "/" + entry.lastPathComponent
      if relativeEntry == "config.json" { continue }
      let destination = mirror.appendingPathComponent(relativeEntry)
      let values = try entry.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
      if values.isDirectory == true, values.isSymbolicLink != true {
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: false)
        try mirrorContents(of: source, at: mirror, relativePath: relativeEntry)
      } else {
        try fileManager.createSymbolicLink(at: destination, withDestinationURL: entry)
      }
    }
  }
}
