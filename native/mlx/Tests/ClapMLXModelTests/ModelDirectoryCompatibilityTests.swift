import Foundation
import Testing
@testable import ClapMLXModel

@Suite("Model directory compatibility")
struct ModelDirectoryCompatibilityTests {
  @Test("normalizes Antares Granite config without changing original bytes")
  func normalizesAntaresConfig() async throws {
    let fixture = try Fixture(config: [
      "model_type": "granitemoehybrid",
      "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
      "layer_types": ["attention", "attention"],
      "time_step_limit": [0.0, ["__float__": "Infinity"]],
    ], nestedFiles: ["weights/model.safetensors": "weights"])
    defer { fixture.remove() }
    let original = try Data(contentsOf: fixture.configURL)
    var mirror: URL?

    try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
      mirror = url
      #expect(url != fixture.directory)
      let config = try config(at: url)
      #expect((config["rope_theta"] as? NSNumber)?.intValue == 10_000_000)
      #expect(config["time_step_limit"] == nil)
      let nested = url.appendingPathComponent("weights/model.safetensors")
      #expect(try String(contentsOf: nested, encoding: .utf8) == "weights")
      #expect(try nested.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true)
      #expect(try url.appendingPathComponent("weights")
        .resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]).isDirectory == true)
    }

    #expect(try Data(contentsOf: fixture.configURL) == original)
    #expect(mirror.map { !FileManager.default.fileExists(atPath: $0.path) } == true)
  }

  @Test("preserves an existing top-level rope theta")
  func topLevelRopeThetaTakesPrecedence() async throws {
    let fixture = try Fixture(config: [
      "model_type": "granitemoehybrid",
      "rope_theta": 42,
      "rope_parameters": ["rope_theta": 99],
      "layer_types": ["attention"],
      "time_step_limit": [0, ["__float__": "Infinity"]],
    ])
    defer { fixture.remove() }

    try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
      let config = try config(at: url)
      #expect((config["rope_theta"] as? NSNumber)?.intValue == 42)
    }
  }

  @Test("passes through non-Granite and unchanged Granite directories")
  func passThrough() async throws {
    for configObject: [String: Any] in [
      ["model_type": "llama", "rope_parameters": ["rope_theta": 100]],
      ["model_type": "granitemoehybrid", "rope_theta": 100],
    ] {
      let fixture = try Fixture(config: configObject)
      defer { fixture.remove() }
      try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
        #expect(url == fixture.directory)
      }
    }
  }

  @Test("keeps tagged Infinity for models with Mamba layers")
  func keepsTaggedInfinityForMamba() async throws {
    let fixture = try Fixture(config: [
      "model_type": "granitemoehybrid",
      "rope_parameters": ["rope_theta": 100],
      "layer_types": ["attention", "mamba"],
      "time_step_limit": [0, ["__float__": "Infinity"]],
    ])
    defer { fixture.remove() }

    try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
      #expect(url != fixture.directory)
      let mirroredConfig = try config(at: url)
      #expect(mirroredConfig["time_step_limit"] != nil)
    }
  }

  @Test("keeps malformed nested compatibility values")
  func keepsMalformedNestedValues() async throws {
    let malformedConfigs: [[String: Any]] = [
      [
        "model_type": "granitemoehybrid",
        "rope_parameters": ["rope_theta": true],
        "layer_types": ["attention"],
        "time_step_limit": [0, ["__float__": "not-infinity"]],
      ],
      [
        "model_type": "granitemoehybrid",
        "rope_parameters": ["rope_theta": "100"],
        "layer_types": ["attention", 1],
        "time_step_limit": [0, ["__float__": "Infinity"]],
      ],
    ]

    for configObject in malformedConfigs {
      let fixture = try Fixture(config: configObject)
      defer { fixture.remove() }
      try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
        #expect(url == fixture.directory)
      }
    }
  }

  @Test("cleans mirror after operation error")
  func cleanupAfterError() async throws {
    let fixture = try Fixture(config: adaptableConfig)
    defer { fixture.remove() }
    var mirror: URL?

    await #expect(throws: TestError.expected) {
      try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
        mirror = url
        throw TestError.expected
      }
    }
    #expect(mirror.map { !FileManager.default.fileExists(atPath: $0.path) } == true)
  }

  @Test("cleans mirror after cancellation")
  func cleanupAfterCancellation() async throws {
    let fixture = try Fixture(config: adaptableConfig)
    defer { fixture.remove() }
    let captured = CapturedURL()
    let task = Task {
      try await ModelDirectoryCompatibility.withCompatibleDirectory(for: fixture.directory) { url in
        await captured.set(url)
        try await Task.sleep(for: .seconds(30))
      }
    }

    while await captured.value == nil { await Task.yield() }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
    let mirror = await captured.value
    #expect(mirror.map { !FileManager.default.fileExists(atPath: $0.path) } == true)
  }

  private var adaptableConfig: [String: Any] {
    [
      "model_type": "granitemoehybrid",
      "rope_parameters": ["rope_theta": 100],
    ]
  }

  private func config(at directory: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
    return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
  }
}

private enum TestError: Error {
  case expected
}

private actor CapturedURL {
  var value: URL?

  func set(_ value: URL) {
    self.value = value
  }
}

private struct Fixture {
  let directory: URL

  var configURL: URL { directory.appendingPathComponent("config.json") }

  init(config: [String: Any], nestedFiles: [String: String] = [:]) throws {
    directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("clap-mlx-compat-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: configURL)
    for (path, contents) in nestedFiles {
      let url = directory.appendingPathComponent(path)
      try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data(contents.utf8).write(to: url)
    }
  }

  func remove() {
    try? FileManager.default.removeItem(at: directory)
  }
}
