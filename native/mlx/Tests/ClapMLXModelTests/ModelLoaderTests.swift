import Foundation
import Testing
@testable import ClapMLXModel

@Suite("Model loader")
struct ModelLoaderTests {
  @Test("rejects a missing model directory")
  func missingDirectory() {
    let path = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString).path
    #expect(throws: ModelLoaderError.invalidModelDirectory(
      "MLX model directory not found: \(path)")) {
      try ModelLoader.validateDirectory(path)
    }
  }

  @Test("rejects a model directory without config.json")
  func missingConfig() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    #expect(throws: ModelLoaderError.invalidModelDirectory(
      "MLX model directory is missing config.json: \(directory.path)")) {
      try ModelLoader.validateDirectory(directory.path)
    }
  }

  @Test("accepts a model directory with config.json")
  func validDirectory() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
    defer { try? FileManager.default.removeItem(at: directory) }
    #expect(try ModelLoader.validateDirectory(directory.path).path == directory.path)
  }
}
