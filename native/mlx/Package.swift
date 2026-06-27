// swift-tools-version: 6.1
import PackageDescription

let package = Package(
  name: "ClapMLX",
  platforms: [.macOS(.v14)],
  products: [
    .executable(name: "clap-mlx", targets: ["clap-mlx"]),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMajor(from: "3.31.3")),
    .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0"),
  ],
  targets: [
    .executableTarget(
      name: "clap-mlx",
      dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ]
    ),
  ]
)
