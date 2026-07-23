# Native Runtime Bundle

Packaged Clap builds look for resident native workers here:

- `libexec/clap-llama` for GGUF / llama.cpp
- `libexec/clap-mlx` for MLX / Swift MLX
- `libexec/mlx.metallib` for the MLX Metal library (required next to `clap-mlx`)

This repository does not include those binaries. Packaging must build or copy real executables before release. Production does not substitute mocks, one-shot subprocesses, or a legacy unversioned protocol — workers speak strict protocol v1 under TypeScript process ownership.

See [`docs/native-runtimes.md`](../docs/native-runtimes.md) for discovery, protocol, and per-launch logs, and [`docs/inference-runtime-baseline.md`](../docs/inference-runtime-baseline.md) for ownership, admission, cache identity, and verification.

```bash
bun run runtime:llama:vendor && bun run runtime:llama:build
bun run runtime:mlx:build   # macOS arm64
CLAP_LLAMA_WORKER=/path/to/clap-llama \
CLAP_MLX_WORKER=/path/to/clap-mlx \
bun run bundle:prepare
bun run bundle:check
```

`bundle:check` fails if any required executable is missing, empty, or not executable. Missing workers are reported as `not_installed` or `unsupported`.
