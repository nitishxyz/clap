# Native Runtime Bundle

Packaged Clap builds look for native worker executables here:

- `libexec/clap-llama` for GGUF / llama.cpp
- `libexec/clap-mlx` for MLX / Swift MLX

This repository does not include those native binaries. Packaging must build or copy real executables into this directory before release:

```bash
CLAP_LLAMA_WORKER=/path/to/clap-llama \
CLAP_MLX_WORKER=/path/to/clap-mlx \
bun run bundle:prepare
bun run bundle:check
```

Use `bun run runtime:llama:vendor && bun run runtime:llama:build` to build and install the direct llama.cpp resident worker. `bun run bundle:check` fails if any required executable is missing, empty, or not executable. Missing workers are reported as `not_installed` or `unsupported`.
