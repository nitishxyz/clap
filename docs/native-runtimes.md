# Native Runtime Workers

Clap ships native runtime workers in `libexec/` and treats missing workers as packaging blockers.
There are no mock, fallback, or placeholder runtime paths in the product flow.

## JSON-Line Protocol

Both workers read exactly one JSON object from stdin and write JSON objects to stdout, one per line.

Input:

```json
{"type":"chat","model":"/path/to/model","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":256,"temperature":0.7}
```

Output:

```json
{"token":"partial text"}
{"done":true}
```

Errors are reported as JSON and cause a non-zero exit:

```json
{"error":"human readable failure"}
```

## llama.cpp GGUF Worker

Build and bundle:

```sh
bun run runtime:llama:vendor
bun run runtime:llama:build
bun run bundle:check
```

`bun run runtime:llama:vendor` clones `ggerganov/llama.cpp` into `vendor/llama.cpp` and initializes submodules.
`bun run runtime:llama:build` configures llama.cpp with CMake, enables Metal on macOS, builds llama.cpp static libraries, links `native/llama/clap-llama.cpp` directly against llama.cpp APIs, and installs `libexec/clap-llama`.
The worker is resident: `load` maps GGUF weights once into the process, `chat` reuses the loaded model/context, and `unload`/`shutdown` free them. It does not shell out to `llama-cli`.

## Swift MLX Worker

Build and bundle on macOS arm64:

```sh
bun run runtime:mlx:build
bun run bundle:check
```

The Swift package in `native/mlx` depends on `mlx-swift-lm`, `mlx-swift`, and Hugging Face tokenizer packages. It builds a real `clap-mlx` executable into `libexec/clap-mlx`.
The worker loads local MLX model directories, runs generation through MLX Swift LM, and writes Clap JSON-line token/content messages. The build script fails immediately on non-macOS-arm64 hosts.

## Runtime Discovery

The control plane checks explicit `CLAP_LLAMA_WORKER` / `CLAP_MLX_WORKER` paths first, then bundled `libexec/clap-llama` and `libexec/clap-mlx` candidates. Missing binaries are reported as `not_installed`; they are not replaced with mocks.

Model inventory endpoints only list installed/cached/local models. Curated aliases are exposed separately at `/clap/v1/aliases` and can still be used with `clap pull <alias>`.
