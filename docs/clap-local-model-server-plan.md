# Clap Local Model Server Plan

## Product Goal

Clap is a single-install local AI server that runs local models behind an OpenAI-compatible API.

The user experience should be:

```bash
clap run qwen2.5:3b
clap serve
clap pull llama3.2:3b
```

Without requiring the user to separately install or run:

- Ollama
- llama.cpp
- MLX Python tooling
- Conda
- model-specific servers
- external daemons

Internally, Clap can bundle multiple native runtimes and route requests to the right backend.

The product promise is **single install, bundled runtimes, no external dependency setup**. It does not need to mean every runtime is physically linked into one executable forever.

## High-Level Architecture

Clap should use a server/client architecture.

```txt
clap CLI / desktop UI / web UI
  -> generated Clap SDK
  -> local Clap server
  -> model resolver
  -> backend router
  -> bundled inference backend
```

The local server is the stable product interface. The CLI should be a client of the server, not the main internal API.

This allows future clients to reuse the same API:

- CLI
- native desktop app
- web UI
- mobile app
- Otto
- Cursor
- Continue
- OpenAI SDK
- Vercel AI SDK
- LangChain

## Core Components

```txt
clap/
  apps/
    cli/
    desktop/              # later
  packages/
    server/               # Bun + Hono local server
    api/                  # Zod schemas, OpenAPI, generated Hey SDK
    models/               # model resolver, aliases, cache, HF downloads
    runtime-router/       # backend selection and lifecycle
    runtime-llama/        # llama.cpp backend wrapper/build scripts
    runtime-mlx/          # Swift MLX backend wrapper/build scripts
  vendor/
    llama.cpp/            # optional vendored runtime source
```

Recommended implementation split:

```txt
Bun/Hono server = API, lifecycle, routing, downloads, cache
Native workers  = heavy inference engines
```

## API Surfaces

Keep two API surfaces.

### OpenAI-Compatible API

This is for third-party clients and provider integrations.

```txt
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings        # later
```

Streaming should use OpenAI-compatible Server-Sent Events:

```txt
data: {"choices":[{"delta":{"content":"hello"}}]}

data: [DONE]
```

### Clap Management API

This is for first-party clients: CLI, desktop app, web UI, mobile app.

```txt
GET  /clap/v1/health
GET  /clap/v1/runtime
GET  /clap/v1/backends
GET  /clap/v1/models
POST /clap/v1/models/pull
POST /clap/v1/models/load
POST /clap/v1/models/unload
GET  /clap/v1/downloads
GET  /clap/v1/logs
```

Use Zod schemas as the source of truth, generate OpenAPI from them, then generate a Hey SDK.

The CLI should call the generated SDK instead of directly calling server internals.

## Runtime Backends

Clap should support multiple bundled backends.

### GGUF / llama.cpp Backend

Use this for:

- local `.gguf` files
- Hugging Face GGUF repos
- Ollama-style aliases that resolve to GGUF
- Linux support
- broad model coverage

Implementation options:

```txt
Option A: bundled clap-llama worker binary
Option B: bundled libllama.dylib / libllama.so via FFI or Node-API
Option C: vendored llama-server hidden behind Clap
```

Recommended first implementation: **worker process**.

Reasons:

- backend crash does not kill the Bun server
- simpler streaming boundary
- easier logs/debugging
- easier platform-specific builds
- easier to replace internals later

### MLX Backend

Use this for:

- MLX-format Hugging Face models
- Apple Silicon optimized inference
- macOS arm64 native performance

Preferred implementation: **Swift MLX worker** using:

- `ml-explore/mlx`
- `ml-explore/mlx-swift`
- `ml-explore/mlx-swift-lm`

Existing projects prove this is feasible, especially native Swift MLX OpenAI-compatible servers such as SwiftLM.

Avoid making Python MLX the core backend. It can be used as a reference or fallback, but the desired product experience should not depend on user-installed Python, Conda, or `mlx-lm`.

## FFI vs Worker Process

Bun can call native libraries through `bun:ffi`, and Bun also supports Node-API addons. These may be useful later, especially for llama.cpp.

However, inference backends are:

- long-running
- memory-heavy
- threaded
- platform-specific
- crash-prone during development
- stateful
- streaming-oriented

For the first production architecture, use worker processes:

```txt
Bun/Hono server
  -> spawn/manage clap-llama or clap-mlx
  -> communicate over stdio, Unix socket, or localhost
  -> stream tokens back to HTTP clients
```

This still satisfies the product goal because the user installs and runs only Clap.

A later optimization can move stable backends in-process through Node-API or FFI.

## Platform Support

### macOS arm64

Ship:

```txt
clap
clap-llama       # llama.cpp with Metal
clap-mlx         # Swift MLX backend
mlx.metallib
```

Supported model formats:

```txt
GGUF -> llama.cpp + Metal
MLX  -> Swift MLX
```

### Linux x64 / arm64

Ship:

```txt
clap
clap-llama       # llama.cpp CPU/CUDA/Vulkan build, depending on release flavor
```

Supported model formats:

```txt
GGUF -> llama.cpp
MLX  -> unsupported or mapped to GGUF equivalent
```

If the user requests an MLX-only model on Linux, Clap should return a useful error or suggest an equivalent GGUF alias.

Example:

```txt
This model requires the MLX backend, which is only available on macOS arm64.
Try: clap run qwen2.5:3b --backend gguf
```

## Model Resolution

Clap should distinguish model source, format, and runtime.

```ts
type ResolvedModel = {
  id: string
  source: "local" | "huggingface" | "alias" | "ollama"
  backend: "llama" | "mlx"
  format: "gguf" | "mlx"
  repo?: string
  file?: string
  localPath: string
  platform: "darwin-arm64" | "linux-x64" | "linux-arm64"
}
```

Routing rules:

```txt
If local path ends in .gguf:
  use llama

If HF repo contains GGUF files:
  use llama

If HF repo has MLX layout and platform is darwin-arm64:
  use mlx

If name matches curated alias:
  choose preferred backend for platform

If Linux:
  prefer GGUF/llama
```

User commands:

```bash
clap run qwen2.5:3b
clap run qwen2.5:3b --backend mlx
clap run qwen2.5:3b --backend gguf
clap run ./model.gguf
clap run mlx-community/Qwen2.5-3B-Instruct-4bit
clap run bartowski/Qwen2.5-3B-Instruct-GGUF --file Q4_K_M.gguf
```

## Ollama Compatibility

Do not depend on the Ollama daemon.

Support Ollama-style usage in stages:

### Stage 1: Curated Aliases

```json
{
  "llama3.2:3b": {
    "darwin-arm64": {
      "backend": "mlx",
      "repo": "mlx-community/Llama-3.2-3B-Instruct-4bit"
    },
    "linux-x64": {
      "backend": "llama",
      "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
      "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    }
  }
}
```

### Stage 2: Import Existing Ollama Models

Detect local Ollama model storage and optionally import or reference compatible GGUF layers.

### Stage 3: Ollama Registry Compatibility

Resolve Ollama manifests/layers directly without requiring the Ollama daemon.

## Background Server / Daemon Model

Clap should run as a background local service when needed.

The desired behavior:

```bash
clap chat qwen2.5:3b
```

If no server is running:

1. start the local Clap server in the background
2. wait for `/clap/v1/health`
3. send the chat request through the generated SDK
4. keep the server alive after the command exits

If a server is already running:

1. reuse it
2. do not start a duplicate process
3. send the request to the existing server

Recommended commands:

```bash
clap server start
clap server stop
clap server status
clap server restart
clap server logs
```

Also provide convenience commands:

```bash
clap serve              # foreground server for debugging
clap chat               # auto-starts background server if needed
clap run <model>        # auto-starts background server if needed
```

### macOS Daemon Strategy

Use `launchd` for a true background service.

Potential behavior:

```bash
clap server install
clap server start
```

This can install a per-user LaunchAgent:

```txt
~/Library/LaunchAgents/dev.clap.server.plist
```

The LaunchAgent should run the Clap server without requiring an open terminal.

Logs should go to:

```txt
~/Library/Logs/clap/server.log
~/Library/Logs/clap/server.err.log
```

State/cache should go to:

```txt
~/Library/Application Support/clap
```

### Linux Daemon Strategy

Use systemd user services when available.

Potential install path:

```txt
~/.config/systemd/user/clap.service
```

Commands can wrap:

```bash
systemctl --user start clap
systemctl --user stop clap
systemctl --user status clap
journalctl --user -u clap
```

Fallback for systems without systemd:

- detached background process
- pidfile
- log file
- health check endpoint

### Server Discovery

Default base URL:

```txt
http://localhost:11435
```

Store server metadata in a state file:

```txt
~/.clap/server.json
```

Example:

```json
{
  "pid": 12345,
  "port": 11435,
  "baseURL": "http://localhost:11435",
  "startedAt": "2026-06-27T00:00:00.000Z"
}
```

The CLI should verify the server by calling `/clap/v1/health`, not by trusting the pidfile alone.

## Packaging

The user-facing promise is:

```txt
one install
one clap command
bundled runtimes
no external dependency setup
```

Implementation can be a bundle:

### macOS Tarball / App Bundle

```txt
Clap.app/
  Contents/MacOS/clap
  Contents/Resources/runtimes/clap-llama
  Contents/Resources/runtimes/clap-mlx
  Contents/Resources/runtimes/mlx.metallib
```

Or CLI tarball:

```txt
clap/
  bin/clap
  libexec/clap-server
  libexec/runtimes/clap-llama
  libexec/runtimes/clap-mlx
```

### Linux Tarball

```txt
clap/
  bin/clap
  libexec/clap-server
  libexec/runtimes/clap-llama
```

Bun single-file executable is useful for the server/CLI control plane, but native inference runtimes should be packaged beside it as bundled resources.

## Milestones

### Milestone 1: API Skeleton

- Bun/Hono server
- Zod schemas
- OpenAPI generation
- Hey SDK generation
- CLI uses generated SDK
- `/clap/v1/health`
- `/v1/chat/completions` schema and model validation

### Milestone 2: Background Server

- foreground `clap serve`
- background `clap server start`
- `clap server stop/status/logs`
- auto-start behavior from `clap chat` and `clap run`
- macOS LaunchAgent support
- Linux systemd user service support

### Milestone 3: GGUF Backend

- bundled llama.cpp backend
- local `.gguf` loading
- streaming tokens through `/v1/chat/completions`
- macOS Metal build
- Linux CPU build

### Milestone 4: MLX Backend

- bundled Swift MLX backend
- macOS arm64 only
- local MLX model directory loading
- streaming tokens through same server API

### Milestone 5: Model Downloads

- Hugging Face downloader/cache
- GGUF repo support
- MLX repo support
- progress tracking through management API

### Milestone 6: Aliases And Routing

- curated aliases like `qwen2.5:3b` and `llama3.2:3b`
- platform-aware backend selection
- explicit `--backend mlx` / `--backend gguf`
- Linux fallback suggestions

## Initial Technical Recommendation

Build Clap as:

```txt
Bun/Hono local server
Zod/OpenAPI/Hey API layer
background daemon/service lifecycle
worker-process inference backends
llama.cpp for GGUF on macOS/Linux
Swift MLX for MLX on macOS arm64
```

Avoid early dependency on:

- Ollama daemon
- user-installed Python
- Conda
- Python MLX server as the main runtime
- CLI commands as internal APIs

This architecture gives Clap the right long-term shape: one local server, multiple bundled runtimes, stable OpenAI-compatible API, and reusable client SDKs.
