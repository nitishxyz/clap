# Clap

Clap is a local model server prototype with an OpenAI-compatible API.

See `docs/clap-local-model-server-plan.md` for the product plan and milestone docs under `docs/` for runnable slices.

## Quick Start

```bash
bun install
bun run generate:openapi
bun run serve
```

Then, in another shell:

```bash
bun run cli server status
bun run cli models
bun run cli pull llama3.2:3b --backend gguf
bun run cli chat --model llama3.2:3b --backend gguf "hello"
bun run cli run llama3.2:3b --backend gguf "hello"
```

## Background Server

`clap serve` runs the server in the foreground for debugging. The `server` subcommands manage a detached local server and verify it with `/clap/v1/health`:

```bash
bun run cli server start
bun run cli server status
bun run cli server logs
bun run cli server restart
bun run cli server stop
```

If `/clap/v1/health` is already healthy but no live managed PID is recorded, `server start`, `server stop`, and `server restart` report that the server is unmanaged instead of writing fake pid metadata. Use `bun run cli server stop --force` or `bun run cli server restart --force` to kill the process listening on the Clap port on macOS/Linux, then start a managed server.

`clap chat` and `clap run` auto-start the background server when no healthy server is available.

State and logs default to `~/.clap`:

```txt
~/.clap/server.json
~/.clap/server.log
~/.clap/server.err.log
```

Set `CLAP_HOME` to move state/logs and `CLAP_BASE_URL` to change the server URL. Long local inference requests use a 240 second Bun socket idle timeout by default; set `CLAP_SERVER_IDLE_TIMEOUT_SECONDS` to another number of seconds (clamped to Bun's 255 second maximum), or `0` to disable the timeout if supported by your Bun version. `bun run cli server install` writes a per-user macOS LaunchAgent or Linux systemd user-service template with start/log commands.

List model IDs the server can serve with `bun run cli models` or `bun run cli models list`. Add `--aliases` to include curated aliases and their download/support status. The default output stays concise: each ID with backend, format, and status for use with `clap run`, `clap chat`, or the OpenAI-compatible API. Add `--json` to print the rich `/clap/v1/models` response with display name, provider/source, served modalities and capabilities, inferred upstream metadata, limits, architecture/model type, quantization, and local cache path.

`GET /v1/models` intentionally returns only OpenAI-compatible model list fields (`id`, `object`, `created`, `owned_by`) by default. Use `GET /clap/v1/models`, `bun run cli models --json`, or opt in on the OpenAI-compatible envelope for rich management metadata:

```bash
curl -s 'http://localhost:11435/v1/models?metadata=1' | jq
```

Served capabilities currently report text input/text output for MLX and GGUF unless Clap has implemented an additional modality in the runtime path; upstream fields are inferred separately from cached Hugging Face files such as `config.json`, `tokenizer_config.json`, `model.safetensors.index.json`, and GGUF filenames.

## Runtime Lifecycle

Clap tracks loaded/warm model entries in a runtime lifecycle registry keyed by public model id, backend, and resolved local path. Use it to inspect which models are loaded, active, pinned, or waiting for idle expiry:

```bash
bun run cli load mlx-community/gemma-4-e4b-it-4bit --backend mlx --keep-alive always
bun run cli load llama3.2:3b --backend gguf --keep-alive 15m
bun run cli models --active
bun run cli unload mlx-community/gemma-4-e4b-it-4bit --backend mlx
```

Management APIs are `POST /clap/v1/models/load`, `POST /clap/v1/models/unload`, and `GET /clap/v1/runtime/models`. Keep-alive accepts durations such as `30s`, `15m`, `1h`, `1d`, or `always`; chat requests automatically load/touch the model and extend expiry. `always` pins the entry until manual unload or server shutdown. Loaded MLX and GGUF entries point at resident JSON-line worker processes (`worker.state: "resident"`, with PID when available). The MLX worker keeps its model container loaded across chats, and the GGUF worker links directly against llama.cpp APIs so it loads GGUF weights once on `load` and reuses the same model/context for subsequent chats until unload, expiry, or shutdown. The server clears lifecycle state and shuts down resident workers on `SIGINT`/`SIGTERM`.

## OpenAI Chat Compatibility

`POST /v1/chat/completions` accepts OpenAI-style text requests plus `tools`, `tool_choice`, `parallel_tool_calls`, `response_format` (`text`, `json_object`, or `json_schema`), `stop`, `seed`, `top_p`, and penalty fields. Clap serializes tool definitions and JSON-output instructions into the local text prompt, then parses generated JSON tool-call shapes back into OpenAI-compatible `tool_calls` for non-streaming responses and streaming deltas. JSON mode/schema support is best effort: valid generated JSON is normalized, while unparsable text is returned as raw assistant content. Text content parts are accepted; `image_url` parts return a precise unsupported-content error until a served multimodal runtime path exists.

`POST /v1/responses` supports the newer OpenAI Responses API shape for local compatibility. It accepts `input` as a string or message array, `instructions`, tools, tool choice, structured text formats, streaming, and sampling fields, then maps them through the same local chat/tool/structured-output pipeline. Stateful continuation via `previous_response_id` is accepted by schema but returns an explicit unsupported error. Legacy `/v1/completions` is intentionally not implemented.

```ts
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:11435/v1", apiKey: "clap" });
const response = await client.responses.create({
  model: "mlx-community/gemma-4-e4b-it-4bit",
  instructions: "Be concise.",
  input: "Say hello in one sentence.",
});
console.log(response.output_text);
```

## Ollama Compatibility

Clap exposes Ollama-compatible routes for common local clients: `GET /api/tags`, `POST /api/show`, `POST /api/pull`, `POST /api/chat`, and `POST /api/generate`. Model names are the same public Clap ids shown by `clap models`, including aliases such as `llama3.2:3b` and cached Hugging Face ids such as `mlx-community/gemma-4-e4b-it-4bit`. `/api/chat` maps tools through Clap's OpenAI tool-call compatibility layer. `/api/delete`, `/api/copy`, `/api/embeddings`, and `/api/embed` return honest `501` responses until implemented. Ollama image inputs are rejected with a text-runtime unsupported error unless a future served multimodal runtime path is active.

```bash
curl -s http://localhost:11435/api/tags | jq
curl -s http://localhost:11435/api/show -d '{"model":"llama3.2:3b"}' | jq
curl -s http://localhost:11435/api/chat -d '{"model":"llama3.2:3b","stream":false,"messages":[{"role":"user","content":"hello"}]}' | jq
```

## Hugging Face Cache

Explicit Hugging Face repos can be pulled into the local cache under `$CLAP_HOME/models/huggingface` (default `~/.clap/models/huggingface`):

```bash
bun run cli pull owner/gguf-repo --file model.Q4_K_M.gguf
bun run cli pull mlx-community/TinyLlama-4bit
```

`POST /clap/v1/models/pull` starts a background download for either a selected `.gguf` file or a detected MLX repo layout and returns a download id immediately. Pulls are idempotent by resolved target: if the same alias/repo/file/backend/cache path is already downloading, a second pull reuses the existing download id instead of starting another network transfer. If the target is already fully cached, pulls return a completed download immediately; pass `--force` or `{ "force": true }` to re-download.

`GET /clap/v1/downloads` reports incremental pull status (`running`, `completed`, `cancelled`, or `failed`), `bytesReceived`, `totalBytes` when known, and the current file being downloaded. `POST /clap/v1/downloads/:id/cancel` cancels a running pull and removes partial files while keeping completed cache files intact. `bun run cli pull ...` polls download progress, renders a progress bar when the total is known, and cancels the active download on Ctrl-C before exiting non-zero. `GET /clap/v1/models` includes cached GGUF files and MLX directories. After a pull, `clap run owner/repo --backend mlx` and `clap chat --model owner/repo --backend mlx` resolve to the cached Hugging Face directory; GGUF repo ids resolve when the cached repo contains a single GGUF file. Set `CLAP_HF_ENDPOINT` for tests or mirrors.

Private or gated Hugging Face repos are supported with a token. Clap checks these variables in order and sends `Authorization: Bearer <token>` on Hugging Face API and file-download requests:

```bash
CLAP_HF_TOKEN=hf_xxx bun run cli pull owner/private-gguf --file model.Q4_K_M.gguf
```

Supported token variables, in priority order: `CLAP_HF_TOKEN`, `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`. If none are set, Clap reads a stored credential so the CLI and background server can reuse one login.

```bash
bun run cli auth login
bun run cli auth status
bun run cli auth logout
```

`auth status` only prints a redacted token preview. On macOS, `auth login` stores the token in Keychain via `security`; on Linux, it uses libsecret via `secret-tool`; if no OS keychain is available, it falls back to `$CLAP_HOME/auth/huggingface-token` with directory mode `0700` and file mode `0600`. Set `CLAP_HF_AUTH_BACKEND=file` to force file-backed storage for tests or headless systems.

When `clap pull` receives a Hugging Face 401/403 and stdin is a TTY, it prompts once for a token without echoing, saves it, and retries the pull one time. Non-interactive pulls fail with guidance to run `clap auth login` or set one of the token env vars.

## Aliases And Routing

Curated Ollama-style aliases currently include `qwen2.5:3b` and `llama3.2:3b`. On macOS arm64 Clap prefers MLX targets; on Linux or with `--backend gguf`, Clap uses GGUF targets:

```bash
bun run cli pull llama3.2:3b --backend gguf
bun run cli run llama3.2:3b --backend gguf "hello"
bun run cli chat --model qwen2.5:3b --backend mlx "hello"
```

If an alias is not cached, chat requests return a pull instruction. Unknown models return model-not-found/not-cached errors; Clap no longer serves placeholder local completions. `clap chat` requires `--model` or `CLAP_DEFAULT_MODEL`. Unsupported MLX requests include a GGUF fallback suggestion.

## Bundled Native Runtimes

Packaged builds should include real native workers at:

```txt
libexec/clap-llama
libexec/clap-mlx
libexec/mlx.metallib
```

Runtime discovery checks explicit environment variables first (`CLAP_LLAMA_WORKER`, `CLAP_MLX_WORKER`), then bundled paths relative to the installed CLI/server executable, then repo development paths under `libexec/`. `clap server status` reports backend reasons as `configured`, `bundled`, or `missing`.

Build the real direct llama.cpp worker before running GGUF models:

```bash
bun run runtime:llama:vendor
bun run runtime:llama:build
```

`runtime:llama:build` configures llama.cpp with CMake, enables Metal on macOS, builds the llama.cpp static libraries, links `native/llama/clap-llama.cpp` directly against llama.cpp APIs, and installs `libexec/clap-llama`. The worker loads GGUF weights once per resident `load` command and reuses them for subsequent `chat` commands; it does not shell out to `llama-cli`.

Before packaging, copy or build real worker binaries and verify them:

```bash
CLAP_LLAMA_WORKER=/path/to/clap-llama \
CLAP_MLX_WORKER=/path/to/clap-mlx \
bun run bundle:prepare
bun run bundle:check
```

`bundle:check` fails if `libexec/clap-llama` or `libexec/clap-mlx` is absent, empty, or not executable. It also requires `libexec/mlx.metallib`, the MLX Metal shader library loaded by the Swift MLX worker.

## GGUF / llama.cpp Worker

Local `.gguf` model paths are routed to the llama backend:

```bash
bun run cli run ./model.gguf "hello"
```

The native worker is discovered from `CLAP_LLAMA_WORKER`, bundled `libexec/clap-llama`, or repo `libexec/clap-llama`, in that order:

```bash
CLAP_LLAMA_WORKER="/path/to/clap-llama" bun run cli run ./model.gguf "hello"
```

`clap-llama` is a direct llama.cpp resident worker. It reads JSON-line `load`, `chat`, `unload`, and `shutdown` commands, keeps the loaded GGUF model/context in process, and emits JSON-line `loaded`, `token`, `done`, or `error` events. Missing workers are reported as backend errors.

Real GGUF residency smoke test:

```bash
bun run cli load ./model.gguf --backend gguf --keep-alive always
bun run cli run ./model.gguf "one short sentence"
bun run cli run ./model.gguf "another short sentence"
bun run cli models --active
bun run cli unload ./model.gguf --backend gguf
```

The active model PID should remain the same across both runs until unload.

## MLX / Swift MLX Worker

Local MLX-format model directories are routed to the MLX backend when they look like Hugging Face/MLX directories, for example containing `config.json` plus tokenizer or `.safetensors` files:

```bash
bun run cli run ./mlx-model "hello"
bun run cli chat --model ./mlx-model "hello"
```

The native worker is discovered from `CLAP_MLX_WORKER`, bundled `libexec/clap-mlx`, or repo `libexec/clap-mlx`, in that order. The real MLX backend is gated to macOS arm64. Missing or unsupported workers are reported as backend errors.

Build the real Swift MLX worker on macOS arm64 with:

```bash
bun run runtime:mlx:build
bun run bundle:check
```

`runtime:mlx:build` builds `native/mlx`, installs `libexec/clap-mlx`, compiles the generated MLX Metal kernels from the `mlx-swift` checkout with `xcrun metal`, and installs `libexec/mlx.metallib` next to the worker. Keep those two files together when packaging or when setting `CLAP_MLX_WORKER`.

The Swift package in `native/mlx` uses `mlx-swift-lm` with Hugging Face tokenizer integration to load local MLX model directories and emit Clap JSON-line token/content messages. Cached Hugging Face MLX repos can be run by repo id after pulling/downloading:

```bash
bun run cli pull mlx-community/gemma-4-e4b-it-4bit --backend mlx
bun run cli server restart
bun run cli run mlx-community/gemma-4-e4b-it-4bit --backend mlx "hello"
```
