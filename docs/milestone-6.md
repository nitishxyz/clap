# Milestone 6: Aliases And Routing

Milestone 6 adds a pragmatic alias resolver and platform-aware backend selection foundation.

## Curated Aliases

The initial alias table includes:

- `qwen2.5:3b`
- `llama3.2:3b`

Each alias has two explicit targets:

- `mlx`: MLX-format Hugging Face repo for Apple Silicon
- `gguf`: GGUF Hugging Face repo and file for llama.cpp

This is intentionally a curated table, not an Ollama registry importer.

## Backend Selection

Default routing prefers MLX when running on macOS arm64, otherwise GGUF.

Explicit overrides are supported in the CLI and API:

```bash
clap pull llama3.2:3b --backend gguf
clap run llama3.2:3b --backend gguf "hello"
clap chat --model qwen2.5:3b --backend mlx "hello"
```

API requests accept `backend: "gguf" | "mlx"` on pull and chat completion requests.

## Cache Integration

Alias pulls resolve to their selected Hugging Face target and use the Milestone 5 cache layout:

```txt
$CLAP_HOME/models/huggingface/<owner>--<repo>/
```

Alias chat resolves to the cached GGUF file or MLX directory. If the target is not cached, the server returns a clear error such as:

```txt
Model qwen2.5:3b is not cached. Run: clap pull qwen2.5:3b --file Qwen2.5-3B-Instruct-Q4_K_M.gguf --backend gguf
```

Unsupported MLX requests report that MLX requires macOS arm64 and suggest `--backend gguf`.

Unknown model names are not treated as local placeholders. Chat requires an explicit cached alias, local `.gguf` file, or MLX directory; otherwise the API returns a model-not-found/not-cached error. The CLI `chat` command requires `--model` or `CLAP_DEFAULT_MODEL`.

## Ollama Compatibility Foundation

Alias IDs intentionally use Ollama-style names (`name:size`). The resolver data structure stores alias ID, display name, backend targets, Hugging Face repos, and GGUF file names so an Ollama manifest/import layer can map into the same shape later.

Out of scope for this milestone:

- Ollama daemon integration
- Ollama manifest/layer import
- automatic quantization ranking beyond curated file names
