# Milestone 4: MLX Backend

Milestone 4 adds a runnable Swift MLX worker-process boundary. The `clap-mlx` executable is built from `native/mlx` and runs real MLX Swift LM inference on macOS arm64.

## Worker Boundary

The server detects local MLX model directories and routes them to the MLX backend. Requests spawn a worker process and use the same newline-delimited JSON stdio protocol as the llama worker.

Request on stdin:

```json
{"type":"chat","model":"./mlx-model","messages":[{"role":"user","content":"hello"}],"stream":true}
```

Response on stdout:

```json
{"token":"hello "}
{"token":"world"}
{"done":true}
```

Workers may also emit `{"content":"..."}`. Worker stderr is written to `~/.clap/mlx-worker.err.log` or `$CLAP_HOME/mlx-worker.err.log`.

## Model Detection

Local directories route to MLX when they contain `config.json` and at least one tokenizer file or `.safetensors` file. `CLAP_MLX_MODEL_PATHS` can also list known local MLX model directories for `/clap/v1/models` metadata.

Downloads, aliases, and remote Hugging Face repo resolution are not implemented in this milestone.

## Worker Discovery

Clap looks for the worker in this order:

1. `CLAP_MLX_WORKER`, for example `/path/to/clap-mlx`.
2. A packaged `libexec/clap-mlx` next to the installed CLI runtime.
3. Repo development paths under `libexec/`.

The real worker is supported only on macOS arm64. Other platforms report `unsupported`.

Build the worker with:

```bash
bun run runtime:mlx:build
```

## Usage

```bash
clap run ./mlx-model "hello"
clap chat --model ./mlx-model "hello"
```

Without a native Swift MLX worker, chat requests fail with a backend error.
