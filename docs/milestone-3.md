# Milestone 3: GGUF Backend

Milestone 3 adds a runnable llama.cpp/GGUF integration boundary. The `clap-llama` worker links directly against llama.cpp APIs and speaks Clap's JSON-line resident worker protocol.

## Worker Boundary

The server routes local `.gguf` model paths to the llama backend and keeps a resident worker process alive according to lifecycle keep-alive policy. The worker protocol is newline-delimited JSON over stdio:

Request on stdin:

```json
{"type":"chat","model":"./model.gguf","messages":[{"role":"user","content":"hello"}],"stream":true}
```

Response on stdout:

```json
{"token":"hello "}
{"token":"world"}
{"done":true}
```

Worker stderr is written to `~/.clap/llama-worker.err.log` or `$CLAP_HOME/llama-worker.err.log`.

## Worker Discovery

Clap looks for the worker in this order:

1. `CLAP_LLAMA_WORKER`, for example `/path/to/clap-llama`.
2. A packaged `libexec/clap-llama` next to the installed CLI runtime.
3. Repo development paths under `libexec/`.

If no worker is found, `/clap/v1/backends` reports llama as `not_installed`, and `.gguf` chat requests return a helpful backend error.

Build the direct llama.cpp worker with:

```bash
bun run runtime:llama:vendor
bun run runtime:llama:build
```

## Usage

```bash
clap run ./model.gguf "hello"
clap chat --model ./model.gguf "hello"
```

The model path must exist and end in `.gguf`. Downloads, aliases, and MLX routing are intentionally left for later milestones.
