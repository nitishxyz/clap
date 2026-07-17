# Milestone 5: Model Downloads

Milestone 5 adds a pragmatic Hugging Face downloader/cache layer for explicit repos. It does not implement curated aliases or automatic model selection.

## Cache Layout

Downloaded files live under:

```txt
$CLAP_HOME/models/huggingface/<owner>--<repo>/
```

If `CLAP_HOME` is unset, Clap uses `~/.clap`.

Examples:

```txt
~/.clap/models/huggingface/bartowski--TinyLlama-GGUF/model.Q4_K_M.gguf
~/.clap/models/huggingface/mlx-community--TinyLlama-4bit/config.json
```

## Pull API

```http
POST /clap/v1/models/pull
Content-Type: application/json

{"model":"owner/repo","file":"model.Q4_K_M.gguf"}
```

- `model` must be an explicit Hugging Face repo in `owner/repo` form.
- `file` is required when a GGUF repo has multiple `.gguf` files.
- Without `file`, a repo with one `.gguf` file is pulled as GGUF.
- Without GGUF files, Clap attempts an MLX repo pull by downloading `config.json`, tokenizer metadata, and `.safetensors` files.

Status is available at:

```http
GET /clap/v1/downloads
```

The implementation records `running`, `completed`, and `failed` statuses. The current pull call completes synchronously and returns the final status, which is sufficient for tests and small dev fixtures.

## Model Inventory

`GET /clap/v1/models` includes cached models:

- cached `.gguf` files route to the llama backend
- cached MLX directories route to the MLX backend

Use the returned `id` as the model path for chat/run commands.

## CLI

```bash
clap pull owner/repo --file model.Q4_K_M.gguf
clap pull mlx-community/Some-MLX-Repo
```

For mirrors or local test fixtures:

```bash
CLAP_HF_ENDPOINT=http://localhost:3000 clap pull owner/repo --file model.gguf
```

For private or gated Hugging Face repos, set a token or store one with the CLI:

```bash
clap auth login
clap auth status
clap auth logout
```

Token lookup priority is explicit env first (`CLAP_HF_TOKEN`, `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`), then stored credential. Stored credentials use macOS Keychain through `security`, Linux libsecret through `secret-tool`, or a file fallback at `$CLAP_HOME/auth/huggingface-token` with directory `0700` and file `0600`; `CLAP_HF_AUTH_BACKEND=file` forces the file backend. Status output redacts tokens.

Hugging Face 401/403 responses are reported as explicit auth errors. For interactive `clap pull`, the CLI prompts once for a token, saves it, and retries the pull; non-interactive pulls print guidance to run `clap auth login` or set an env token.

## Out Of Scope

- curated aliases like `qwen2.5:3b`
- choosing among many GGUF quantizations without `--file`
- resumable/ranged downloads
- authentication/private repos
