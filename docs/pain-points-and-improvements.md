# Clap: Pain Points And Improvement Plan

Audit snapshot: typecheck passes, all 91 tests pass. The problems are not build health;
they are end-to-end UX gaps and architectural shortcuts that conflict with the product
goal ("single tool, run local inference, downloads models, great UX").

## P0 — Blocks the core product promise

### 1. Streaming is fake end-to-end

- `packages/runtime-router/src/resident.ts:59` — `chat()` resolves only after the full
  generation is done; tokens are buffered into one string.
- `packages/server/src/index.ts` `streamResidentResponse` awaits the whole completion,
  then emits one big SSE "delta". Same for Ollama `/api/chat` (forced `stream: false`,
  re-wrapped as a single ndjson line) and `/v1/responses` (pre-baked event list).
- The native worker (`native/llama/clap-llama.cpp` `generate()`) already emits
  token-by-token JSON lines — the TS layer throws that away.
- Consequence: time-to-first-token equals total generation time; every OpenAI/Ollama
  client shows a frozen spinner. This alone makes the tool feel broken.
- Fix: make `ResidentWorkerHandle.chat` an `AsyncIterable<string>` (or accept an
  `onToken` callback), pipe worker `token` events straight into SSE/ndjson chunks,
  and run the tool/reasoning parser incrementally (or only on the final buffer while
  still forwarding raw text deltas).

### 2. Long requests are killed by the socket idle timeout

- `packages/server/src/index.ts` clamps idle timeout to Bun's 255s max. A non-streaming
  completion on a 7B+ model can exceed that; the connection dies mid-generation.
- Real streaming (P0.1) fixes this for free since bytes flow continuously; until then,
  send SSE keep-alive comments or heartbeat writes.

### 3. `clap run <model>` does not download the model

- `resolveAvailableModel` (`packages/server/src/index.ts`) returns a 404 with
  "Run: clap pull ..." text. The plan doc promises `clap run qwen2.5:3b` just works.
- Fix: in `run`/`chat` CLI paths, when the server answers `not_downloaded`/`not_found`
  for a known alias or HF repo, auto-invoke the pull flow (with progress bar), then
  retry the chat. Prompt for confirmation only on large downloads or with `--yes`.

### 4. `clap chat` / `clap run` are single-shot, no interactive session

- `apps/cli/src/index.ts` `chat()`/`run()` send one prompt and exit; `run` even invents
  a default prompt "Hello from clap run". There is no REPL, no multi-turn history.
- Fix: `clap run <model>` with no prompt should open an interactive chat loop
  (readline, history array, `/bye`, Ctrl-D), streaming tokens as they arrive.

### 5. No installable artifact — everything is `bun run cli ...`

- No compiled binary, no npm package, no `bun build --compile` script, no CI/release
  pipeline (`.github/` absent). Users must clone the repo and have Bun installed,
  which contradicts "single-install".
- `libexec/` binaries (~200MB+) are checked into git instead of being release assets.
- Fix: `bun build --compile apps/cli/src/index.ts` producing a `clap` binary,
  per-platform release archives with `libexec/` runtimes, an install script
  (`curl | sh`) or Homebrew tap, and CI that builds llama/mlx workers per platform.

## P1 — Correctness and robustness

### 6. Hand-rolled JSON parsing in the native llama worker

- `native/llama/clap-llama.cpp` `extract_string`/`messages_from_request` scan raw JSON
  with substring searches. Any message content containing `"role"`, `"content"`,
  escaped quotes, or nested JSON (tool results are serialized as JSON!) corrupts the
  parsed prompt silently.
- Fix: vendor a small JSON parser (nlohmann/json single header or llama.cpp's own
  bundled nlohmann copy) and parse the request properly.

### 7. Hardcoded 4095 generation cutoff in the worker

- `native/llama/clap-llama.cpp` `generate()`: `if (n_pos >= 4095) break;` ignores
  `CLAP_LLAMA_CONTEXT`. With a larger configured context, generation still silently
  stops at position 4095. Use `llama_n_ctx(loaded.ctx) - 1`.

### 8. Context reservation is inconsistent

- Server defaults `max_tokens` to 4096 (`chat-compat.ts prepareChatRequest`), but the
  worker only reserves `min(max_tokens, 256)` output tokens in its overflow check, so
  requests can pass the check and then fail (or truncate) mid-generation.

### 9. No KV/prompt cache reuse

- `generate()` calls `llama_memory_clear` per chat, so every turn re-ingests the whole
  conversation. Multi-turn chats on long histories get slower every turn.
- Fix: keep the KV cache and only decode the suffix when the new prompt extends the
  previous one (llama.cpp common prefix reuse), like llama-server/Ollama do.

### 10. No cancellation of in-flight generation

- Client disconnects (SSE abort, Ctrl-C in the CLI) do not reach the worker; it keeps
  burning GPU until `max_tokens`. There is no `cancel` control message in the JSON-line
  protocol and the server never watches `c.req.raw.signal`.
- Fix: add a `cancel` message keyed by request id; abort on request signal.

### 11. Resident-worker protocol has correlation/concurrency hazards

- `resident.ts handleLine`: non-JSON lines and messages without `id` are attributed to
  "the first pending request" — with two concurrent chats to the same model, output can
  interleave into the wrong response. There is also no per-worker request queue and no
  per-request timeout (a hung worker leaves promises pending forever).
- Fix: strictly require ids from the worker (it already echoes them), queue requests
  per worker, and add a watchdog timeout + worker restart policy.

### 12. Token usage numbers are fabricated

- `usageFor` estimates `chars/4` for prompt and completion. Clients that budget by
  usage get wrong numbers. The worker knows the real token counts — report them in the
  `done` message (`prompt_tokens`, `completion_tokens`) and thread them through.

### 13. Sampling parameters are silently dropped

- Worker reads only `temperature`, `top_p`, `seed`, `max_tokens`. `stop` is applied
  post-hoc in TS (generation keeps going after a stop string), and `presence_penalty`/
  `frequency_penalty`/`top_k` are accepted by the schema then ignored. Apply stop
  sequences and penalties inside the worker's sampler chain.

### 14. Download manager gaps

- No resume: partial files are deleted on any failure/cancel (`packages/models/src/index.ts`
  `downloadFile`); a 20GB pull that dies at 95% restarts from zero. Use HTTP `Range`
  with the existing `.part` file.
- No integrity checks: HF provides sha256 (LFS metadata); nothing is verified.
- Sequential MLX file downloads; no parallelism.
- Download registry is in-memory only (`downloads` map in `server/src/index.ts`):
  restart the server and `clap pull` polling loses its download; the map also grows
  unboundedly.
- Pin the revision (`resolve/main` today) and record it in the cache for reproducibility.

### 15. MLX routing by string sniffing

- `packages/runtime-mlx/src/index.ts` `isMlxModel`: `lower.includes("mlx-community")`
  treats any id containing that substring as a local MLX path. Route on actual resolved
  cache layout instead of name heuristics.

## P2 — UX polish and product gaps

### 16. Model discovery is nonexistent

- Only 2 curated aliases (`qwen2.5:3b`, `llama3.2:3b`); everything else requires knowing
  exact HF repo ids. Add `clap search <query>` (HF API `?search=`), grow the alias table
  (or generate it), and show model sizes + disk requirements in `clap models`.

### 17. No way to delete models / manage disk

- No `clap rm <model>`; Ollama `/api/delete` returns 501. Caches accumulate forever.
  Add `clap rm` + `DELETE /clap/v1/models/:id`, and show per-model size / total disk
  usage in `clap models`.

### 18. No hardware-aware recommendations

- The resolver recommends Q4_K_M regardless of installed RAM/VRAM. Compare artifact
  size against `os.totalmem()` (and Metal limits) to warn "this model likely won't fit"
  before a 40GB download, and pick quants accordingly.

### 19. CLI ergonomics

- `parseFlags` silently swallows unknown flags into positional args (typos like
  `--stram` become the prompt). Error on unknown flags.
- No `clap --version`. No `clap ps` (alias for `models --active`), no `clap stop <model>`.
- Errors from the server print raw messages; `printError` suggests `clap server start`
  even when the server is running.
- `clap run <model>` with no prompt should not send "Hello from clap run".

### 20. First-run experience

- First `clap run` silently spawns a background server, downloads nothing, then errors.
  Print what is happening ("starting server...", "model not cached, downloading X GB...").
  A `clap doctor` command (checks workers, metallib, platform, disk, port conflicts)
  would fold `bundle:check` diagnostics into the user-facing tool.

### 21. Repo/dev hygiene

- 200MB+ binaries (`libexec/clap-llama`, `clap-mlx`, `llama-cli`, `mlx.metallib`)
  belong in releases or a download-on-first-run step, not git.
- `libexec/llama-cli` appears unused now (worker links llama.cpp directly) — remove.
- No linter/formatter config (biome/eslint/prettier) and no CI to run
  `typecheck`/`test` — add a minimal GitHub Actions workflow.
- `clapHome()` is duplicated in `packages/models` and `apps/cli/src/server-lifecycle.ts`.

## Suggested order of attack

1. Real token streaming through resident workers → SSE/ndjson (fixes #1, #2 mostly).
2. Auto-pull on `run`/`chat` + interactive REPL (fixes #3, #4).
3. Native worker correctness: real JSON parsing, ctx cutoff, stop/penalties, real
   usage counts, cancel message (fixes #6–#13).
4. Download resume + delete/disk management (fixes #14, #17).
5. Packaging: compiled `clap` binary, release pipeline, install script (fixes #5, #21).
6. Discovery/aliases/hardware sizing polish (fixes #16, #18–#20).
