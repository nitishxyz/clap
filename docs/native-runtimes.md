# Native Runtime Workers

Clap ships resident native workers in `libexec/`. Missing or invalid workers are explicit installation/platform failures; production does not substitute mocks, one-shot subprocesses, or an unversioned protocol.

The complete ownership and operator model is documented in [Final Inference Runtime Architecture](inference-runtime-baseline.md).

## Runtime discovery

The TypeScript control plane checks `CLAP_LLAMA_WORKER` or `CLAP_MLX_WORKER` first, then bundled candidates around the executable and repository `libexec` directory. A configured value may contain a command plus arguments. The executable must exist and be nonempty.

- GGUF discovery failure reports `not_installed` with bundle diagnostics.
- MLX requires macOS arm64. Other platforms report `unsupported`.
- MLX discovery also requires `mlx.metallib` or `Resources/mlx.metallib` next to the worker.

Inspect discovery without launching a model at `GET /clap/v1/backends`. Validate packaged artifacts with:

```sh
bun run bundle:check
```

## Strict protocol v1

Both workers use newline-delimited JSON over stdin/stdout. Every envelope contains `protocol: 1`; requests have nonempty `request_id` and explicit request types. A worker must emit one `ready` event before any other event.

Example load request:

```json
{"protocol":1,"type":"load","request_id":"load_1","model":"/absolute/path/to/model"}
```

The corresponding event progression is scoped and sequenced:

```json
{"protocol":1,"type":"accepted","request_id":"load_1","sequence":0}
{"protocol":1,"type":"started","request_id":"load_1","sequence":1}
{"protocol":1,"type":"completed","request_id":"load_1","sequence":2,"result":{"kind":"loaded","effective_model_capabilities":{},"token_capabilities":{}}}
```

The abbreviated capability objects above illustrate event order only; real `loaded` results must satisfy the strict capability schemas in `packages/worker-protocol/src/schemas.ts`.

Generation requests carry a server-derived `cache_identity` and optional typed `structured_output`; callers do not write worker commands directly. Request types are `load`, `generate`, `cancel`, `set_max_active`, `unload`, and `shutdown`. Event types are `ready`, `accepted`, `started`, `token`, `content`, `prefill_progress`, `completed`, `failed`, `telemetry`, and `diagnostic`.

Stdout is protocol-only. Native diagnostics go to stderr and are captured per launch. Unknown events, malformed envelopes, version mismatch, sequence/state violations, or non-JSON stdout are protocol faults; there is no legacy parser fallback.

## llama.cpp GGUF worker

Build and package:

```sh
bun run runtime:llama:vendor
bun run runtime:llama:build
bun run bundle:check
```

`runtime:llama:vendor` provisions the pinned llama.cpp source. `runtime:llama:build` configures the Clap-owned CMake project, builds the Rust cache FFI and focused C++ modules, links the resident worker, and installs `libexec/clap-llama`.

Production code is split across protocol, worker/application state, model runtime, request preparation, prompt/sampling/stop handling, bounded generation stepping, scheduling, structured output, cache identity/execution, environment, and telemetry modules under `native/llama/src`. Physical KV mutation is confined to the cache executor; logical reuse policy remains in Rust.

Useful resource controls include `CLAP_LLAMA_CONTEXT`, `CLAP_LLAMA_MAX_SESSION_CTX`, `CLAP_LLAMA_MAX_OUTPUT`, `CLAP_LLAMA_BATCH`, `CLAP_LLAMA_UBATCH`, `CLAP_LLAMA_GPU_LAYERS`, `CLAP_LLAMA_KV_TYPE`, and `CLAP_LLAMA_RETAINED_MAX`.

## Swift MLX worker

Build and package on macOS arm64:

```sh
bun run runtime:mlx:build
bun run bundle:check
```

The Swift package builds the resident `clap-mlx` executable and packages its Metal library. Production code is split into application, protocol, scheduling, model, configuration, telemetry, tool, support, generation-adapter, and cache-adapter modules under `native/mlx/Sources/clap-mlx`, with reusable worker/cache policy modules in sibling Swift targets.

Physical MLX cache copy/trim/clear is confined to the cache adapter. Scheduling and bounded generation state are protocol-independent. Logical cache planning and generations remain Rust-owned.

Useful controls include `CLAP_MLX_CONTEXT`, `CLAP_MLX_MAX_SESSION_CTX`, `CLAP_MLX_MAX_OUTPUT`, `CLAP_MLX_KV_TYPE`, and retained-cache initial/max/budget/watermark/growth settings documented in the final architecture guide.

## Process lifecycle and logs

Only `ResidentWorkerProcess` spawns native inference workers. A process loads one physical model identity and remains resident across requests. Load is single-flight; unload and shutdown are explicit protocol requests. The control plane waits for graceful shutdown and then uses `CLAP_WORKER_SHUTDOWN_TIMEOUT_MS` as the forced-stop bound.

Every process start has independent artifacts:

```text
$CLAP_HOME/logs/workers/<backend>/<model-identity-hash>/<launch-id>.stderr.log
$CLAP_HOME/logs/workers/<backend>/<model-identity-hash>/<launch-id>.json
```

The sidecar records launch identity, command, PID, protocol, phase, timestamps, exit status, and crash classification. Retention is controlled with `CLAP_WORKER_LOG_MAX_LAUNCHES_PER_MODEL` and `CLAP_WORKER_LOG_MAX_BYTES_PER_BACKEND`.

## Verification

Run backend-specific tests independently:

```sh
bun run runtime:cache:test
bun run runtime:llama:test
bun run runtime:mlx:test
```

Run all production builds and native gates:

```sh
bun run native:build
bun run native:test
bun run bundle:check
```

For physical cache probes with provisioned pinned models:

```sh
bun run native:cache:tier-b
bun run native:cache:tier-c
```

The MLX build/test commands require macOS arm64. Rust and supported GGUF checks remain independently runnable on other hosts.
