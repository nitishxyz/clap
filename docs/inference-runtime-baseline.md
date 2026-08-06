# Final Inference Runtime Architecture

This document supersedes the Phase 0 baseline. It describes the production architecture after the inference-runtime hardening program. The old unversioned worker protocol, backend-owned process loops, permissive memory provenance, and monolithic native entrypoints are not supported compatibility surfaces.

For build and runtime discovery details, see [Native Runtime Workers](native-runtimes.md). Performance work intentionally deferred beyond hardening is tracked in [Inference Runtime Benchmark Follow-up](inference-runtime-benchmark-follow-up.md).

## System ownership

| Layer | Owner | Responsibilities | Explicitly does not own |
| --- | --- | --- | --- |
| HTTP/control plane | `packages/server`, `packages/runtime-router` | API validation, authentication, cache-identity derivation and rotation, model resolution, global load admission, resident lifecycle, worker spawning, strict protocol decoding, parser selection, structured-output policy, telemetry projection | Native token generation, physical cache mutation, cache reuse policy |
| Shared worker contract | `packages/worker-protocol` | Protocol-v1 TypeScript types, Zod schemas, validation, request/event discriminators, capability and memory contracts | Process lifecycle or backend behavior inference |
| Cache policy | `native/cache` | Backend-neutral Rust coordinator, authenticated namespace decisions, donor/target selection, generations, leases, eviction, plan/commit/abort, C ABI | llama.cpp or MLX object mutation |
| GGUF adapter | `native/llama` | llama.cpp model/runtime integration, prompt/token handling, bounded generation steps, scheduler, structured constraints, physical KV operations, telemetry, protocol-v1 transport | Cache-policy decisions or TypeScript process ownership |
| MLX adapter | `native/mlx` | MLX model/runtime integration, bounded generation state, scheduling, physical cache operations, telemetry, protocol-v1 transport | Cache-policy decisions or TypeScript process ownership |

The native entrypoints are intentionally thin: `native/llama/src/main.cpp` and `native/mlx/Sources/clap-mlx/main.swift` only assemble their applications. Production logic lives in focused modules. Ownership is enforced by `native:structure:check`, `native:ownership:check`, and `runtime:process:ownership`.

## TypeScript control plane and process ownership

`ResidentWorkerRegistry` performs global model admission and owns residency across both backends. Admission is serialized, reserves estimated bytes before launch, protects active/pinned/loading/target/reserved residents, evicts only eligible idle residents, and replans stale snapshots. It fails closed when available memory is unavailable or only estimated unless explicit headroom makes admission safe.

`packages/runtime-router/src/process/resident-worker-process.ts` is the sole native inference spawn and request-correlation owner. It:

1. resolves a configured or bundled worker command;
2. creates unique per-launch paths and metadata;
3. spawns one resident worker for one physical model identity;
4. negotiates protocol v1 through `ready`;
5. registers request IDs before writing commands;
6. validates sequence and state transitions;
7. maps typed worker results and telemetry;
8. unloads and shuts down gracefully, then applies a bounded forced-stop timeout;
9. finalizes launch metadata exactly once even when protocol and exit paths race.

Backend packages discover binaries and validate model formats. They do not spawn inference processes or parse worker stdout.

## Strict worker protocol v1

Workers communicate as newline-delimited JSON, but every line is a strict protocol-v1 envelope. There is no legacy shape fallback.

### Requests

Every request has `protocol: 1`, a non-empty `request_id`, and one of these explicit `type` values:

- `load`
- `generate`
- `cancel`
- `set_max_active`
- `unload`
- `shutdown`

`generate` requires the server-derived `cache_identity`; callers cannot supply authoritative identity material directly. Structured-output contracts are typed as `json_object` or `json_schema` with `best_effort` or `required` strength.

### Events

The worker first emits exactly one unscoped `ready` event containing strict worker capabilities. Request-scoped events carry a nonnegative, contiguous `sequence` and are one of `accepted`, `started`, `token`, `content`, `prefill_progress`, `completed`, or `failed`. Unscoped `telemetry` and `diagnostic` events cannot carry request scope. Completed results have explicit kinds: `loaded`, `generated`, `cancelled`, `max_active_updated`, `unloaded`, or `shutdown`.

The decoder rejects oversized lines, malformed JSON, version mismatches, missing/duplicate readiness, unknown event types, malformed envelopes, scope violations, unknown request IDs, sequence gaps/regressions, and invalid state transitions. Fault codes are defined in `packages/runtime-router/src/protocol/errors.ts`; request-scoped faults reject that request, while worker-scoped faults terminate the unsafe launch.

## Cache policy and physical adapters

Rust owns logical cache correctness. The C++ and Swift adapters execute a validated plan against backend-native state, then commit or abort it. Physical mutation cannot occur outside the designated cache adapters, and generation primitives cannot occur outside the generation owners.

A reuse transaction follows this order:

1. derive authenticated cache and physical model identities;
2. ask Rust to plan against current generations, leases, capabilities, and boundaries;
3. validate donor/target and physical capability preconditions;
4. copy, trim, clear, or continue native state through the backend adapter;
5. confirm realized state and commit the logical plan;
6. invalidate/clear and abort on any failed physical or logical transition;
7. report planned versus realized reuse and classified candidate rejection reasons.

This ordering prevents a logical hit from being published before physical materialization succeeds.

## Authenticated cache identity and rotation

The server derives an opaque v1 identity from the authenticated principal, installation secret generation, bounded display labels, project/harness/agent/session scope, and physical model identity. Secret-bearing labels are HMAC-derived before crossing the worker boundary. The wire contract contains fixed-width fingerprints and a numeric namespace ID; raw API keys and installation secrets are never sent to native workers or persisted in cache-decision records.

The isolation namespace is derived from exactly three inputs: the authenticated tenant root, the caller's explicit `cache.namespace` label, and the physical model domain. Scope labels (`project`/`harness`/`agent`/`session`) select the reuse scope and rank donors but intentionally do not partition the namespace — sessions of one authenticated caller on one model share a namespace so the coordinator's cross-session branch/anchor reuse is reachable. Verified end-to-end: distinct sessions sharing a declared harness prefix reuse ~3,100 tokens with branch TTFT within ~1.1x of exact continuation.

Remote requests carrying cache intent require a valid API key. Loopback/embedded behavior follows server authentication policy. Rotate identity material with:

```sh
curl -X POST http://127.0.0.1:11435/clap/v1/cache/identity/rotate \
  -H 'Authorization: Bearer <key>'
```

Rotation takes the installation-secret writer lock, blocks new identity derivation, lets already-dispatched old-generation requests reach terminal state, drains all residents, atomically installs a new generation, and reports `previousGeneration`, `newGeneration`, `rotatedAt`, and `clearedResidents`. Old-generation cache entries cannot match the new generation.

## Capabilities, scheduling, and priority

Behavior-sensitive choices use effective capabilities reported after model load, not backend-name sniffing. The capability surface includes:

- cache: suffix trim, prefix branch, whole-state copy, prompt-boundary snapshots, quantized KV;
- generation: structured-output modes and tool-template support;
- modalities: accepted input and output modalities;
- token limits: model/effective context, input/output bounds, allocation cap, and configured override.

Backend labels remain valid only for mechanics such as binary selection, model-format validation, physical identity, and backend-specific configuration/error guidance.

The only request/cache priority vocabulary is `interactive`, `normal`, and `background`. Both native schedulers preserve FIFO within a class, advance every runnable class before weighted extras, and do not destructively preempt active work. Interactive work receives greater weight and a larger prefill quantum; starvation tests ensure sustained interactive load cannot block normal or background requests indefinitely.

## Parsers and structured output

Assistant-output parser selection is trait-driven in `packages/server/src/parsers`:

1. an exact user parser selection wins;
2. discovered template family traits select user or built-in parsers;
3. tool, response-format, tool-template, or reasoning traits select the generic structured parser;
4. otherwise the plain parser is used.

A model-name substring is not evidence of an output protocol. Parsing is centralized; native backends do not duplicate assistant-output parsers.

For each requested structured contract, the loaded model advertises `native`, `post_validate`, or `unsupported`. `native` sends the typed constraint to the worker. `post_validate` validates the generated result in the control plane. `required` constraints fail when the effective capability cannot satisfy them; `best_effort` may use the supported fallback mode. Outcomes are classified for telemetry without storing caller schemas.

## Honest memory and global admission

Memory values carry all three parts: bytes, source, and basis.

- `measured` requires a positive observation and a measured basis such as resident RSS or a runtime/worker allocator.
- `estimated` may be zero and identifies its estimate basis, such as artifacts, architecture metadata, configured cache, cache components, or context configuration.
- `unavailable` requires `bytes: null` and a reason basis.

A missing measurement is never encoded as measured zero. Prometheus omits unavailable values and numeric values without provenance instead of inventing labels. Estimated retained bytes are separate from observed retained bytes.

Global admission combines canonical model identity, artifact/configuration estimates, measured OS availability, configured OS/runtime headroom, existing reservations, and resident observations. Critical pressure can evict one eligible idle resident only after stale-state revalidation; active, pinned, loading, always-resident, target, and reserved models are protected.

## Operator configuration

Configuration precedence is defaults, `/etc/clap/clap.toml`, `$CLAP_HOME/clap.toml`, then explicit environment. `GET /clap/v1/config` shows effective config and loaded sources; `PATCH /clap/v1/config` writes the user file. Authentication and per-model worker settings can apply live or on the next worker start; server listener and limit changes require restart.

Important controls:

| Area | Configuration |
| --- | --- |
| Runtime discovery | `CLAP_LLAMA_WORKER`, `CLAP_MLX_WORKER`, `CLAP_HOME` |
| Lifecycle/admission | `CLAP_KEEP_ALIVE`, `CLAP_MAX_ACTIVE`, `CLAP_MODEL_OS_HEADROOM_BYTES`, `CLAP_MODEL_RUNTIME_HEADROOM_BYTES`, `CLAP_UNKNOWN_MODEL_MEMORY_MIN_BYTES` |
| Request queue | `[limits] max_inflight`, `queue_depth`, `max_active` (environment equivalents are consumed by the server) |
| GGUF | `[llama]` or `CLAP_LLAMA_CONTEXT`, `CLAP_LLAMA_KV_AUTOFIT`, `CLAP_LLAMA_MAX_SESSION_CTX`, `CLAP_LLAMA_MAX_OUTPUT`, `CLAP_LLAMA_BATCH`, `CLAP_LLAMA_UBATCH`, `CLAP_LLAMA_PREFILL_BUDGET`, `CLAP_LLAMA_GPU_LAYERS`, `CLAP_LLAMA_KV_TYPE`, `CLAP_LLAMA_RETAINED_MAX` |
| MLX | `[mlx]` or `CLAP_MLX_CONTEXT`, `CLAP_MLX_MAX_SESSION_CTX`, `CLAP_MLX_MAX_OUTPUT`, `CLAP_MLX_KV_TYPE`, `CLAP_MLX_PREFILL_QUANTUM`, retained initial/max/budget/watermark/growth controls |
| Cache fairness | `[cache] max_anchors_per_session` or `CLAP_CACHE_MAX_ANCHORS_PER_SESSION` (default 4; includes automatic checkpoints) |
| Checkpoints | `[cache.checkpoints]` or `CLAP_CACHE_CHECKPOINTS_ENABLED`, minimum/interval/max, `max_checkpoints_per_session` (default 2), and budget controls |
| Launch logs | `CLAP_WORKER_LOG_MAX_LAUNCHES_PER_MODEL`, `CLAP_WORKER_LOG_MAX_BYTES_PER_BACKEND`, `CLAP_WORKER_SHUTDOWN_TIMEOUT_MS` |
| Auth/telemetry | `[auth] require_api_key`, `CLAP_REQUIRE_API_KEY`, `[telemetry] cache_decisions_enabled`, maximum MiB and age |

Per-model `[models."owner/name"]` sections can set lifecycle and worker values applied to that model on its next start.

## Operator diagnostics

Use these paths before inspecting process state manually:

| Path | Purpose |
| --- | --- |
| `GET /clap/v1/health` | Liveness, version, uptime |
| `GET /clap/v1/backends` | Worker discovery and platform/install reasons |
| `GET /clap/v1/runtime` | Runtime and configuration summary |
| `GET /clap/v1/runtime/models` | Loaded state, memory provenance, retention, limits, effective capabilities, launch links |
| `GET /clap/v1/dashboard` and `/clap/v1/dashboard/stream` | Current operational snapshot and live events |
| `GET /clap/v1/cache-decisions` and `/:id` | Persisted cache decision evidence |
| `GET /metrics` | Prometheus counters, queues, priorities, residency, structured outcomes, and honest memory series |
| `GET /openapi.json` | Machine-readable HTTP contract |

### KV reuse capacity: what limits the reusable prefix

Reuse is bounded by bytes, not by intent. A cached prefix (an anchor) costs:

```txt
anchor_bytes = fixed_state + tokens * bytes_per_token
```

`bytes_per_token` comes only from layers whose KV grows with sequence length,
and `fixed_state` from layers that carry a constant-size state. Two measured
examples show how far apart model families sit:

| Model | Growing layers | `bytes_per_token` | `fixed_state` | 8k-token anchor |
| --- | --- | --- | --- | --- |
| Gemma-4-E4B-4bit | 7 full-attention (35 sliding capped at 512) | 14 KiB | 35 MiB (sliding window cap) | ~150 MiB |
| Qwen3.6-27B-4bit | 16 full-attention (48 linear-attention) | 64 KiB | 72 MiB (recurrent state) | ~585 MiB |

Anchors are funded from the automatic-checkpoint share of the retained-KV
budget, and that share is divided across the anchors retained concurrently, so
the longest reusable prefix is:

```txt
reusable_tokens = (checkpoint_share / concurrent_anchors - fixed_state) / bytes_per_token
```

When that value falls below the prompt length, every turn re-prefills the
remainder: reuse degrades smoothly into a near-cold request rather than
failing loudly. A model with a large `fixed_state` is hit twice, because the
constant cost is charged to each anchor.

### Anchor placement: why a snapshot at prompt end is not reusable

A non-rewindable cache (MLX) can only reuse an anchor whose *whole* state is a
prefix of the new prompt; it cannot trim a longer anchor down. Placement
therefore decides everything.

A rendered prompt ends with a generation prompt — the assistant header, and for
thinking templates an empty reasoning block. Once that turn becomes history the
template re-renders it differently, so the tail of turn N's prompt is *not* a
prefix of turn N+1's. A snapshot captured at prompt end is unusable next turn,
and reuse silently falls back to the coarser automatic checkpoint grid.

The worker therefore resolves a boundary through the final message, rendered
with `add_generation_prompt: false`, and offers it to the coordinator as a
stable boundary. That position is exactly what the next turn shares. Templates
that are append-only resolve to the position the session slot already holds,
where this is a no-op; llama.cpp is unaffected because it trims instead of
anchoring.

Measured on Qwen3.6-27B-4bit with a ~6.1k-token prompt:

| Anchor available | Reused | TTFT |
| --- | --- | --- |
| checkpoint grid only | 2,048 (24%) | 121 s |
| message boundary | 6,083 (99%) | 2.2 s |

Automatic checkpoints are a fallback for prompts with no matching semantic
boundary, not a substitute for one. They are charged to the same byte budget,
so a fine grid is actively harmful on heavy models: a 512-token grid spent
~1.9 GiB of a 5.7 GiB budget on intermediate checkpoints for one 8.4k prompt
and starved the message-boundary anchor, producing 47 evictions and reuse
collapsing from 98% to 59% mid-conversation. The shipped 2048 grid keeps
budget available for the anchor that actually carries a conversation.

Under real multi-session load the binding constraint is anchor bytes, not hit
rate. Eight concurrent otto sessions against Qwen3.6-27B (~8.4k prompts,
~600 MiB per anchor) exceeded the budget on a 32 GiB host: the hit rate stayed
high at 95.8% because every session still matched the shared prefix anchor,
while reuse depth fell to the shared portion (~60%) because per-session
anchors could not be retained. Check `pressureState`, `evictionCount`, and
`retainedBytes` against `highWatermarkBytes` in `/clap/v1/runtime/models`
before concluding a model or template is at fault.

Sizing knobs, in the order worth reaching for:

- `[cache.checkpoints] budget_fraction` — share of the retained budget usable
  for anchors (`CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS`)
- `[mlx] retained_budget_gib` / `retained_budget_percent` — the retained budget
  itself, which by default is derived from memory available at startup
- `[cache.checkpoints] interval_tokens` — anchors are planted on multiples of
  this interval, so a coarse interval leaves up to `interval - 1` tokens to be
  re-prefilled even when the budget is ample

### KV reuse metrics: two intentionally different denominators

`/metrics` reports cache reuse twice, because the two questions differ:

| Series | Denominator | Answers |
| --- | --- | --- |
| `clap_kv_cache_total{outcome}`, `clap_kv_cache_eligibility_total`, `clap_kv_cache_miss_class_total` | Requests that supplied a `cache` intent block **and** received an admission decision | "Is the coordinator honoring the reuse contract callers asked for?" |
| `clap_kv_cache_admitted_total{outcome}`, `clap_kv_cache_admitted_reused_tokens_total`, `clap_kv_cache_admitted_prompt_tokens_total` | Every request that received an admission decision, intent or not | "How much prompt work is the KV cache actually saving?" |

Standard OpenAI clients never send `cache`, so they are `not_eligible` for the
first family by design and would otherwise appear as zero reuse. The
`admitted` family is a superset that covers them, so reuse is visible out of
the box; the dashboard tiles read from it and fall back to the intent-gated
KPI only when talking to a server that predates these fields.

Neither family counts requests that failed before admission — those have no
reuse outcome to report and stay out of both denominators rather than being
recorded as misses.

Per-launch artifacts are under:

```text
$CLAP_HOME/logs/workers/<backend>/<model-identity-hash>/<launch-id>.stderr.log
$CLAP_HOME/logs/workers/<backend>/<model-identity-hash>/<launch-id>.json
```

Metadata includes launch/model identity fingerprints, PID, command, protocol version, lifecycle phase/timestamps, exit status, and crash classification. Paths use canonical model fingerprints rather than raw model paths. Finalized pairs are pruned by per-model count and per-backend bytes; active, unfinished, malformed, and invalid entries are not silently deleted.

Persisted cache decisions are stored under `$CLAP_HOME/telemetry` with configured byte/age retention. They include launch IDs so historical rows are distinguishable from current resident cache state.

Typical operator error classes:

- discovery/platform: worker missing, unsupported MLX platform, missing MLX Metal library, model path/layout invalid;
- protocol: the strict fault codes listed above, surfaced with the per-launch log hint;
- admission: unavailable memory, insufficient headroom, no safe eviction candidate, or model estimate above capacity;
- structured output: required capability unavailable or generated output invalid;
- cache: identity authentication required, plan/materialization mismatch, generation mismatch, busy lease, or fail-closed coordinator error;
- native resource failure: backend-specific message plus launch log path; GGUF guidance may suggest a smaller quant/context/batch/GPU-layer configuration.

## Cache correctness tiers and assets

The correctness program separates deterministic logic from physical-model coverage:

- **Tier A**: locked Rust coordinator tests, C++ tests, Swift tests, protocol isolation, and gate wiring. It needs no downloaded model.
- **Tier B**: default pinned physical GGUF and MLX assets, run through the backend probes and canonical scenarios.
- **Tier C**: explicitly provisioned architecture/format expansion matrix. Missing declared assets are skipped unless assets are required.

Asset pins and expected probe metadata live in `config/cache-correctness-assets.json`. Cases, architectures, scenarios, timeouts, and resident-byte ceilings live in `config/cache-correctness-matrix.json`. Asset validation rejects mutable revisions and mismatched files/probe facts. Reports and sanitized logs are written by `scripts/run-cache-correctness-matrix.ts`; secrets and local asset paths must not enter committed output.

Run:

```sh
bun run native:cache:tier-a
bun run native:cache:tier-b
bun run native:cache:tier-c
```

To make absent pinned assets fail instead of skip:

```sh
CLAP_CACHE_TEST_REQUIRE_ASSETS=1 bun run native:cache:tier-b
CLAP_CACHE_TEST_REQUIRE_ASSETS=1 bun run native:cache:tier-c
```

## Required verification

Run from the repository root with Bun:

```sh
bun run typecheck
bun test apps packages
bun run native:build
bun run native:test
bun run bundle:check
```

Individual architectural gates are also directly runnable:

```sh
bun run cache:identity:check
bun run structured-output:gate
bun run capability-priority:gate
bun run memory:honesty:check
bun run native:structure:check
bun run native:ownership:check
bun run runtime:process:ownership
bun run native:probe:leak:check
bun run native:cache:gate:check
```

`native:test` includes the identity, structured-output, capability/priority, memory, structure, ownership, process, and Tier A gates. `native:build` builds both native workers and therefore requires macOS arm64 for MLX. On other hosts, run the GGUF/cache gates supported by that host and use a macOS arm64 runner for the full matrix.
