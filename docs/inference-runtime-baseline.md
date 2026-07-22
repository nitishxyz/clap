# Inference Runtime Baseline

This Phase 0 baseline freezes the native runtime's current responsibilities and observable behavior before decomposition. It is descriptive, not a new protocol contract.

## Snapshot

Line counts use `wc -l` on the working tree. The audit counts supplied before the Phase 0 characterization changes are retained for comparison. Current counts were rechecked on 2026-07-22 at 07:04 IST.

| File | Pre-Phase 0 audit | Current baseline |
| --- | ---: | ---: |
| `native/llama/clap-llama.cpp` | 1,811 | 1,765 |
| `native/mlx/Sources/clap-mlx/main.swift` | 2,116 | 2,107 |
| `native/mlx/Sources/clap-mlx/CacheCoordinator.swift` | 362 | 361 |
| `packages/runtime-router/src/resident.ts` | 998 | 997 |

The current values reflect concurrent extraction/characterization work already present in the working tree. This document does not attribute those deltas to the CMake gate.

## Ownership And Responsibilities

- `native/llama/clap-llama.cpp`: GGUF/llama.cpp model lifecycle, JSON-lines transport, prompt rendering/tokenization, sampling, stop and UTF-8 handling, request admission, decode-first batching, physical KV-slot mutation, Rust cache-plan application, telemetry, and terminal events.
- `native/llama/CMakeLists.txt`: sole explicit manifest of production C++ translation units, vendored llama.cpp configuration, imported Rust cache archive linkage, worker installation, and CTest characterization registration.
- `native/mlx/Sources/clap-mlx/main.swift`: MLX model lifecycle, JSON-lines transport, prompt/token processing, generation, physical MLX KV-cache mutation, scheduler integration, telemetry, and terminal events.
- `native/mlx/Sources/clap-mlx/CacheCoordinator.swift`: Swift-to-C cache coordinator ownership, plan/view translation, slot/generation operations, plan commit/abort, and coordinator error propagation.
- `native/cache`: backend-neutral Rust cache policy and C ABI. It decides eligible reuse, target/donor slots, evictions, leases, generations, and commit outcomes; it does not mutate llama.cpp or MLX cache objects.
- `packages/runtime-router/src/resident.ts`: server-side worker process lifecycle, request correlation, model residency/retention state, cancellation and load/unload control, worker-output parsing, and propagation of tokens, usage, cache, timing, memory, and capability telemetry.

## Legacy JSON-Lines Protocol

There is no required protocol-version envelope. Each stdin line is decoded independently and each stdout line is one JSON object. Request IDs are optional strings; unsolicited telemetry omits `id`.

### Inbound Variants

| Variant | Current fields and behavior |
| --- | --- |
| Load | `type: "load"`, `id?`, `model`; returns model capabilities and retention. |
| Unload | `type: "unload"`, `id?`; rejected while active requests exist. |
| Shutdown | `type: "shutdown"`, `id?`; acknowledges completion and exits. |
| Cancel | `type: "cancel"`, `id?`; an empty/missing ID is treated as active-request wildcard, while queued cancellation requires an exact non-empty ID. |
| Concurrency update | `type: "set_max_active"`, `max_active`, plus optional previous value, limiting/adjustment metadata, retained-growth reserve, global resident bytes, and pressure state. |
| Chat/generation | Any other `type` value, including omitted `type`, is treated as chat. Fields include `id?`, `model`, `messages`, `stream?`, token/sampling controls, stop as string or string array, penalties, tools/tool choice where supported, and cache intent/identity. |

The backends accept overlapping but not formally versioned field sets. Unknown commands therefore fall through to chat instead of producing an unknown-command error.

### Outbound Variants

| Variant | Observable shape |
| --- | --- |
| Started | `{id, started: true, retention?}` after admission. |
| Stream text | `{id, token}`; MLX also has a legacy `{id, content}` field in its message envelope. |
| Prefill progress | `{id, prefill: {done, total}}` where emitted by the backend. |
| Load completion | `{id, loaded: true, done: true, token_capabilities, retention?}`. |
| Unload completion | `{id, unloaded: true, done: true}`. |
| Control acknowledgement | `{id, done: true, retention?}` for shutdown/concurrency updates. |
| Chat completion | `{id, done: true, finish_reason, cancelled, usage, cache, timing?, memory?}`; queued cancellation may emit only the cancellation terminal fields. |
| Error | `{id?, error, code?}`. The machine-readable `code` is optional. |
| Unsolicited telemetry | `{retention}` or memory/retention telemetry without an `id`. |

All fields are additive/optional in the legacy parser; terminal messages are recognized by shape rather than an explicit event discriminator.

## Cache Outcome Baseline

- A coordinator plan can select continue-in-slot (`reuse_kind: "slot"`), restore from an anchor (`"anchor"`), branch/copy from a donor (`"branch"`), or no realized reuse (`reuse_kind: null`, `hit: false`).
- Successful outcomes report planned and realized reuse token counts, target and optional donor slot/generation, scope, evictions, decision time, stable-boundary/checkpoint details, and candidate eligibility/rejection reasons.
- Capacity or busy-slot planning outcomes apply backpressure and leave the request queued.
- Non-capacity coordinator planning failures fail closed (`coordinator_plan_failed_closed`).
- Runtime fallback telemetry currently includes coordinator unavailable, coordinator unavailable with no cache, coordinator advance failure, and decode retry with full prefill.
- Materialization validates target/donor bounds and physical capabilities before commit. Failure clears the target physical slot and does not report successful reuse.
- Generation/advance failures invalidate coordinator state where possible. Cache behavior remains backend-owned physically and Rust-owned logically.

## Scheduler Ordering Baseline

- GGUF rounds are decode-first: runnable decode requests retain active-vector order, followed by runnable prefill requests in the same stable order. Non-runnable requests are omitted. The characterization test also freezes duplicate/mixed request-ID ordering independently of ID value.
- MLX gives highest priority to requests near their first token (residual prefill at or below 256), then first decode, then established decode, then other prefill. Equal-priority requests use ascending admission order.
- Every runnable MLX request appears once per round, preventing arrival-based starvation. Prefill quantum is 512 when uncontended and 96 when contended. Near-first-token prefills may receive additional turns, capped at five.
- Cancelled MLX requests are excluded. GGUF cancellation marks matching active requests and exact-ID queued requests according to the legacy asymmetry above.

## Native Build And Test Gates

- `runtime:cache:test` runs the locked Rust workspace independently.
- `runtime:llama:test` configures the Clap-owned CMake project, builds only the C++ characterization executable, and runs its named CTest.
- `runtime:mlx:test` builds the release Rust FFI archive needed by SwiftPM, then runs the Swift package tests independently.
- `native:test` runs the three gates in cache, llama, MLX order.
- `runtime:llama:build` configures and builds the `clap-llama` CMake target, then installs the runtime component into `libexec`; source and link manifests are not duplicated in the Bun script.

## Verification Matrix

The complete Phase 0 matrix below was run on 2026-07-22 between 07:02 and 07:04 IST on macOS 26.4.1 arm64 with Bun 1.3.14, Rust 1.92.0, CMake 4.3.0, and Apple Swift 6.3.1. Status is one of `PASS`, `FAIL`, or `SKIP`; `SKIP` includes its reason.

| Gate | Exact command | Status | Notes |
| --- | --- | --- | --- |
| TypeScript typecheck | `bun run typecheck` | PASS | Exit 0; root `tsc --noEmit -p tsconfig.json` produced no diagnostics. |
| TypeScript tests | `bun test apps packages` | PASS | 271 passed, 0 failed, 1,214 expectations across 25 files in 7.84 seconds. |
| Rust cache | `bun run runtime:cache:test` | PASS | 54 integration tests passed (36 coordinator, 10 retention, 8 ABI), 0 failed; unit/doc targets with no tests also passed. |
| C++ characterization | `bun run runtime:llama:test` | PASS | CMake configured and built `clap-llama-characterization`; CTest passed 1/1. |
| Swift/MLX tests | `bun run runtime:mlx:test` | PASS | 33 XCTest tests and 3 Swift Testing tests passed, 0 failed. |
| GGUF production build/package | `bun run runtime:llama:build` | PASS | Built `clap-llama` with static llama.cpp and Rust cache dependencies; installed/up-to-date at `libexec/clap-llama`. |
| MLX production build/package | `bun run runtime:mlx:build` | PASS | Production Swift build completed; packaged `libexec/clap-mlx` and `libexec/mlx.metallib`. |
| Bundle validation | `bun run bundle:check` | PASS | Validated `clap-llama`, `clap-mlx`, and `mlx.metallib`. |
| Legacy JSON/JSONL fixture parse | `bun -e 'import { readdir } from "node:fs/promises"; const dir="packages/runtime-router/fixtures/legacy-worker-protocol"; for (const name of (await readdir(dir)).filter((name) => name.endsWith(".json") \|\| name.endsWith(".jsonl")).sort()) { const text=await Bun.file(`${dir}/${name}`).text(); if (name.endsWith(".jsonl")) { const lines=text.split(/\r?\n/).filter((line) => line.trim()); lines.forEach((line, index) => { try { JSON.parse(line); } catch (error) { throw new Error(`${name}:${index + 1}: ${error}`); } }); console.log(`${name}: ${lines.length} JSONL records parsed`); } else { JSON.parse(text); console.log(`${name}: JSON parsed`); } }'` | PASS | Parsed manifest JSON plus 7 request, 13 llama-event, and 14 MLX-event JSONL records. |
| Patch whitespace validation | `git diff --check` | PASS | Exit 0 with no diagnostics across the current workspace diff. |
| Linux GGUF release gate | `bun run runtime:llama:build` | SKIP | Current host is macOS arm64; Linux CPU/CUDA release coverage requires a Linux runner. |

### Classified Diagnostics

- No required matrix command failed.
- `runtime:llama:test` and `runtime:llama:build` emitted vendored llama.cpp configuration warnings: ccache was unavailable, OpenMP was not found, and ARM feature detection fell back to `-mcpu=native`. These are pre-existing host/toolchain warnings, not Phase 0 regressions: both targets built successfully using the detected CPU, Accelerate, BLAS, and Metal backends, and CTest passed.
- `bun test apps packages` printed `[clap] ignoring invalid profile in broken.json: missing "name"`. This is expected output from the passing invalid-profile characterization test, not a suite failure.
- An earlier package-local TypeScript invocation reported TS6310 project-reference diagnostics. The required root gate did not reproduce them: `bun run typecheck` passed with no diagnostics. The package-local invocation is therefore classified as a pre-existing alternate-invocation/project-reference issue outside this matrix, not a Phase 0 regression.
