# Inference Runtime Hardening and Organization Plan

Status: draft.

Scope: preserve Clap's custom control over request parsing, model-native prompt
rendering, scheduling, KV-cache execution, output interpretation, lifecycle,
and observability while making the system easier to maintain and safer to run.

Benchmarks are explicitly deferred. This plan prioritizes architecture,
correctness, isolation, and operational safety. Performance work may be added
later without changing the boundaries established here.

README policy for this work: do not read or modify any README file during plan
implementation. README updates are deferred until the plan is complete and
will be handled as a final, explicit documentation task.

## Product decision

Clap will continue to own its native C++ llama worker and native Swift MLX
worker. They will not become pure pass-through wrappers around upstream
servers. Direct control is required for:

- exact model-native template rendering and tokenization;
- cache identity, stable-boundary proof, branching, restoration, and eviction;
- backend-specific scheduling and cancellation;
- parsing and structured-output behavior;
- physical cache validation and telemetry;
- model lifecycle and machine-wide memory policy.

We own policy and physical integration, but not kernels. llama.cpp and MLX
remain responsible for tensor execution, model implementations, tokenization
primitives, and backend acceleration.

## Goals

1. Split both native workers into small modules with explicit ownership and
   dependency direction.
2. Make the worker transport strict, versioned, multiplex-safe, and testable.
3. Prevent concurrent model loads and resident weights from exhausting the
   machine even when request concurrency is already at its minimum.
4. Bind cache reuse to authenticated identity with cryptographic keyed
   fingerprints.
5. Verify that logical cache decisions and physical backend state produce the
   same deterministic result as a fresh execution.
6. Give each worker launch independent diagnostics.
7. Remove duplicate worker process and protocol implementations.
8. Add decode-time structured-output guarantees where the backend supports
   them and accurately report weaker modes elsewhere.
9. Distinguish measured, estimated, and unavailable memory values.
10. Publish explicit runtime capabilities and use one priority vocabulary from
    server admission through scheduling and cache retention.

## Non-goals

- Replacing native workers with upstream server processes.
- Rewriting llama.cpp or MLX kernels.
- Requiring identical physical schedulers for llama.cpp and MLX.
- Expanding into every local-inference modality or compatibility endpoint.
- Performance benchmarking in this plan.
- Moving template rendering or tokenization into TypeScript.
- Updating README files before all implementation phases are complete.

## Ownership model

The architecture must enforce these ownership boundaries.

### TypeScript control plane

Owns:

- public API compatibility and request normalization;
- authenticated client identity;
- model resolution and download lifecycle;
- server-wide fairness and overload behavior;
- model residency, load reservations, and idle-model eviction;
- worker process lifecycle and log indexing;
- output parser selection and public response formatting;
- capability-aware routing and public telemetry.

Does not own:

- model chat-template rendering;
- model tokenization;
- physical KV-cache mutation;
- backend execution scheduling details.

### Rust cache policy

Owns:

- logical cache identity and isolation rules;
- donor and target selection;
- continuation, branch, restore, and fresh decisions;
- slot leases and generation guards;
- retained-entry and byte-pressure policy;
- stable-boundary and automatic-checkpoint authorization;
- cache priority and deterministic eviction policy.

Does not own:

- backend cache objects;
- llama or MLX API calls;
- model loading;
- public API parsing.

### Native backend adapters

Own:

- model-native template rendering and tokenization;
- backend model and cache object lifetimes;
- execution of Rust-authorized physical cache operations;
- backend-appropriate scheduling and stepping;
- sampling, stop handling, token streaming, and cancellation;
- physical-state reconciliation and backend telemetry.

Do not own:

- cross-request cache policy;
- authenticated tenant selection;
- server-wide load admission;
- public API compatibility.

## Structural rules

These rules apply to all reorganization work.

1. Refactor before redesign. The first extraction phases must preserve the
   worker protocol, scheduling order, cache decisions, and emitted output.
2. Every module has one primary reason to change. Do not create generic
   `utils`, `helpers`, or `common` dumping grounds.
3. Entrypoints only compose dependencies and start the run loop. Target fewer
   than 100 lines for each native `main` file.
4. Prefer modules below 300-400 lines. A file above 500 lines requires a
   documented reason or a follow-up extraction task. A file above 700 lines is
   not accepted as the end state.
5. State has one owner. Other modules receive narrow references, immutable
   snapshots, or explicit commands.
6. Protocol DTOs do not depend on model runtime APIs.
7. Cache policy adapters do not parse public API requests.
8. Schedulers operate on prepared requests; they do not render prompts or
   choose cache donors.
9. Backend modules may depend on the protocol and Rust bridge, but the protocol
   must not depend on backend modules.
10. Each extraction lands with tests and a buildable worker. No long-lived
    branch may contain a half-migrated worker.

## Target native layout

Names may be adjusted during implementation, but responsibilities and
boundaries should remain stable.

### llama C++ worker

```text
native/llama/
  include/clap/llama/
    worker-types.h          # backend-neutral worker domain structs
    protocol.h              # protocol DTOs and event writer interface
    environment.h           # typed environment/config reads
    model-runtime.h         # loaded model/context RAII owner
    prompt.h                # messages, templates, tokenization, boundaries
    sampling.h              # sampling params and llama sampler construction
    stop-buffer.h           # UTF-8 and stop-sequence streaming state
    cache-executor.h        # Rust plan to physical llama memory operations
    request-state.h         # one prepared/in-flight request
    scheduler.h             # queue, admission, mixed-batch stepping
    telemetry.h             # token, retention, timing, memory snapshots
    worker.h                # command dispatch and worker run loop
  src/
    main.cpp
    protocol.cpp
    environment.cpp
    model-runtime.cpp
    prompt.cpp
    sampling.cpp
    stop-buffer.cpp
    cache-executor.cpp
    request-state.cpp
    scheduler.cpp
    telemetry.cpp
    worker.cpp
  policy/
    active-concurrency.h
    stable-boundary.h
  cache/
    cache-adapter.h
  tests/
    protocol.test.cpp
    prompt.test.cpp
    sampling.test.cpp
    stop-buffer.test.cpp
    cache-executor.test.cpp
    scheduler.test.cpp
```

Dependency direction:

```text
protocol/environment/stop-buffer
              |
              v
model-runtime/prompt/sampling/telemetry
              |
              v
cache-executor/request-state
              |
              v
scheduler
              |
              v
worker -> main
```

Important constraints:

- `ModelRuntime` is the sole owner of `llama_model` and `llama_context` and
  releases both through RAII.
- `CacheExecutor` is the only module allowed to call sequence copy, trim,
  remove, clear, or coordinator commit/invalidate operations.
- `Scheduler` is the only module allowed to mutate active and waiting queues.
- `StopBuffer` owns incomplete UTF-8 and partial-stop holdback behavior.
- `ProtocolWriter` is the only code that writes worker events to stdout.
- `main.cpp` performs backend initialization, constructs `Worker`, and exits
  with its status.

The direct compiler invocation in `scripts/build-llama.ts` must compile an
explicit source list or invoke a dedicated CMake target. The source list must
not be maintained in multiple scripts. Prefer a small Clap-owned CMake target
that links the vendored llama and Rust cache libraries, with the Bun script
remaining the packaging orchestrator.

### MLX Swift worker

```text
native/mlx/Sources/clap-mlx/
  main.swift                       # constructs and runs WorkerApplication
  Application/
    WorkerApplication.swift        # command dispatch and top-level run loop
    WorkerState.swift              # loaded runtime and coordinated state
  Protocol/
    WorkerRequest.swift            # versioned inbound DTOs
    WorkerEvent.swift              # versioned outbound DTOs
    JSONLineTransport.swift        # stdin buffering and stdout serialization
  Configuration/
    WorkerConfiguration.swift      # environment parsing and typed defaults
  Model/
    ModelRuntime.swift             # model/container/tokenizer ownership
    ModelLoader.swift              # load/unload and metadata discovery
    PromptRenderer.swift           # templates, tools, fallback policy
    TokenCapabilities.swift        # context/output capability derivation
  Generation/
    PreparedRequest.swift          # immutable admission result
    ActiveRequest.swift            # mutable request-local generation state
    GenerationStepper.swift        # prefill/decode steps
    SamplingParameters.swift
    StopMatcher.swift
  Scheduling/
    RequestQueue.swift
    WorkerScheduler.swift          # admission and latency-aware rounds
  Cache/
    CacheCoordinator.swift         # existing Rust bridge wrapper
    CacheSlot.swift                # physical slot state
    CacheExecutor.swift            # plan validation and physical apply/commit
    AnchorManager.swift            # authorized snapshot materialization
    CacheIdentity.swift
  Telemetry/
    WorkerTelemetry.swift
    RetentionTelemetry.swift
    TimingTelemetry.swift
    MemoryTelemetry.swift
  Tools/
    ToolTemplateAdapter.swift      # template-compatible tool representation
  Support/
    Errors.swift                   # typed worker errors only
  Tests/                           # executable-target-focused tests as feasible
```

Swift package notes:

- SwiftPM discovers source subdirectories automatically for the executable
  target; no monolithic source manifest is required.
- `WorkerApplication` owns the event loop but delegates all model, cache,
  generation, and scheduling mutations.
- `ModelRuntime` owns MLX model and tokenizer references.
- `CacheExecutor` is the only module allowed to copy, trim, replace, or clear
  `[KVCache]` objects in response to a cache plan.
- `WorkerScheduler` selects turns but does not choose cache reuse or render
  prompts.
- `GenerationStepper` advances exactly one bounded turn and has no queue or
  model-switching policy.
- Tool-template conversion belongs beside prompt rendering, not in the worker
  event loop.

## Phase 0 - Freeze behavior and define extraction gates

Purpose: make large-file decomposition safe before changing behavior.

Tasks:

- Record the current inbound commands, outbound message shapes, cache outcomes,
  and scheduler ordering in characterization tests.
- Add golden protocol fixtures for load, unload, chat, stream token, cancel,
  error, telemetry, and completion messages.
- Add tests for UTF-8 token splits, split stop sequences, cancellation while
  queued, cancellation while active, and mixed request IDs.
- Add deterministic scheduler tests for llama decode-first mixed batching and
  MLX latency-aware round ordering.
- Add native build/test commands that can be run independently.
- Capture current line counts and module responsibilities as the refactor
  baseline.

Exit criteria:

- Existing TypeScript and Rust suites pass from a clean baseline.
- Both native workers build.
- Characterization fixtures cover all currently emitted message variants.
- No production behavior has changed.

## Phase 1 - Decompose native workers without behavior changes

This phase is first because later safety work should land in focused modules,
not add more logic to the two monoliths.

### 1A. Extract pure and low-coupling code

C++ extraction order:

1. environment and memory helpers;
2. protocol emit/error helpers and stdin reader;
3. stop matching and UTF-8 holdback;
4. sampling request parsing and sampler construction;
5. telemetry serialization.

Swift extraction order:

1. protocol DTOs and line transport;
2. typed configuration and errors;
3. tool/message conversion and prompt fallback helpers;
4. telemetry DTOs and mappers;
5. model metadata and token-capability derivation.

Each move must be a code move plus tests, not a redesign.

### 1B. Extract model and prompt ownership

- Introduce `ModelRuntime` in both workers.
- Move load, unload, context capability, tokenizer/template, and model metadata
  logic behind it.
- Make prepared prompt tokens and exact stable boundaries an explicit output
  of `PromptRenderer`/`prompt`.
- Preserve native template and tokenizer usage.

### 1C. Extract cache physical execution

- Move all Rust coordinator interaction and physical cache mutation behind
  `CacheExecutor`.
- Define an explicit transaction shape: plan, validate, materialize, commit;
  on failure, abort/invalidate and reconcile physical state.
- Move anchor/checkpoint materialization into `AnchorManager` or the equivalent
  C++ cache module.
- Prohibit scheduler code from performing direct cache copies or trims.

### 1D. Extract request state and generation stepping

- Separate immutable admission facts from mutable generation state.
- Move prefill, decode, sampled-token handling, stop handling, usage, and
  completion into request/generation modules.
- Preserve current llama mixed-batch execution and MLX bounded interleaving.

### 1E. Extract scheduler and application loop

- Make the scheduler the sole owner of active/waiting queues.
- Move control command dispatch into `Worker`/`WorkerApplication`.
- Reduce both main files to dependency construction and process exit.
- Remove old monolithic code only after parity tests pass.

Exit criteria:

- `native/llama/src/main.cpp` and `native/mlx/Sources/clap-mlx/main.swift` are
  each below 100 lines.
- No native production source exceeds 700 lines; the target is below 400.
- Direct physical cache calls exist only in cache modules.
- Queue mutations exist only in scheduler modules.
- Protocol output exists only in transport/protocol modules.
- Existing request, cache, cancellation, and streaming behavior remains
  unchanged.

## Phase 2 - Strict versioned worker protocol

Introduce a shared protocol contract while retaining JSON Lines as the local
transport.

### Contract

Create a protocol package and fixtures:

```text
packages/worker-protocol/
  src/
    schemas.ts
    types.ts
    validation.ts
  fixtures/v1/
    requests/
    events/
```

The protocol includes:

- a required protocol version;
- a startup `ready` handshake;
- a typed event name on every line;
- a request ID on every request-scoped event;
- monotonically increasing per-request sequence numbers for stream events;
- explicit unsolicited event types for worker telemetry and diagnostics;
- declared worker and model capabilities;
- structured error code, message, retryability, and fatality;
- an explicit terminal event for every accepted request.

Example event:

```json
{
  "protocol": 1,
  "type": "token",
  "request_id": "req_123",
  "sequence": 14,
  "text": "hello"
}
```

Rules:

- Non-JSON stdout is never generated content.
- Missing or unknown IDs are never assigned to the first pending request.
- A malformed request-scoped event fails that request.
- An unscoped protocol violation marks the worker unhealthy, rejects pending
  work, preserves diagnostics, and restarts through the normal backoff path.
- Unknown optional fields are tolerated within a protocol version; unknown
  required event types are rejected.
- Server and worker negotiate a supported version during startup.

Implementation order:

1. Add schemas, fixtures, and TypeScript decoder.
2. Add native v1 event encoders and ready handshakes.
3. Teach the router to support old and v1 workers temporarily.
4. Switch bundled workers to v1.
5. Remove legacy first-pending fallback and old protocol support.

Exit criteria:

- Concurrent streams cannot cross request IDs in fault-injection tests.
- Invalid stdout cannot appear in a client response.
- Version mismatch returns an actionable worker error.
- Every accepted request terminates exactly once.

## Phase 3 - Worker lifecycle cleanup and diagnostics

### Per-launch logs

Use independent paths:

```text
$CLAP_HOME/logs/workers/<backend>/<model-hash>/<launch-id>.stderr.log
$CLAP_HOME/logs/workers/<backend>/<model-hash>/<launch-id>.json
```

The metadata sidecar records launch ID, model ID and path fingerprint, backend,
PID, command, protocol version, start/end timestamps, exit status, and crash
classification.

Add bounded retention by launch count and total bytes. Never rotate or unlink
another running worker's log.

### One process abstraction

Remove independent one-shot subprocess parsing from the runtime packages.
All inference uses the same `ResidentWorkerProcess` protocol implementation.
A one-shot request, if still needed, is a resident worker configured with zero
keep-alive and unloaded in `finally`.

Exit criteria:

- One implementation owns spawn, handshake, decode, cancel, shutdown, logs,
  crash classification, and restart backoff.
- Concurrent models of the same backend have independent logs.
- Dashboard and errors link to the exact launch log.

## Phase 4 - Authenticated cache identity

The public caller supplies cache intent, never authoritative tenant identity.

Tasks:

- Create or load an installation secret under `CLAP_HOME` with owner-only file
  permissions.
- Derive an authenticated tenant root from API key identity; use a dedicated
  local identity for trusted loopback use.
- Derive tenant/project/harness/agent/session fingerprints with HMAC-SHA256 or
  keyed BLAKE3 and domain-separated inputs.
- Bind the physical namespace to backend, resolved model revision, tokenizer,
  context allocation, KV format, and cache layout version.
- Pass opaque, server-derived identity material to the native worker.
- Ensure caller-provided labels cannot escape the authenticated tenant root.
- Define secret rotation behavior: existing cache identities become invalid
  and resident cache state is cleared.
- Rename any non-HMAC helper or environment field that currently implies HMAC
  without implementing it.

Exit criteria:

- Identical caller labels under different API keys cannot reuse cache state.
- A caller cannot claim another tenant by supplying public cache fields.
- Fingerprints are stable for one installation and domain-separated by scope.
- Rotation safely invalidates prior identities.

## Phase 5 - Global model-load memory admission

Request concurrency control does not prevent several resident model weights
from exhausting memory. Add machine-wide model residency admission.

Proposed control-plane modules:

```text
packages/runtime-router/src/residency/
  coordinator.ts       # reservations and serialized load admission
  estimates.ts         # backend/model cost estimates
  eviction.ts          # idle unpinned victim selection
  types.ts
```

Load flow:

1. Resolve the model and estimate load cost.
2. Acquire a serialized or capacity-aware load reservation.
3. Read current available memory and resident worker RSS.
4. Reserve OS and runtime headroom.
5. Evict idle, unpinned workers in deterministic LRU/value order if needed.
6. Never evict active, loading, or pinned models.
7. Start the worker load.
8. Replace the estimate with observed RSS/allocator measurements when available.
9. Commit or roll back the reservation.
10. Return a structured `insufficient_model_memory` error if safety cannot be
    established.

Estimation sources:

- GGUF file size, configured context, KV type, known architecture metadata,
  prior observed RSS, and a conservative margin;
- MLX weight files, prior observed active memory, configured cache budget, and
  a conservative margin;
- unknown models use a deliberately conservative fallback.

Multiple loads must not independently pass a stale memory check. Reservations
are included in subsequent admission decisions.

Exit criteria:

- Two simultaneous large loads cannot overcommit the same free memory.
- Idle unpinned workers are unloaded before machine-critical pressure.
- Active and pinned models are never evicted.
- Failed loads release reservations.
- Admission reasons and evictions are observable.

## Phase 6 - Physical cache correctness suite

This is correctness validation, not benchmarking.

Add opt-in native integration tests using small pinned GGUF and MLX models.
Test cases:

- cold generation versus exact continuation;
- branch from a shared prefix;
- anchor restore;
- same tokens under different authenticated namespaces;
- cancellation during prefill and decode;
- eviction and slot generation reuse;
- worker restart and cache reset;
- long prompt with automatic checkpoints;
- sliding-window cache behavior;
- recurrent/hybrid cache behavior where representative models are available;
- coordinator failure after physical preparation;
- physical failure before coordinator commit.

The primary invariant is:

> With deterministic sampling, a cache-assisted execution must produce the
> same next-token result as an equivalent fresh full-prefill execution.

Where practical, add a test-only next-logit or selected-token fingerprint so
correctness does not depend solely on decoded text.

Test tiers:

- Tier A: unit and fixture tests, always run;
- Tier B: small local real-model tests, run in native CI where assets permit;
- Tier C: architecture matrix, scheduled or release-gated.

Exit criteria:

- Logical resident lengths and physical cache lengths are checked at every
  commit boundary in test builds.
- Fresh and cached deterministic results match across supported operations.
- Cancellation and failure leave no busy lease or reusable corrupt state.

## Phase 7 - Structured output and parser organization

Clap retains control of output interpretation, but parsing and generation
constraints must be separate concepts.

### Parser layout

Split the server parsing layer into explicit profiles:

```text
packages/server/src/parsing/
  types.ts
  registry.ts
  stream-state.ts
  plain-text.ts
  native-tools.ts
  json-tools.ts
  xml-tools.ts
  reasoning.ts
  fixtures/
```

A parser profile declares supported model/template traits and implements
incremental `feed` plus terminal `finish`. Partial token boundaries, malformed
calls, reasoning separation, and repair behavior are fixture-tested.

### Constraint modes

Report one of:

- `grammar_constrained` - invalid token paths are masked during decoding;
- `validated_retry` - output is validated and may be retried;
- `best_effort_validated` - prompt plus parse/validation, without a guarantee;
- `none`.

Tasks:

- Integrate llama.cpp grammar/schema sampling through the lower-level APIs used
  by the custom worker.
- Keep parser validation after constrained generation as defense in depth.
- For MLX, expose the actual supported mode; do not claim grammar guarantees
  until token masking is implemented.
- Return a structured capability error when a caller requires a stronger mode
  than the selected backend provides.

Exit criteria:

- API responses state the effective structured-output mode.
- llama schema requests use decode-time constraints when supported.
- MLX behavior is accurately labeled.
- Parser profiles are isolated and stream-boundary tested.

## Phase 8 - Memory semantics and observability

Every memory value carries a source and confidence class.

Use explicit states:

- `measured` - authoritative runtime or OS observation;
- `estimated` - derived from model/configuration/history;
- `unavailable` - no defensible value.

Do not serialize unavailable GGUF retained bytes as a measured zero.

Example:

```json
{
  "retained_bytes": null,
  "retained_bytes_source": "unavailable",
  "estimated_retained_bytes": 123456789,
  "estimated_retained_bytes_source": "context_configuration"
}
```

Machine-safety policy uses process RSS and available system memory. Estimated
KV values may guide policy but must not be represented as exact.

Exit criteria:

- Dashboard, API, and Prometheus distinguish measured and estimated values.
- GGUF unavailable byte telemetry is not reported as zero.
- Critical pressure can unload an idle worker, not only reduce active request
  concurrency.

## Phase 9 - Capabilities and unified priority semantics

### Capability handshake

Workers report protocol, backend, model, cache, generation, and modality
capabilities at ready/load time. Examples:

- fused multi-sequence batching;
- interleaved scheduling;
- partial suffix trim;
- partial prefix branch;
- whole-state copy;
- prompt-boundary snapshots;
- quantized KV;
- grammar-constrained output;
- tool-template support;
- embeddings or multimodal input if later added.

Routing and validation consume capability data rather than backend-name checks.
Public endpoints expose effective capabilities per loaded model.

### Priority vocabulary

Use one enum end to end:

```text
interactive
normal
background
```

Define its meaning at each layer:

- server admission order and per-client fairness;
- worker scheduling weight and prefill quantum;
- cache retention and eviction value;
- whether preemption is permitted;
- metrics labels and dashboard presentation.

Queue responsibilities remain distinct:

- server queue: authenticated fairness and global overload;
- worker queue: model-specific execution availability;
- Rust coordinator: cache leases and retained capacity.

No layer invents a conflicting priority interpretation.

Exit criteria:

- Capability-gated behavior has no backend string sniffing.
- Priority is preserved from public request normalization to cache labels.
- Interactive work cannot be starved by background work.
- Background cache entries remain preferred eviction candidates.

## Phase 10 - Final cleanup and documentation

This phase happens only after all implementation and verification phases.

Tasks:

- Remove temporary protocol compatibility code.
- Remove obsolete one-shot process code and unused helpers.
- Enforce native module dependency and line-count checks where practical.
- Update architecture and operator documentation.
- Only now, explicitly read and update README files as a dedicated final task.
- Record deferred benchmark work as a separate follow-up plan.

Exit criteria:

- No legacy protocol or duplicate process path remains.
- Native entrypoints and modules meet the structural rules.
- All documentation describes measured capabilities accurately.
- README updates are isolated to this final phase.

## Implementation sequence

The intended order is:

1. Phase 0: characterization and gates.
2. Phase 1: native decomposition with no behavior changes.
3. Phase 2: strict worker protocol.
4. Phase 3: process unification and per-launch logs.
5. Phase 4: authenticated cache identity.
6. Phase 5: global model-load admission.
7. Phase 6: physical cache correctness suite.
8. Phase 7: constrained output and parser profiles.
9. Phase 8: honest memory semantics.
10. Phase 9: capability and priority unification.
11. Phase 10: cleanup and final documentation.

Phase 5 design can begin in parallel with native decomposition, but its runtime
integration should wait until protocol and process ownership are stable.
Phase 6 fixtures should be added continuously as cache modules are extracted.
No phase may weaken cache correctness to simplify the refactor.

## Change-unit strategy

Keep implementation changes reviewable and reversible. Suggested atomic units:

1. Add characterization tests only.
2. Add C++ multi-source build target without moving behavior.
3. Extract one C++ pure module at a time.
4. Extract one Swift pure module at a time.
5. Introduce model runtime ownership in one backend at a time.
6. Introduce cache executor ownership in one backend at a time.
7. Introduce scheduler ownership in one backend at a time.
8. Reduce each entrypoint only after all dependencies are extracted.
9. Add protocol v1 alongside legacy decoding.
10. Switch bundled workers, then delete legacy behavior separately.

Avoid combining structural moves with policy changes. If a move exposes a bug,
add a failing regression test and fix it in a separate behavioral change unless
the old code cannot be safely extracted without the correction.

## Verification matrix

Run the relevant subset after every atomic change and the full matrix at phase
boundaries.

```text
bun run typecheck
bun test apps packages
cargo test --workspace --locked             # from native/cache
swift test --package-path native/mlx
bun run runtime:llama:build
bun run runtime:mlx:build                    # macOS arm64
bun run bundle:check
```

Add dedicated commands during Phase 0 for C++ unit tests and native protocol
fixture conformance. Linux GGUF builds and macOS arm64 GGUF/MLX builds are
release gates.

Required fault-injection cases:

- malformed worker output;
- unknown request ID;
- duplicate terminal event;
- worker exit during load, prefill, and decode;
- cancellation racing completion;
- coordinator plan commit failure;
- physical cache materialization failure;
- simultaneous model load reservations;
- critical memory pressure with active, idle, and pinned workers.

## Completion definition

This plan is complete when:

- both native workers are composed from cohesive modules rather than monolithic
  source files;
- custom cache and scheduling control is preserved;
- worker communication is strict and versioned;
- cache identity is cryptographically bound to authenticated tenancy;
- model loads cannot independently overcommit machine memory;
- physical cache reuse has deterministic real-model correctness coverage;
- logs are launch-specific;
- process management has one implementation;
- structured-output guarantees are capability-accurate;
- memory telemetry states what is measured, estimated, or unavailable;
- capability and priority semantics are consistent across all layers;
- all verification gates pass;
- README files are updated only in the final explicit documentation task.
