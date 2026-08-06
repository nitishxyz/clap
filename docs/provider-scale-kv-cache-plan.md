# Provider-Scale KV Cache Plan

Status: Phases 1 through 3 implemented and validated on clean A100 pods on
2026-08-06.
This plan extends `docs/scale-plan.md` and
`docs/rust-cache-coordinator-plan.md` from a strong single-host cache into a
cache-aware inference fleet. It is intentionally phased so every checkpoint
can be committed, pushed, installed on a clean GPU pod, and measured before the
next phase starts.

## Executive decision

Clap should build one backend-neutral cache control plane, not an independent
cache product for every runtime.

Shared Clap components own:

- cache identity and compatibility
- session and tenant lifecycle
- prefix indexing and cache-location metadata
- admission, quotas, TTLs, and eviction policy
- cache-aware request routing
- reuse-versus-queue-versus-transfer cost decisions
- observability, benchmarking, and autoscaling signals

Each runtime gets a narrow physical-cache adapter that advertises capabilities
and executes only operations the runtime can prove correct:

- llama.cpp/GGUF: keep the public sequence API as the portable baseline; add an
  optional, narrow block-table extension only if upstream APIs cannot provide
  provider-grade density and accounting
- MLX/MLX-LM: adapt MLX cache arrays and cache classes; add block sharing or
  copy-on-write in the adapter/runtime layer without duplicating fleet policy
- vLLM, SGLang, TensorRT-LLM, and other paged engines: integrate with their
  native block managers and telemetry rather than replacing them
- remote OpenAI-compatible providers: request routing only; report cache state
  as unavailable unless the provider offers an explicit cache contract

Clap should not build model loading, quantization, attention kernels, CUDA
kernels, Metal kernels, or a complete inference runtime merely to control KV.
That would multiply maintenance cost and delay the parts that distinguish Clap:
portable policy, routing, isolation, and observability.

## Problem statement

The current llama.cpp path can retain a backend-defined number of logical
sequence entries. On the tested Qwen deployment that ceiling was 128 entries,
typically one writable generation plus up to 127 immutable anchors. A logical
entry is not a user and is not a turn. One long tool-heavy request can publish
several semantic boundaries and automatic checkpoints, so a modest number of
requests can fill the sequence table.

This remains correct under pressure: Clap replaces entries and a cache miss
falls back to prefill. The scaling problem is efficiency:

- overlapping full-sequence snapshots can duplicate physical KV
- a hot session can publish many anchors and crowd out other sessions
- a single worker cannot retain every returning user's working set
- unrelated requests can land on workers that do not own their useful prefix
- current cache metadata is local to one worker
- cache limits are expressed partly as entry ceilings rather than universally
  byte-accurate, reference-counted blocks
- no fleet cost model decides whether to wait, recompute, or transfer KV

The objective is not to make all users permanently resident. The objective is
to retain the most valuable hot working set, share common prefixes physically,
and route each request to the cheapest correct execution path.

## Provider-scale target

A mature deployment should look like this:

```text
Clients
   |
Edge/API admission
   |
Cache-aware model router
   |-- session directory
   |-- prefix-location directory
   |-- queue and load state
   |-- tenant budgets
   |-- placement cost model
   |
   +-- worker A -- device KV -- host KV -- local spill
   +-- worker B -- device KV -- host KV -- local spill
   +-- worker C -- device KV -- host KV -- local spill
             \\-- optional KV transfer fabric --//
```

Correctness must never depend on cache presence. Every cached object is an
optimization over deterministic prompt reconstruction. If metadata is stale,
a worker crashes, a transfer fails, or compatibility cannot be proven, Clap
must recompute rather than use uncertain KV state.

## Design principles

1. **Correctness before reuse.** A false hit is data corruption; a false miss
   is only slower.
2. **Capabilities, not backend assumptions.** Policy requests an operation only
   after the adapter advertises and proves support.
3. **Immutable shared prefixes.** Shared objects are immutable; mutations use
   copy-on-write or a private writable tail.
4. **Bytes and saved work are first-class.** Entry count is a hard safety limit,
   not the primary economic metric.
5. **Cache locality is a routing input.** It is balanced against queue delay,
   transfer time, and worker pressure, not treated as an absolute pin.
6. **Bound every owner.** Session, tenant, model, worker, and fleet budgets are
   explicit and observable.
7. **Graceful degradation.** Under pressure: stop creating low-value anchors,
   evict, recompute, queue within bounds, then return 429. Never silently
   truncate or overcommit.
8. **No prompt-content control plane.** Fleet metadata uses keyed fingerprints,
   sizes, locations, and compatibility domains, not raw prompts or KV bytes.
9. **Backend-native fast paths.** Existing paged engines keep their optimized
   physical manager.
10. **Every phase ships measurable value.** No long-lived architectural branch
    that cannot be installed and tested independently.

## Terminology

- **Session:** a logical conversation or agent execution identity.
- **Prefix:** an exact token sequence from the beginning of a request.
- **Boundary:** a template-aware point that may be valuable to restore.
- **Anchor:** an immutable restorable prefix represented by a backend.
- **Block:** a fixed or bounded physical KV allocation unit.
- **Block table:** the ordered block references that materialize a prefix.
- **Writable tail:** private state receiving newly prefetched and decoded tokens.
- **Compatibility domain:** every input that must match before KV can be reused.
- **Resident:** materialized in a worker's active physical cache tier.
- **Directory:** metadata describing likely cache locations, not authoritative KV.

## Cache identity and compatibility

A cache key must commit to all state that can change hidden activations:

- exact model repository and weight revision or file digest
- model architecture and runtime compatibility version
- tokenizer files and tokenizer revision
- chat template and template revision
- exact token IDs in the prefix
- positional encoding and context parameters
- KV data type and quantization format
- adapter, LoRA, prompt adapter, and multimodal projector identity
- runtime/backend and physical cache format version where import is involved
- tenant sharing domain and authorization policy

Use keyed, versioned fingerprints. Never use a caller-provided hash as proof of
identity. Caller hints may partition reuse but may not broaden authorization.

The logical prefix identity should be backend-neutral. A transferable physical
object additionally carries a physical format identity. Two workers may agree
that prompts match while still being unable to import each other's KV.

## Backend capability contract

The existing Rust coordinator capability model should evolve into a versioned
adapter contract. The logical operation set is:

```text
inspect       report exact resident prefixes, tiers, bytes, and leases
continue      append to an existing writable session
restore       materialize an immutable prefix into a writable target
fork          create a copy-on-write branch from a prefix
trim          remove a proven suffix when the cache class supports it
snapshot      publish an exact immutable boundary
release       remove a logical reference
export        serialize compatible blocks or state for transfer/offload
import        materialize an exported object after validating compatibility
promote       move a cache object to a faster tier
demote        move a cache object to a slower tier
```

Capabilities should describe constraints rather than a single boolean:

- full-state versus block-granular restore
- whole-state copy versus copy-on-write
- suffix trimming and minimum trim granularity
- prompt-boundary snapshot support
- safe busy-donor reads
- block size or allocation granularity
- host offload support
- import/export format and transport support
- KV quantization support
- byte-accurate accounting support
- recurrent/hybrid cache restrictions

Every adapter needs conformance tests that intentionally request unsupported
operations and verify fail-closed behavior.

## Logical cache model

Clap's logical index should become a radix/prefix DAG over token fingerprints,
while physical representation remains adapter-specific.

Each logical node records:

- parent and covered token range
- compatibility domain
- owning tenant and optional share group
- sessions referencing it
- known worker/tier locations
- physical bytes by location when known
- reference and lease counts
- creation, last-use, expiry, and reuse counters
- measured prefill time or compute cost saved
- transfer cost estimates
- structural class: harness, tool schema, message boundary, automatic
  checkpoint, side request, or private tail
- health and confidence of directory observations

Logical references allow several sessions to share one prefix. A sequence-only
backend may emulate a node with a complete snapshot; a paged backend maps it to
a block table with reference-counted blocks.

## Physical block model

The target representation is immutable fixed-size KV blocks plus a private
writable tail:

```text
system + tools + turn 1 + turn 2
  B1       B2       B3       B4

session A -> [B1, B2, B3] + writable A
session B -> [B1, B2, B3] + writable B
anchor T2 -> [B1, B2, B3, B4]
```

Required invariants:

- a shared block is immutable
- a block is reclaimed only when its references and leases reach zero
- a block table is published atomically
- incomplete imports and snapshots are never visible
- generations prevent late completions from mutating recycled storage
- byte accounting includes allocator and cache-class overhead
- eviction removes references first and physical blocks only when unreferenced
- adapter recovery can reconstruct or invalidate logical locations after crash

Block size is backend-dependent. The control plane should not require a global
block size, but it should receive byte and token granularity for cost decisions.

## Retention policy

### Session budgets

Add enforceable defaults for every session:

- one writable session state
- a small bounded number of semantic anchors
- a smaller bounded number of automatic checkpoints
- a retained-byte budget
- an idle TTL
- an absolute maximum TTL unless explicitly pinned
- a cache-creation rate budget to prevent anchor churn

When a newer boundary strictly extends an older compatible boundary for the
same session, the older boundary is a replacement candidate unless it is a
valuable branch point. Keeping every turn forever is not the goal.

Suggested initial sequence-backend defaults, to be validated:

- semantic anchors per session: 3
- automatic checkpoints per session: 2
- automatic interval: 8192 tokens for server workloads
- idle session TTL: 15 minutes
- structural shared-prefix TTL: 60 minutes with reuse-based renewal
- background and side-request anchors: 1 or none unless reused

Defaults must be model- and memory-aware; these values are a test hypothesis,
not constants to bury in code.

### Tenant budgets

Local advisory eviction ranking is not enough for multi-tenant service. Add:

- hard retained-device-byte budget
- hard retained-host-byte budget
- active generation limit
- queue depth and request-rate limits
- cache insertion rate limit
- maximum session count with retained state
- optional share-group budget for organization-wide harnesses

Exceeding a retained-cache budget must reject new cache publication or evict
that tenant's unleased state; it must not reject inference that can safely run
without retaining the result.

### Eviction value

Rank unleased candidates by estimated future value per retained byte:

```text
value =
  probability_of_reuse
  * recompute_cost_saved
  * locality_bonus
  * structural_weight
  / retained_bytes
```

Inputs include recency, frequency, observed return interval, prefix depth,
prefill throughput, tenant policy, queue pressure, transfer cost, and whether
another equivalent location exists. Exact policies can begin as deterministic
weighted ordering and become adaptive only after telemetry is trustworthy.

Never evict:

- a write-leased target
- a read-leased donor
- a block referenced by an active generation unless another reference remains
- state whose physical mutation has started but has not committed or aborted

### Admission under pressure

Use this order:

1. reuse an existing compatible writable state
2. restore or fork from the best local prefix
3. skip low-value checkpoint publication
4. reclaim expired and dominated state
5. evict lowest-value unleased state to low watermark
6. demote to a slower tier when cheaper than recomputation
7. run without retaining optional results
8. queue within the bounded admission limit
9. return 429 with `Retry-After`

## Cache-aware routing

### Directory

Maintain soft-state records keyed by compatibility domain and prefix
fingerprint:

```text
(model_domain, share_domain, prefix_fingerprint)
  -> worker, tier, token_count, bytes, last_seen, lease/load hints
```

Worker heartbeats publish bounded summaries and invalidation generations.
Directory records expire quickly and are never trusted as proof that state
exists. The selected worker revalidates locally before reuse.

Start with exact session affinity and known structural prefixes. A global
fine-grained prefix directory can follow after cardinality and privacy tests.

### Worker scoring

For each compatible worker estimate:

```text
total_cost =
  queue_wait_ms
  + missing_prefill_ms
  + transfer_ms
  + decode_contention_ms
  + memory_pressure_penalty
  + cold_model_load_ms
```

Route to the lowest expected total cost. This allows correct choices such as:

- wait briefly for a worker holding a 40k-token prefix
- recompute a 500-token prefix on an idle worker
- avoid transferring KV when network time exceeds prefill time
- break session affinity when the cached worker is unhealthy

The decision and all estimates must be emitted in request telemetry.

### Failure behavior

- stale directory hit: local miss, then recompute
- worker death: expire locations and retry according to request policy
- partition: workers continue with local policy; global routing degrades to load
  and sticky hashing
- duplicate execution: request IDs and cancellation prevent accidental double
  publication; inference idempotency remains an API-level concern
- rolling upgrade: compatibility versions prevent importing old formats

## Multi-tier cache

Implement tiers only after byte accounting and routing costs are reliable:

1. device or Apple unified-memory hot tier
2. host-memory warm tier on discrete-GPU systems
3. local NVMe cold tier for expensive long prefixes
4. optional remote transfer/cache tier for specialized deployments

Promotion and demotion must be transactional. Transfer is worthwhile only when:

```text
transfer_time + import_time < recompute_time - routing_penalty
```

Measure with real model/backend pairs. Do not assume SSD or network offload is
universally faster than prefill.

KV encryption at rest and in transit, per-tenant keys or isolation domains,
checksums, bounded deserialization, and cache-format validation are mandatory
before persistent or remote tiers are enabled.

## Backend strategy

### llama.cpp / GGUF

Stage A uses public APIs and remains permanently supported:

- sequence continuation
- exact whole-state anchors
- sequence copy/branch where valid
- hybrid/recurrent fail-closed restrictions
- per-session anchor/checkpoint bounds
- exact local telemetry and expiry

Stage B investigates a narrow upstreamable extension exposing:

- physical block identifiers and block-table snapshots
- block reference counts
- copy-on-write sequence forks
- allocation and exact byte telemetry
- block-range export/import where representation permits
- independent logical reference release

Acceptance requires a measured density or latency gain large enough to justify
maintaining the extension. If llama.cpp evolves equivalent public APIs, use
them and delete the patch.

Do not fork model execution, samplers, quantization, or device kernels.

### MLX / MLX-LM

The adapter owns MLX cache objects and must account for cache-class behavior:

- ordinary attention caches may slice, clone, or share immutable arrays
- rotating/sliding-window and recurrent caches cannot claim arbitrary trim
- unified memory changes device/host tier semantics
- older state may be quantized only if output correctness gates pass

First add session budgets, TTL, and cache-location telemetry to existing slots.
Then prototype immutable chunk sharing or copy-on-write in the narrowest MLX
layer that can preserve cache-class correctness.

### Native paged engines

For vLLM, SGLang, TensorRT-LLM, and similar engines:

- let the runtime own block allocation, prefix caching, and kernel integration
- pass stable session/share identities when supported
- ingest runtime hit, reused-token, block, eviction, and pressure telemetry
- route to replicas using cache locality
- use native import/export only through a versioned adapter

Clap should not impose its own physical allocator on these engines.

## Security and privacy

Provider-scale cache sharing creates side channels unless deliberately scoped.

Required controls:

- cache namespace derives from authenticated identity, not request labels
- cross-tenant sharing is disabled by default
- organization sharing requires an explicit share domain and policy
- directory and metrics avoid raw prompt text and raw unkeyed hashes
- external APIs do not reveal whether another tenant populated a prefix
- timing-sensitive cross-tenant hits may require padding or no sharing
- cache export/import is authenticated, encrypted, checksummed, and size-bounded
- released tenant state is cryptographically or physically isolated according
  to the backend's threat model
- admin operations and quota changes are audited
- cancellation releases leases and cannot leave reusable partial state

A security review is a release gate for any cross-tenant or remote-KV feature.

## Observability

Per request:

- selected worker and routing reason
- queue estimate and actual wait
- local/remote cache candidate count
- operation: continue, restore, fork, import, fresh
- planned and realized reused tokens
- prefix and suffix prefill time
- decode time and tokens per second
- transfer bytes/time and source tier
- cache publication accepted/skipped reason
- evicted/demoted bytes and reason
- cancellation and retry outcome

Per worker/model/tenant, with safe cardinality:

- active, queued, and retained sessions
- logical anchors and physical blocks
- referenced, unique, leased, and reclaimable bytes
- device/host/disk cache occupancy and budgets
- hit rate and token-weighted reuse ratio
- prefill time saved
- eviction/demotion/promotion rates by reason
- directory hit accuracy
- cache churn and average object lifetime
- restore/import failures
- working-set estimate
- no-capacity and overload counts

Global hit percentage alone is insufficient. The primary economic indicators are
reused tokens, prefill milliseconds saved, and saved work per retained byte.

## Configuration surface

Configuration should layer through existing Clap TOML/env machinery. Proposed
logical keys:

```toml
[cache]
mode = "sequence"                    # sequence | block | backend_native
session_idle_ttl = "15m"
structural_idle_ttl = "60m"
semantic_anchors_per_session = 3
automatic_checkpoints_per_session = 2
automatic_checkpoint_interval_tokens = 8192

[cache.device]
budget_bytes = 0                     # 0 = derive safely
high_watermark_basis_points = 9000
low_watermark_basis_points = 8000

[cache.host]
enabled = false
budget_bytes = 0

[routing]
enabled = true
node_id = "gpu-router-a"             # optional stable deployment identity
local_replicas = 1                   # 1..8
worker_ttl_ms = 15000
session_ttl_ms = 900000
max_workers = 1024
max_locations = 50000
default_prefill_tokens_per_second = 1000
queue_wait_ms_per_request = 250
pressure_penalty_ms = 1000
cold_load_ms = 5000

[cache.tenants.default]
retained_device_bytes = 0
retained_host_bytes = 0
retained_sessions = 64
cache_insertions_per_minute = 240
```

Backend-specific settings remain under `[llama]`, `[mlx]`, or model overrides
only when they describe physical mechanisms. The generic policy should not be
reimplemented in each config parser.

## Delivery workflow

All implementation happens on one feature branch. Each phase follows this
mandatory loop:

1. implement a bounded phase locally
2. run focused unit, integration, formatting, and type checks
3. review the diff and update this document's status/evidence
4. commit the ready-to-test phase
5. push the branch to GitHub
6. create a fresh GPU pod for phases requiring CUDA validation and explicitly
   expose the Clap HTTP/dashboard port through the provider proxy
7. start Clap on `0.0.0.0`, open the provider's public dashboard URL, and verify
   the dashboard HTML and `/clap/v1/health` before workload testing
8. connect through an interactive terminal/SSH session
9. clone the branch from GitHub; do not copy an uncommitted working tree
10. install Bun and declared system dependencies on the clean pod
11. build Clap and the CUDA worker from source
12. run correctness, saturation, cancellation, and telemetry probes
13. record sanitized commands, hardware, model digest, configuration, and
    measurements in a checked-in validation artifact
14. fix discovered issues locally, commit, push, and pull the updated branch on
    the pod before retesting
15. only mark the phase complete after local and clean-pod gates pass
16. stop and delete idle paid pods after artifacts are collected

Never commit secrets, pod SSH keys, signed URLs, API keys, or unsanitized prompt
content. Pod creation and deletion IDs may be recorded only in ephemeral notes.

## Phased execution plan

### Phase 0 — Baseline and reproducible provider-scale harness

Deliverables:

- this architecture and execution plan
- one branch with explicit phase commits
- a scripted workload with configurable users, sessions, turns, shared-prefix
  size, tool-result size, think time, and concurrency
- deterministic cache trace replay against Rust policy
- clean-pod bootstrap/runbook using Bun and public repository access
- validation artifact schema recording Git SHA, model digest, hardware, driver,
  worker capabilities, environment overrides, and all result paths
- baseline tests on llama.cpp CUDA and local MLX where available

Gates:

- workload distinguishes API requests, human turns, semantic anchors,
  automatic checkpoints, logical entries, and physical bytes
- baseline reproduces hard-ceiling churn without confusing entries with users
- cancellation, overload, cache miss, and worker restart are represented
- no secrets or raw private prompts in artifacts

### Phase 1 — Fleet-safe bounded sequence cache

Status: COMPLETE. Local verification and clean A100 CUDA validation completed
on 2026-08-06 at implementation SHA
`4c9168b3797dbdc1910af8a3cf3205416d5d2485`; the checked-in evidence commit is
`cbda5871ed50edf5ab506e2ed51993823cdfe7dc`. The branch includes total
retained-anchor and automatic-checkpoint caps per non-zero session, an optional
policy-accounted byte cap that stays separate from measured physical memory, a
15-minute default idle TTL, explicit coordinator and adapter session release,
physical-slot reconciliation in llama.cpp and MLX, ABI v6,
lifecycle/publication telemetry, Prometheus series, and provider-shaped
hard-ceiling churn tests. Defaults are four total anchors and two automatic
checkpoints per session; the byte cap defaults to disabled until an operator
supplies a model-appropriate budget. Session zero remains the explicitly
unscoped/structural path and is governed by global and byte budgets. An external
server/admin endpoint for explicit session release remains outside Phase 1.

Local evidence (2026-08-06):

- `bun run hardening:verify` passed end to end
- 554 TypeScript tests passed
- 71 Rust coordinator/retention/ABI/provider-scale tests passed
- 26 native llama.cpp tests passed, including expired-slot mirror reconciliation
- 133 MLX XCTest/Swift Testing cases passed (one physical model probe skipped by
  design when no model fixture is configured)
- native production builds, worker-protocol conformance, generated OpenAPI/web
  drift, bundle checks, structure/ownership gates, and docs checks passed
- provider-scale trace: 100 anchors from one noisy session plus 79 other users
  saturated a 32-entry pool while preserving the four-anchor session bound and
  allowing a fresh post-saturation generation

Deliverables:

- per-session semantic-anchor limit
- per-session automatic-checkpoint limit
- per-session retained-byte accounting and optional hard limit
- idle TTL and explicit session release
- dominated-anchor replacement that preserves valuable branch points
- publication admission: inference may proceed while optional snapshot creation
  is skipped under owner pressure
- config/API/worker-protocol wiring for llama.cpp and MLX
- metrics for publication skip, expiration, per-session occupancy, and churn

Gates:

- thousands of mostly idle logical users cannot monopolize a worker forever
- one tool-heavy session cannot consume the complete retained-entry table
- active and leased state is never expired or evicted
- fresh continuation after saturation remains correct
- lower limits cause predictable recomputation, not failed generation
- llama.cpp CUDA and MLX conformance suites pass

This is the highest-priority implementation phase because it improves fairness
and predictability without requiring distributed infrastructure or custom
kernels.

### Phase 2 — Cache-aware multi-worker routing

Status: COMPLETE. Local verification and clean A100 CUDA multi-replica
validation completed on 2026-08-06 at implementation SHA
`ea508176a8b4d403e811be1eb7d0c33897d4d507`; the checked-in evidence commit is
`f42a407ac48263b52d45a2d4e54778f380c2f609`. The initial deployment has opaque
stable worker IDs, generation-aware heartbeats, a bounded in-memory
session/prefix location directory, compatibility isolation by physical model
and keyed namespace, deterministic sticky fallback, and a cost model combining
queue delay, missing prefill, pressure, and cold-load estimates.
`[routing].local_replicas` enables one router process to manage up to eight
local worker replicas per model. A failed pre-dispatch load invalidates that
worker generation and retries another compatible replica; a stale location can
only cause a local miss and recomputation. Model unload, idle expiry, and
process shutdown synchronously invalidate replica heartbeats and their soft
locations. Request telemetry records the chosen worker, reason, cost
components, candidate count, stale records, and fallback, persists it with
cache decisions, exposes directory diagnostics at `/clap/v1/router`, and
renders it in the dashboard request detail.

Local evidence (2026-08-06):

- `bun run hardening:verify` passed end to end
- 566 TypeScript tests passed, including six deterministic routing-directory
  tests and server/dashboard integration coverage
- all Rust, native llama.cpp, MLX, worker-protocol, bundle, generated-artifact,
  ownership, structure, capability, memory-honesty, and documentation gates
  passed
- `scripts/provider-routing-probe.py` is syntax-checked and verifies two live
  replicas, exact-session locality, overload recomputation, restart
  invalidation, the router diagnostics endpoint, and the public dashboard
- after live validation exposed lifecycle cleanup gaps, the focused
  runtime-router/server suites passed 108 tests and the repository typecheck
  passed with the production-lifecycle unload regression included

Deliverables:

- stable worker IDs and worker heartbeats
- soft session-to-worker directory
- model/compatibility-aware sticky routing
- worker load, queue, pressure, and local reusable-prefix summaries
- routing score comparing queue delay and missing prefill
- retry/fallback on stale directory records and worker loss
- bounded directory cardinality and TTL cleanup
- routing decision telemetry

Initial deployment may use one router process with several local or remote
workers. The metadata protocol must permit a replicated directory later.

Gates:

- follow-up turns prefer the worker owning the longest useful prefix
- an overloaded cached worker loses to an idle worker when recomputation is
  predicted cheaper
- stale metadata never causes a false hit or failed request
- rolling worker restarts drain or invalidate directory records
- tenant identity and model compatibility are always enforced

### Phase 3 — Versioned physical-cache adapter contract

Status: COMPLETE. Local verification and clean A100 CUDA sequence-fallback
validation completed on 2026-08-06 at implementation SHA
`9d1c2320d5b0a6b80d2095f267bc34b5698f0da9`; the checked-in evidence commit is
`c20e765d1afd23cfc973aa0a4e7a369732fb4420`. Rust defines physical adapter
contract v1 independently of coordinator policy. The checked adapter negotiates
the exact contract version, validates operation support and request shape before
physical mutation, requires exact format and transfer identity for import,
rejects invalid outcomes, and invalidates uncertain targets after execution
failure or abort. Physical bytes are represented as known or explicitly unknown
with a reason; a backend that claims exact accounting cannot return unknown
bytes, while the llama.cpp sequence path honestly advertises unknown accounting
and MLX advertises estimated accounting.

The contract operations are `inspect`, `continue`, `restore`, `fork`, `trim`,
`snapshot`, `release`, `export`, `import`, `promote`, and `demote`. Descriptors
also declare sequence/paged kind, physical format version, model compatibility
domain, KV type, optional block granularity, restore granularity, whole-copy or
copy-on-write fork semantics, trim granularity, busy-donor safety, cache tiers,
and optional transfer format. Transfer remains disabled for current adapters.

Current adapters:

- llama.cpp: public sequence API, token trim and copy-on-write prefix fork for
  non-hybrid caches, whole-state fork for hybrid/recurrent caches, optional
  prompt-boundary snapshots, device tier, physical bytes unknown
- MLX: cache-array sequence emulation, whole-state restore/fork and snapshots,
  no trim, device tier, physical bytes estimated
- paged engine skeleton: `inspect` only; all mutations fail closed until a real
  engine integration supplies block granularity and conformance evidence

The worker capability protocol and public loaded-model schema carry the same
strict descriptor. Version 1 is the only accepted version; descriptors with
duplicate operations/tiers, inconsistent trim/snapshot/transfer constraints,
or a live paged kind without block size fail validation. The field remains
optional so an older worker can still use the existing sequence fallback, but
an advertised unknown version is rejected during model load.

Local evidence (2026-08-06):

- 11 Rust adapter conformance tests cover unsupported and malformed operations
  before mutation, crash and abort invalidation, invalid backend outcomes,
  invalidation failure visibility, version skew, import compatibility, honest
  unknown bytes, llama.cpp/MLX parity, and the fail-closed paged skeleton
- all 568 TypeScript tests passed through the full hardening run
- 26 native llama.cpp tests passed with descriptor parity assertions
- 101 MLX tests passed, including two adapter descriptor tests (one unrelated
  physical model probe remained intentionally skipped without a fixture)
- Rust workspace tests, clippy with warnings denied, native production builds,
  protocol checks, cache correctness gates, ownership/structure gates, bundle
  checks, and OpenAPI/web generation passed

Deliverables:

- backend-neutral versioned adapter API
- capability constraints and format identity
- adapter conformance suite
- sequence emulation adapter for current llama.cpp
- MLX adapter parity
- backend-native adapter skeleton for one paged engine
- byte-accurate telemetry contract with explicit unknown values

Gates:

- unsupported operations fail before physical mutation
- adapter crash/abort invalidates all uncertain targets
- public sequence fallback remains fully functional
- protocol version skew fails safely during rolling upgrades

### Phase 4 — Reference-counted logical blocks

Status: IMPLEMENTED AND LOCALLY HARDENED; clean A100 sequence-fallback
regression pending. The Rust core now has immutable exact-token logical blocks,
opaque monotonic block IDs, versioned prefix/session block tables, counted table
references, exclusive write leases, shared read leases, and copy-on-write table
forks. A logical block remains physically resident while any table reference or
lease exists, and invariant validation reconciles every table reference, block
index entry, physical object, lease, and byte counter.

Publish, import, release, eviction, and last-lease reclamation use a narrow
prepare/commit/abort physical-backend transaction. Logical state changes only
after a valid physical outcome commits; prepare failures, commit failures,
invalid outcomes, and stale table/block generations leave references and tables
unchanged. The in-memory fake backend stages exact creates/releases and supports
deterministic failure injection. It is intentionally Rust-only and is not
advertised by llama.cpp, MLX, the C ABI, or the public worker protocol.

Exact-token identity is scoped by namespace and model compatibility domain.
Overlapping tables share one physical block and unique-byte telemetry counts it
once, while logical referenced bytes count every table occurrence. Eviction
plans remove whole versioned tables in deterministic least-recently-used order
and report bytes only when all references to an unleased block are selected.
Sequence traces can be deterministically chunked and replayed into stable block
references, providing the migration bridge without changing the validated
sequence adapters.

Local evidence (2026-08-06):

- 10 focused logical-block tests pass, covering shared COW forks, append
  isolation, exact reference counts, read/write lease exclusion, leased-orphan
  reclamation, stale generations, physical-outcome validation, transactional
  failure rollback, shared-byte eviction, trace replay, namespace/model
  isolation, and conflicting byte metadata
- a generated state-machine property test executes 8,000 deterministic
  publish/append/replace/fork/release/lease/evict operations across 32 seeds and
  reconciles logical and fake-physical invariants after every operation
- all 92 Rust workspace tests pass and workspace clippy passes with warnings
  denied
- `bun run hardening:verify` passes end to end: all 568 TypeScript tests, 26
  native llama.cpp tests, 135 MLX XCTest/Swift Testing cases (one physical model
  probe skipped by design), native production builds, worker-protocol checks,
  bundle and generated-artifact drift checks, ownership/structure/capability/
  memory-honesty gates, and documentation checks are green
- the public sequence adapter contract and all existing coordinator, retention,
  provider-scale, FFI, and ABI tests remain unchanged and green

Deliverables:

- Rust logical block IDs and immutable block metadata
- block tables for prefixes and sessions
- reference counts and read/write leases
- copy-on-write logical fork
- block-level byte accounting and eviction planning
- transactional publish/import/release
- trace migration from sequence entries to logical block references

The first implementation may use an in-memory fake physical backend. This phase
must prove coordinator invariants before coupling to CUDA or Metal storage.

Gates:

- property tests show no referenced or leased block is reclaimed
- overlapping prefixes account shared blocks once
- aborts and stale generations leak no references
- deterministic replay produces stable decisions
- sequence adapters can emulate the contract with honest reduced capabilities

### Phase 5 — Provider-grade llama.cpp physical path

Deliverables:

- benchmark public llama.cpp APIs against required block operations
- propose or implement the smallest upstreamable extension if needed
- physical block-table inspection and exact memory reporting
- copy-on-write prefix forks or the closest proven primitive
- optional block-range export/import only if safe and beneficial
- build-time/runtime fallback to unmodified llama.cpp

Go/no-go rule:

- maintain an extension only if clean A/B tests demonstrate a material gain in
  retained concurrent working set, TTFT, or throughput at equal memory
- reject an extension that requires owning unrelated inference machinery or
  creates fragile version coupling without a fallback

Gates:

- output correctness matches fresh prefill across branch/restore scenarios
- hybrid/recurrent architectures remain fail-closed
- no allocator leaks during churn and cancellation soak
- CUDA and CPU paths pass; Metal behavior is documented if applicable
- density improvement is measured in unique shared bytes, not entry count

### Phase 6 — MLX shared physical cache

Deliverables:

- cache-class-specific immutable chunk representation
- copy-on-write or safe array sharing for ordinary attention caches
- exact restrictions for rotating, sliding-window, and recurrent caches
- unified-memory pressure telemetry
- optional old-KV quantization behind correctness and quality gates

Gates:

- ordinary attention sessions share physical prefix storage measurably
- unsupported cache classes never advertise trim or partial restore
- memory pressure remains bounded during rapid branch churn
- macOS Apple Silicon correctness and soak tests pass

### Phase 7 — Native paged-engine integrations

Deliverables:

- production adapter for at least one of vLLM or SGLang
- stable session/share identities
- native prefix-hit and block-pressure telemetry ingestion
- replica cache-locality routing
- capability-based optional import/export

Gates:

- Clap does not duplicate the engine's physical allocation
- routing improves token-weighted reuse versus round robin
- backend upgrade/version mismatch degrades to fresh execution safely
- native and Clap metrics reconcile within documented tolerances

### Phase 8 — Multi-tier offload and KV transfer

Deliverables:

- host-memory tier for discrete GPUs
- optional local-NVMe tier
- transfer manifests, checksums, encryption, and compatibility validation
- cost model using measured bandwidth and recompute throughput
- transactional promote/demote/import/export
- cancellation and partial-transfer cleanup

Gates:

- transfer is selected only when measured cheaper than recomputation
- corrupted, truncated, oversized, or incompatible objects fail closed
- no raw KV crosses tenant/share boundaries
- device high/low watermarks remain stable during soak
- failure of a lower tier cannot fail inference that can recompute

### Phase 9 — Fleet quotas, autoscaling, and directory resilience

Deliverables:

- enforced fleet tenant budgets
- distributed or replicated soft-state directory
- worker draining and rolling-upgrade semantics
- autoscaling signals based on queue, prefill, decode, cache pressure, and
  working-set churn
- warm-prefix replication for newly added workers
- cache-aware scale-down and graceful location invalidation

Gates:

- tenant load cannot evict protected active state of another tenant
- directory node loss does not break inference correctness
- scale-out reaches useful warm state within a measured SLO
- scale-down drains leases and invalidates locations without false hits
- quota and routing decisions are auditable

### Phase 10 — Disaggregated prefill/decode research

This is optional and begins only when fleet data demonstrates that coupled
workers materially underutilize hardware.

Research:

- separate prefill and decode pools
- KV transfer topology and format
- model-parallel and expert-parallel interaction
- chunked prefill scheduling across pools
- failure/retry semantics after prefill publication

Go/no-go gates:

- measurable fleet utilization or latency gain over simpler cache-aware replicas
- transfer overhead leaves meaningful net savings
- operational complexity has a clear owner and rollback path

## Test matrix

### Workload shapes

- short independent chat sessions
- long returning conversations
- many users with long think time
- organization-wide 10k-40k shared harness
- tool-heavy agent turns with small and very large results
- branches from old boundaries
- side requests and summarization jobs
- prompt growth to context limit
- mixed tenants with noisy-neighbor pressure
- cold fleet startup and rolling restart

### Correctness

For every backend and supported operation compare against fresh prefill:

- greedy token equality where deterministic
- seeded sampling equality where backend guarantees it
- logits tolerance where available
- exact reused-token accounting
- no reuse across compatibility or authorization boundaries
- cancellation leaves no partial reusable object

### Performance

Capture:

- TTFT p50/p95/p99
- end-to-end latency and queue wait
- prompt and decode throughput
- token-weighted reuse
- unique versus referenced KV bytes
- saved prefill milliseconds per GiB-hour retained
- churn, eviction, and promotion rates
- maximum hot sessions at a fixed latency SLO
- energy/cost where hardware telemetry permits

### Soak and chaos

- hours of session creation, idle, return, and expiry
- repeated branch/fork/release cycles
- worker kill during prefill, decode, snapshot, import, and export
- router restart and directory loss
- network delay and partition
- storage full/corrupt for offload tiers
- rolling mixed-version deployment

## Phase evidence format

Each phase adds a dated section or artifact containing:

```text
phase:
git_sha:
branch:
date:
hardware:
accelerator:
driver_runtime:
model_id_and_digest:
backend_and_version:
clap_config:
commands:
local_checks:
pod_checks:
workload:
result_summary:
regressions:
known_limits:
artifact_paths:
pod_deleted:
```

Raw logs should be sanitized and retained only when useful. Summary tables must
link to reproducible commands or scripts.

## Phase 1 validation evidence — 2026-08-06

```text
phase: 1 — fleet-safe session retention
git_sha: 4c9168b3797dbdc1910af8a3cf3205416d5d2485
branch: feat/provider-scale-kv-cache
date: 2026-08-06
hardware: RunPod secure-cloud NVIDIA A100-SXM4-80GB, 81920 MiB
accelerator: CUDA, all 23 model layers and the 44 MiB unified KV pool on CUDA0
driver_runtime: NVIDIA driver 550.127.05; CUDA toolkit 12.8
model_id_and_digest: tensorblock/tinyllama-GGUF tinyllama-Q3_K_M.gguf; sha256 d8025766484965a5499a760af3df0ea8999ab6e247dac21e9e06fef6e85b489a
backend_and_version: llama.cpp 0ed235ea2c17a19fc8238668653946721ed136fd; Clap worker protocol v1
pod_deleted: yes, after evidence push and final local synchronization
```

The clean pod cloned the feature branch at the SHA above, installed Bun
1.3.14 and Rust 1.97.1, built the Rust cache in release mode, and built the
llama.cpp worker for `CMAKE_CUDA_ARCHITECTURES=80`. Dynamic linkage resolved
CUDA 12, cuBLAS, and `libcuda`; the worker reported one A100, assigned every
model layer to CUDA0, and allocated its KV pool on CUDA0. The provider-shaped
probe used a 2,048-cell context and these retention overrides to make pressure
observable with a small public fixture:

```text
CLAP_LLAMA_RETAINED_MAX=32
CLAP_CACHE_MAX_ANCHORS_PER_SESSION=4
CLAP_CACHE_MAX_ANCHOR_BYTES_PER_SESSION=0
CLAP_CACHE_CHECKPOINT_MAX_PER_SESSION=2
CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS=16
CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS=16
CLAP_CACHE_SESSION_IDLE_TTL_MS=20000
```

The production default checkpoint threshold remains larger; the 16-token
threshold was test-only so a 1.1B model could exercise checkpoint and anchor
churn quickly. `CLAP_HOME` was placed on the pod's local root filesystem, not
`/workspace`, because RunPod's FUSE volume reports mode `0777` even after
`chmod`; the cache-identity secret correctly rejected that insecure mount.

Commands executed on the clean clone included:

```sh
bun install --frozen-lockfile
cargo build --manifest-path native/cache/Cargo.toml --release --locked
bun run runtime:llama:vendor
CLAP_CUDA_ARCHS=80 bun run runtime:llama:build
python3 /workspace/clap-validation/provider_gpu_probe.py
cargo test --manifest-path native/cache/Cargo.toml --locked --test provider_scale
cargo test --manifest-path native/cache/Cargo.toml --locked --test coordinator \
  idle_expiry_and_explicit_release_preserve_busy_and_other_sessions -- --exact
```

| Checkpoint | Retained total | Anchors | Evictions | Session-policy evictions | Expired | Budget rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| after one noisy session | 10 | 8 across noisy + warmup sessions | 11 | 11 | 0 | 0 |
| after 44 additional users | 32 | 31 | 89 | 11 | 0 | 0 |
| fresh user + noisy-session return at saturation | 32 | 31 | 93 | 11 | 0 | 0 |
| after the 20-second idle TTL | 5 | 4 | 93 | 11 | 32 | 0 |

All 58 live chat requests completed without an API or worker error. The
dashboard recorded 56 physical cache hits, two misses, 6,112 physical prompt
tokens, and 4,109 reused tokens. A fresh user and the original noisy session
both generated successfully after the pool reached its hard 32-entry ceiling.
Prometheus independently reported 81 anchor publications, 11 per-session
policy evictions, 32 expired slots, and zero session-budget rejections. The
VRAM sampler observed a 1,097 MiB maximum with 56% peak GPU utilization; no
OOM or worker restart occurred.

The two deterministic provider-scale tests passed on the pod. The focused
lifecycle test also proved that idle expiry and explicit internal release do
not reclaim busy state or another session. An external admin/server release
endpoint is intentionally still outside Phase 1. Physical retained bytes
remain unavailable through the pinned llama.cpp public API, so the live
telemetry correctly kept observed retained bytes unavailable while reporting
policy-accounted anchor bytes separately.

Known limits of this checkpoint are the single-GPU, small-model fixture and
sequence-cache semantics. It does not validate multi-GPU transfer, fleet
routing, byte-accurate block sharing, or a public session-release API. Those
remain later phases; no custom llama.cpp block extension is justified by this
checkpoint alone.

## Phase 2 validation evidence — 2026-08-06

```text
phase: 2 — cache-aware multi-worker routing
git_sha: ea508176a8b4d403e811be1eb7d0c33897d4d507
branch: feat/provider-scale-kv-cache
date: 2026-08-06
hardware: RunPod secure-cloud NVIDIA A100-SXM4-80GB, 81920 MiB
accelerator: CUDA, two local llama.cpp worker replicas loaded concurrently
             during the routing workload
driver_runtime: NVIDIA driver 580.126.16; CUDA toolkit 12.8
model_id_and_digest: tensorblock/tinyllama-GGUF tinyllama-Q3_K_M.gguf; sha256 d8025766484965a5499a760af3df0ea8999ab6e247dac21e9e06fef6e85b489a
backend_and_version: llama.cpp 0ed235ea2c17a19fc8238668653946721ed136fd; Clap worker protocol v1
pod_deleted: yes, after evidence commit f42a407 was pushed and synchronized
```

The clean pod cloned the public feature branch, installed dependencies with Bun
1.3.14, built the Rust cache in release mode, and built the pinned llama.cpp
worker for `CMAKE_CUDA_ARCHITECTURES=80`. The server bound to
`0.0.0.0:11435`, RunPod exposed that HTTP port, and external checks reached
both `/clap/v1/health` and the production dashboard HTML before and after the
workload. RunPod's Cloudflare edge rejected Python `urllib` by browser signature
(error 1010), so the deterministic probe ran against the pod-local listener;
the public endpoint itself was independently verified from outside the pod.

The validation configuration used two local replicas and deliberately simple
cost constants so locality and queueing decisions were reproducible:

```toml
[limits]
max_inflight = 16
queue_depth = 64
max_active = 1

[routing]
enabled = true
node_id = "phase2-a100"
local_replicas = 2
worker_ttl_ms = 30000
session_ttl_ms = 900000
max_workers = 16
max_locations = 1000
default_prefill_tokens_per_second = 1000
queue_wait_ms_per_request = 1000
pressure_penalty_ms = 0
cold_load_ms = 0

[llama]
retained_max = 32
context = 2048
max_output = 1024
gpu_layers = 999
```

The final command was:

```sh
CLAP_BASE_URL=http://127.0.0.1:11435 \
CLAP_VALIDATION_MODEL=/root/clap-phase2/models/tinyllama-Q3_K_M.gguf \
PYTHONDONTWRITEBYTECODE=1 python3 scripts/provider-routing-probe.py
```

All 16 requests completed without an API or worker error. Twelve seed sessions
were distributed across two opaque workers. A follow-up with 68 of 71 prompt
tokens available on its owner selected that worker with
`reason=session_locality`, a directory hit, and an estimated 3 ms missing
prefill cost. While a 512-token request occupied that owner, the next follow-up
selected the other replica with `reason=lowest_cost`, no directory hit, and a
105 ms recomputation estimate instead of waiting behind the cached worker.

The probe then unloaded the model and required `/clap/v1/router` to contain no
worker heartbeats or locations. A request after unload recreated the two
candidates but did not trust the old directory: it reported
`reason=sticky_fallback`, `directoryHit=false`, zero matched tokens, and a new
worker generation. This proves the routing directory remains soft state across
unload/restart and that stable worker identity is not confused with a live
worker generation. The final diagnostic snapshot contained two healthy
candidate records and one new post-restart location within the configured
16-worker/1,000-location bounds.

The final server dashboard reported 16 successful requests, zero errors or
cancellations, 13 physical cache hits, three physical misses, 1,561 physical
prompt tokens, and 680 reused tokens. Across the validation session, the GPU
sampler observed 2,187 MiB maximum VRAM while both replicas were loaded, 92%
peak GPU utilization, and 1,095 MiB after the post-unload request left one
replica resident. No OOM or worker crash occurred.

Live validation found and fixed two issues before the final pass: workers now
refresh heartbeats while idle instead of expiring solely because no request is
arriving, and production lifecycle removal now invalidates and shuts down all
replicas rather than only the primary resident key. The final fix has a server
regression test that supplies the lifecycle exactly as the production server
does and asserts that unload empties routing state.

Sanitized pod artifacts were recorded under
`/workspace/clap-phase2/evidence/`, including `provider-routing-probe.log`,
`phase2-summary.json`, `router-final.json`, `server.log`, `vram.log`,
`validated-commit.txt`, the model digest, CUDA build logs, and web build logs.

Known limits are the single-node, single-GPU, small-model fixture and an
in-memory directory. This phase validates placement, overload recomputation,
and lifecycle invalidation, not replicated directory consensus, remote workers,
KV transfer, multi-GPU routing, or comparative throughput against a single
replica. Physical sequence-cache density remains a later measurement; this
result does not by itself justify a custom llama.cpp block extension.

## Phase 3 validation evidence — 2026-08-06

```text
phase: 3 — versioned physical-cache adapter contract
git_sha: 9d1c2320d5b0a6b80d2095f267bc34b5698f0da9
branch: feat/provider-scale-kv-cache
date: 2026-08-06
hardware: RunPod secure-cloud NVIDIA A100-SXM4-80GB, 81920 MiB
accelerator: CUDA, all 23 model layers and the 44 MiB KV buffer on CUDA0
driver_runtime: NVIDIA driver 580.126.16; CUDA toolkit 12.8; compute capability 8.0
model_id_and_digest: tensorblock/tinyllama-GGUF tinyllama-Q3_K_M.gguf; sha256 d8025766484965a5499a760af3df0ea8999ab6e247dac21e9e06fef6e85b489a
backend_and_version: llama.cpp 0ed235ea2c17a19fc8238668653946721ed136fd; Clap worker protocol v1
pod_deleted: yes, after evidence commit c20e765 was pushed and synchronized
```

The clean pod cloned the public feature branch at the SHA above, installed
dependencies with Bun 1.3.14 and Rust 1.97.1, ran the focused Rust contract
suite, and built the pinned llama.cpp worker for
`CMAKE_CUDA_ARCHITECTURES=80`. The worker log assigned all 23 model layers to
CUDA0, reported a 495.44 MiB CUDA model buffer and 44.00 MiB CUDA KV buffer,
and reused CUDA graphs during generation. The loaded-model API independently
reported `compiled=cuda`, `CUDA0`, and the A100 device identity.

The focused clean-pod commands included:

```sh
bun install --frozen-lockfile
cargo test --manifest-path native/cache/Cargo.toml --locked \
  --test adapter_contract
bun run runtime:llama:vendor
CLAP_CUDA_ARCHS=80 bun run runtime:llama:build
bun run build:web
curl -X POST http://127.0.0.1:11435/clap/v1/models/load ...
curl -X POST http://127.0.0.1:11435/v1/chat/completions ...
```

All 11 adapter conformance tests passed on the clean pod. The live descriptor
assertion required contract version 1, sequence kind, llama.cpp
`llama-sequence` format version 1, whole-state restore, copy-on-write fork,
token trim, prompt-boundary snapshots, device tier, explicit unknown physical
byte accounting, and no transfer format or export/import operations. The
assertion passed before workload generation.

Two identical cache-eligible chat requests then completed successfully. The
first was a physical miss and the repeat was a physical hit that reused 60 of
61 prompt tokens. The dashboard reported two successful requests, zero errors
or cancellations, 122 physical prompt tokens, 60 physically reused tokens, one
physical hit, and one physical miss. Runtime telemetry retained one session and
two anchors across three physical sequence slots. Because the public llama.cpp
API does not expose measured per-sequence bytes, `retainedBytes` correctly
remained `null` with source `unavailable`; the separate context-based policy
estimate was 25,165,824 bytes.

The CUDA worker used 1,082 MiB according to the compute-process sample. The
one-second sampler observed 1,091 MiB peak device memory and 13% peak GPU
utilization during this intentionally small correctness workload. No worker
crash, restart, OOM, API error, or queue rejection occurred.

The server bound to `0.0.0.0:11435`, and RunPod exposed that HTTP port. External
checks reached `/clap/v1/health` and the production dashboard HTML before the
workload and again after generation; the final public health response was
`status=ok`, version 0.2.2. This verifies the required public dashboard path in
addition to the pod-local workload listener.

Sanitized artifacts were copied off the pod before teardown under
`/tmp/clap-phase3-evidence/`, including the adapter assertion and descriptor,
focused contract log, CUDA build log, worker launch metadata and stderr,
generation responses, dashboard/runtime summaries, public health/dashboard
responses, VRAM samples, model digest, and per-file checksums. The frozen tar
archive `/tmp/clap-phase3-evidence.tar.gz` has SHA-256
`05621194cb3dad18e08c008c7de20b3ded24ef0b230572e6342ceafad60bb3dd`.
After commit `c20e765` reached the public feature branch, RunPod confirmed pod
`ao17x4nhl0z5bg` deleted and a subsequent lookup returned `404 not_found`.

Known limits are the single-node, single-GPU, small-model fixture and public
sequence-cache semantics. This validates contract negotiation, capability
publication, honest unknown byte accounting, and fallback generation; it does
not validate a native paged engine, exact physical block bytes, remote KV
transfer, multi-GPU movement, or comparative throughput. Those remain later
phases, and this result does not justify a custom llama.cpp block extension by
itself.

## Success criteria

Clap reaches provider-scale cache maturity when:

- user population is decoupled from resident cache-entry count
- a bounded hot working set remains stable under high session churn
- common prefixes are physically stored once on capable backends
- returning sessions are routed using locality and load rather than blind round
  robin
- every cache operation is capability-checked and transactional
- cache identity prevents model, template, adapter, and tenant contamination
- byte accounting is honest and reconciles with backend memory within a known
  tolerance
- failures always degrade to recomputation or bounded overload responses
- multiple runtime families share one policy and telemetry model
- clean-hardware validation demonstrates better TTFT/throughput or lower memory
  at equal correctness, not merely higher nominal hit counts

## Explicit non-goals

- caching every user forever
- treating 128 logical entries as 128 users
- building a new general-purpose inference engine
- replacing native paged attention in engines that already implement it well
- remote KV transfer before measured cost and security controls exist
- cross-tenant sharing by default
- persistent prompt or KV storage without explicit policy
- hiding cache misses, evictions, or unsupported operations from telemetry

## Immediate next action

Commit and push the locally hardened Phase 4 logical-block checkpoint, then
validate the focused Rust property/transaction suite and live CUDA sequence
fallback from a clean A100 pod with the HTTP port exposed and the public
dashboard verified first. Preserve the Phase 3 descriptor and do not advertise
paged support, begin remote transfer, or add a custom llama.cpp block extension.
