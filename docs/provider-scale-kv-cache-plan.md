# Provider-Scale KV Cache Plan

Status: proposed. This plan extends `docs/scale-plan.md` and
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

[cache.routing]
mode = "local"                       # local | sticky | cache_aware
directory_ttl = "30s"
max_locality_wait = "250ms"

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
6. create a fresh GPU pod for phases requiring CUDA validation
7. connect through an interactive terminal/SSH session
8. clone the branch from GitHub; do not copy an uncommitted working tree
9. install Bun and declared system dependencies on the clean pod
10. build Clap and the CUDA worker from source
11. run correctness, saturation, cancellation, and telemetry probes
12. record sanitized commands, hardware, model digest, configuration, and
    measurements in a checked-in validation artifact
13. fix discovered issues locally, commit, push, and pull the updated branch on
    the pod before retesting
14. only mark the phase complete after local and clean-pod gates pass
15. stop and delete idle paid pods after artifacts are collected

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

Status: IN PROGRESS; substantial local implementation complete and GPU
validation intentionally deferred until the phase is consolidated. The branch
now includes total retained-anchor and automatic-checkpoint caps per non-zero
session, an optional policy-accounted byte cap that stays separate from
measured physical memory, a 15-minute default idle TTL, explicit coordinator
and adapter session release, physical-slot reconciliation in llama.cpp and
MLX, ABI v6, lifecycle/publication telemetry, Prometheus series, and
provider-shaped hard-ceiling churn tests. Defaults are four total anchors and
two automatic checkpoints per session; the byte cap defaults to disabled until
an operator supplies a model-appropriate budget. Session zero remains the
explicitly unscoped/structural path and is governed by global and byte budgets.
An external server/admin endpoint for explicit session release and a clean CUDA
pod run remain before the phase is complete.

Local evidence (2026-08-06, branch checkpoint pending):

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

Start Phase 0 and Phase 1 on a feature branch:

1. establish the provider-scale workload and baseline artifact
2. add per-session anchor/checkpoint budgets and idle expiry to the Rust policy
3. wire configuration and telemetry through llama.cpp and MLX adapters
4. test locally
5. push the ready-to-test commit
6. validate a clean branch clone on an A100-class pod
7. commit the sanitized evidence and pull the final branch state from GitHub

Do not start a custom llama.cpp block extension until bounded sequence-cache
behavior and cache-aware routing have reliable measurements. Those two phases
provide immediate multi-user value and the evidence needed to decide which
low-level extension is actually worth maintaining.
