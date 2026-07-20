# Rust Cache Coordinator Plan

Status: proposed architecture. Scope: give Clap one cache policy and prefix-management foundation across llama.cpp and MLX while leaving physical KV tensors inside their native inference engines.

## Decision

Clap should add an embedded Rust library that coordinates logical KV-cache state for every resident model worker.

The Rust library should own:

- exact token-prefix indexing
- slot and anchor metadata
- cache planning and leases
- session, project, agent, and tenant policy
- admission and eviction decisions
- normalized cache telemetry
- backend-independent policy tests

The Rust library should not own:

- llama.cpp sequence memory
- MLX `KVCache` objects
- Metal, CUDA, or CPU tensor allocation
- attention or recurrent-state kernels
- model tokenization or chat templates
- a cross-backend KV tensor format

The operating rule is:

> Rust decides and indexes. The backend executes and owns. Transactions keep both sides synchronized. TypeScript supplies intent and exposes results.

"Shared cache" means one policy implementation linked into both workers. It does not mean MLX and llama.cpp can exchange live KV tensors.

## Product Goal

Clap should efficiently serve many related conversations on one machine, especially organizational workloads where requests often share expensive prefixes:

- the same company-wide agent harness
- the same safety and behavioral instructions
- the same tool definitions
- project-specific additions to a common harness
- separate planner, builder, reviewer, and research agents
- multiple conversations under each agent
- short side requests branching from an active conversation

The cache should automatically discover exact shared token prefixes, retain the most valuable boundaries, and select the cheapest legal reuse mechanism for each backend.

A user should not need to understand KV slots. Administrators should be able to control budgets, isolation, and policy, while cache behavior remains visible and explainable.

## Existing Foundation

Clap already has substantial caching behavior that the coordinator must preserve.

### Model files

Downloaded model artifacts are persisted under:

```text
$CLAP_HOME/models/huggingface/
```

This is model storage, not an inference KV cache. GGUF files and MLX model directories survive process restarts; warm prompts do not.

### TypeScript control plane

The Bun/TypeScript server currently owns:

- HTTP and OpenAI-compatible APIs
- model download, inventory, and resolution
- resident worker lifecycle
- request routing and streaming
- cache telemetry collection
- metrics and dashboard integration

`packages/runtime-router` communicates with workers over JSON lines. It receives cache results such as hit, reused token count, reuse kind, reuse scope, and slot.

### llama.cpp worker

The C++ worker currently provides:

- resident model and context loading
- multiple sequence slots, defaulting to 16
- exact token-level longest-prefix matching
- same-slot continuation
- cross-slot prefix branching with `llama_memory_seq_cp`
- suffix removal with `llama_memory_seq_rm`
- dedicated anchors for shared prefixes
- LRU slot recycling
- proactive idle-slot eviction under KV pressure
- continuous batching
- optional unified KV storage and KV quantization
- special handling for recurrent and hybrid models

### MLX worker

The Swift worker currently provides:

- resident MLX model loading
- multiple KV slots, defaulting to 4
- exact token-level longest-prefix matching
- same-slot continuation
- cross-slot cache copy and trim when supported
- anchors for rotating and non-rewindable caches
- prompt- and conversation-boundary snapshots
- LRU slot recycling
- interleaved concurrent generation
- optional KV quantization
- special handling for sliding-window, recurrent, and hybrid caches

### Existing behavior to preserve

The Rust coordinator must not regress:

- exact token-prefix correctness
- cache reuse from idle or safe busy donors
- same-conversation continuation
- shared-prefix branching
- prompt-boundary restoration after decode
- anchor behavior for non-trimmable caches
- cancellation and failure cleanup
- continuous or interleaved batching
- backend-specific context limits
- cache quantization configuration
- current cache telemetry

## Target Architecture

```text
Clap clients
    |
    v
TypeScript control plane
- API and streaming
- model routing
- tenant/project/session identity
- configuration and metrics
    |
    | JSON-lines request with high-level cache hints
    v
+-----------------------------+  +-----------------------------+
| C++ llama worker            |  | Swift MLX worker            |
| - template and tokenize     |  | - template and tokenize     |
| - schedule inference        |  | - schedule inference        |
| - llama cache adapter       |  | - MLX cache adapter         |
|            |                |  |            |                |
|   embedded Rust manager     |  |   embedded Rust manager     |
|   same library/semantics    |  |   same library/semantics    |
|            |                |  |            |                |
| - llama_memory operations   |  | - KVCache operations        |
| - llama.cpp KV tensors      |  | - MLX/Metal KV tensors      |
+-----------------------------+  +-----------------------------+
```

Each resident model worker has its own Rust manager instance. Managers do not share live state across workers or processes.

## Ownership Boundaries

### TypeScript owns intent

TypeScript should provide information that exists above the model runtime:

- tenant or organization namespace
- project identity
- harness identity and revision
- agent role
- conversation/session identity
- interactive or background priority
- whether the request is a side request
- administrative budgets and policy settings

These are policy hints. They never override exact token matching.

### Rust owns logical cache policy

Rust should be the source of truth for committed logical cache metadata:

- which exact token sequence a slot represents
- which slots are busy, leased, idle, or anchored
- which prefixes are materialized by physical backend state
- which donor and target should be used
- how much of a prompt can safely be reused
- which entries should be retained or evicted
- how cache outcomes are classified and measured

### Native adapters own mechanism

The C++ and Swift adapters should translate plans into backend operations.

The llama adapter owns calls such as:

- clear sequence
- remove a suffix
- copy a full or partial sequence
- create an anchor sequence
- report physical KV occupancy when available

The MLX adapter owns operations such as:

- copy cache objects
- trim supported cache layers
- capture exact boundary snapshots
- restore prompt or conversation boundaries
- report cache offsets and physical memory when available

Adapters report actual outcomes. They do not independently choose donors, targets, LRU victims, or anchor policy.

### Engines own physical state

llama.cpp and MLX remain authoritative for:

- tensor layouts
- device memory
- cache allocation
- copy and trim legality
- sliding-window rotation
- recurrent state
- decode and prefill execution

Rust never receives or dereferences engine object pointers.

## Core Rust Components

The Rust workspace should conceptually contain:

```text
native/cache/
  Cargo.toml
  crates/
    clap-cache-core/      # pure policy and data structures
    clap-cache-ffi/       # stable C ABI
  include/
    clap_cache.h
  tests/
    traces/               # backend-independent policy traces
```

The build should produce a static library and generated C header suitable for both the C++ build and a SwiftPM C target.

### Cache domain

One cache manager represents one loaded model and one physical worker context.

A domain has an epoch that changes whenever the model or physical cache is globally reset. Plans from an older epoch are invalid.

A cache namespace fingerprint should include all properties that affect token identity or physical compatibility:

- backend
- model/checkpoint digest
- tokenizer digest or revision
- chat-template digest or revision
- KV format and quantization
- context and RoPE configuration
- backend cache-layout version
- tenant isolation namespace

Two entries can only reuse one another inside a compatible namespace.

### Logical slot record

A logical slot should track:

- stable slot ID
- monotonically increasing generation
- exact resident token sequence or radix terminal
- actual resident length confirmed by the backend
- state: empty, session, prompt boundary, or anchor
- busy and lease state
- last-used time or logical clock
- session, project, harness, and agent labels
- semantic scope for telemetry
- estimated physical cost
- observed reuse frequency and saved prefill cost
- current backend capabilities

Semantic labels influence retention and reporting. Exact tokens remain the correctness condition.

### Prefix index

Use a compressed radix tree over model token IDs.

```text
organization base + shared harness
    |
    +-- project A additions
    |      |
    |      +-- planner agent
    |      +-- builder agent
    |      +-- reviewer agent
    |
    +-- project B additions
    |      |
    |      +-- planner agent
    |      +-- builder agent
    |
    +-- generic research agent
```

The tree represents logical token sharing. Only nodes backed by a committed physical cache entry are reusable.

A radix node does not automatically own KV state. The coordinator may decide to materialize an important node as an anchor, but the backend must capture that exact boundary first.

At the current 4-16 slot sizes, a radix tree is not needed for lookup speed. Its value is a durable representation of shared organizational prefixes, multiple anchors, value-based retention, and larger future working sets.

## Transactional Planning

Rust metadata and backend state must never drift.

Every cache operation should follow a plan/execute/commit contract:

```text
worker templates and tokenizes request
    |
    v
Rust finds prefix, reserves slots, and returns a plan
    |
    v
native adapter validates generations and executes operations
    |
    +-- success --> commit actual state to Rust
    |
    +-- failure --> abort plan and invalidate uncertain target state
```

A plan may contain operations such as:

- continue an idle session slot
- trim a target to an exact prefix
- branch a donor prefix into a target
- restore a whole-state anchor
- clear and use a fresh slot
- evict one or more idle entries
- capture an anchor at a future prefill boundary

Planning should:

- lease mutation targets exclusively
- read-lease donors
- capture slot generations and the domain epoch
- avoid holding Rust locks during engine work
- reject stale plans before execution
- release every lease on commit or abort

Commit should record only the state the backend confirms actually exists. A failed or partially applied mutation makes the target invalid unless the backend can prove its exact resulting state.

Global cache reset, unload, or model reload increments the domain epoch and invalidates all plans and entries.

## Backend Capability Model

The coordinator must choose plans from explicit capabilities rather than assuming both backends behave alike.

Capabilities may include:

- partial suffix trim
- partial prefix branch
- whole-state copy
- safe copy from a busy donor
- zero-copy or shared-cell branch
- prompt-boundary snapshot
- sliding-window behavior
- recurrent or hybrid state
- unified physical storage
- reliable physical resident length
- KV quantization

Capabilities may vary by model, cache layer, slot, and current state. For example, an MLX cache may become non-rewindable after its window rotates.

The adapter should report current capabilities and failed operations. Rust should fall back safely rather than treating unsupported reuse as an error.

Expected examples:

| Backend state | Preferred behavior |
|---|---|
| llama attention with unified KV | partial branch, often sharing physical cells |
| llama recurrent/hybrid | exact whole-state continuation or anchor; full prefill when rewind is unavailable |
| MLX trimmable attention | copy and trim to the shared boundary |
| MLX rotating/non-trimmable cache | restore an exact boundary anchor or prefill fresh |

Consistent semantics do not require identical physical plans.

## Organizational Prefix Model

Organizational prompts usually form a hierarchy rather than one universal system prompt.

A useful conceptual composition is:

```text
stable organization instructions
+ shared harness and tool contract
+ project-specific instructions
+ agent-role instructions
+ dynamic task context
+ conversation history
```

The radix index naturally represents this hierarchy when the final tokenized prompts preserve that ordering.

### Shared organization harness

If many users share an identical company harness and tool schema, that prefix is a strong anchor candidate because it has:

- high reuse frequency
- high prefill cost
- broad fan-out across sessions
- relatively low invalidation frequency

The coordinator should retain it longer than an ordinary conversation slot.

### Minor project variations

When project prompts differ only after a common base, the base remains reusable and each project becomes a branch.

```text
shared harness
    +-- project A policy and repository context
    +-- project B policy and repository context
    +-- project C policy and repository context
```

If the variation appears near the start of the prompt, causal attention prevents reusing matching text that occurs after the divergence. Prefix reuse cannot skip a differing section and resume later.

Prompt construction should therefore place the most stable and widely shared material first, while preserving intended model behavior and template requirements.

### Multiple agent prompts

Planner, builder, reviewer, and research agents should become sibling branches under their common harness or project prefix.

```text
org + harness + project
    +-- planner instructions
    +-- builder instructions
    +-- reviewer instructions
    +-- research instructions
```

Frequently used agent branches may each earn their own anchor. Rare roles should normally branch from the common project or harness anchor and prefill their unique suffix.

Agent labels are useful for policy and metrics, but the coordinator still verifies exact token prefixes.

### Tool schemas

Large stable tool schemas can dominate prefill cost. They should be treated as part of a versioned harness identity.

A tool addition, removal, reorder, or description change may alter tokens and naturally creates a new branch. Old tool-schema anchors should age out rather than being globally flushed unless a namespace revision explicitly changes.

### Side requests

A summarization, planning, validation, or tool-repair request may branch from an active conversation while the main request is still generating.

The coordinator should:

- allow a busy slot to be a read-only donor only when the backend supports it
- never destructively trim a busy donor
- allocate a separate target for the side request
- score side-request cache retention lower unless it becomes a continued session
- preserve the primary interactive conversation under pressure

## Prompt Structure And Boundaries

Cache correctness is based on the exact final token sequence after the native chat template runs.

High-level request metadata should describe intent:

```json
{
  "cache": {
    "namespace": "org-acme",
    "project": "payments",
    "harness": "coding-v4",
    "agent": "planner",
    "session": "conversation-123",
    "priority": "interactive",
    "side_request": false
  }
}
```

The native worker should provide Rust with:

- final token IDs
- known stable boundary indices when they can be determined reliably
- requested output reserve
- current backend capabilities

Useful boundaries include:

- end of organization base
- end of harness/tool schema
- end of project instructions
- end of agent instructions
- end of prompt before generation
- stable conversation continuation boundary

Boundaries improve anchor placement and telemetry, but an incorrect label must never permit reuse beyond exact token equality.

The system should avoid depending on fragile text searches for semantic boundaries. Boundary information should come from structured prompt composition, template-aware differential tokenization, or exact worker-observed snapshot positions.

## Cache Policies

Policy should be configurable but deterministic and testable.

### Reuse policy

A candidate donor is eligible only when:

- its namespace is compatible
- its committed token sequence exactly matches the prompt prefix
- its physical state is confirmed valid
- the backend can legally execute the required operation
- its leases permit donation
- reuse exceeds a configurable minimum value

The selected donor should maximize expected saved work after considering:

- reusable token count
- measured or estimated prefill cost per token
- copy cost
- physical memory cost
- whether the donor can be continued in place
- whether the donor is already busy
- fallback risk for the backend/cache type

The longest prefix is usually best, but not always. A shorter zero-copy llama branch may be cheaper than a deep MLX cache copy with poor expected reuse.

### Anchor promotion policy

Not every prefix should become a dedicated anchor. Promotion should be based on observed value:

- prefix token length
- reuse count and fan-out
- prefill time saved
- recency
- expected future demand
- physical snapshot cost
- project or organization priority
- invalidation frequency

Likely anchor classes are:

- organization harness
- harness plus tool schema
- popular project base
- popular project-agent combination
- rolling conversation prompt boundary

Anchors should be demoted when their retained value falls below competing entries.

### Eviction policy

Eviction should be value-aware, not only LRU.

Busy and leased entries are ineligible. Among idle entries, the score should account for:

- recency
- reuse frequency
- prefix length and prefill cost
- number of dependent branches
- anchor scope
- physical bytes retained
- whether storage is shared or copied
- tenant/project quota pressure
- request priority
- cost to reconstruct

LRU remains a safe fallback when cost information is unavailable.

Anchors should be protected, not immortal. A stale 30k-token project anchor should not permanently starve active conversations.

### Admission and capacity policy

The coordinator should combine logical demand with backend-reported physical limits:

- context capacity
- output reserve
- active sequence count
- estimated bytes per token
- available memory budget
- physical occupancy when available
- cost of copies versus shared cells

Logical token counts must not be treated as exact physical bytes. llama unified branches may share cells, while MLX copies may duplicate memory.

The backend remains authoritative when an allocation or operation fails.

### Isolation policy

Default reuse should be restricted to the same tenant namespace.

Even exact-token reuse across organizations can reveal timing or cache-presence information. Cross-tenant reuse should require an explicit administrative mode and should never apply to private prompt snapshots by default.

Per-tenant and per-project budgets should prevent one workload from occupying the full cache.

### Policy profiles

The final system may expose profiles such as:

- `balanced`: value-aware anchors and eviction
- `interactive`: protect active conversations and TTFT
- `throughput`: favor widely shared long prefixes
- `private`: session-local reuse only
- `batch`: lower retention and background priority

Profiles should compile into explicit settings rather than hiding undocumented behavior.

## Stable C ABI

The Rust library should expose a versioned C ABI with opaque handles.

Conceptual API:

```c
clap_cache_t *clap_cache_create(...);
clap_cache_status_t clap_cache_plan(..., clap_cache_plan_t **out);
clap_cache_status_t clap_cache_commit(...);
void clap_cache_abort(...);
clap_cache_status_t clap_cache_advance(...);
clap_cache_status_t clap_cache_reset(...);
void clap_cache_destroy(...);
```

ABI requirements:

- fixed-width IDs and token types
- version and `struct_size` on public structures
- caller-provided slices as pointer plus length
- explicit ownership and destroy functions
- opaque manager and plan handles
- numeric status codes
- no Rust panic across FFI
- no C++ exception or Swift error across FFI
- no engine pointers stored or dereferenced by Rust
- no backend callback while a Rust lock is held

Swift should consume the same C header through a SwiftPM C target or module map.

## Concurrency Model

The coordinator should be thread-safe even if a worker currently confines most scheduling to one executor.

Rules:

- one exclusive writer lease for every mutation target
- read leases for donor slots
- no eviction of leased or busy entries
- optional busy-donor reads only when explicitly supported
- short lock duration during lookup and reservation
- engine execution outside coordinator locks
- stale-plan detection through domain epoch and slot generations
- deterministic handling of duplicate concurrent anchor creation
- no GPU scheduling or request fairness inside Rust

Worker schedulers continue controlling continuous batching, interleaving, and decode fairness.

## Failure And Recovery

Safe degradation is always full prefill, eviction, or queueing; never silent state reuse.

Expected behavior:

- unsupported operation: abort plan and replan or prefill fresh
- failed trim or copy: invalidate the uncertain target
- decode failure: commit only a backend-confirmed stable boundary
- cancellation: preserve only the exact state the backend confirms
- worker crash: discard the in-process manager with the physical cache
- model reload: increment epoch and clear all logical state
- stale transaction: reject and replan
- metadata/engine mismatch: invalidate, do not infer recovery
- Rust panic: contain at FFI boundary and fail closed

The current worker restart and crash-isolation behavior must remain intact.

## Observability

Both backends should emit the same cache decision record:

```json
{
  "hit": true,
  "reused_tokens": 1832,
  "reuse_kind": "branch",
  "reuse_scope": "project",
  "namespace": "org-acme",
  "harness": "coding-v4",
  "agent": "planner",
  "donor_slot": 0,
  "target_slot": 3,
  "evicted_slots": [],
  "decision_us": 34,
  "planned_reuse_tokens": 1832,
  "realized_reuse_tokens": 1832,
  "fallback": null
}
```

Metrics should cover:

- hit and miss count by model/backend/scope
- reused tokens and estimated prefill time saved
- slot, branch, and anchor reuse
- planned versus realized reuse
- operation failures and full-prefill fallbacks
- anchor promotions and demotions
- evictions and estimated reconstruction cost
- active slots, anchors, leases, and prefix nodes
- physical occupancy where the backend can report it
- decision latency
- tenant/project budget pressure

Metrics should avoid exposing raw private prompt tokens or content.

## Persistence Extension

Live coordination should not depend on persistence, but the architecture should leave room for it.

Rust could later own a snapshot catalog and compatibility manifest. The backend would serialize and restore opaque physical state.

A useful snapshot identity must include:

- model and tokenizer fingerprints
- chat-template revision
- exact token-prefix digest
- backend and backend-library version
- cache format and KV quantization
- context/RoPE configuration
- checksum and byte size
- tenant security scope

Snapshots are backend-specific and generally version-specific:

- llama snapshots restore only into compatible llama workers
- MLX snapshots restore only if MLX exposes a stable complete serialization contract
- no snapshot is portable between llama and MLX

Persistent shared harnesses may contain proprietary instructions and tool definitions. Persistence must therefore be opt-in with restrictive permissions, quotas, TTL/deletion controls, and an encryption-at-rest path.

Persisting only Rust token metadata does not preserve prefill work and should not be described as persistent KV caching.

## Configuration Surface

The final configuration should support global defaults and per-model overrides. A possible shape is:

```toml
[cache]
policy = "balanced"
min_reuse_tokens = 16
max_anchors = 8
namespace_isolation = "tenant"

[cache.capacity]
memory_budget_percent = 25
max_slots = 32

[cache.eviction]
recency_weight = 1.0
reuse_weight = 2.0
prefill_cost_weight = 2.0
memory_cost_weight = 1.0

[models."org/model".cache]
policy = "throughput"
max_anchors = 16
```

Configuration names and defaults should remain backend-neutral where semantics match. Backend-specific physical controls such as KV type can remain in their existing sections.

Policy changes that affect only Rust metadata may apply live. Changes to physical context allocation or KV representation require worker drain and reload.

## Testing Foundation

The coordinator should be testable without loading a model.

### Deterministic traces

A trace describes:

- backend capabilities
- existing slots and exact tokens
- busy/lease state
- request tokens and identity
- available capacity
- expected semantic decision

The same trace can produce different legal physical plans under different capability sets while preserving the same correctness rules.

### Property-based invariants

Tests should enforce:

- reused tokens are always an exact prompt prefix
- reuse never crosses namespaces or epochs
- busy donors are never mutated
- leased entries are never evicted
- anchors are never extended as sessions
- failed plans do not publish metadata
- commit cannot exceed backend-confirmed resident length
- stale slot generations cannot commit
- eviction always has a deterministic fallback
- radix insert/remove operations preserve prefix lookup correctness

### Adapter conformance

Each native adapter should run a shared scenario suite covering:

- fresh prefill
- exact session continuation
- partial branch
- anchor restore
- non-rewindable fallback
- busy donor
- cancellation
- failed copy/trim
- model reset
- cache pressure and eviction

Production cache-decision traces should be replayable against new policy versions before rollout.

## Security And Privacy

The coordinator handles token IDs rather than text, but token sequences still encode private content.

Requirements:

- no raw token sequence in normal logs or metrics
- tenant namespace isolation by default
- bounded in-memory metadata
- prompt fingerprints use keyed or cryptographic hashes where exported
- snapshot files inherit strict credential-like permissions
- cache inspection APIs require administrative authorization
- explicit deletion by tenant, project, model, and session
- cache state disappears when a worker exits unless persistence is enabled

## Success Criteria

The architecture is successful when:

- llama and MLX use one cache-policy implementation
- all existing cache reuse behavior remains available
- backend-specific mechanisms remain independently optimized
- exact-token correctness is enforced centrally
- organizational harness, project, and agent prefixes can coexist
- valuable shared prefixes survive ordinary session churn
- cache pressure degrades through eviction or prefill, not crashes
- cache outcomes have identical meanings across backends
- policy is deterministic, traceable, and testable without GPUs
- adding a new runtime requires an adapter and capability declaration, not a new policy implementation
- Rust metadata never claims physical state that the backend has not confirmed

## Explicit Non-Goals

This design does not attempt to:

- make llama and MLX KV tensors interoperable
- replace llama.cpp or MLX memory allocators
- implement custom attention kernels
- guarantee persistent KV support
- cache arbitrary matching prompt fragments after a divergence
- make semantic similarity sufficient for KV reuse
- move tokenization into Rust
- move worker scheduling into Rust
- put TypeScript in the token-level planning loop

## Product Questions To Refine

The foundation supports these policy decisions without requiring an architectural rewrite:

- Which prefix order best balances organization, project, and agent reuse?
- Should tool schemas belong to the organization harness or project branch?
- How many project and agent anchors deserve protection?
- Should interactive and batch workloads use separate budgets?
- When is an MLX cache copy more expensive than re-prefill?
- Should side requests inherit the parent session's priority?
- What tenant isolation mode should single-user local installs use?
- Which prefixes, if any, are safe and valuable enough to persist?
- How should administrators inspect cache value without exposing prompt content?

These are policy choices. The Rust coordinator, transactional adapter boundary, exact-token radix index, and namespace model are the stable foundation beneath them.
