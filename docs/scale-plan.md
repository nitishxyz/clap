# Clap at Scale: Org Deployment Plan

Status: draft. Scope: make a single clap server on one machine serve an
organization (10-1000 clients) reliably. Multi-machine routing/load balancing
is intentionally deferred; the design here must not preclude it.

## Where we are

Verified working today (single box, single GPU, RunPod 96GB test rig):

- OpenAI-compatible API (chat, streaming, tools), Ollama compat, dashboard
- GGUF via llama.cpp worker (CUDA auto-provisioned) and MLX via Swift worker
- Multi-session KV slot cache: per-slot prefix matching, LRU recycling,
  unified KV pool, proactive idle-slot eviction under pressure, ingest
  self-heal as last resort
- Per-model resident workers with keep-alive lifecycle and crash isolation
- Per-model context: model's trained context by default with halving fallback
  when the KV allocation cannot fit; `CLAP_LLAMA_CONTEXT` pins explicitly
- Opt-in KV cache quantization: `CLAP_LLAMA_KV_TYPE=q8_0|q4_0|f16` (default f16)
- UTF-8-safe streaming from byte-level BPE workers

Known constraint (the scaling wall): each model worker decodes one request at
a time; concurrent requests queue behind it.

## Design tenets

1. The heavy machinery (attention KV storage, batched decode) belongs to the
   runtimes (llama.cpp, MLX). We own policy, not kernels.
2. Both backends get the same policy features unless physically impossible;
   the worker JSON-lines protocol is the contract that keeps them swappable.
3. Every capacity policy is admin-configurable, observable, and defaults sane.
4. Degradation is graceful: evict/re-prefill/queue — never crash, never
   silently truncate.

## Tier 1 — Concurrency correctness

### 1. Continuous batching (llama worker) — the big one
Restructure the worker generation loop from request-at-a-time to a scheduler
step: all active sequences advance one token per `llama_decode` batch; prefill
chunks interleave with decode steps. llama.cpp natively supports multi-
sequence batches (this is llama-server's `--parallel` mode); our slots already
map 1:1 to `seq_id`s.

- Protocol: worker accepts overlapping `chat` requests, each keyed by `id`
  (already true); emits per-id token streams (already true). Only the
  serialization bottleneck inside the loop goes away.
- Server: resident router must multiplex requests to a busy worker instead of
  awaiting exclusivity.
- Expected effect: 10-50 concurrent sessions per model on one GPU.
- MLX: same restructuring later via `BatchedKVCache` (mlx-lm has batched
  generation); acceptable for MLX to lag one phase behind.

### 2. Admission control
Before ingest: check `prompt_tokens + max_tokens` against per-session ctx cap
and pool capacity. Reject with a structured 400 (includes actual numbers) or
trim per policy. No more mid-ingest failures for oversized prompts.

### 3. Queue fairness + backpressure
Bounded per-model queue with per-client (API key) fair scheduling. When
saturated: 429 + `Retry-After`. Queue depth and wait time exposed in metrics
and dashboard.

### 4. Worker watchdog
Auto-restart crashed workers with exponential backoff; crash counters in
dashboard; alert threshold. A native fault must never require manual
`server stop/start`.

## Tier 2 — Memory economics (users-per-GPU)

### 5. Shared-prefix KV dedup
Org harnesses share one system prompt + tool schema (10-40k tokens) across
all sessions of a project. Store that prefix KV once (radix/prefix tree over
token ids), sessions branch from it; new session prefill skips the shared
prefix entirely.

- Data structure pairs naturally with continuous batching sequences.
- llama.cpp primitive: `llama_memory_seq_cp` (copy-on-branch), or the
  llama-server prompt-cache approach.
- Wins: near-instant TTFT on session start, massive KV savings — the single
  most org-shaped optimization we have.

### 6. Adaptive capacity policy
Replace fixed `4 slots` with derived values: measure per-token KV cost at
load, then `slots ≈ f(VRAM budget, per-session ctx cap, expected sessions)`.
Admin overrides in config. Per-session context caps (`max_session_ctx`)
instead of promising train_ctx (262k) to every session.

### 7. KV quantization surfacing
Worker flag exists (`CLAP_LLAMA_KV_TYPE`). Add:
- Config: `[models.<name>] kv_type = "q8_0"` and a global default
- Admin API: `PATCH /clap/v1/models/:id/settings` — applies on next worker
  (re)start since KV type is fixed at context creation; endpoint triggers a
  graceful reload (drain, restart, warm)
- Dashboard: per-model settings panel with the same semantics

### 8. Session-aware eviction (needs Tier 3 identity)
Eviction ordered by policy, not just LRU: idle time, client priority
(interactive > batch), per-key quota pressure. Eviction cost is only a
re-prefill; report evictions per session in metrics.

## Tier 3 — Multi-tenant operations

### 9. AuthN/AuthZ
API keys (`Authorization: Bearer`): hashed at rest, per-key rate limits and
token quotas, key management via CLI + admin API + dashboard. An org box on
`0.0.0.0` without auth is a non-starter. SSO/OIDC later.

### 10. Config file
`clap.toml` (TOML for comments; Bun parses natively). Layering:
defaults < system file < user file < env < flags. Admin API writes the same
file. Machine state stays JSON under `~/.clap/state`.

Sketch:

```toml
[server]
host = "0.0.0.0"
port = 11435

[auth]
require_api_key = true

[limits]
max_session_ctx = 65536
queue_depth = 64

[models."unsloth/Qwen3.6-27B-GGUF"]
pinned = true
kv_type = "q8_0"
slots = 8
```

### 11. Observability
- Prometheus `/metrics`: queue depth, TTFT/TPS percentiles, KV occupancy,
  evictions, VRAM, crash counts, per-key usage
- Structured JSON logs with request ids
- Dashboard reads the same series (it is the first consumer, not a fork)

### 12. Model lifecycle policy
Pin models resident, warm on boot, per-model VRAM budgets, admission control
when multiple models share a GPU. Today's keep-alive becomes one policy among
several.

## Tier 4 — More hardware (single logical deployment)

### 13. Multi-GPU tensor split (llama worker)
llama.cpp built-in (`split_mode`, `tensor_split`, NCCL already linked when
present). Plumb through config; default `layer` split. Enables 70B+ Q8 or
more concurrent sessions.

### 14. MLX distributed (research track)
MLX has a distributed layer (ring/MPI; RDMA over Thunderbolt 5 on macOS
Tahoe). A distributed MLX worker is a drop-in replacement thanks to the
process-boundary worker protocol. Explore when a multi-Mac test rig exists.

## Deferred (explicitly out of scope for now)

- Multi-machine router tier / load balancing with session affinity
- Speculative decoding / draft models
- Cross-machine KV migration

## Execution order

1. Continuous batching (T1.1)
2. Admission control (T1.2)
3. Watchdog (T1.4)
4. API keys (T3.9)
5. Config file (T3.10) — carries KV-type/slot policy surfacing (T2.7)
6. Queue fairness (T1.3) — needs keys for per-client fairness
7. Prometheus metrics (T3.11)
8. Shared-prefix dedup (T2.5)
9. Adaptive capacity + session ctx caps (T2.6)
10. Session-aware eviction (T2.8)
11. Multi-GPU split (T4.13)

Rationale: batching unlocks concurrency; keys+config unlock every per-client
policy; dedup is the biggest org-shaped memory win but rides on batching's
sequence machinery.

## Verification bar per phase

- Synthetic multi-client stress: N parallel sessions with org-sized prompts
  (10-40k shared prefix + per-session divergence), assert zero failed
  requests, bounded TTFT p95, no worker restarts
- Soak: hours-long run with session churn (create/idle/expire), assert no
  VRAM growth, no slot leaks
- Chaos: kill worker mid-decode, assert watchdog recovery within SLO
