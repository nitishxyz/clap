# Inference Runtime Benchmark Follow-up

## Status and scope

This is a deferred performance program. It is explicitly **outside the completed inference-runtime hardening scope**. Hardening established ownership, strict protocol behavior, cache correctness, authenticated identity, admission safety, capability-driven behavior, scheduler fairness, and honest telemetry. Those invariants must not be weakened to improve a benchmark.

No performance target in this document is a release claim. Work begins only when representative assets and repeatable runners are available.

## Questions to answer

1. What are cold-load latency and peak resident memory by backend, architecture, quantization, and context allocation?
2. What are time to first token (TTFT), decode throughput, and end-to-end latency at interactive, normal, and background priorities?
3. How much do exact continuation, branch, anchor, and checkpoint reuse improve TTFT and prefill work?
4. What is the CPU/GPU/RSS/allocator cost of retained cache growth and each active-concurrency level?
5. Where do scheduler quantum, batch/ubatch, context size, KV format, and MLX retained-budget settings trade throughput for tail latency?
6. How does global model admission behave under mixed model loads and real memory pressure?

## Non-negotiable correctness constraints

Every benchmark run must retain:

- strict worker protocol v1 and one TypeScript process/parser path;
- Rust ownership of logical cache decisions and native-adapter ownership of physical mutation;
- authenticated cache identity and generation isolation;
- plan/materialize/commit ordering and fail-closed cache outcomes;
- no starvation across `interactive`, `normal`, and `background`;
- capability-driven behavior rather than backend-label behavior sniffing;
- measured/estimated/unavailable memory provenance without fake zeroes;
- per-launch logs and machine-readable run metadata.

Before accepting a tuning change, run `bun run native:test` in addition to the benchmark.

## Proposed benchmark suites

### 1. Load and residency

Measure cold load, warm resident access, unload, and replacement for each pinned asset. Capture wall time, model estimate, measured RSS before/after, allocator telemetry when available, OS availability, launch ID, and configuration fingerprint.

### 2. Single-request generation

Use fixed prompts and output token counts for:

- cold prompt;
- exact continuation;
- branch from shared prefix;
- prompt-boundary anchor restore;
- below-checkpoint/no-reuse control;
- native structured output;
- post-validation structured output.

Report p50/p95/p99 TTFT, prompt tokens, residual prefill tokens, prefill time, first-decode/first-emit time, decode tokens per second, total duration, and planned versus realized reuse.

### 3. Concurrent scheduling

Run controlled mixes at each priority with fixed arrival traces. Report queue delay and completion latency by priority, aggregate throughput, fairness, and maximum time without progress for every runnable class. Include sustained interactive arrival while normal/background work remains queued.

### 4. Cache pressure

Sweep retained entry/byte budgets and checkpoint settings. Measure hit/reuse rate, copy/trim/materialization time, eviction count/reason, retained measured and estimated bytes, and failure/fallback categories. Separate logical hit rate from physically realized token reuse.

### 5. Multi-model admission

Load mixed GGUF/MLX models under bounded headroom. Record reservation estimates, measured post-load RSS, estimate/observed ratio, eviction ordering, stale replan count, admission rejection reason, and time waiting on the global admission lock.

## Assets and reproducibility

Reuse immutable physical assets from:

- `config/cache-correctness-assets.json`
- `config/cache-correctness-matrix.json`

Benchmark metadata must include repository commit, dirty status, asset revision and digest, backend/architecture, OS and hardware identity, Bun/Rust/CMake/Swift versions, power mode, worker configuration, context/output limits, cache settings, priority mix, warmup count, sample count, and launch IDs.

Do not commit local model paths, API keys, installation secrets, prompts containing private data, or unsanitized worker logs. Extend the existing cache-log sanitizer when benchmark output adds new fields.

## Runner and output design

Add a separate benchmark runner rather than overloading correctness probes. Suggested files, to be implemented in the follow-up:

```text
config/inference-benchmark-matrix.json
scripts/run-inference-benchmarks.ts
scripts/sanitize-inference-benchmark-log.ts
```

The runner should:

1. validate immutable assets and host prerequisites;
2. build or verify the exact workers under test;
3. create an isolated `CLAP_HOME` and explicit config;
4. run warmups separately from recorded samples;
5. save one JSON record per sample plus a summarized report;
6. link each sample to worker launch metadata without copying stderr into results;
7. classify skipped cases with reasons;
8. enforce per-case timeout and resident-byte ceilings;
9. preserve raw measurements so summaries can be recomputed;
10. compare only equivalent hardware/configuration cohorts.

Prefer JSON/JSONL as the canonical artifact. Human tables are derived output, not the source of record.

## Statistical policy

- Predetermine warmup and sample counts per case.
- Report distributions, not only means.
- Preserve failures and timeouts in denominators.
- Do not mix cold and warm samples.
- Compare medians and tail percentiles with confidence intervals where practical.
- Require repeated runs before labeling a regression or improvement.
- Treat changes below measured run-to-run noise as inconclusive.
- Never replace correctness gates with benchmark thresholds.

## Candidate tuning sequence

Change one dimension at a time:

1. establish untouched final-architecture reference results;
2. profile control-plane queue/admission overhead;
3. profile native scheduler and bounded-step overhead;
4. tune GGUF batch/ubatch and priority prefill quanta;
5. tune MLX retained budgets and scheduling quanta;
6. evaluate checkpoint interval/count/budget;
7. evaluate model/context/KV configurations;
8. test combined settings only after isolated effects are understood.

Any tuning that introduces duplicate spawn/parser paths, backend capability sniffing, legacy priority names, unproven cache reuse, or dishonest memory telemetry is rejected regardless of speed.

## Entry and exit criteria

Start this follow-up only when:

- at least one immutable representative asset per supported backend is provisioned;
- the target hardware runner is stable and identifiable;
- benchmark prompts are redistributable and deterministic;
- output sanitization has tests;
- the full hardening verification suite passes.

A benchmark milestone is complete when the runner and schema are reviewed, reference artifacts are reproducible, variance is characterized, regressions have agreed thresholds, and operator-facing tuning recommendations distinguish measured facts from estimates. README performance claims, if any, require a separate review and are not part of this document change.

## Commands available before runner implementation

Use existing correctness and telemetry gates as prerequisites:

```sh
bun run typecheck
bun test apps packages
bun run native:build
bun run native:test
CLAP_CACHE_TEST_REQUIRE_ASSETS=1 bun run native:cache:tier-b
CLAP_CACHE_TEST_REQUIRE_ASSETS=1 bun run native:cache:tier-c
```

There is intentionally no `benchmark` package script yet. Adding the runner, its script entry, pinned benchmark matrix, sanitizer tests, and collected reference results is future implementation work.
