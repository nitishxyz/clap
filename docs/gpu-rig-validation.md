# GPU rig validation runbook

Purely mechanical validation session for the three GPU-dependent features
that shipped code-complete but hardware-unvalidated:

1. KV per-token byte estimate vs measured VRAM
   (`native/llama/include/clap/llama/kv-fit.h`)
2. KV-derived active-slot autoderivation (`native/llama/active-concurrency.h`,
   `[llama] slot_memory_budget_bytes`)
3. Multi-GPU split (`native/llama/include/clap/llama/gpu-split.h`,
   `[llama] split_mode` / `main_gpu` / `tensor_split`)

Every measurement lands in `bench-results/` (gitignored) via `bun run bench`;
the acceptance results get summarized in this file's "Results" section at the
bottom when the session is done.

## Hardware requirements

- Linux box with 2+ NVIDIA GPUs. 2×4090 (24 GB each) or a rented 2×A100 for
  a few hours is enough. Layer split does not need NVLink; row-split numbers
  are only meaningful with NVLink or equivalent interconnect — note which
  one the rig has in the results.
- Models on disk:
  - a mid-size GGUF that fits one GPU (e.g. ~8B Q8) — experiments 1 and 2
  - a GGUF that does NOT fit one GPU (70B Q8 on 2×A100, ~34B Q8 on 2×4090)
    — experiment 3
- `nvidia-smi` present (the server's GPU sampler uses it too).

## Setup

```bash
# On the rig
./scripts/build-linux.sh
~/bin/clap serve --port 11435

# From the workstation
ssh -N -L 11435:127.0.0.1:11435 <rig> &
curl -s http://localhost:11435/clap/v1/health
```

VRAM readings: prefer one loop on the rig writing a timestamped log so every
experiment shares the same source of truth:

```bash
while true; do
  echo "$(date -u +%FT%TZ) $(nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader)"
  sleep 2
done | tee /tmp/vram.log
```

Cross-check that the dashboard payload (`/clap/v1/dashboard`) reports the
same per-GPU numbers — `packages/server/src/gpu-usage.ts` is also untested
on real NVIDIA hardware; file anything inconsistent as a bug.

## Experiment 1 — KV estimate vs measured VRAM

llama.cpp allocates the full unified KV pool at context creation, so two
loads at different pinned contexts isolate pool cost from weights.

For each KV type in `f16`, `q8_0`, `q4_0`:

1. In `clap.toml`: `[llama] context = 16384`, `kv_type = "<type>"`. Load the
   mid-size model, wait for steady state, record per-process VRAM
   (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`).
2. Unload. Set `context = 65536` (clamp to train context if smaller; adjust
   the delta arithmetic to the actual cell counts). Load, record again.
3. `measured_bytes_per_token = (vram_hi - vram_lo) / (ctx_hi - ctx_lo)`.
4. Compare with the worker's estimate: unset `context` once with
   `CLAP_LLAMA_KV_AUTOFIT=1` and read the estimate from the autofit log line
   in the per-launch stderr log, or compute `estimate_kv_bytes_per_token`
   from model metadata (layers × 2 × kv_heads × head_dim × bytes/element).

Acceptance:

- estimate within ±20% of measured for f16
- estimate ≥ measured for q8_0 and q4_0 (the constants in `kv-fit.h` are
  deliberately conservative upper bounds — confirm they actually are)

If the estimate is below measured for any type, raise the per-element
constants in `kv_type_half_bytes` and rerun; that is a correctness fix, not
tuning.

## Experiment 2 — slot autoderivation honesty

1. Compute the free-VRAM budget after weights: load the model with
   `context = 16384`, read weights+scratch VRAM, subtract from total, keep
   ~1 GiB headroom.
2. In `clap.toml`:

   ```toml
   [llama]
   max_session_ctx = 16384
   slot_memory_budget_bytes = <budget from step 1>
   ```

3. Restart the worker; read `max_active` and the limiting reason from the
   retention telemetry (`/clap/v1/runtime/models`). Expected:
   `max_active ≈ budget / (kv_bytes_per_token × 16384)`, reason
   `memory_ceiling` or `context_ceiling`.
4. Saturation proof — run exactly `max_active` streams with ~16k-token
   prompts:

   ```bash
   bun run bench -- --model <id> --base-url http://localhost:11435 \
     --matrix config/inference-benchmark-matrix-gpu-rig.json \
     --suite concurrency --streams <max_active> --label rig-slots-at-max
   ```

   Watch `/tmp/vram.log`: VRAM must plateau under the budget; no OOM, no
   worker crash, zero failed samples.
5. Overload proof — same command with `--streams <max_active + 2>` and label
   `rig-slots-over-max`. The two extra streams must queue or reject cleanly
   (fair limiter / admission), never OOM the worker.
6. Optional sharpness check: bump `slot_memory_budget_bytes` by ~50%,
   restart, rerun step 4 at the new `max_active`. If it OOMs, the estimate
   from experiment 1 is too low — fix that first.

Acceptance: the derived `max_active` is the largest stream count that
survives step 4, and step 5 degrades gracefully.

## Experiment 3 — multi-GPU split

Placement checks (mid-size model, cheap):

| `clap.toml` `[llama]`                  | Expected                                    |
| -------------------------------------- | ------------------------------------------- |
| (defaults)                             | weights on both GPUs (layer split)          |
| `tensor_split = "3,1"`                 | ~75/25 VRAM ratio GPU0:GPU1                 |
| `split_mode = "none"`, `main_gpu = 1`  | everything on GPU 1, GPU 0 idle             |
| `split_mode = "diagonal"`              | load FAILS with described error in the per-launch stderr log |

The last row is the part unit tests cannot prove: that the env actually
reaches the real `llama_model_load_from_file` and fails closed.

Performance (the T4.13 payoff — use the model that does not fit one GPU):

```bash
# layer split (default)
bun run bench -- --model <big-id> --base-url http://localhost:11435 \
  --matrix config/inference-benchmark-matrix-gpu-rig.json \
  --label rig-split-layer

# row split (set [llama] split_mode = "row", restart worker)
bun run bench -- --model <big-id> --base-url http://localhost:11435 \
  --matrix config/inference-benchmark-matrix-gpu-rig.json \
  --label rig-split-row
```

Compare per case (`summary.json`): TTFT p50/p95 and decode p50 — decode-only
tok/s is exactly the metric that shows whether row-split tensor parallelism
helps or interconnect traffic hurts. Also scrape
`clap_request_decode_tokens_per_second` from `/metrics` before/after as the
server-side cross-check.

Acceptance:

- layer split serves the big model at usable decode speed (it cannot load on
  one GPU at all — that alone validates the feature)
- the layer-vs-row comparison is recorded honestly (either direction), and
  the README default recommendation is updated from data if row wins on the
  rig's interconnect

Also grab one `CLAP_LLAMA_UBATCH` sweep (256/512/1024) on the big model
under the concurrency suite while the rig is up — the remaining T1.1b item.

## Wrap-up checklist

- [ ] Sanitize any log excerpts before committing quotes
      (`bun run bench:sanitize -- <input> <output>`)
- [ ] Fill in the Results section below (hardware, driver, model IDs, the
      three acceptance verdicts, chosen constants/defaults)
- [ ] If `kv_type_half_bytes` constants changed: update `kv-fit.h` + tests
- [ ] Update `docs/scale-plan.md` T2.6 / T4.13 "Remaining" lines
- [ ] File dashboard GPU-sampling issues found in setup, if any

## Results

### Session 1 — single A100-SXM4-80GB (RunPod, driver CUDA 12.8)

Single-GPU pod, so experiment 3 (multi-GPU split) is still unvalidated. What
this session did establish:

**CUDA worker provisioning.** Release binaries embed a CPU-only `clap-llama`
and re-extract it into `~/.clap/libexec/<build-id>/` on every start, so
hand-copying a GPU worker there is silently undone. `ensureCudaWorker`
(`apps/cli/src/cuda-worker.ts`) is the supported path: it downloads
`clap-llama-cuda-<tag>-linux-x64.tar.gz` into `~/.clap/libexec/cuda-<version>/`.
That asset is not published for v0.2.0 yet, so the pod fell back to CPU until
`CLAP_LLAMA_WORKER` was pointed at a locally built worker
(`scripts/pod-build-cuda-worker.sh`). Verify offload from the worker stderr
log — `ggml_cuda_init: found 1 CUDA devices` plus
`load_tensors: layer N assigned to device CUDA0` — not from wall-clock alone.

| Qwen3.6-27B-Q4_K_M | CPU worker | CUDA worker |
| --- | --- | --- |
| `clap load` | 34.2s | 6.4s |
| VRAM used | 0 MiB | 52435 MiB (n_ctx 262144) |
| 147-token reply | 71.0s | 25.8s |

**Cache reuse (`scripts/cache-probe.py`, 6.1k-token shared system prompt).**

| Model | Sessions x turns | Hit rate | Per-turn reuse |
| --- | --- | --- | --- |
| gemma-4-E4B-it-GGUF | 1 x 4 | 75.0% (3/4) | 99% (`kind=slot`) |
| Qwen3.6-27B-GGUF | 1 x 5 | 0.0% (0/5) | 0% |
| Qwen3.6-27B-GGUF | 4 x 4 | 0.0% (0/16) | 0% |

The gemma row is the expected shape: turn 1 is a cold miss, every later turn
trims and continues. Qwen3.6 reused nothing because hybrid models were denied
prompt-boundary anchors; fixed by advertising
`CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT` for them. Re-measured on the same
pod with the rebuilt worker:

| Model | Sessions x turns | Hit rate | Per-turn reuse |
| --- | --- | --- | --- |
| Qwen3.6-27B-GGUF | 1 x 5 | 80.0% (4/5) | 99% (`kind=anchor`) |
| Qwen3.6-27B-GGUF | 4 x 4 | 100.0% (16/16) | 99% (`kind=anchor`) |

`scripts/otto-session-probe.py` replays an agent-shaped transcript (tool
schemas in the system prompt, growing tool/assistant turns) rather than
filler text, and reaches 80% (4/5) at 82-93% per turn on the same model — the
lower per-turn figure is real, because each agent turn appends a larger
share of new tokens than the probe's one-line question.

**Container memory.** `totalmem()` reported the 2016 GiB host inside a pod
limited to 234 GiB. Fixed by `packages/server/src/container-memory.ts`.
