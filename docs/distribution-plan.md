# clapd Distribution Plan — one binary, install and run

Goal: `curl -fsSL clapd.sh | sh` (or `npm i -g clapd`, or `brew install clapd`) on any
supported machine, then `clap run <model>` just works — GPU included, no toolchains,
no separate worker files, no post-install steps.

## End product

One self-contained executable per platform:

| artifact | contents | GPU |
| --- | --- | --- |
| `clap-darwin-arm64` | CLI + server + dashboard + `clap-llama` + `clap-mlx` + metallib | Metal (always on) |
| `clap-linux-x64` | CLI + server + dashboard + `clap-llama` + ggml backend plugins (CUDA, CPU) | CUDA when driver present, CPU fallback |
| `clap-linux-arm64` (later) | same as linux-x64 | CUDA (Jetson/GH200) / CPU |

Everything is embedded in the binary (Bun compile assets, as today) and extracted to
`~/.clap/libexec/<build-id>/` on first run. Version-addressed, pruned automatically.
No network access needed at install time beyond downloading the binary itself.

## The GPU problem, solved properly (no variants, no flags)

A CUDA-linked worker hard-fails on machines without the CUDA runtime. Shipping
`-cpu` / `-cuda` variants pushes the choice onto the user — wrong answer.

llama.cpp already solves this: **dynamic ggml backends** (`GGML_BACKEND_DL`).
Build ggml backends as plugins, load the best one at runtime
(`ggml_backend_load_all_from_path`). This is exactly Ollama's architecture.

- Build `clap-llama` with `-DGGML_BACKEND_DL=ON`:
  - `libggml-cpu-*.so` (several CPU feature levels; ggml picks best for the host)
  - `libggml-cuda.so` (compiled in CI with nvcc — compiling needs no GPU)
- Embed the plugins in the binary alongside the worker; extract with it.
- Worker startup: `ggml_backend_load_all_from_path(<extracted dir>)` — CUDA loads
  when `libcuda.so` (driver) exists, otherwise silently falls back to CPU.
- CUDA runtime libs (`libcudart`, `libcublas`) are statically linked into
  `libggml-cuda.so` (`-DGGML_STATIC_CUDA=ON` / `CUDA_USE_STATIC_CUDA_RUNTIME`),
  so the only host requirement is the NVIDIA *driver* — same bar as Ollama.
- macOS keeps the current static Metal build (no plugins needed — Metal is universal
  on Apple silicon).

Size budget: CUDA plugin (fatbin for sm_70..sm_120) ≈ 200-400 MB. Linux binary lands
~350-500 MB — same ballpark as Ollama's. Acceptable; correctness > size. If we later
care: `--compress` the assets (zstd) and decompress on extract.

## Install channels (all serve the same binaries from GitHub Releases)

1. **Shell installer (primary)** — `curl -fsSL https://clapd.sh/install | sh`
   - detects OS/arch, downloads `clap-<platform>` from the latest GitHub release,
     verifies sha256, installs to `~/.clap/bin/clap` + PATH hint (or
     `/usr/local/bin` when writable). Identical UX to Bun/Ollama installers.
   - Until the domain exists: `curl -fsSL https://raw.githubusercontent.com/nitishxyz/clap/main/install.sh | sh`
2. **npm** — `npm i -g clapd`
   - `clapd` meta package: `bin` shim + `optionalDependencies` on
     `@clapd/darwin-arm64`, `@clapd/linux-x64` (each with `os`/`cpu` fields, binary
     inside). npm fetches only the matching one. esbuild/biome pattern.
   - npm's 2 GB pack limit is fine for our sizes.
3. **Homebrew** — `brew install nitishxyz/tap/clapd` (cask-like formula that downloads
   the release binary; no build from source).
4. **Direct** — grab the binary from GitHub Releases, `chmod +x`, done (works today).

`clap upgrade` command: hits the GitHub releases API, downloads the new binary for the
current platform, sha256-checks, atomically replaces itself. Keeps every channel fresh
without waiting on package managers.

## CI / release pipeline (tag `v*` → everything publishes)

1. `build-macos-arm64` (macos-14): llama (Metal) + MLX workers, compile binary,
   tar + sha256. Already exists.
2. `build-linux-x64` (ubuntu-22.04, big runner): install CUDA **toolkit** (no GPU
   needed), build llama with `GGML_BACKEND_DL` + CPU/CUDA plugins, compile binary
   with plugins embedded, tar + sha256. Smoke-test CPU path in CI
   (`clap run` a 30 MB model); CUDA path smoke-tested on RunPod manually per release
   until we add a self-hosted GPU runner.
3. `release`: attach all tarballs + checksums; generate notes.
4. `publish-npm`: pack `@clapd/<platform>` from the same binaries + the `clapd` shim,
   `npm publish --provenance` (versions locked to the git tag).
5. `update-tap`: bump the Homebrew formula (sha + url) via PR to `nitishxyz/homebrew-tap`.

Secrets needed: `NPM_TOKEN`, tap repo push token. Everything else is stock GITHUB_TOKEN.

## Platform behavior matrix (runtime)

- darwin-arm64: GGUF (Metal) + MLX. Current behavior, unchanged.
- linux-x64 + NVIDIA driver: GGUF on CUDA (all layers offloaded by default,
  `CLAP_LLAMA_GPU_LAYERS` override). MLX models: clear error suggesting the GGUF
  equivalent (resolver already prefers GGUF on linux).
- linux-x64 CPU-only: GGUF on CPU, works everywhere (musl excluded; glibc ≥ 2.31).
- Windows: explicitly out of scope for now (WSL2 covers it); revisit on demand.

## Phases

**P1 — Linux works (few hours)**
Commit + push current state. Verify `scripts/build-linux.sh` end-to-end on the RunPod
box (CUDA static build, current non-plugin path), fix whatever Linux-only issues
surface (paths, `ps` flags, systemd template). Smoke: pull + run + stream + tools.
Deliverable: working `clap` on the pod, patterns proven.

**P2 — Dynamic backends (1-2 days)**
Switch linux llama build to `GGML_BACKEND_DL` with CPU + CUDA plugins; embed plugins;
worker loads from extraction dir; CPU fallback verified by hiding the driver.
Deliverable: ONE linux binary correct on both GPU and CPU boxes.

**P3 — Release automation (1 day)**
linux-x64 release job (CUDA toolkit in CI), sha256s, CI smoke test, `install.sh` at
repo root, README install section rewritten around it.
Deliverable: tag → downloadable, installable binaries for both platforms.

**P4 — Package managers (1 day)**
`clapd` npm shim + `@clapd/*` platform packages + publish job; Homebrew tap + formula
+ bump job.
Deliverable: `npm i -g clapd`, `brew install nitishxyz/tap/clapd`.

**P5 — Self-update + polish (half day)**
`clap upgrade`, version-check nudge in `clap --version`/dashboard footer
(daily, cached, `CLAP_NO_UPDATE_CHECK=1` opt-out), binary size pass (zstd assets).

## Decisions locked

- Name: **clapd** on npm (`clapd` shim + `@clapd/*` platform packages).
- One binary per platform; no cpu/cuda variants; plugins inside.
- GitHub Releases is the single source of truth; all channels redistribute it.
- GPU support bar: NVIDIA driver only (statically linked CUDA runtime), Metal built-in.
