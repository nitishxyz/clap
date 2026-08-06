#!/usr/bin/env bash
# Builds the CUDA-enabled clap-llama worker on a Linux/CUDA host (e.g. a RunPod pod).
# Mirrors .github/workflows/release.yml:build-linux-x64-cuda-worker but targets the
# local GPU arch only, which keeps the nvcc pass short.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/clap-src}
REF=${REF:-main}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CLAP_CUDA_ARCHS=${CLAP_CUDA_ARCHS:-80}
export PATH="$CUDA_HOME/bin:$HOME/.bun/bin:$HOME/.cargo/bin:$PATH"

command -v bun >/dev/null || { curl -fsSL https://bun.sh/install | bash; }
command -v cargo >/dev/null || { curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal; }
export PATH="$HOME/.bun/bin:$HOME/.cargo/bin:$PATH"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/nitishxyz/clap.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --depth 1 origin "$REF" && git checkout -f FETCH_HEAD

bun install --frozen-lockfile
cargo build --manifest-path native/cache/Cargo.toml --release --locked
bun run runtime:llama:vendor
bun run runtime:llama:build

ls -la "$REPO_DIR/libexec/clap-llama"
echo "CUDA worker built: $REPO_DIR/libexec/clap-llama"
