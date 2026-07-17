#!/usr/bin/env bash
# One-shot Linux build for clap: run this ON the Linux machine (e.g. a RunPod
# GPU pod). Produces a self-contained ./dist/clap with the llama.cpp (GGUF)
# backend embedded; CUDA is enabled automatically when nvcc is present.
#
#   git clone <repo> clap && cd clap && bash scripts/build-linux.sh
#
# MLX is Apple-silicon-only; on Linux the MLX backend is disabled and MLX
# model requests return a clear unsupported error.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v bun >/dev/null 2>&1; then
  echo "installing bun..."
  curl -fsSL https://bun.sh/install | bash
  export PATH="$HOME/.bun/bin:$PATH"
fi

missing=()
command -v cmake >/dev/null 2>&1 || missing+=(cmake)
command -v g++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || missing+=(g++)
command -v git >/dev/null 2>&1 || missing+=(git)
if [ ${#missing[@]} -gt 0 ]; then
  echo "installing build tools: ${missing[*]}"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq "${missing[@]}"
  else
    echo "please install: ${missing[*]}" >&2
    exit 1
  fi
fi

bun install --frozen-lockfile
bun run runtime:llama:vendor
bun run runtime:llama:build
bun run build:binary

echo
echo "done: ./dist/clap"
echo "  serve:  CLAP_HOST=0.0.0.0 ./dist/clap server start"
echo "  models: ./dist/clap pull unsloth/gemma-4-E4B-it-GGUF"
