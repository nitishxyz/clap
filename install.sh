#!/bin/sh
# Clap installer
#   curl -fsSL https://raw.githubusercontent.com/nitishxyz/clap/main/install.sh | sh
# Options (environment variables):
#   CLAP_VERSION      install a specific tag (default: latest release)
#   CLAP_INSTALL_DIR  install directory (default: /usr/local/bin, falls back to ~/.local/bin)
set -eu

REPO="nitishxyz/clap"

say() { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"

os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Darwin)
    case "$arch" in
      arm64) target="darwin-arm64" ;;
      *) fail "unsupported macOS architecture: $arch (only arm64 builds are published)" ;;
    esac
    ;;
  Linux)
    case "$arch" in
      x86_64|amd64) target="linux-x64" ;;
      *) fail "unsupported Linux architecture: $arch (only x64 builds are published)" ;;
    esac
    ;;
  *) fail "unsupported OS: $os" ;;
esac

version="${CLAP_VERSION:-}"
if [ -z "$version" ]; then
  version=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | head -n 1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
  [ -n "$version" ] || fail "could not determine the latest release tag"
fi

archive="clap-$version-$target.tar.gz"
url="https://github.com/$REPO/releases/download/$version/$archive"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

say "downloading clap $version ($target)..."
curl -fL --progress-bar -o "$tmp/$archive" "$url" || fail "download failed: $url"

if command -v shasum >/dev/null 2>&1 || command -v sha256sum >/dev/null 2>&1; then
  say "verifying checksum..."
  curl -fsSL -o "$tmp/$archive.sha256" "$url.sha256" || fail "checksum download failed: $url.sha256"
  expected=$(awk '{print $1}' "$tmp/$archive.sha256")
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$tmp/$archive" | awk '{print $1}')
  else
    actual=$(shasum -a 256 "$tmp/$archive" | awk '{print $1}')
  fi
  [ "$expected" = "$actual" ] || fail "sha256 mismatch: expected $expected, got $actual"
else
  say "warning: no sha256 tool found, skipping checksum verification"
fi

tar -xzf "$tmp/$archive" -C "$tmp"
[ -f "$tmp/clap" ] || fail "archive did not contain a clap binary"

install_dir="${CLAP_INSTALL_DIR:-/usr/local/bin}"
if [ -n "${CLAP_INSTALL_DIR:-}" ]; then
  mkdir -p "$install_dir"
fi
if [ ! -d "$install_dir" ] || [ ! -w "$install_dir" ]; then
  if [ -d "$install_dir" ] && command -v sudo >/dev/null 2>&1 && [ -t 0 ]; then
    say "installing to $install_dir (requires sudo)..."
    sudo install -m 755 "$tmp/clap" "$install_dir/clap"
    say "installed clap $version to $install_dir/clap"
    exit 0
  fi
  install_dir="$HOME/.local/bin"
  mkdir -p "$install_dir"
fi

install -m 755 "$tmp/clap" "$install_dir/clap"
say "installed clap $version to $install_dir/clap"

case ":$PATH:" in
  *":$install_dir:"*) ;;
  *) say "note: $install_dir is not on your PATH; add it with:"
     say "  export PATH=\"$install_dir:\$PATH\"" ;;
esac

say ""
say "get started:"
say "  clap run llama3.2:3b"
