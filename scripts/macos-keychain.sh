#!/usr/bin/env bash
set -euo pipefail

command=${1:-}
keychain=${CLAP_KEYCHAIN_PATH:-"${RUNNER_TEMP:-${TMPDIR:-/tmp}}/clap-signing.keychain-db"}

require() {
  local name
  for name in "$@"; do
    if [[ -z ${!name:-} ]]; then
      echo "error: required release secret $name is not set" >&2
      exit 1
    fi
  done
}

case "$command" in
  import)
    require APPLE_CERTIFICATE APPLE_CERTIFICATE_PASSWORD KEYCHAIN_PASSWORD APPLE_SIGNING_IDENTITY
    certificate=$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/clap-certificate.XXXXXX.p12")
    trap 'rm -f "$certificate"' EXIT
    printf '%s' "$APPLE_CERTIFICATE" | base64 -D > "$certificate"
    security create-keychain -p "$KEYCHAIN_PASSWORD" "$keychain"
    security set-keychain-settings -lut 21600 "$keychain"
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$keychain"
    security import "$certificate" -k "$keychain" -P "$APPLE_CERTIFICATE_PASSWORD" -T /usr/bin/codesign -T /usr/bin/security
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KEYCHAIN_PASSWORD" "$keychain" >/dev/null
    security find-identity -v -p codesigning "$keychain" | grep -F -- "$APPLE_SIGNING_IDENTITY" >/dev/null || {
      echo "error: APPLE_SIGNING_IDENTITY was not found in the imported certificate" >&2
      exit 1
    }
    printf 'CLAP_KEYCHAIN_PATH=%s\n' "$keychain"
    ;;
  cleanup)
    security delete-keychain "$keychain" 2>/dev/null || true
    rm -f "$keychain"
    ;;
  *)
    echo "usage: $0 import|cleanup" >&2
    exit 2
    ;;
esac
