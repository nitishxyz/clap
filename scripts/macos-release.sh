#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

entitlements=${CLAP_ENTITLEMENTS:-config/macos/clap.entitlements.plist}
version=${CLAP_RELEASE_VERSION:-local}
dmg="dist/clap-${version}-darwin-arm64.dmg"
archive="dist/clap-${version}-darwin-arm64.tar.gz"
dry_run=${CLAP_SIGNING_DRY_RUN:-0}

require() {
  local name
  for name in "$@"; do
    if [[ -z ${!name:-} ]]; then
      echo "error: required release secret $name is not set" >&2
      exit 1
    fi
  done
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" != 1 ]]; then "$@"; fi
}

sign_executable() {
  local path=$1
  if [[ "$dry_run" == 1 ]]; then
    run codesign --force --sign "$APPLE_SIGNING_IDENTITY" --keychain "$CLAP_KEYCHAIN_PATH" --options runtime --timestamp --entitlements "$entitlements" "$path"
  else
    printf '+ codesign --force --sign [REDACTED] --keychain %q --options runtime --timestamp --entitlements %q %q\n' "$CLAP_KEYCHAIN_PATH" "$entitlements" "$path"
    codesign --force --sign "$APPLE_SIGNING_IDENTITY" --keychain "$CLAP_KEYCHAIN_PATH" --options runtime --timestamp --entitlements "$entitlements" "$path"
  fi
  run codesign --verify --strict --verbose=2 "$path"
}

if [[ "$dry_run" == 1 ]]; then
  APPLE_SIGNING_IDENTITY=${APPLE_SIGNING_IDENTITY:-"Developer ID Application: DRY RUN"}
  CLAP_KEYCHAIN_PATH=${CLAP_KEYCHAIN_PATH:-/tmp/clap-dry-run.keychain-db}
  APPLE_ID=${APPLE_ID:-dry-run@example.invalid}
  APPLE_PASSWORD=${APPLE_PASSWORD:-dry-run}
  APPLE_TEAM_ID=${APPLE_TEAM_ID:-DRYRUN0000}
else
  [[ $(uname -s) == Darwin ]] || { echo "error: signed macOS releases require macOS" >&2; exit 1; }
  require APPLE_SIGNING_IDENTITY CLAP_KEYCHAIN_PATH APPLE_ID APPLE_PASSWORD APPLE_TEAM_ID
  [[ -f libexec/clap-llama && -f libexec/clap-mlx ]] || {
    echo "error: build libexec/clap-llama and libexec/clap-mlx before signing" >&2
    exit 1
  }
fi
[[ -f "$entitlements" ]] || { echo "error: missing entitlements: $entitlements" >&2; exit 1; }

# Worker signatures must be part of the content-addressed embedded manifest.
sign_executable libexec/clap-llama
sign_executable libexec/clap-mlx

run bun run build:binary
sign_executable dist/clap

smoke_home=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/clap-release-smoke-$$
run rm -rf "$smoke_home"
run env CLAP_HOME="$smoke_home" dist/clap --help
if [[ "$dry_run" != 1 ]]; then
  extracted_llama=$(find "$smoke_home/libexec" -type f -name clap-llama -print -quit)
  extracted_mlx=$(find "$smoke_home/libexec" -type f -name clap-mlx -print -quit)
  [[ -n "$extracted_llama" && -n "$extracted_mlx" ]] || {
    echo "error: smoke run did not extract both signed workers" >&2
    exit 1
  }
  codesign --verify --strict --verbose=2 "$extracted_llama"
  codesign --verify --strict --verbose=2 "$extracted_mlx"
  cmp libexec/clap-llama "$extracted_llama"
  cmp libexec/clap-mlx "$extracted_mlx"
fi
run rm -rf "$smoke_home"

stage=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/clap-dmg-stage-$$
run rm -rf "$stage" "$dmg"
run mkdir -p "$stage"
run cp dist/clap "$stage/clap"
run hdiutil create -volname Clap -srcfolder "$stage" -ov -format UDZO "$dmg"
run rm -rf "$stage"
if [[ "$dry_run" == 1 ]]; then
  run codesign --force --sign "$APPLE_SIGNING_IDENTITY" --keychain "$CLAP_KEYCHAIN_PATH" --timestamp "$dmg"
else
  printf '+ codesign --force --sign [REDACTED] --keychain %q --timestamp %q\n' "$CLAP_KEYCHAIN_PATH" "$dmg"
  codesign --force --sign "$APPLE_SIGNING_IDENTITY" --keychain "$CLAP_KEYCHAIN_PATH" --timestamp "$dmg"
fi
run codesign --verify --strict --verbose=2 "$dmg"

notary_result=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/clap-notary-result-$$.json
if [[ "$dry_run" == 1 ]]; then
  run xcrun notarytool submit "$dmg" --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID" --wait --output-format json
else
  printf '+ xcrun notarytool submit %q --apple-id [REDACTED] --password [REDACTED] --team-id [REDACTED] --wait --output-format json\n' "$dmg"
  xcrun notarytool submit "$dmg" --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID" --wait --output-format json > "$notary_result"
  /usr/bin/python3 - "$notary_result" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as result:
    status = json.load(result).get("status")
if status != "Accepted":
    raise SystemExit(f"error: notarization status is {status!r}, expected 'Accepted'")
PY
  rm -f "$notary_result"
fi
run xcrun stapler staple "$dmg"
run xcrun stapler validate "$dmg"
run spctl --assess --type open --context context:primary-signature --verbose=2 "$dmg"

# Gatekeeper-assess the exact notarized CLI from the stapled DMG before also
# publishing those signed bytes in the noninteractive installer tarball.
mount_point=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/clap-dmg-mount-$$
run rm -rf "$mount_point" "$archive"
run mkdir -p "$mount_point"
run hdiutil attach -readonly -nobrowse -mountpoint "$mount_point" "$dmg"
run codesign --verify --strict --verbose=2 "$mount_point/clap"
run spctl --assess --type execute --verbose=2 "$mount_point/clap"
run cmp dist/clap "$mount_point/clap"
run hdiutil detach "$mount_point"
run rm -rf "$mount_point"
run tar -czf "$archive" -C dist clap

if [[ "$dry_run" == 1 ]]; then
  run shasum -a 256 "$dmg"
  run shasum -a 256 "$archive"
else
  printf '+ checksum and verify %q and %q\n' "$dmg" "$archive"
  (
    cd dist
    shasum -a 256 "$(basename "$dmg")" > "$(basename "$dmg.sha256")"
    shasum -a 256 "$(basename "$archive")" > "$(basename "$archive.sha256")"
    shasum -a 256 -c "$(basename "$dmg.sha256")"
    shasum -a 256 -c "$(basename "$archive.sha256")"
  )
fi
