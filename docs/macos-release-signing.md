# macOS release signing

Clap's macOS release is signed with a Developer ID Application certificate and notarized as a DMG. Release secrets are scoped to the protected GitHub `release` environment; normal CI and local builds remain unsigned and need no Apple credentials.

## GitHub environment and secrets

Create a protected environment named `release`, require appropriate reviewers, and restrict it to release tags. Configure these environment secrets without putting secret values in shell history:

```sh
# Export the Developer ID Application certificate and private key as a password-protected PKCS#12 file.
base64 < developer-id-application.p12 | gh secret set --env release APPLE_CERTIFICATE
gh secret set --env release APPLE_CERTIFICATE_PASSWORD < certificate-password.txt
gh secret set --env release KEYCHAIN_PASSWORD < temporary-keychain-password.txt
gh secret set --env release APPLE_SIGNING_IDENTITY < signing-identity.txt
gh secret set --env release APPLE_ID < apple-id.txt
gh secret set --env release APPLE_PASSWORD < app-specific-password.txt
gh secret set --env release APPLE_TEAM_ID < apple-team-id.txt
```

`APPLE_CERTIFICATE` is the base64 encoding of the `.p12`. `APPLE_PASSWORD` is an app-specific password, not the Apple ID account password. Secret files should be created with restrictive permissions and securely deleted after use.

## Release order

The macOS architecture job:

1. Builds Rust cache components, llama, and MLX workers.
2. Imports the PKCS#12 into an ephemeral keychain and confirms `APPLE_SIGNING_IDENTITY` exists.
3. Signs and verifies `libexec/clap-llama` and `libexec/clap-mlx` with the hardened runtime, timestamping, and `config/macos/clap.entitlements.plist`.
4. Runs `build:binary` only after worker signing, ensuring the content manifest hashes the signed bytes.
5. Signs and verifies `dist/clap`, smoke-runs it with an isolated `CLAP_HOME`, and byte-compares and verifies both extracted worker signatures.
6. Creates and signs a DMG, waits for an accepted `notarytool` result, staples and validates it, then runs Gatekeeper assessment.
7. Mounts the stapled DMG noninteractively, verifies the CLI's signature and notarization with `codesign` and Gatekeeper, and confirms it is byte-identical to `dist/clap`.
8. Packages that verified signed CLI as `clap-<tag>-darwin-arm64.tar.gz` for `install.sh`, then generates and verifies SHA-256 checksums for both the tarball and DMG.

There is deliberately no ad-hoc fallback for tag releases. Missing or invalid signing/notarization credentials fail the release clearly. The workflow deletes the imported certificate file and temporary keychain in an `always()` cleanup step.

Each architecture job uploads a workflow artifact. The macOS artifact contains the stapled DMG, the installer tarball, and a checksum for each. A single final job downloads all required artifacts, rejects missing or unexpected files, verifies every checksum, and publishes one GitHub release.

## Local validation

Validate the release command order without credentials, signing, notarizing, or modifying build outputs:

```sh
bun run release:macos:validate
bun test scripts/macos-release.test.ts
```

A real signed release should only run in the protected GitHub environment. Do not store certificates, passwords, or keychains in the repository.
