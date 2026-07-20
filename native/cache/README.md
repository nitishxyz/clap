# Clap cache coordinator

This workspace builds the backend-independent cache policy and its stable C ABI.
It does not own or call backend KV-cache objects.

```sh
cargo build --release -p clap-cache-ffi
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The release build emits `libclap_cache_ffi.a` (and a platform dynamic library).
`include/clap_cache.h` is the checked-in ABI contract consumed by C++ and a
future SwiftPM C target. Keep it synchronized with the `#[repr(C)]` declarations
in `clap-cache-ffi`; `cbindgen.toml` provides generation scaffolding for ABI
review, but generation is intentionally not part of ordinary builds.

## Transaction ownership

A successful `clap_cache_plan` exclusively leases every mutation target and
read-leases its donor. Execute the returned operation without holding a Rust
lock, then call exactly one of `clap_cache_commit` or `clap_cache_abort` before
`clap_cache_plan_destroy`. Abort invalidates the target because physical state
may be uncertain. Destroy all plans before destroying their manager.

Every public input struct starts with `version` and `struct_size`. Initialize
`version` to `CLAP_CACHE_ABI_VERSION`, zero reserved fields, and set
`struct_size` to `sizeof(the_struct)`. Token buffers are borrowed only for the
duration of a call; Rust copies committed logical token metadata and never
stores engine pointers.
