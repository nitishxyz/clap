# Clap cache coordinator

This workspace builds the backend-independent cache policy and its stable C ABI.
The production sequence coordinator does not own or call backend KV-cache
objects. The Rust-only logical-block module uses a narrow transactional backend
trait and an in-memory fake to prove paged ownership invariants before native
engine integration.

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

## Logical block ownership

`clap-cache-core::logical_block` models immutable exact-token blocks, versioned
prefix/session block tables, reference counts, read/write leases, copy-on-write
forks, unique-byte eviction planning, and all-or-nothing publish/import/release.
`InMemoryBlockBackend` stages physical creates and releases until commit and is
only a Phase 4 conformance backend; it is not wired through the C ABI or used by
llama.cpp or MLX. Sequence traces can be deterministically chunked into logical
references while namespace and model-domain compatibility remain fail-closed.

Every public input struct starts with `version` and `struct_size`. Initialize
`version` to `CLAP_CACHE_ABI_VERSION`, zero reserved fields, and set
`struct_size` to `sizeof(the_struct)`. Token buffers are borrowed only for the
duration of a call; Rust copies committed logical token metadata and never
stores engine pointers.
