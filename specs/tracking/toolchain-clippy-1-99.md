# Toolchain — rustc / clippy 1.99.1 gate breakage

Filed 2026-10-25 during Block 109, independent of it (its own commit).
Precedent: `toolchain-clippy-1-98.md`.

## Finding

`cargo clippy --workspace --all-targets -- -D warnings` failed on an
unmodified `bead72b` checkout with rustc / clippy 1.99.1 (`58cf8f9c3`,
2026-10-13):

| Site | Lint | Fix |
|---|---|---|
| `yawgpu-hal/src/lib.rs` `next_submission_index` | `deprecated`: `AtomicU64::fetch_update` renamed to `try_update` | `try_update` (same signature/semantics) |
| `yawgpu-core/src/command_encoder.rs` timestamp-conversion immediates | `clippy::chunks_exact_to_as_chunks` | `as_chunks_mut::<4>().0` |
| `yawgpu-hal/src/vulkan/encode.rs` test helpers `read_query_results`, `read_f32s` | `clippy::chunks_exact_to_as_chunks` | `as_chunks::<N>().0` |
| `yawgpu-hal/src/metal/surface.rs` test module (`--features metal`) | `unused_imports` (`super::super::*`) | import removed |

No behaviour change. The replacement APIs need Rust >= 1.95.0; the
workspace declares no MSRV.

## Gate after the fix

Strict clippy (`-D warnings`, no `-A` overrides) clean on default,
`metal`, `vulkan`, `gles`, `vulkan,tiled`, `metal,tiled`; `cargo fmt
--check` clean; `yawgpu-hal` / `yawgpu-core` lib tests green.
