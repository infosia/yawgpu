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

## Addendum (2026-10-25, Linux host, rustc 1.98.1)

The first fix covered library code only. `--all-targets` with
`--features vulkan` / `vulkan,tiled` / `vulkan,tiled,shader-passthrough`
still failed on `clippy::chunks_exact_to_as_chunks` in the real-GPU test
files: `tests/common/lazy_init.rs`, `e2e_vulkan_{timestamp_query, tiled,
layouts, framebuffer_fetch, shader_passthrough}.rs` (10 sites). Rewritten to
`as_chunks::<N>().0` / `as_chunks_mut`; the six affected binaries re-run
green on the Linux NVIDIA Vulkan host. Strict clippy is now clean on default,
`vulkan`, `vulkan,tiled`, `vulkan,tiled,shader-passthrough`, `gles`,
`gles,tiled,shader-passthrough`; `cargo fmt --check` clean. (`metal*`
combinations not checkable on Linux.)

## Addendum (2026-10-25, macOS host, rustc 1.99.1)

The same lint remained in eleven sites across ten real-GPU Metal test files
(`e2e_metal_{compute, f16, framebuffer_fetch, immediates, render,
shader_passthrough, subgroups, texture_formats_tier2, tiled,
timestamp_query}.rs`), not reachable from the Linux host. Rewritten the same
way. Strict clippy `--all-targets` now clean on `metal`, `metal,tiled`,
`metal,shader-passthrough`, `metal,tiled,shader-passthrough` and `gles` (incl.
the new `e2e_gles_class_audit.rs`); `cargo fmt --check` clean; the ten
binaries plus `e2e_metal_lazy_init` (shared `common/lazy_init.rs`) pass on
the M2, 39/0.
