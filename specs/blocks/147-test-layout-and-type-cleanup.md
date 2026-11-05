# Block 147 — Test-module layout, a test-only shim, the pass timestamp type, and `HalError: Clone`

Status: **COMPLETE (2026-11-05)**. Phase Review: 0 CRITICAL, 0 MAJOR, 4 MINOR; all resolved, see
below.
Source: backlog E9 (Block 102 Phase Review m12 (b), (c), (d)) and E10, started by user
request on 2026-11-05.

No behaviour changes. Every rule below is a refactor; the C API is unchanged.

## Problem

1. **Tests above the imports.** In 25 `#[cfg(test)] mod tests` modules, one or more
   `#[test]` functions come before the module's `use super::*;` line. For example
   `yawgpu-core/src/adapter.rs` starts its test module with
   `adapter_info_passes_through_hal_identity` and imports `super::*` after it. Found with:
   for each file containing `use super::*`, a `mod tests {` whose first `#[test]` precedes
   its first `    use super::*;` line. The affected files, as of `0e5f6de`:
   - `yawgpu-core/src/`: `adapter.rs`, `bind_group.rs`, `bind_group_layout.rs`,
     `buffer.rs`, `command_encoder.rs`, `compute_pipeline.rs`, `device.rs`, `queue.rs`,
     `render_pipeline.rs`, `shader_tint.rs`, `texture.rs`, `texture_view.rs`;
   - `yawgpu-hal/src/`: `lib.rs`, `metal/encode.rs`, `metal/mod.rs`, `metal/surface.rs`,
     `gles/queue.rs`, `gles/surface.rs`, `vulkan/encode.rs`, `vulkan/mod.rs`,
     `vulkan/pipeline.rs`, `vulkan/surface.rs`, `vulkan/texture.rs`;
   - `yawgpu/src/ffi/`: `mod.rs`, `surface.rs`.
2. **A test-only shim in library code.** `yawgpu-core/src/bind_group.rs` has
   `#[cfg(test)] pub(crate) fn validate_bind_group_entry`, a wrapper over
   `validate_bind_group_entry_with_usage(.., internal: false)` that only the tests call.
   A doc comment in `yawgpu-core/src/queue.rs` names it as if it were the library's
   validator.
3. **A render-pass type used by compute passes.** `yawgpu-core` exports
   `RenderPassTimestampWrites` (`render_pass.rs`), and compute passes use it too
   (`CommandEncoder::begin_compute_pass`, the compute-pass descriptor, the timestamp
   helpers in `command_encoder.rs`, and `yawgpu/src/conv/descriptors.rs`).
4. **`HalError` is not `Clone`.** The GLES HAL copies one by hand in
   `rebuild_hal_error` (`yawgpu-hal/src/gles/mod.rs`), with a `TODO` to derive `Clone`.
   Every variant holds only `&'static str` and `String`, so `Clone` can be derived.

## Rules

R1. **Imports first.** In every test module listed above, and any other test module in
`yawgpu`, `yawgpu-core`, and `yawgpu-hal` with the same layout, the `use` items move to
the top of the module, before any function. The tests themselves, their order, and
their bodies do not change. Nested test modules follow the same rule.

R2. **No test-only shim in library code.** `validate_bind_group_entry` leaves the
library part of `bind_group.rs`. The tests call `validate_bind_group_entry_with_usage`
with `internal: false` directly, or through a private helper inside the test module.
The `queue.rs` doc comment names the function the library actually uses. Any other
`#[cfg(test)]` item outside a test module whose only purpose is a test entry point is
handled the same way; `#[cfg(test)]` helpers that genuinely need crate-level visibility
across modules (for example `test_helpers`) stay.

R3. **One name for both passes.** `RenderPassTimestampWrites` is renamed
`PassTimestampWrites`, defined in the module both passes can reach without one
depending on the other's name (for example `pass.rs`), and exported from `yawgpu-core`
under the new name only. Every use in `yawgpu-core` and `yawgpu` follows. Its doc
comment says it serves render and compute passes. No alias for the old name.

R4. **`HalError` derives `Clone`.** `rebuild_hal_error` and its `TODO` are deleted; its
call sites use `.clone()` (for example `.map_err(Clone::clone)` or
`.map_err(|error| error.clone())`). Any other hand-written copy of a `HalError` in the
workspace goes the same way. `HalError` stays `#[non_exhaustive]`.

R5. **Unchanged.** Behaviour, error messages, the C API, test names, and test count.

## Tests

- R4: a unit test in `yawgpu-hal/src/error.rs` that clones a variant with a `String`
  message and compares the `Display` output of both.
- Everything else is covered by the existing tests: the workspace test count is the
  same as before plus the R4 test.
- Claude: `cargo test --workspace`, fmt, the clippy matrix, the HAL Metal and Vulkan
  lib tests, and the GLES `--features gles` build (the GLES HAL does not run on this
  host). No GPU or CTS run: the changes do not reach code generation or backend
  behaviour.

## Out of scope

- Moving or reorganising tests between files.
- Other cosmetic cleanups.

## Slices

- **S1 (coding agent):** R1-R5 with the R4 unit test.
- **S2 (Claude):** verification, Phase Review.

## Verification (2026-11-05)

Run on Apple M2, macOS 27.2.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,437 passed, 0 failed (1,436 at `44b867b` plus the R4 test) |
| `#[test]` count per crate against `44b867b` (Phase Review) | `yawgpu` 319 / 319, `yawgpu-core` 813 / 813, `yawgpu-hal` 746 / 747; test names unchanged |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough` | clean |
| HAL lib tests, `metal` and `vulkan` | 201 passed; 373 passed |
| R1 scan (scope-aware, nested modules, every backend) | 26 modules fixed: the 25 listed above and `yawgpu-core/src/render_pass.rs`; none left |

R2 also moved the test-only `buffer_ranges_overlap`, `PendingWriteBatch::pool_sizes` and
`in_flight_submission_indices`, `hal_command_execution` and
`hal_command_execution_with_ops`, `FutureRegistry::len`, and the C API `cache_len` into
their test modules, unchanged. The `#[cfg(test)]` items left outside test modules are
called from other modules' tests or are test hooks threaded through library code.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Backlog rows E9 and E10 and this block's status still open | Closed in the backlog; status updated |
| m2 | The problem list names 25 files; `render_pass.rs` was the 26th | Recorded in Verification |
| m3 | A test in `vulkan/pipeline.rs` repeats the module-level `use ash::vk::Handle;` in its body | Removed |
| m4 | The `PassTimestampWrites` doc comment read as if the passes were used by validation | Reworded |
