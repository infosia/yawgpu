# Block 157 — GLES lazy clears upload zeros from a bounded staging block

Status: **S1 landed (2026-11-13)**; Phase Review pending; Linux GLES hardware run owed. Tier 2 (GLES), ranked last as the backlog directs.
Source: whole-code review at `92e38c7` (`specs/tracking/code-review-2026-11.md`), finding R16.

## Problem

`submit_texture_clear_zero_upload` (`yawgpu-hal/src/gles/queue.rs` ~3292) clears a
non-renderable color subresource by allocating one zero-filled `Vec<u8>` of
`mip_width × mip_height × layer_count × bytes_per_pixel` and uploading it with one
`tex_sub_image_*` call. For a large valid texture that is hundreds of megabytes of host
memory, and `vec![0; n]` aborts the process if the allocation fails.

## Rules

R1. **Bounded staging.** The zero upload uses one zero-filled block of at most
`ZERO_UPLOAD_BLOCK_BYTES` (4 MiB) per call, uploaded repeatedly: whole rows per call
(as many rows of one layer as fit; a single row larger than the block is split into
column ranges), then the next layer. The result is identical to the current single
upload. `UNPACK_*` pixel-store state is set as today and restored as today.

R2. **Fallible allocation.** The block is allocated with `Vec::try_reserve_exact`; a
failure returns `HalError::OutOfMemory` instead of aborting.

R3. **Scope.** Only the zero-upload clear path changes. Formats, targets (2D, 2D array,
3D, cube) and layer handling stay as they are.

## Slices

S1: R1–R3 in `yawgpu-hal/src/gles/queue.rs`, with inline tests of the pure tiling plan
(a helper returning the sequence of `(layer, x, y, width, height)` uploads for given
dimensions, bytes-per-pixel and block size: covers one call, many rows, a row wider
than the block, multiple layers, and that the union covers the subresource exactly once).

## Verification

- Agent gates: `cargo test -p yawgpu-hal --features gles --lib`,
  `cargo clippy -p yawgpu-hal --features gles --all-targets -- -D warnings`.
- Manual (Linux NVIDIA EGL host): the `e2e_gles_*` lazy-init suites. Not runnable on this
  Mac; recorded as owed until the next Linux session.

## Verification record

S1: `yawgpu-hal --features gles --lib` 269/0 on macOS (eight `zero_upload_*` plan tests);
clippy clean with `gles`. Uploads are per layer, `(x, y, width, height)` tiles from a lazy
plan, with a single block of `min(subresource bytes, 4 MiB)` allocated through
`try_reserve_exact`. The `e2e_gles_*` lazy-init suites on the Linux NVIDIA EGL host are
owed to the next Linux session.
