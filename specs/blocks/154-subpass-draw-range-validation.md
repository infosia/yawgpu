# Block 154 — Subpass draws run the regular draw-range validation

Status: **PLANNED (2026-11-13)**.
Source: whole-code review at `92e38c7` (`REPORT-review.md`), finding R08.
Oracle: subpass rendering (`tiled`) has no Dawn counterpart; each rule copies what
yawgpu's regular render pass already does (`pass.rs` `validate_render_draw_state`, which
follows Dawn `CommandBufferStateTracker::ValidateIndexBufferInRange` / `ValidateBufferInRange`
and `RenderPassEncoder` strip-index-format checks).

## Problem

`validate_subpass_draw_state` (`subpass.rs` ~833) checks bind groups, required buffer
sizes, the presence of used vertex buffers, and the presence of an index buffer. It
uses only `kind.is_indexed()` from the `RenderDrawKind`, so it skips the three checks
that `validate_render_draw_state` (`pass.rs` ~677) runs for a regular pass:
`validate_strip_index_format`, `validate_vertex_buffer_oob`, `validate_index_buffer_oob`.
A subpass draw that reads past a bound vertex or index range, or a strip pipeline whose
`strip_index_format` disagrees with the bound index format, is recorded and sent to the
HAL instead of being rejected.

## Rules

R1. **Same checks, same errors.** `validate_subpass_draw_state` runs, after its existing
checks: strip-index-format validation for indexed draws, then — per draw kind, exactly as
`validate_render_draw_state` —
- `Direct`: vertex/instance range against every used vertex buffer;
- `IndexedDirect`: index range against the bound index buffer, then the instance range
  against the instance-step vertex buffers;
- `Indirect` / `IndexedIndirect`: no range check (unchanged).
The error messages match the regular pass's.

R2. **One implementation.** The three helpers stop taking `&PassEncoderState` and take
only what they read (the pipeline plus a view of the bound vertex buffers and index
buffer: slot → (offset, size) and the index format/offset/size), so both passes call the
same code. A second copy of the range arithmetic is out of scope. The regular pass's
behaviour and messages are unchanged.

## Slices

S1: R1–R2 in `yawgpu-core` (`pass.rs`, `subpass.rs`), inline tests under
`--features tiled`.

## Verification

- Inline tests in `subpass.rs`: a direct draw past the vertex range, an instance range past
  an instance-step buffer, an indexed draw past the index range, and a strip pipeline with
  a mismatched index format are each rejected with the regular pass's message; the same
  draws within range are accepted; indirect draws are unaffected. The existing `pass.rs`
  tests keep passing unchanged.
- Agent gates: `cargo test -p yawgpu-core --features tiled --lib`,
  `cargo test -p yawgpu-core --lib`,
  `cargo test -p yawgpu --features tiled --test tiled_subpass_ffi`, clippy with and
  without `tiled`.
- Claude: workspace test; real-GPU `e2e_metal_tiled*` and `e2e_vulkan_tiled*`, and the
  `tiled` examples with `--verify`, to confirm no valid draw is now rejected.
