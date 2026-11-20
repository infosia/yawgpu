# Block 162 — Metal render passes write occlusion results through a per-pass scratch buffer

Status: **COMPLETE pending Phase Review (2026-11-20)**.
Source: measured while verifying Block 159 S2 (see that block's verification record).
No CTS case covers it; Dawn attaches the query set's visibility buffer the same way and
is expected to share the defect.

## Problem

The Metal HAL sets the occlusion query set's own visibility buffer as the render pass
descriptor's `visibilityResultBuffer`, and `BeginOcclusionQuery { index }` sets
`MTLVisibilityResultModeCounting` at offset `index * 8`. Measured on the M2
(2026-11-20, two-query set, both slots first written with draws):

| later pass | slot 0 | slot 1 |
|---|---|---|
| begins only query 1, with a draw | **0** (never used by the pass) | correct |
| begins only query 0, with a draw | correct | kept |
| begins only query 0, no draw | 0 | kept |
| no occlusion query | kept | kept |

The GPU writes every visibility slot below the pass's highest used offset. A query
written by an earlier pass or submit, and still available (Block 159), is therefore
overwritten with 0 by a later pass that uses only higher indices of the same set.

## Rules

R1. **Scratch visibility buffer per pass.** For a render pass with an occlusion query
set, the Metal HAL assigns each distinct query index begun in the pass a dense slot
`j = 0, 1, …` in begin order, allocates a private `MTLBuffer` of `max(k, 1) * 8` bytes
(`k` = number of distinct begun indices), and sets it as the descriptor's
`visibilityResultBuffer`. `BeginOcclusionQuery { index }` uses offset `j * 8`. The query
set's own visibility buffer is never attached to a render pass. The scratch buffer needs
no explicit lifetime handling beyond encoding: command buffers are created with retained
references (`commandBuffer`), which keep it alive until completion; keep that invariant
documented next to the allocation.

R2. **Copy back only the used slots.** After the render encoder ends, one blit encoder
copies 8 bytes from scratch slot `j` to `index * 8` of the set's visibility buffer for
each query that enclosed a draw, and fills `index * 8` with zero for each draw-free
query (Block 159 S2's rule, now applied to the set's buffer instead of relying on
Metal). No other slot of the set's buffer is written. Passes with an occlusion query set
but no begun query attach no visibility buffer and encode no blit.

R3. **Every Metal render-pass path.** Apply R1/R2 to every Metal path that honours
`occlusion_query_set` (the command-stream render pass; the tiled subpass path has no
occlusion queries today — keep it that way and assert it in a comment only if the type
still lacks the field). Render bundles cannot begin occlusion queries (validation), so
bundle replay only counts draws.

R4. **Unchanged.** Core, availability, resolve encoding (the resolve still reads the
set's visibility buffer), Vulkan/GLES.

## Slices

S1: R1–R4 in `yawgpu-hal/src/metal/`. Tests:
- GPU-free: the dense slot assignment for a command stream (repeat begins are not
  possible after validation; still map by first occurrence), and the copy/fill plan
  (draw vs draw-free) derived from it;
- `#[ignore = "manual real Metal backend test"]`: a three-query set written with draws in
  pass A (all non-zero); pass B begins only query 2 with a draw; resolve shows slots 0
  and 1 unchanged and slot 2 non-zero; pass C begins only query 1 without a draw;
  resolve shows slot 1 = 0 and slots 0, 2 unchanged. Keep the existing S2 test passing.

## Verification

- Agent gates as in Block 159 S2 (Metal HAL lib tests, Metal + tiled clippy, workspace
  clippy, fmt).
- Claude, on review: the ignored Metal tests on the M2; CTS
  `webgpu:api,operation,command_buffer,queries,occlusionQuery:*` on Metal (244 pass);
  `e2e_metal_*` render targets that use occlusion queries, if any.

## Verification record

S1 (2026-11-20, M2): `yawgpu-hal --features metal` lib tests including every ignored
real-device test 287/287, among them `metal_queue_occlusion_pass_preserves_unused_query_slots`
(three-query set; a pass using only query 2, then a draw-free pass using only query 1, leave
the other slots intact) and the Block 159 S2 regression. Clippy (default, metal, metal+tiled)
and fmt clean.
