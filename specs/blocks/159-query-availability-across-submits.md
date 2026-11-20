# Block 159 — Query availability persists across submits

Status: **COMPLETE (2026-11-20)**. Phase Review (with Blocks 160–161): 0 CRITICAL; findings in Block 161.
Source: webgpu-native-cts finding **F-171**
(`webgpu:api,operation,command_buffer,queries,occlusionQuery:occlusion_query,multi_resolve:*`
fails on Metal and MoltenVK with `result(2): queryIndex: 1, passed: false, expected: true`;
Dawn passes).

## Problem

`resolveQuerySet` zero-fills the destination range and then copies only the queries
listed in `HalResolveQuerySet::written_queries`. Core computes that list from the
command ops of the **current submit only**:

- occlusion: `resolve_written_occlusion_queries` (`yawgpu-core/src/queue.rs`) scans every
  `RenderPass` op in the slice it is given;
- timestamp: the inline filter over `WriteTimestamp` ops next to it;
- the submit walk in `Queue::submit` passes `&all_ops[..=op_index]` for occlusion and
  only the current command buffer's prefix for timestamps.

A query written in an earlier submit and resolved in a later one therefore resolves to
`0`. In `multi_resolve`, the second submit's render pass uses the occlusion set without
beginning any query, so every odd query in `readBuffer3` reads `0`.

The WebGPU model (and Dawn's `QuerySetBase::GetQueryAvailability` /
`SetQueryAvailability`, set at submit) is that a query stays available once a submitted
command has written it, until the query set is destroyed. All three HALs already keep
the written value across submits: Vulkan resets only the query being rewritten
(`vkCmdResetQueryPool` per index), the Metal visibility buffer and counter sample buffer
are never cleared, and the GLES result array persists. The fix is core-only.

## Rules

R1. **Persistent availability on the query set.** `QuerySetInner` holds the set of
available query indices (one bit per query, sized from `count`; at most 4096 queries per
set). It starts empty and only grows; it is never cleared by a resolve.

R2. **Availability at the resolve point.** During the submit walk, the
`written_queries` for a `ResolveQuerySet` op are, intersected with
`[first_query, first_query + query_count)` and sorted ascending without duplicates:
- every query of that set available from previous successful submits (R1), plus
- every query of that set written by an op **strictly before** the resolve in this
  submit, across all command buffers of the submit in order.

A query written only by an op *after* the resolve is not included (today the occlusion
path includes it, so an unwritten query copies uninitialised visibility memory). The
per-command-buffer prefix special case for timestamp resolves is removed; timestamp and
occlusion follow the same rule.

R3. **What counts as a write.** An occlusion query is written by a render pass that
has the set as `occlusion_query_set` and whose `written_occlusion_queries` contains the
index (i.e. a begin/end pair ran for it, whether or not anything passed). A timestamp
query is written by a `WriteTimestamp` op for that set and index, including those the
core lowers from pass `timestampWrites`. Verify where pass `timestampWrites` are
lowered and make sure they are counted; if they are not represented as
`CommandExecution::WriteTimestamp`, count them where they are.

R4. **Commit only on success.** The writes recorded during the walk are merged into the
persistent availability (R1) only when the submit's HAL execution succeeds, the same
point at which the lazy-init overlay commits (Block 153 R1). A submit rejected by
validation, a lowering error, or a HAL error leaves availability unchanged.

R5. **Error and destroyed sets.** An error query set (no HAL object) keeps today's
behaviour (its resolve is skipped). A destroyed set is already rejected at submit; no
availability handling is needed for it.

R6. **Unchanged.** `HalResolveQuerySet`, every HAL's resolve encoding, the timestamp
conversion pass, and validation are unchanged.

## Slices

S1: R1–R6 in `yawgpu-core` (`query_set.rs`, `queue.rs`, and `command_encoder.rs`/
`pass.rs` only if R3 needs it). Inline unit tests on the Noop backend, asserting the
lowered `HalResolveQuerySet::written_queries` (Noop results stay all-zero, so the lowered
list is the observable). Factor the walk so the lowered copies of a submit can be
inspected in tests, or reuse an existing test hook; do not add a public API for it.
Tests:
- occlusion: write odd queries in submit 1, resolve in submit 2 with a pass that uses the
  set but begins no query → `written_queries` = the odd indices (the F-171 shape);
- timestamp: `WriteTimestamp` in submit 1, resolve in submit 2 → included;
- resolve before the writing pass in one command buffer → not included; a second resolve
  after it in the same command buffer → included;
- writes in command buffer A, resolve in command buffer B of the same submit → included
  (timestamps too);
- range intersection: available queries outside `[first, first+count)` are excluded;
- R4: a submit that fails (e.g. a destroyed buffer or a lowering error) does not mark
  its writes available for a later submit;
- two different query sets do not share availability.

S2: Metal zero-fills draw-free occlusion queries (Dawn toggle
`MetalFillEmptyOcclusionQueriesWithZero`, crbug.com/dawn/1707, enabled by default on
every Apple GPU in `PhysicalDeviceMTL.mm`; implementation in `CommandBufferMTL.mm`,
`EmptyOcclusionQueries`). S1 makes every completed begin/end pair available, but Apple
GPUs may leave the previous result in the visibility buffer when a query encloses no
draw. The M2 used for S1 verification wrote zero (CTS `occlusion_query,empty` passes), so
this is parity with Dawn for other Apple GPUs, not a fix for an observed M2 failure.
- R7. While encoding a render pass, the Metal HAL records each occlusion query index
  whose begin/end pair enclosed no draw command (`Draw`, `DrawIndexed`, `DrawIndirect`,
  `DrawIndexedIndirect`, or an executed bundle containing any draw; mirror what Dawn
  counts). After the pass's render encoder ends, it encodes one blit
  `fillBuffer:range:value:0` of 8 bytes at `index * 8` of the set's visibility buffer per
  such index, before any later command of the submission. Apply the same rule to every
  Metal render-pass encoding path that honours `occlusion_query_set` (check the tiled
  subpass path too).
- Tests: a GPU-free helper that derives the empty-query list from a command stream
  (non-ignored), and an `#[ignore = "manual real Metal backend test"]` test that writes a
  non-zero result with draws, then runs a draw-free begin/end on the same index in a
  later submission and resolves `0`.

## Verification

- Agent gates (redirect output to a file; report exit codes):
  `cargo test -p yawgpu-core --lib`, `cargo test -p yawgpu --lib`,
  `cargo clippy --workspace --all-targets -- -D warnings`.
- Claude, on review: `cargo test --workspace`; real-GPU `e2e_metal_*` / `e2e_vulkan_*`
  query targets; CTS `webgpu:api,operation,command_buffer,queries,*` on Metal and
  MoltenVK (`multi_resolve` must pass for occlusion and timestamp).

## Verification record

S1 (2026-11-20, M2): `cargo test --workspace` 1500/0; fmt and clippy clean. Real
Metal: `e2e_metal_timestamp_query` 6/6, including
`metal_stamp_written_in_one_submit_resolves_in_a_later_submit`. MoltenVK under the
Khronos validation layer: `e2e_vulkan_timestamp_query` 5/5, no VUIDs. CTS
`webgpu:api,operation,command_buffer,queries,*` on Metal: every `occlusionQuery` case
passes (244), including `multi_resolve` (F-171), `empty` and `reuse:noop="after"`.

S2 (2026-11-20, M2): `yawgpu-hal --features metal` occlusion tests 5/5 including the
real-device `metal_queue_empty_occlusion_query_clears_previous_result`; clippy and fmt
clean; CTS `occlusionQuery:*` on Metal 244/244.

Measured while verifying S2 (M2, two-query set, both slots first written with draws):
a later pass that begins only query 1 (with a draw) leaves query 1 correct but
**writes 0 to query 0**, which that pass never used; a pass that begins only query 0
leaves query 1 intact, and a pass without occlusion queries leaves both intact. The
Apple GPU writes every visibility slot below the pass's highest used offset. Dawn
attaches the query set's visibility buffer to the pass the same way. No CTS case covers
it. Tracked separately (not part of this block).
