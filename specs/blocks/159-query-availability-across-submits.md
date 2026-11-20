# Block 159 — Query availability persists across submits

Status: **IN PROGRESS (2026-11-20)**.
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
