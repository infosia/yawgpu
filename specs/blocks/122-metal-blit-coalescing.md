# Block 122 — Metal: one blit encoder per run of copies

Status: **COMPLETE (2026-11-02)**. Phase Reviews: S1 and S1b each 0 CRITICAL, 0 MAJOR,
3 MINOR; all fixed. P-006's first diagnosis was wrong; R6 records the measured cause.
Source: backlog E14, perf finding P-006 (`specs/tracking/perf-dawn-baseline.md`
Run 8). `queue/write_buffer_4kb` is 2.98× Dawn on Metal: yawgpu 1,935 ns, Dawn 649 ns.
Oracle: Dawn at the pinned `third_party/dawn`:
`native/metal/CommandRecordingContext.mm` (`EnsureBlit`, `EndBlit`),
`native/metal/DeviceMTL.mm` (`CopyFromStagingToBuffer`).

## Problem

`MetalQueue::submit_copies` (`yawgpu-hal/src/metal/queue.rs`) opens a new
`MTLBlitCommandEncoder` for every blit-type `HalCopy` and ends it after one command.
Queue writes reach the HAL as one `HalCopy::Buffer` each, so a submit carrying N
writes creates N encoders. Since Block 121, a write loop flushes about 4,096 copies
at its 16 MiB threshold, and creating an encoder per copy costs about 1.3 µs each.

Dawn keeps one blit encoder open in its recording context (`EnsureBlit`) and records
every staging copy into it. It ends the encoder only when another encoder type is
needed or the command buffer is committed.

## Rules

R1. In `submit_copies`, a maximal run of consecutive copies of the kinds below is
encoded into a single blit encoder, in order:
- `HalCopy::Buffer`
- `HalCopy::BufferClear`
- `HalCopy::BufferToTexture`
- `HalCopy::TextureToBuffer`
- `HalCopy::TextureToTexture`

R2. The open blit encoder is ended before any other `HalCopy` kind is encoded, and
before the command buffer is committed. Every other kind keeps its current encoding
unchanged. That includes `WriteTimestamp`, whose blit encoder carries a sample-buffer
descriptor, and `ResolveQuerySet`, which may encode an event signal and wait first.

R3. Queue order holds within a run. A later copy observes the effect of an earlier
one, including writes to the same range. Metal's hazard tracking on tracked
resources provides this, and Dawn relies on the same within its single encoder.
Verify that yawgpu creates no untracked resource that a copy can reach. If one
exists, R1 must not include it in a run.

R4. If encoding a copy fails, the open encoder is ended before the error is
returned, as each arm does today.

R5. The grouping decision is a pure function of the copy kinds, unit-tested without a
GPU: runs, boundaries, an empty list, and a run at the start and at the end.

R6. **The measured cost is in core, not in the encoder.** With R1-R5 in place,
`queue/write_buffer_4kb` stayed at 1,951 ns. A `sample` profile put the largest
self time in `PendingWriteBatch::recycle_completed` (`yawgpu-core/src/queue.rs`). That
function runs on every write, and it moves every in-flight staging chunk into a newly
allocated `Vec` even when none has completed. After a 16 MiB flush, hundreds of
chunks are in flight, so each write pays O(in-flight) work plus an allocation.
Recycling must cost O(completed chunks) per call, and nothing when none completed.
Chunks are retired in non-decreasing submission-index order, because `retire` runs
under the `submission` lock with the index just recorded. A queue ordered by index can
therefore stop at the first chunk that is still in flight. A unit test pins the order
the recycle relies on. R1-R5 stay: they match Dawn and cost nothing.

## Tests

- Unit (R5): the grouping function over mixed sequences.
- Metal HAL `#[ignore]` (R1, R3): one submit with many buffer copies into the same
  range, read back to check that the last one wins. A mixed sequence of buffer copy,
  buffer clear, texture upload, texture readback, and a compute pass reads back the
  expected values.
- Existing Metal e2e and the Block 121 e2e pass, especially
  `writes_across_the_threshold_keep_queue_order`, which submits about 4,096 copies
  to 8 regions in one flush.
- Timestamp e2e (`e2e_metal_timestamp_query`) passes, because R2 keeps its encoders
  separate.
- Bench: `queue/write_buffer_4kb` near Dawn's figure, and no regression elsewhere.
- CTS on Metal, `webgpu:*`: `fail=2 crash=0`.

## Out of scope

- Vulkan. Each copy adds a memory barrier there, but all copies share one command
  buffer. It is not measured for this cost.
- Coalescing across command buffers or submits.

## Slices

- **S1 (coding agent):** R1-R5 with the unit test and the Metal HAL ignored tests.
- **S1b (coding agent):** R6, with its unit tests.
- **S2 (Claude):** real-GPU runs, bench, CTS, Phase Review. Also investigate
  `pipeline/compute_unique` (0.97× in Run 1, 1.12-1.15× in Runs 7 and 8) and record
  the result in `specs/tracking/perf-dawn-baseline.md`.

## Verification (2026-11-02)

Run on Apple M2, macOS 27.2.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,307 passed, 0 failed, with the retire-order `debug_assert!` live |
| fmt, clippy on default, `metal`, `metal,tiled`, `metal,shader-passthrough`, `vulkan`, `vulkan,tiled`, `vulkan,shader-passthrough`, `gles` | clean |
| Metal HAL ignored tests | 61 passed; the two new ones also pass under `MTL_DEBUG_LAYER=1` |
| Metal e2e, `metal,tiled,shader-passthrough` | 150 passed, 0 failed |
| Vulkan e2e, MoltenVK under the Khronos layer | 138 passed, 1 failed (known `imageView2DOn3DImage`) |
| webgpu-native-cts, Metal, `webgpu:*` | `fail=2 crash=0`; subcase counts unchanged |
| Bench `queue/write_buffer_4kb` | 1,935 ns to 1,058 ns (Dawn 649 ns) |

R1-R5 alone left `queue/write_buffer_4kb` at 1,951 ns. The gain came from R6. The
GPU tests pin ordering inside a coalesced run, not the coalescing itself: they also
pass without it.

The verification also found a broken test from Block 119. The
`shader-passthrough`-only test
`resolve_shader_passthrough_render_pipeline_uses_explicit_layout_and_entries` still
destructured four elements after `ResolvedRenderPipelineParts` gained a fifth, so
`clippy --features metal,shader-passthrough` failed to compile it. It is fixed here.
Earlier blocks reported that configuration as clean, and that report was wrong.

## Phase Review

| Slice | ID | Finding | Disposition |
|---|---|---|---|
| S1 | m1 | Grouping test re-implemented the grouping; predicate had a wildcard | Exhaustive `match`; test asserts every non-blit kind |
| S1 | m2 | No write-after-read step in a run | Added to the mixed GPU test |
| S1 | m3 | Arm list and predicate kept apart | Invariant comment on the predicate |
| S1b | m1 | Retire-order check only in core unit tests | Gated on `debug_assertions` |
| S1b | m2 | Order test missed four retire paths | Threshold flush, rejected submit, lowering failure with copies, `wait_idle` added |
| S1b | m3 | No test that an order violation recycles late, never early | Out-of-order `in_flight` test added |

## S2: `pipeline/compute_unique`

A `sample` profile puts about 98% of the case in the Metal compiler service, through
`newLibraryWithSource` and `newComputePipelineState`. Tint takes about 2%. The case
reuses each shader source in every batch, because the constant comes from the
iteration index (`benches/src/main.cpp`, `uniqueComputeWgsl`). It therefore measures
Metal's compiled-shader cache lookups, not unique compiles: truly unique sources take
about 5 ms each. The move from 0.97× to 1.15× is in the Metal compiler service, across
macOS 26.5 to 27.2. yawgpu has no lever on it short of an in-process `MTLLibrary`
cache (backlog E3).

Dawn's MSL also differs: Dawn prepends `#pragma METAL fp math_mode(relaxed)` on
macOS 15 and later, unless strict math is requested. yawgpu leaves Metal's default
math mode. This did not change compile time in a direct test, but it may change
floating-point semantics. It is recorded as backlog A15.

