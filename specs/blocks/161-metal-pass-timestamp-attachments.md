# Block 161 — Metal samples pass `timestampWrites` through the pass descriptor

Status: **COMPLETE pending Phase Review (2026-11-20)**.
Source: webgpu-native-cts finding **F-170**, remainder after Block 160:
`timestampQuery:many_query_sets:numQuerySets=65536` (both stages).

## Problem

Core records a pass's `timestampWrites` as two `CommandExecution::WriteTimestamp` ops
around the pass (Block 102), and the Metal HAL encodes each `HalCopy::WriteTimestamp`
in its own encoder: in stage-boundary mode a 1-byte mock blit encoder whose
`startOfEncoderSampleIndex` samples the stamp; in command-boundary mode a blit encoder
with `sampleCountersInBuffer:atSampleIndex:withBarrier:`. A pass with both stamps
therefore costs three encoders.

With 65,536 query sets in one command buffer (one pass each, begin and end stamps):
- `compute`: the command buffer fails and the device is lost. Measured on the M2
  (2026-11-20, empty compute passes with begin/end stamps): 49,152 passes succeed,
  63,488 fail, independent of the counter-sample-buffer count (31 pool buffers).
  Dawn passes this case.
- `render`: the Metal driver aborts the process with the assertion
  `Failed to allocate IOGPUDeviceShmem` (`IOGPUMetalDeviceShmem.m:43`). Dawn loses the
  device instead (F-169, in Dawn's own expectations).

Dawn (`third_party/dawn/src/dawn/native/metal/CommandBufferMTL.mm`,
`SetSampleBufferAttachments`, `CreateMTLComputePassDescriptor`,
`CreateMTLRenderPassDescriptor`) samples pass timestamps through the pass descriptor's
`sampleBufferAttachments[0]` in stage-boundary mode: compute
`startOfEncoderSampleIndex` / `endOfEncoderSampleIndex`, render
`startOfVertexSampleIndex` / `endOfFragmentSampleIndex`, `MTLCounterDontSample` for an
absent index. One encoder per pass.

## Rules

R1. **Pass timestamp writes reach the HAL pass command.** The HAL compute-pass and
render-pass commands (and the tiled subpass render-pass command, if it can carry
`timestampWrites`) gain an optional `HalPassTimestampWrites { query_set, beginning_index:
Option<u32>, end_index: Option<u32> }`, filled by core lowering from the pass's
`timestampWrites`.

R2. **Boundary stamps stay explicit for the other HALs.** Core still emits the two
`WriteTimestamp` ops around the pass, now marked as pass-boundary stamps
(`HalWriteTimestamp::pass_boundary: bool`, false for `writeTimestamp`). Vulkan and GLES
keep executing them exactly as today and ignore the field in R1. Query availability
(Block 159) is unchanged: it still counts these ops.

R3. **Metal stage-boundary mode uses the descriptor.** When the device uses
stage-boundary sampling, the Metal HAL skips pass-boundary `WriteTimestamp` copies and
instead sets `sampleBufferAttachments[0]` on the pass descriptor from R1, with sample
indices through `MetalQuerySet::sample_index` (Block 160): compute
`startOfEncoderSampleIndex` / `endOfEncoderSampleIndex`, render
`startOfVertexSampleIndex` / `endOfFragmentSampleIndex`, and `MTLCounterDontSample` for an
absent index. A compute pass then needs a descriptor-based compute encoder
(`computeCommandEncoderWithDescriptor:`). Command-boundary mode and plain
`writeTimestamp` keep today's encoding.

R4. **Real-GPU gate.** Block 102 found that, on the M2, an end-of-encoder sample on a
blit encoder that follows another blit encoder fails the whole command buffer, and that
the shared-event serialization before a timestamp resolve is required. Those findings
were for blit encoders; whether pass-descriptor sampling is affected is unknown. R3 is
accepted only if the real-Metal e2e and CTS checks below pass. If a pass kind fails on
hardware, that pass kind keeps the per-stamp encoders; record the measurement here.
The resolve serialization is unchanged.

R5. **Unchanged.** Validation, availability, the conversion pass, resolve encoding, and
the Vulkan/GLES/Noop execution of pass timestamps.

## Slices

S1: R1–R3 (core lowering + `yawgpu-hal` command types + Metal HAL). Tests:
- core (Noop): lowering a compute and a render pass with `timestampWrites` (begin only,
  end only, both) fills R1 on the pass command and marks the two boundary
  `WriteTimestamp` copies; `writeTimestamp` copies are not marked;
- Metal: a GPU-free helper that maps R1 to the four descriptor indices (base index added,
  `MTLCounterDontSample` for absent), with normal tests; `#[ignore]` real-device tests
  that run a compute pass and a render pass with begin/end stamps after a buffer copy and
  after a clear, and check `0 < begin <= end` after resolve.

S2 (R4 outcome for compute, measured on the M2 2026-11-20 with S1 in place):
- A compute encoder created with sample attachments but **no dispatch** takes no samples:
  both stamps resolve to 0, with or without a preceding blit; the same pass with one or
  more dispatches samples correctly. CTS `many_query_sets` uses empty passes.
- Two attachments on the same sample buffer in one mock blit encoder (both
  start-of-encoder) sample only one of them.
- One mock blit encoder (the existing 1-byte fill) with `sampleBufferAttachments[0]`
  `startOfEncoderSampleIndex = begin` and `endOfEncoderSampleIndex = end` samples both,
  including directly after a buffer copy or clear blit. With that encoding for empty
  compute passes, CTS `queries,*` on Metal gives 263 pass / 1 fail: `many_query_sets:65536`
  compute passes and render loses the device without aborting the process, as Dawn (F-169).

R6. **Empty timestamped compute pass.** In stage-boundary mode, a
`ComputePassCommandStream` with `timestamp_writes` and no dispatch is encoded as one mock
blit encoder (reuse the lazily allocated `mock_blit` buffer and its 1-byte fill) whose
`sampleBufferAttachments[0]` uses the sample buffer from R1 with
`startOfEncoderSampleIndex = begin` and `endOfEncoderSampleIndex = end`
(`MTLCounterDontSample` when absent). Passes with dispatches keep the S1 descriptor
encoding. Tests: the existing real-device test runs empty and non-empty passes
interleaved (`[d], [], [d], [], [d, d]`) and requires `0 < begin <= end` for each; the
descriptor-index helper covers the blit case.

Claude, on review (R4): `e2e_metal_timestamp_query` (all), `e2e_vulkan_timestamp_query`
on MoltenVK, CTS `timestampQuery:*` on Metal including `many_query_sets:65536` (compute
must pass; render must at least not abort the process), the probe of 65,536 empty
compute passes, and a full `webgpu:*` sweep before closing F-170.

## Verification record

S1 + S2 (2026-11-20, M2): `cargo test --workspace` 1502/0; fmt and clippy clean;
`yawgpu-hal --features metal` lib tests including every ignored real-device test 286/286;
all `e2e_metal_*` targets (except `leak`, `compile_churn`, `surface`,
`shader_passthrough`) 170/170. MoltenVK under the Khronos validation layer, all
`e2e_vulkan_*` targets: 163/164; the one failure
(`e2e_vulkan_layouts::vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`) and all
8 VUIDs (`imageView2DOn3DImage-04459`, `vkDestroyDevice-device-05137`) reproduce
identically at the previous commit (MoltenVK portability limits, unrelated). CTS
`webgpu:api,operation,command_buffer,queries,*` on Metal: 263 pass, 1 fail —
`many_query_sets:numQuerySets=65536;stage="render"` loses the device (no process
abort), matching Dawn's F-169 behaviour; the compute case passes.

R4 outcome: render passes sample through the descriptor; compute passes with dispatches
sample through the descriptor; empty compute passes use R6.

`e2e_metal_timestamp_query::metal_pass_timestamp_writes_with_only_an_end_index_leave_the_other_slot_zero`
no longer requires the render pass's end stamp to follow the compute pass's: the two
passes share no resource and the GPU overlapped them (compute end 15 ms after render end).

## Phase Review (2026-11-20, fresh-context reviewer over `01324f0..f2695f8`, Blocks 159–161)

0 CRITICAL, 1 MAJOR, 3 MINOR.
- M1: an empty timestamped compute pass with only an end index encodes a mock blit with
  `start = DontSample, end = end` — the end-of-encoder-after-blit shape Block 102 measured
  as fatal on the M2; R6 was measured only with both indices. Fix in S3.
- m1 (Block 159 S2): Dawn does not count a draw with `instanceCount == 0` (or
  `indexCount == 0` for indexed draws) as a draw for the empty-query fill; yawgpu does. Fix
  in S3.
- m2 (Block 160): pool fragmentation / per-process limit — recorded under Block 160
  "Known limits".
- m3: block statuses — updated when the phase closes.

S3 (Metal HAL): for an empty pass, sample with start-of-encoder only when one index is
absent (`[end, DontSample]` for end-only, `[begin, DontSample]` for begin-only; keep
`[begin, end]` for both, as measured); add begin-only and end-only empty-pass cases after a
copy and after a clear to the real-device test. In `render_command_draws`, a direct or
indexed draw with `instance_count == 0`, or an indexed draw with `index_count == 0`, is not
a draw (bundle replay included); add unit-test cases.

S3 (2026-11-20, M2): `yawgpu-hal --features metal` lib tests including every ignored
real-device test 287/287; the empty-pass test now covers begin-only and end-only passes
directly after a copy and after a clear. Clippy (default, metal) and fmt clean.
