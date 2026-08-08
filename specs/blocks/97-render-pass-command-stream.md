# Block 97 — Render pass command stream

Closes finding **P-003** from `specs/tracking/perf-dawn-baseline.md`.

## The problem

`HalRenderPass` (`yawgpu-hal/src/command.rs:426`) describes a whole render pass
*and one draw*: colour targets, framebuffer-fetch slots, depth-stencil
attachment, bind buffers/textures/samplers/external textures, vertex buffers,
index buffer, indirect buffer, viewport, scissor, blend constant, stencil
reference, occlusion query state, immediate data — plus a single
`draw: Option<HalDraw>`.

`RenderPass::draw` (`yawgpu-core/src/render_pass.rs`) therefore emits one
`RenderPassCommand` per draw, each snapshotting the entire pass state:
`bind_group_layouts().to_vec()` plus clones of `attachment_texture_uses`,
`attachment_textures`, `bind_groups`, `vertex_buffers`, `index_buffer`,
`occlusion_query_set` and `immediate_data`, and a reload of the attachment list.

No backend merges consecutive entries. `MetalQueue::submit_copies`
(`yawgpu-hal/src/metal/queue.rs:307`) creates a
`renderCommandEncoderWithDescriptor`, encodes, and calls `endEncoding` for every
`HalCopy::RenderPass`; the Vulkan path is equivalent.

Two costs follow, and the second is the one that matters:

- **CPU**: ~7 heap allocations and a batch of `Arc` traffic per draw. Measured
  at 148 ns/draw against Dawn's 42 ns — 3.5×.
- **GPU**: a WebGPU render pass with N draws executes as N GPU render passes,
  each loading and storing its attachments. On a tiled GPU that resolves tile
  memory once per draw. Measured by `submit/render_100_draws_wait` at 5.14 ms
  against Dawn's 0.25 ms — **20.4×**.

The render-pass-per-draw `storeOp` forcing already recorded in the CTS work is a
symptom of this shape, not an independent quirk.

## Target shape

One pass, one GPU encoder, a stream of commands inside it — what Dawn, wgpu and
every native API do.

- **R1** The HAL gains a render-pass type that carries pass-level state once
  (attachments, load/store ops, occlusion query set) plus an ordered list of
  commands executed inside it: set-pipeline, set-bind-group, set-vertex-buffer,
  set-index-buffer, set-viewport, set-scissor, set-blend-constant,
  set-stencil-reference, set-immediates, begin/end-occlusion-query, draw,
  indexed draw, indirect draw, and render-bundle execution.
- **R2** A backend creates **one** encoder per pass and replays the command
  list into it. `endEncoding` happens once, at the end of the pass.
- **R3** Core records only what changed. Setting a pipeline emits a
  set-pipeline; a draw emits a draw. No draw may clone whole-pass state.
- **R4** Pass-level `loadOp`/`storeOp` are honoured once, at the real pass
  boundaries. The per-draw `storeOp` forcing that exists today to keep
  consecutive one-draw passes from discarding each other's output must be
  removed, not carried forward — it is only correct for the current shape and
  becomes a bug in the new one.
- **R5** Validation is unchanged. Every rule that fires today must fire
  identically, with the same message, and the usage-scope tracking that
  `record_pipeline_usage_scope` performs must keep the same semantics.

## Risk

This is the highest-risk change in the performance work so far. The current
shape means every draw re-establishes complete state; the new one carries state
across draws inside a pass, so any state a backend fails to set — or wrongly
retains between passes — becomes a wrong-pixels bug rather than a crash.

- **R6** Real-GPU verification is mandatory on both Tier 1 backends. Noop cannot
  observe any of this.
- **R7** `webgpu-native-cts` must be re-run on Metal and Vulkan/MoltenVK and may
  not move a single pass/fail count. The `api,operation,rendering` and
  `api,operation,render_pass` trees are the load-bearing ones here and must be
  included alongside the trees the Block 96 baseline used.
- **R8** Lazy zero-init (F-138) interacts directly with load/store ops. Its
  existing behaviour must be preserved; the failure mode recorded for it was an
  *inverse* bug where a spurious read-clear wiped just-written data, which only
  appeared on real hardware.

## Slices

- **S1** Introduce the HAL command-stream types alongside the existing
  `HalRenderPass`, with the Noop backend consuming them. No core changes, no
  behaviour change.
- **S2** Core emits the command stream: pass-level state once, per-draw deltas.
  Metal and Vulkan encoders replay it into a single encoder per pass. Remove the
  per-draw `storeOp` forcing (**R4**).
- **S3** Delete the old per-draw `HalRenderPass` path once nothing emits it.
- **S4** Re-measure and re-run CTS on both backends; record a new run section in
  `specs/tracking/perf-dawn-baseline.md`.

Each slice ends green on Noop; S2 and S4 additionally require real-GPU runs.

## Expected outcome

`submit/render_100_draws_wait` should fall from 20.4× to roughly parity, and
`encode/render_draw` from 3.5× to the same order as Dawn's 42 ns. The GPU-side
figure is the one to quote; the recording figure is secondary.
