# Whole-codebase code review — 2026-10-25

A read-only review of all Rust crates (`yawgpu`, `yawgpu-core`,
`yawgpu-hal` Noop/Metal/Vulkan/GLES, `yawgpu-tint`) at `bead72b`, split
into six areas with one fresh reviewer each. Claude re-verified the items
marked **verified** against the code; **repro** means a probe or e2e run
reproduced it; the rest are reviewer claims still to be confirmed when
their block is planned.

Fix blocks: **Block 109** (CRITICAL) — **COMPLETE 2026-10-25** (`eee477b`,
`2e25131`, `f759bbd` + Phase Review fixes); later blocks take the MAJOR tiers.
**Block 110** (core / FFI MAJORs + 3 core MINORs) — COMPLETE 2026-10-25.
**Block 111** (Metal / Vulkan / Tint / GLES MAJORs) — slices committed
2026-10-25; first Vulkan sync-validation run found and fixed 3 more
pre-existing hazard classes (R16–R18). **Block 112** (queue submit serialization + writable-binding aliasing) —
COMPLETE 2026-10-25. Still open: GLES context release across threads,
Vulkan per-submit object churn + retire waits under `queue_access` (perf),
Tint double IR lowering (perf), MINOR list; owed: GLES Block 109/111 e2e
on Windows ANGLE.
Block 109 also closed the Vulkan MAJOR "transient attachments end in
`TRANSFER_SRC_OPTIMAL`" (R7) and added a surface double-acquire fix (R8).

## CRITICAL → Block 109

| ID | Area | Finding | Status |
|---|---|---|---|
| C1 | GLES HAL | `gles/` never enables `GL_DEPTH_TEST` or calls `glDepthFunc` / `glDepthMask` / `glPolygonOffset` per draw; depth-tested pipelines render with no occlusion and no depth writes, silently | verified (grep) |
| C2 | Vulkan HAL | the per-sync-index in-flight fence is reset at acquire while an older retire-ring entry (the present transition) still holds it → ring waits forever on a reset fence. Hangs with 3 present-only frames then a rendered frame (any image count), with all-render frames at 2 images, and on reconfigure after an unpresented acquire | **repro** on MoltenVK (`e2e_vulkan_surface` Block 109 tests time out) |
| C3 | FFI | `Drop for WGPUBufferImpl` / `WGPUTextureImpl` call `core.destroy()`; releasing the last C handle while a command buffer / view / bind group still uses it makes the next submit fail ("destroyed buffer/texture"). Dawn does not destroy on last external release | **repro** (Noop probe) |
| C4 | FFI | `slice::from_raw_parts(NULL, 0)` in `wgpuQueueWriteBuffer` and the three `SetImmediates` entry points (UB; debug build aborts on the legal `(NULL, 0)` call) | **repro** (Noop probe) |
| C5 | FFI | `validate_render_pass_descriptor_devices` builds a slice from a null `colorAttachments` with non-zero count (null deref); `map_render_pass_descriptor` / bundle `colorFormats` silently decode null+count as holes | **repro** (Noop probe) |

## MAJOR (later blocks)

### core validation
- BGL entries are not sorted by `binding`; dynamic offsets, alignment and HAL binding follow declaration order (`bind_group_layout.rs:327`, `pass.rs:1865`, `queue.rs:2892`) — verified
- dynamic offset bounds ignore `effectiveSize` for whole-size bindings (`pass.rs:1925`) — verified
- attachment signatures don't trim trailing null color formats (`render_pipeline.rs:51`) — verified
- writable-binding aliasing check at draw/dispatch missing (render under-strict, compute over-strict on disjoint ranges)
- map race: mapAsync → unmap → mapAsync swaps callback outcomes and leaves the buffer unmapped (`buffer.rs:481-620`) — repro (probe)
- every buffer allocates + zero-fills full-size host storage (`buffer.rs:391`) — verified
- mapAsync with a null callback never resolves the pending map (`ffi/mod.rs:1971`) — verified
- concurrent `Queue::submit` lowering vs HAL submit order (lazy-init clears, layout tracking) not serialized

### FFI
- `dstFactor` Undefined maps to `One` (spec: `Zero`) (`conv/pipeline.rs`) — verified; several enum mappers default invalid values silently
- pending callbacks never cancelled on instance teardown; Arc cycle via pending results

### Metal HAL
- color / resolve attachments ignore `view_format` (`metal/encode.rs:979`) — verified
- 2D-array ↔ 3D texture copy issued as one blit with z on the 2D side (`metal/encode.rs:577`) — verified
- tiled subpass drops resolve target / level / slice (`tiled` only)

### Vulkan HAL
- acquire semaphore waited at `COLOR_ATTACHMENT_OUTPUT` only (`encode.rs:155`) — verified
- no outgoing dependency for vertex/fragment storage writes after a render pass
- transient attachments end in `TRANSFER_SRC_OPTIMAL` in the non-tiled path
- layout tracking committed at record time, not at successful submit

### Tint shim
- workgroup storage size measured over the whole module, not the entry point; failures return `Ok(0)` (`tint_shim.cpp:2376`) — repro (probe)

### GLES HAL
- context never released after `with_current_context` (second thread fails)
- render-pass viewport / resolve rects use mip 0 size
- scissor test left enabled into resolve / present blits
- primitive restart never enabled
- buffer→texture copy ignores the stencil aspect

## MINOR / IMPROVEMENT

Recorded in the review transcript summary; re-raised when the owning
block is planned (Metal `StorageModeShared` textures, Vulkan per-submit
object churn, per-draw validation redundancy, `wgpuInstanceWaitAny`
busy-spin, `destroy()` not freeing GPU memory, Tint shim exception
safety, `build.rs` rerun triggers, etc.).
