# Block 111 — whole-codebase review: HAL / Tint MAJORs

Status: **COMPLETE (2026-10-25)** — S1 `34c89e8`, S2 `77e3197`, S3 `b2ecf46`,
S4 `553776f`, S2b (R16–R18) `45cd4fd`, Phase Review fixes in the closing
commit (0 CRITICAL / 0 MAJOR / 6 MINOR; m3 double-lowering and m6 deferred).
GLES R8–R11 real-GPU: `e2e_gles_class_audit.rs` (Linux NVIDIA EGL, 2026-10-25): R8 (mip-1
viewport, MSAA resolve into mip 1), R9 (resolve under a draw scissor; next-pass
clear), R10 (uint16 strip restart at `0xFFFF`), R11 (combined-DS stencil
upload → Internal error; `Stencil8` upload drives the stencil test) all pass,
and each goes red with its fix reverted — except the R9 `RenderPassCleanup`
reset, which no path observes (pass begin, resolve, lazy-init clears and
present each disable the scissor themselves).
Source: `specs/tracking/code-review-2026-10.md` (Metal, Vulkan, Tint, GLES
MAJOR rows). Claude re-verified each item against the code at `c60989b`
unless marked *(reviewer claim)*.

Four slices with disjoint files, then a Phase Review:

| Slice | Rules | Files | Real-GPU gate (Claude) |
|---|---|---|---|
| S1 Metal | R1–R3 | `yawgpu-hal/src/metal/**` | new `e2e_metal_*` cases on the M2 |
| S2 Vulkan | R4–R6 | `yawgpu-hal/src/vulkan/**` | all `e2e_vulkan_*` on MoltenVK under the layer, incl. sync validation |
| S3 Tint | R7 | `yawgpu-tint/**`, `yawgpu-core/src/shader_tint.rs`, `compute_pipeline.rs` call site | Noop + Metal compute e2e |
| S4 GLES | R8–R11 | `yawgpu-hal/src/gles/**` | `e2e_gles_class_audit` (Linux NVIDIA EGL ☑ 2026-10-25) |

## S1 — Metal

### R1. Color and resolve attachments use the view format

`render_pass_descriptor_parts` (`metal/encode.rs`) attaches the raw
texture even when `HalRenderColorTarget.view_format` /
`resolve_view_format` differ from the texture format (srgb ↔ non-srgb
`viewFormats`). Metal requires the attachment pixel format to match the
pipeline's (which is the view format), and load-op clears / MSAA resolve
are encoded in the texture's format. When the formats differ, attach a
`newTextureViewWithPixelFormat` view (same level / slice handling as
today) for the color attachment and, independently, for the resolve
target — as Vulkan does. Unit-test the decision (formats equal → raw
texture; differ → view) device-free where possible; Claude adds an e2e:
clear an `rgba8unorm` texture through an `rgba8unorm-srgb` view and read
back the sRGB-encoded bytes, plus an MSAA resolve through a reinterpreted
view.

### R2. 2D / 2D-array ↔ 3D texture copies are encoded per slice

When exactly one side of a texture-to-texture copy is 3D
(`metal/encode.rs` copy path), loop over the copy depth and issue one blit
per z with depth 1: the 2D-array side uses `slice = origin.z + i`,
`origin.z = 0`; the 3D side uses `slice = 0`, `origin.z = origin.z + i`
(Dawn `CommandBufferMTL.mm` copy-texture-to-texture). 3D ↔ 3D and
2D ↔ 2D keep their current single blit. Claude adds an e2e copying a
3-layer 2D-array into a 3D texture and back.

### R3. Tiled subpass attachments honour resolve target, level and slice (`tiled`)

The subpass render-pass path (`metal/encode.rs`, `subpass_attachment_texture`
and the persistent-attachment builder) drops `resolve_target` and never
sets level / slice. Either honour them (`setResolveTexture`, resolve
level / slice, `setLevel`, `setSlice`) or reject them with a `HalError` in
the HAL; core validation is unchanged either way. State which in the
report; unit-test the builder.

## S2 — Vulkan

### R4. Acquire semaphore is waited at `ALL_COMMANDS`

The swapchain acquire semaphore wait (`vulkan/encode.rs`, the
`pending_acquire` consumer, and the present-without-render path) uses
`ALL_COMMANDS`, as Dawn does, so the first use of the image (layout
transition, copy, compute storage write) cannot run before the
presentation engine releases it.

### R5. Render-pass shader writes are made visible to later work *(reviewer claim; confirm with sync validation)*

A render pass whose bound pipelines write storage buffers / storage
textures from vertex or fragment stages has no outgoing dependency for
those writes: `dependency_out` covers attachment writes → `TRANSFER`
only. After such a pass (or, simpler, after every pass that had writable
storage bindings), record a global memory barrier
`VERTEX_SHADER | FRAGMENT_SHADER` / `SHADER_WRITE` →
`buffer_write_read_barrier_dst_stage_mask()` and its access mask. Unit-test
the barrier masks; Claude runs the Vulkan e2e set with
`VK_LAYER_KHRONOS_validation` **synchronization validation** enabled
(`VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION` via
`vk_layer_settings.txt` / `VK_LAYER_ENABLES`) before and after, and
records the hazard counts.

### R6. Layout tracking commits only after a successful submit

Image layout tracking (`vulkan/layout.rs`, via `transition_image_range`)
is updated while commands are recorded. If recording or `vkQueueSubmit`
then fails, tracking is ahead of the GPU and the next submission's barrier
uses the wrong `oldLayout`; two threads recording concurrently can also
commit in the opposite order of their submits. Stage layout changes in a
per-submission overlay (read-through to the committed state) and commit
it only after `vkQueueSubmit` succeeds, holding one per-queue lock from
the start of recording to the commit (the core `InitOverlay` pattern).
Unit-test the overlay (commit on success, discard on failure,
read-through).

## S3 — Tint shim

### R7. Workgroup storage size is measured per entry point

`yawgpu_tint_workgroup_storage_size` (`yawgpu-tint/shim/tint_shim.cpp`)
runs `GetWorkgroupInfo` on the whole lowered module: it sums every
`var<workgroup>` in the module and takes the workgroup size from the
*last* entry point, and any override / substitution / info failure
returns `Ok(0)`. Consequences (reproduced by the reviewer's probe): a
32 KiB compute entry point followed by a fragment entry point reports 0
and skips the 16 KiB limit; two compute entry points with 12 KiB each
report 24 KiB and wrongly reject a valid pipeline.

Take an entry-point name and mirror `yawgpu_tint_resolved_workgroup_size`:
lower → `SingleEntryPoint(ep)` → `SubstituteOverrides` →
`GetWorkgroupInfo`; return `false` with `*err` on any failure instead of
`0`. Update the Rust wrapper(s) and the core call sites
(`shader_tint.rs`, the `compute_pipeline.rs` limit check) so a failure
becomes a pipeline-creation error, not a silently passed limit check.
Unit tests in `yawgpu-tint` for both probe shapes and the override-failure
path; core test that the 16 KiB limit fires for the multi-entry-point
module.

## S4 — GLES (Tier 2)

### R8. Render-pass viewport and resolve rectangles use the mip extent

The default viewport and the resolve blit rectangles
(`gles/queue.rs`) come from the texture's mip-0 width/height. Use the
attachment's mip-level extent (`max(1, size >> level)`), as Vulkan does.

### R9. No scissor test during resolve / present blits

`glBlitFramebuffer` is clipped by the scissor test, which the draw path
enables and pass cleanup leaves on. Disable `GL_SCISSOR_TEST` in
`RenderPassCleanup` and before the resolve blit and
`blit_back_buffer_to_window` (`gles/surface.rs`).

### R10. Primitive restart is enabled

Enable `GL_PRIMITIVE_RESTART_FIXED_INDEX` (core in ES 3.0) once per
context, so strip draws with `0xFFFF` / `0xFFFFFFFF` restart like Vulkan
and Metal.

### R11. Stencil-aspect buffer→texture copies are rejected

A buffer-to-texture copy into the stencil aspect of a combined
depth-stencil texture currently uploads with the combined format and
corrupts depth. Return a `HalError` (catalogued in the 67-gles matrix by
Claude), matching the texture-to-buffer path's existing stencil rejection.

GL context release across threads (review G2) is deferred to its own
block: it changes the context ownership model.

## S5 — Phase Review

Fresh no-context reviewer over the block's cumulative diff.

## S2b — pre-existing sync hazards found by sync validation

Running every `e2e_vulkan_*` on MoltenVK with
`VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`
(first time this suite ran under sync validation): 42 `SYNC-HAZARD` lines
at `339729c` (vulkan/ reverted to the pre-S2 code), 12 after S2
(`77e3197`). All surface write-after-read hazards were the acquire wait
stage (R4). The remaining 12 occur identically before and after S2 in six
tests and fall into two classes:

### R16. A color attachment barrier covers the load-op read

`vkCmdBeginRenderPass` with `loadOp = LOAD` reads the attachment
(`COLOR_ATTACHMENT_READ`), but the barrier that transitions it into
`COLOR_ATTACHMENT_OPTIMAL` (`transition_image_range` →
`access_mask_for_layout`) only makes it available for
`COLOR_ATTACHMENT_WRITE` (and the depth equivalents). Include
`COLOR_ATTACHMENT_READ` (and `DEPTH_STENCIL_ATTACHMENT_READ` for depth)
in the destination access of transitions into attachment layouts.
Seen in `lazy_init_store_op_discard_makes_the_next_read_zero`,
`vulkan_render_*_triangle_readback`, `vulkan_render_two_color_attachments_*`,
`vulkan_non_uniform_discard_then_fwidth_*`.

### R17. Buffer writes are visible to vertex / index / indirect reads

After a `vkCmdCopyBuffer` (queue write / copy) into a buffer later used as
a vertex buffer, `vkCmdDraw` reads it at `VERTEX_ATTRIBUTE_INPUT` while
the post-copy barrier's destination scope lacks `VERTEX_INPUT` /
`VERTEX_ATTRIBUTE_READ` (`buffer_write_read_barrier_dst_*`). Add
`VERTEX_INPUT` + `VERTEX_ATTRIBUTE_READ | INDEX_READ` and `DRAW_INDIRECT`
+ `INDIRECT_COMMAND_READ` to that destination (check which are already
present). Seen in `vulkan_msaa_resolve_writes_resolve_target`.

Gate (Claude): the same sync-validation run reports 0 `SYNC-HAZARD` lines.

### R18. A query resolve makes its destination writes visible to every later consumer

With R16/R17 the e2e sync-validation run is clean (0 `SYNC-HAZARD`), but
the Vulkan HAL ignored tests report `SYNC-HAZARD-READ-AFTER-WRITE` in
`vulkan_resolve_timestamp_query_set_zero_fills_unwritten_slots` and
`vulkan_write_timestamp_then_resolve_yields_increasing_ticks` (pre-existing:
identical at `77e3197`). `encode_resolve_query_set` ends with a barrier only
for timestamp sets and only towards the compute timestamp-conversion
shader; a later `copyBufferToBuffer` / map / shader read of the
destination has no dependency (occlusion sets get none at all). End every
non-empty resolve with a `TRANSFER_WRITE` → general buffer-read barrier
(`buffer_write_read_barrier_dst_stage_mask()` / access mask, which already
include compute shader reads) on the destination range, for all query
kinds; keep the fill → copy barrier. Unit-test the barrier masks; Claude
re-runs the HAL ignored tests under sync validation (0 hazards).

## S5 — Phase Review (2026-10-25)

Fresh reviewer over `339729c..45cd4fd`: **0 CRITICAL / 0 MAJOR / 6 MINOR**.
Gates before review: `cargo test --workspace` green; strict clippy (all
backend feature combos) + fmt clean; Metal e2e incl.
`e2e_metal_view_format_copies` (both cases fail on the pre-fix Metal code,
pass after, also under `MTL_DEBUG_LAYER`); MoltenVK all `e2e_vulkan_*` +
Vulkan HAL ignored tests under sync validation: 0 `SYNC-HAZARD`
(pre-existing `vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`
MoltenVK limit unchanged); webgpu-native-cts on Metal:
`api,validation,compute_pipeline` 11842/0, `shader_io,compute_builtins`
1512/0 (6 feature skips), `shader_module,overrides` 2/0,
`api,operation,compute,basic` 1/0.

| ID | Finding | Disposition |
|---|---|---|
| m1 | R5 shader-write barrier is emitted after every render pass, serialising consecutive passes on tile-based GPUs | fix (R19): emit only when the pass bound a writable storage resource in a vertex / fragment stage (incl. via bundles) |
| m2 | spec asked for an MSAA-resolve-through-reinterpreted-view e2e; only the descriptor is unit-tested | fix: Claude adds the e2e |
| m3 | `shader_tint.rs` fast-path doc comment garbled; IR lowered twice for override-bearing modules | fix the comment; the double lowering is **deferred** (compile-time only, results memoised; merging the two shim calls is an ABI change for no correctness gain) |
| m4 | GLES R11 error text says "stencil formats" though `Stencil8` uploads are allowed | fix wording |
| m5 | 67-gles matrix lacks the R11 B2T stencil-aspect `HalError`; this spec file untracked | fix (Claude) |
| m6 | retire-ring waits run while holding `queue_access` | **deferred** — not a deadlock (every waited fence was submitted under the same lock); a throughput issue, logged with the Vulkan per-submit object-churn perf item in the review ledger |

### Phase Review fix gate

R19 (conditional post-pass shader-write barrier), m3 doc, m4 wording, m5
matrix row, m2 e2e `metal_msaa_resolve_through_srgb_views_stores_srgb_encoded_bytes`
(fails on the pre-R1 Metal code, passes after). `cargo test --workspace`
green; strict clippy on default / metal / vulkan / gles / vulkan,tiled /
metal,tiled + fmt clean; Vulkan HAL ignored 58/58 and all `e2e_vulkan_*`
under sync validation: 0 `SYNC-HAZARD` (only the pre-existing MoltenVK
3D-attachment fail / VUIDs).
