# Block 109 — whole-codebase review: CRITICAL fixes

Status: **COMPLETE (2026-10-25).** Phase Review 0 CRITICAL / 2 MAJOR /
4 MINOR, all fixed (R8–R12) or catalogued (m3); see S4 below.
- **S1 DONE** (`eee477b`) — R1–R3 with Noop FFI unit tests; `ffi_slice`
  helper; every `from_raw_parts` in `yawgpu/src` audited (no other unsafe
  null/zero site).
- **S2 + S2b DONE** (`2e25131`) — R4/R5 (fence) and R7 (final layout).
  S2b was added mid-block: the new S2 frame-loop e2e (surface without
  `CopySrc`) surfaced `VUID-vkCmdBeginRenderPass-initialLayout-00898` on
  every rendered frame. MoltenVK under `VK_LAYER_KHRONOS_validation`:
  `e2e_vulkan_surface` 7/7 (both hang regressions timed out before the
  fix), Vulkan HAL ignored tests 58/58, all 29 `e2e_vulkan_*` binaries
  114 pass / 1 fail; the failure and its two VUIDs
  (`vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`,
  `imageView2DOn3DImage-04459` portability error + the follow-on
  `vkDestroyDevice-device-05137`) reproduce identically at `bead72b`, so
  they are pre-existing MoltenVK 3D-attachment limits, not a regression.
  Metal e2e (35 binaries) 124/0.
- **S3 DONE** (`f759bbd`) — R6 with 9 pure-fn unit tests; mapping-matrix
  row added; `e2e_gles_render` depth-occlusion case **passes on the Linux NVIDIA
  EGL host** (`5cd8c6a`, 2026-10-25).
- Gates: `cargo test --workspace` green; clippy clean on default / vulkan /
  gles / vulkan+tiled except the pre-existing rustc 1.99.1 lints
  (`deprecated` `fetch_update`, `chunks_exact_to_as_chunks`, metal-test
  unused import), fixed separately as a toolchain commit.
Source: the 2026-10-25 whole-codebase review; ledger
`specs/tracking/code-review-2026-10.md` (rows C1–C5).

Five CRITICAL findings, in three independent slices (disjoint files):

| Slice | Findings | Crate | Gate |
|---|---|---|---|
| S1 | C3, C4, C5 | `yawgpu` (FFI) | Noop unit tests |
| S2 | C2 | `yawgpu-hal` Vulkan | HAL unit tests + `e2e_vulkan_surface` on MoltenVK (Claude) |
| S3 | C1 | `yawgpu-hal` GLES | pure-fn unit tests; `e2e_gles_render` on real GLES (Linux NVIDIA EGL ☑ 2026-10-25) |
| S4 | — | — | Phase Review |

## S1 — FFI handle lifetime and null slices

### R1 (C3). Releasing the last C reference does not destroy the resource

`wgpuBufferRelease` / `wgpuTextureRelease` dropping the last *external*
reference must not mark the core resource destroyed. Objects that hold
the core resource internally (texture views, bind groups, command
buffers, render bundles, surface textures) keep it usable; the backend
resource is freed when the last core `Arc` drops.

- Oracle: Dawn `TextureBase::WillDropLastExternalRef` only clears the view
  cache; `BufferBase` has no override. Neither calls `Destroy`.
- Keep `abort_pending_map()` in the buffer's drop: a pending map on a
  buffer nobody can reach any more still resolves its callback as aborted
  (current behaviour, unchanged).
- Remove the `core.destroy()` calls from `Drop for WGPUBufferImpl` and
  `Drop for WGPUTextureImpl` (`yawgpu/src/ffi/mod.rs`); drop the texture
  `Drop` impl if it becomes empty.
- Tests (FFI unit tests, Noop):
  - encode `copyBufferToBuffer(src → dst)`, finish, `wgpuBufferRelease(src)`,
    submit → no device error.
  - create a texture + view, `wgpuTextureRelease(texture)`, render a
    clear pass to the view, submit → no device error.
  - explicit `wgpuBufferDestroy` / `wgpuTextureDestroy` followed by submit
    still errors (regression guard: R1 does not weaken destroy).

### R2 (C4). `(NULL, 0)` data is a legal empty slice

Every FFI entry that turns a `(ptr, len)` pair into a slice must not call
`std::slice::from_raw_parts` with a null pointer, including `len == 0`
(undefined behaviour; the debug precondition check aborts the process).

- Sites: `wgpuQueueWriteBuffer` (`ffi/queue.rs`), and the three
  `SetImmediates` entries (`ffi/bundle.rs`, `ffi/compute_pass.rs`,
  `ffi/render_pass.rs`).
- Shape: `len == 0` → `&[]`; `len > 0 && ptr.is_null()` → the existing
  validation error. The two already-correct sites
  (`wgpuCommandEncoderWriteBuffer` in `ffi/encoder.rs`,
  `wgpuQueueWriteTexture` in `ffi/queue.rs`) show the pattern. A single
  shared helper used by all six sites is preferred over six copies.
- Audit every other `from_raw_parts` in `yawgpu/src` for the same shape
  and list the result in REPORT.md.
- Tests: each of the four entry points called with `(NULL, 0)` returns
  without a device error (in a debug test build this is also the UB
  regression guard).

### R3 (C5). Null array with non-zero count is a validation error, never a deref

- `WGPURenderPassDescriptor.colorAttachments == NULL` with
  `colorAttachmentCount > 0` → `beginRenderPass` returns an error pass
  and the encoder is invalidated with a validation error, as for any other
  invalid render pass descriptor. No raw pointer is read.
- `WGPURenderBundleEncoderDescriptor.colorFormats == NULL` with
  `colorFormatCount > 0` → `createRenderBundleEncoder` produces an error
  encoder plus a validation error.
- `map_render_pass_descriptor` (and the bundle mapper) must no longer
  silently decode null+count as `None` holes; the decision lives in one
  place, before any slice is formed.
- `validate_render_pass_descriptor_devices` dereferences raw pointers;
  make it `unsafe fn` with a `# Safety` section (or fold it into the
  already-unsafe caller).
- Tests: both descriptors with null+count → error object + exactly one
  validation error, no crash.

## S2 — Vulkan swapchain in-flight fence (C2)

### Problem

`VulkanSurface::acquire_next_texture` waits on
`in_flight_fences[sync_index]` and then **resets** it. The same fence was
earlier passed to the present-transition submit
(`transition_swapchain_image_to_present`, `vulkan/encode.rs`), which
registers it in the surface `RetireRing` with `destroy_fence = false`.
If that ring entry has not been cleaned up yet, the ring now holds a
reset fence that nothing will signal until the *next* present on that
sync index. Any `retire_tracked` that lands on that slot, or any
`wait_all` (configure / unconfigure / drop), blocks forever on it.

Reproduced on MoltenVK (3 swapchain images, ring size 3) by the two
Block 109 tests in `yawgpu/tests/e2e_vulkan_surface.rs`:
`vulkan_surface_present_only_frames_then_render_does_not_hang` and
`vulkan_surface_reconfigure_after_unpresented_acquire_does_not_hang` (both
time out before the fix).

### R4. The retire ring never holds a fence that has been reset

Before `acquire_next_texture` resets the in-flight fence, every retire-ring
entry (slots **and** overflow) whose fence is that in-flight fence is
cleaned up. This is safe: the fence was just waited signaled, so those
entries' work is complete. Their `destroy_fence` is `false`, so cleanup
must not destroy the fence. Implement as a `RetireRing` method (e.g.
`release_signaled_fence(device, fence)`) under the existing
`pending_state` lock.

### R5. A failed present leaves the in-flight fence signaled

If `present` fails after the acquire reset the fence but before the
transition submit was issued successfully, the fence must end up
signaled (for example with a fence-only empty `vkQueueSubmit`, as
`submit_empty` does), so the next acquire on that sync index does not
block forever. An acquire followed by `configure` / `unconfigure` without
present is already covered: both recreate the sync objects.

### Tests

- HAL unit test (Vulkan-gated, `#[ignore]` real-GPU like the other
  Vulkan HAL tests): drive a `RetireRing` with a real device's fences
  through the reset-at-acquire sequence and assert that no wait blocks
  (use a bounded wait or check `vkGetFenceStatus` instead of waiting
  forever in the test).
- Claude runs `e2e_vulkan_surface` (all tests) on MoltenVK under
  `VK_LAYER_KHRONOS_validation`: the two Block 109 tests pass, the others
  stay green, 0 validation lines.

## S2b — Vulkan render-pass final layout vs image usage (found by S2's e2e)

### Problem

The S2 frame-loop e2e configures the surface with `RenderAttachment` only
(no `CopySrc`) and, under `VK_LAYER_KHRONOS_validation`, reports
`VUID-vkCmdBeginRenderPass-initialLayout-00898` on every rendered frame:
the regular render path (`vk_color_attachment_description` and the
resolve / other attachment descriptions in `vulkan/encode.rs`) always sets
`finalLayout = TRANSFER_SRC_OPTIMAL` and records
`IMAGE_LAYOUT_TRANSFER_SRC` after the pass, but a swapchain image only
carries `TRANSFER_SRC` usage when the surface was configured with
`CopySrc`. Ordinary textures always get `TRANSFER_SRC | TRANSFER_DST`
(`vulkan/format.rs`), transient textures never do. The existing surface
e2e tests all configured `CopySrc`, which hid it. Any windowed app that
renders to the swapchain without `CopySrc` hits it. The same root cause is
review finding "transient attachments end in `TRANSFER_SRC_OPTIMAL` in the
non-tiled path" (ledger, Vulkan MAJOR); the tiled path already handles the
transient case via `subpass_color_final_layout`.

### R7. A render pass never ends an attachment in a layout its image cannot use

For every color attachment and resolve target in every render-pass path
(regular, tiled/subpass, and any other `AttachmentDescription` with
`TRANSFER_SRC_OPTIMAL` final layout), the final layout and the layout
recorded in the subresource tracker after the pass are:

- `TRANSFER_SRC_OPTIMAL` / `IMAGE_LAYOUT_TRANSFER_SRC` when the `VkImage`
  was created with `VK_IMAGE_USAGE_TRANSFER_SRC_BIT`;
- otherwise `COLOR_ATTACHMENT_OPTIMAL` / `IMAGE_LAYOUT_COLOR_ATTACHMENT`.

The decision is made from the image's actual `VkImageUsageFlags` (record
them, or a derived flag, on the Vulkan texture at creation for regular,
transient and swapchain images), not from the WebGPU format or the
`transient` bit, so swapchain images follow it automatically. The tiled
`subpass_color_final_layout` / `subpass_color_tracked_layout` helpers fold
into the same predicate. Later transitions already start from the tracked
layout, so no other barrier changes are expected; verify the present
transition and copy-from-attachment paths still work from
`COLOR_ATTACHMENT_OPTIMAL`.

Tests: Vulkan HAL unit tests for the predicate (usage with / without
`TRANSFER_SRC`) and for each attachment-description builder; Claude runs
all `e2e_vulkan_*` under the validation layer on MoltenVK (0 VUID lines,
including `e2e_vulkan_surface` without `CopySrc`).

## S3 — GLES depth state (C1)

### R6. Depth state is applied on every draw

Mirror Dawn `RenderPipelineGL.cpp` (`ApplyDepthStencilState` and the
depth-bias block in `ApplyNow`). Per draw, from the pipeline's
`Option<HalDepthStencilState>`:

- `GL_DEPTH_TEST` enabled iff a depth-stencil state is present **and**
  not (`depth_compare == Always` and `!depth_write_enabled`); disabled
  when the pipeline has no depth-stencil state.
- `glDepthMask(depth_write_enabled)` (`false` with no depth-stencil
  state).
- `glDepthFunc(depth_compare)` through the existing
  `gles_compare_function`.
- Depth bias: enabled iff `depth_bias != 0 || depth_bias_slope_scale != 0`
  → `glEnable(GL_POLYGON_OFFSET_FILL)` +
  `glPolygonOffset(depth_bias_slope_scale, depth_bias as f32)`;
  otherwise disable `GL_POLYGON_OFFSET_FILL`. GLES 3.1 has no
  `glPolygonOffsetClamp`; a non-zero `depth_bias_clamp` is applied
  unclamped, as Dawn does without the extension. Record this in the
  `specs/blocks/67-gles-backend.md` mapping matrix (Claude edits the
  spec; the agent reports it).
- Compute the GL state in a **pure function** (no GL calls) returning a
  small struct (`test_enabled`, `write_mask`, `func`, `polygon_offset:
  Option<(factor, units)>`), and apply it next to `apply_stencil_state`
  in `run_render_draw`. Unit-test the pure function: no state,
  `Always`+no-write (test off), `Less`+write, `Always`+write (test on so
  writes happen), bias zero / non-zero / slope-only.
- The render-pass clear path keeps forcing `glDepthMask(true)` for the
  depth clear; R6's per-draw application restores the pipeline value.

### Verification

GLES real-GPU runs are manual on Windows ANGLE (Tier 2). Claude adds a
depth-occlusion case to `yawgpu/tests/e2e_gles_render.rs` (near quad
first, far quad second with `Less` → near colour wins; then reversed
order); the run on ANGLE is recorded as owed in the ledger until done.

## S4 — Phase Review

Fresh no-context reviewer over the block's cumulative diff, per
`specs/reference/workflow.md`.

## S4 — Phase Review (2026-10-25)

Fresh no-context reviewer over `bead72b..f759bbd`: **0 CRITICAL / 2 MAJOR /
4 MINOR**, all confirmed by Claude against the code. None dropped.

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| PR-M1 | MAJOR | `wgpuSurfaceGetCurrentTexture` called twice before present acquires again: the second acquire resets the next in-flight fence and overwrites `pending_acquire`, so the first fence stays reset and untracked forever (same class as C2) and the first image is never released | fix → R8 |
| PR-M2 | MAJOR | after R7, consecutive render passes on an image without `TRANSFER_SRC` stay `COLOR_ATTACHMENT → COLOR_ATTACHMENT` (no pre-pass barrier) and `dependency_in.src_access_mask` is empty, so the previous pass's attachment writes are never made available (RAW/WAW hazard; depth always had it) | fix → R9 |
| PR-m1 | MINOR | R5 recovery submit signals the fence but leaves the acquire / render-finished binary semaphore signaled | fix → R10 |
| PR-m2 | MINOR | null attachment array (or device-ownership error) on an already-finished encoder dispatches two errors | fix → R11 |
| PR-m3 | MINOR | GLES depth bias does not apply Dawn's `GLDepthBiasModifier` halving (desktop GL / ANGLE-on-GL / SwiftShader) | catalogue in the 67-gles matrix (Claude) |
| PR-m4 | MINOR | `RenderPassCleanup` does not reset `DEPTH_TEST` / `POLYGON_OFFSET_FILL` / depth mask | fix → R12 |

### R8. One acquired surface texture per present

While a surface texture is acquired and not yet presented (or the surface
reconfigured / unconfigured), `wgpuSurfaceGetCurrentTexture` returns the
**same** texture again (a new reference to it, status as before) without a
new HAL acquire — Dawn `Surface::GetCurrentTexture` semantics. `present`,
`configure` and `unconfigure` clear the current texture. Implemented
backend-independently in the FFI surface (Metal benefits too: a second
`nextDrawable` would otherwise hold a second drawable). Tests: Noop FFI
unit test (two calls → same core texture, one HAL acquire); Claude adds
a double-acquire frame to `e2e_vulkan_surface`.

### R9. Render passes make the previous attachment writes available

`render_pass_dependencies` `dependency_in` uses
`src_access_mask = render_attachment_access_flags()` (both regular and
framebuffer-fetch variants), so a pass following a pass on the same
attachment (color or depth, tracked layout unchanged) has a memory
dependency. Unit-test the dependency masks.

### R10. The R5 recovery submit consumes the pending semaphore

The fence-only recovery submit in `present` waits on the semaphore the
present would have waited on (`render_finished_sem` if consumed, else
`acquired_sem`) at `ALL_COMMANDS`, so no binary semaphore stays signaled.

### R11. One error per failed `beginRenderPass`

When the FFI mapping (null array) or device-ownership check fails on an
encoder that is not recording, exactly one error is dispatched (the
encoder-state error), matching Dawn. Test on a finished encoder.

### R12. GLES pass cleanup resets depth state

`RenderPassCleanup` (`gles/queue.rs`) also disables `DEPTH_TEST` and
`POLYGON_OFFSET_FILL` and restores `glDepthMask(true)` (GL default), like
the other per-draw state it already resets.

### Phase Review fix gate

R8–R12 implemented (FFI: surface current-texture tracking + one error per
failed `beginRenderPass`; Vulkan: `dependency_in` source access, recovery
submit waits the pending semaphore; GLES: cleanup resets depth state);
PR-m3 catalogued in the 67-gles matrix. `cargo test --workspace` green,
strict clippy + fmt clean. MoltenVK under `VK_LAYER_KHRONOS_validation`:
all 29 `e2e_vulkan_*` binaries 115 pass / 1 fail — the fail and its VUIDs
are the pre-existing `vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`
MoltenVK `imageView2DOn3DImage` limit (identical at `bead72b`);
`e2e_vulkan_surface` 8/8 incl. the new double-`getCurrentTexture` case;
Vulkan HAL ignored tests 58/58 with 0 VUID; `e2e_metal_surface` 6/6.

### Owed

- ~~`e2e_gles_render::gles_render_depth_test_occludes_later_far_draw` on
  real GLES~~ — passes on the Linux NVIDIA EGL host (`5cd8c6a`,
  2026-10-25).
- The default Khronos layer does not run synchronization validation; a
  sync-validation pass over the Vulkan e2e set would confirm R9 directly.
