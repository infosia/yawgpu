# Block 149 — External textures in subpass draws, color-target format check, suboptimal surfaces

Status: **COMPLETE (2026-11-09)**. Phase Review: 0 CRITICAL, 2 MAJOR (both fixed), 6 MINOR;
four fixed, one moved to the backlog, one declined, see below. A focused second review of
the fixes found one MINOR, fixed.
Source: backlog A29 and A30 and the examples' MoltenVK acquire failure, started by user
request on 2026-11-09.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`RenderPipeline.cpp` `ValidateColorTargetState` for R2 and `vulkan/SwapChainVk.cpp`
(`GetCurrentTextureImpl`, `PresentImpl`) for R3. Subpass rendering (`tiled`) is a yawgpu
extension with no oracle; R1 follows the regular render pass.

## Problem

1. **A29: subpass pipelines cannot use external textures.** The backlog row said subpass
   draws drop `bindings.external_textures`. That path cannot be reached:
   `create_hal_subpass_render_pipeline` (`yawgpu-core/src/render_pipeline.rs`) rejects
   any subpass pipeline with an external-texture binding with the internal error
   "external textures are not supported in the tiled subpass pipeline path", added by
   Block 146 as out of scope. Behind it, `HalSubpassDraw` has no external-texture field,
   `hal_subpass_draw_execution` (`yawgpu-core/src/queue.rs`) does not carry them, and the
   Metal subpass encoder passes `bind_external_textures: &[]`. A new real-GPU e2e on the
   M2 shows the rejection for a single subpass sampling an external texture and for a
   subpass that reads an input attachment at group 0 and samples an external texture at
   group 1.
2. **A30: depth formats pass as color targets.** `validate_fragment_state`
   (`yawgpu-core/src/render_pipeline.rs`) checks `caps.renderable` only. Depth and
   stencil formats are renderable, so a WGSL render pipeline with a `depth24plus` color
   target, `writeMask` 0, and no fragment output at that location is accepted. Dawn's
   `ValidateColorTargetState` rejects it: "Color format (%s) is not color renderable."
   when the format is not a color format or not renderable. `render_bundle.rs` and
   `subpass.rs` already check `!caps.aspects.color || !caps.renderable`.
3. **Suboptimal surfaces.**
   - Library: the Vulkan surface ORs `VK_SUBOPTIMAL_KHR` from every acquire and every
     present into one flag (`yawgpu-hal/src/vulkan/surface.rs`), cleared only by
     `configure`. After one suboptimal present, every later
     `wgpuSurfaceGetCurrentTexture` reports `SuccessSuboptimal`. Dawn reports the status
     of the current acquire only and treats a suboptimal present as success.
   - Examples: on MoltenVK every windowed example stopped at the first frame with
     "failed to acquire surface texture, status=2". The macOS framework's `CAMetalLayer`
     kept `contentsScale` 1.0 while the surface is configured with the Retina
     framebuffer size, so MoltenVK reported every acquire as suboptimal, and the
     examples treated `SuccessSuboptimal` as a failure.

## Rules

R1. **External textures in subpass draws.** A subpass render pipeline may use external
textures, on Noop, Metal, and Vulkan, with the same bindings, ABI, and results as a
regular render pipeline:
- The rejection in `create_hal_subpass_render_pipeline` and its unit test go away.
- On Metal the subpass pipeline gets the same plane and params slots as a regular
  pipeline. On Vulkan it gets the expanded plane 1 and params descriptor bindings of
  Block 146 R1.
- `HalSubpassDraw` carries the draw's bound external textures, filled by
  `hal_subpass_draw_execution` from the resolved bindings.
- The Metal subpass encoder binds them as the regular draw path does. The Vulkan
  subpass descriptor writes include the plane views and the params buffer, and the
  plane images are in the shader-read layout before the subpass render pass begins, as
  for regular passes.
- Everything a regular render pass does for an external texture at submit also happens
  for a subpass pass: the lazy clears and initialization marks of Block 144, usage
  tracking, and keeping the planes and the params buffer alive until the submission
  completes.
- Input attachments and external textures in one subpass pipeline work together.
- If a backend cannot do part of this, it returns a `HalError` naming the subpass path.
  It never drops a binding silently.

R2. **Color targets must be color-renderable.** In `validate_fragment_state`, a defined
color target format that is not a color format or not renderable fails with "render
pipeline color target format must be color-renderable". This applies to every render
pipeline (WGSL, passthrough, and subpass), whatever the `writeMask` and the shader
outputs. The `writeMask` bit check stays before it, as in Dawn.

R3. **Suboptimal is per acquire.** The Vulkan surface reports `SuccessSuboptimal` only
when the acquire that produced the current texture returned `VK_SUBOPTIMAL_KHR`. A
suboptimal present is a success and does not affect any later status. Both match Dawn
`SwapChainVk.cpp`. Every path of `wgpuSurfaceGetCurrentTexture` that reports a success
status, including a repeat call for an already-acquired texture, reports the status of
that texture's acquire.

R4. **Examples (Claude).** The macOS framework sets the layer's `contentsScale` to the
window's `backingScaleFactor`. A framework helper `yawgpu_surface_texture_acquired`
accepts `SuccessOptimal` and `SuccessSuboptimal` with a texture, and every windowed
example uses it.

R5. **Unchanged.** The C API, the external-texture ABI, regular render passes, and the
Metal and GLES surfaces.

## Tests

- R1 (coding agent): core unit tests showing that a subpass pipeline with an external
  texture builds and that `hal_subpass_draw_execution` carries its external textures.
  Metal and Vulkan HAL unit tests for the subpass descriptor and slot wiring where the
  HAL tests can reach it.
- R1 (Claude): `yawgpu/tests/common/tiled_external_texture.rs`, run by
  `e2e_metal_tiled_external_texture.rs` and `e2e_vulkan_tiled_external_texture.rs`. One
  subpass samples an RGBA external texture, and a second subpass reads an input
  attachment at group 0 and samples an external texture at group 1. The expected
  texels are checked on every pixel. Both fail before R1 with the rejection.
- R2 (coding agent): unit tests for a WGSL render pipeline with a `depth24plus` color
  target, `writeMask` 0, and no output at that location, and for the passthrough and
  subpass paths. A C API Noop test through `wgpuDeviceCreateRenderPipeline`.
- R3 (coding agent): HAL unit tests showing that a suboptimal present does not mark the
  next acquire and that a suboptimal acquire does. Factor the flag update out if the
  swapchain cannot be faked.
- R4 (Claude): every windowed example with `--verify` on Metal and on MoltenVK.
- Claude: `cargo test --workspace`, fmt, the clippy matrix, HAL Metal and Vulkan lib
  and ignored tests, Metal e2e, every `e2e_vulkan_*` on MoltenVK with the validation
  layer, and webgpu-native-cts Metal `webgpu:api,*`.

## Out of scope

- External textures on GLES (backlog B3) and the other `tiled` gaps (backlog E5).
- Reconfiguring the surface in the examples when the window moves to a display with
  another scale.

## Slices

- **S1 (coding agent):** R1, R2, R3 with their unit tests.
- **S2 (Claude):** R4, the e2e, verification, Phase Review.

## Verification (2026-11-09)

Run on Apple M2, macOS 27.2. MoltenVK runs used the Khronos validation layer.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,443 passed, 0 failed; with `shader-passthrough` 1,493 passed, 0 failed |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough`, `metal,shader-passthrough`, `vulkan,shader-passthrough` | clean |
| HAL lib tests including ignored, `metal,tiled` | 274 passed, 0 failed |
| HAL lib tests including ignored, `vulkan,tiled`, MoltenVK | 471 passed, 0 failed, 0 VUIDs |
| Metal e2e, every integration test, `metal,tiled,shader-passthrough`, `--ignored` | 181 passed, 0 failed |
| Vulkan e2e, every `e2e_vulkan_*`, `vulkan,tiled,shader-passthrough`, MoltenVK | 173 passed, 1 failed: `vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`, the known MoltenVK `imageView2DOn3DImage` failure, with its VUIDs only |
| New e2e `*_subpass_samples_external_texture`, `*_subpass_reads_input_attachment_and_external_texture` | pass on Metal and MoltenVK; before R1 both failed at pipeline creation with the rejection |
| New e2e `*_subpass_unused_external_texture_keeps_buffer_sizes` (Phase Review M2) | pass on Metal and MoltenVK; before the M2 fix Metal read `arrayLength` 0 instead of 64 |
| Examples `--verify`: `surface_smoke`, `triangle`, `hello_triangle`, `triangle_passthrough`, `tiled_deferred` on Metal and MoltenVK; `tiled_msaa` on MoltenVK | all exit 0; before R4 every windowed example failed on MoltenVK at the first acquire |
| webgpu-native-cts, Metal, `webgpu:api,*` | pass 589,574, skip 101,793, fail 0, crash 0; identical to Block 148. Run before the Phase Review fixes, which change only the Metal subpass path and remove an unreachable check |

## Phase Review

A fresh review of the S1 diff, then a focused review of the fixes.

| ID | Finding | Disposition |
|---|---|---|
| M1 | The passthrough unit test `native_render_runs_reflection_free_validators_in_wgsl_order` still expected the old "no output class" error for a depth color target; the workspace run without `shader-passthrough` does not build it | Expectation updated |
| M2 | The Metal subpass encoder wrote the immediates block before textures, external textures, samplers and vertex buffers. An external texture the entry point does not use has a params slot the immediate-slot chooser may pick, so its params buffer overwrote the block (Block 118 R7a) | The block is written last; e2e `*_subpass_unused_external_texture_keeps_buffer_sizes` |
| m1 | The passthrough-only "no output class" check became unreachable | Removed |
| m2 | Dawn checks the blend state before `writeMask` and the format; yawgpu checks it after the format | Declined: both reject the same descriptors, only the first message differs, and the order predates this block |
| m3 | Metal external-texture errors on the subpass path did not name it | Wrapped as "subpass external texture binding failed" |
| m4 | Subpass passes do no usage-scope validation for any bind-group resource, external planes included | Predates this block; added to backlog E5 |
| m5 | The new examples helper split a comment from `yawgpu_wait_for_future` | Moved |
| m6 | The e2e covers RGBA only; NV12 in a subpass is covered by the core and Vulkan unit tests | Accepted |
| m7 (second review) | Block 118 R7a still said the subpass path may write after the block and has no external textures | R7a updated |
