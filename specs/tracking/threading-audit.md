# Threading audit — "validated/accepted but not threaded to the HAL" (silently wrong)

A proactive audit (5 parallel trace agents, FFI → `yawgpu-core` → `yawgpu-hal` → Metal/Vulkan) for the
recurring pattern behind F-035 (`blend_constant`), F-038 (`stencil_reference`), and F-043 (`depthSlice`):
**a WebGPU field is parsed/validated in `yawgpu-core` but its value never reaches the HAL backend, so the
GPU uses a default and the result is silently wrong.** The CTS currently passes these because only the
default values are exercised on real hardware; non-default values would surface them (future CTS findings).

**Clean (no gap found):** the full `WGPUDepthStencilState` pipeline state (depthWrite/compare/bias/slope/
clamp, stencil ops/compare/masks, format) and the full `WGPUSamplerDescriptor` (all address modes incl.
`addressModeW`, filters, lod clamps, compare, maxAnisotropy). All draw/dispatch/binding/copy params
(firstVertex/firstInstance/baseVertex/firstIndex, dynamic offsets, vertex/index offsets, copy
offset/bytesPerRow/rowsPerImage/origin.z) are applied on both backends.

**Findings — RESOLVED in a single combined slice (user revised 2026-06-04; the groups below
are the per-group structure of that one HANDOFF, not separate phases). Real-GPU verified on
Metal + Vulkan/MoltenVK; Clean Review clean (no CRITICAL/MAJOR):**

| Group | Field(s) | Fix | Status |
|---|---|---|---|
| A | `primitive.cullMode`, `primitive.frontFace` | threaded FFI→core `PrimitiveState`→`HalRenderPipelineDescriptor`; Vulkan rasterization state (replaced hardcoded NONE/CCW), Metal render-encoder `setCullMode`/`setFrontFacingWinding` (stored on `MetalRenderPipeline`), GLES `glCullFace`/`glFrontFace` | RESOLVED |
| B | `setViewport`, `setScissorRect` | stored on `PassEncoderState`, snapshotted per draw into `RenderPassCommand`, threaded to `HalRenderPass.{viewport,scissor_rect}` (`Option`, None = full-attachment); applied on all 3 backends | RESOLVED |
| C | `depthReadOnly`, `stencilReadOnly` | read-only aspect now maps to `Load` + preserve (Store) in `queue.rs`, not Clear; per aspect | RESOLVED |
| D | `multisample.mask`, `multisample.alphaToCoverageEnabled` | Vulkan applies both (`sample_mask` + `alpha_to_coverage_enable`); Metal applies a2c, **rejects non-default sample mask with `HalError`** (see Tier-1 note); GLES rejects both with `HalError` | RESOLVED |
| E | `primitive.unclippedDepth` | threaded to HAL + backend application code (Vulkan `depth_clamp_enable`, Metal `setDepthClipMode`), but core **rejects** `unclippedDepth=true` (validation error) pending a `depth-clip-control` feature — so the apply paths are currently unreachable-by-design | RESOLVED |
| F | `dispatchWorkgroupsIndirect` | records an indirect `ComputePassCommand`→`HalComputeDispatch::Indirect`; executed via `cmd_dispatch_indirect`(Vk)/`dispatchThreadgroupsWithIndirectBuffer`(Metal)/`glDispatchComputeIndirect`(GLES) | RESOLVED |

GLES (Tier 2) rejections catalogued in `specs/blocks/67-gles-backend.md`; no core rule relaxed.
Verification: `yawgpu/tests/e2e_{metal,vulkan}_threading_audit.rs` — 6 probes each (cull keep/discard,
scissor, viewport, read-only depth preserved, indirect dispatch), all green on both backends; the cull
keep/discard pair is **identical across Metal and Vulkan**, pinning WebGPU↔Vulkan Y/winding parity.

**Known Tier-1 limitation (Clean Review MINOR, accepted):** Metal rejects a non-default
`multisample.mask` (`!= 0xFFFF_FFFF`) with `HalError` because the pinned `objc2-metal` bindings do not
expose `MTLRenderPipelineDescriptor.sampleMask`. Default mask is honored; only non-default masks error on
Metal (Vulkan applies them).

**Follow-up — RESOLVED (tiled subpass slice):** the `tiled`-feature subpass path
(`SubpassRenderPassEncoder`) now threads group B (`setViewport`/`setScissorRect` → `SubpassDrawState` →
`SubpassDrawExecution` → `HalSubpassDraw.{viewport,scissor_rect}` → both backends; None = full-attachment)
and applies group A on the **Metal** subpass encode (`setCullMode`/`setFrontFacingWinding`/`setDepthClipMode`
in `encode_subpass_draw` — previously skipped, so tiled subpass draws used the encoder defaults; Vulkan bakes
cull/frontFace into the subpass `VkPipeline` at creation, so it was already correct). Verified on real-GPU
Metal + Vulkan/MoltenVK: `e2e_{metal,vulkan}_tiled` gained 4 probes each (scissor, viewport, cull keep CCW,
cull discard CW), all green, with the existing tiled e2e (deferred two-subpass) still passing. Clean Review:
no CRITICAL/MAJOR.

While doing this, found the `tiled` feature had **not compiled since ~F-034/F-040/F-043** — no gate builds
it (CI was Noop-default only; the per-backend review gates omitted `tiled`). Accumulated compile rot fixed
(`metal/device.rs` fragment-entry `Option`, `vulkan/pipeline.rs` fragment `Option`, `vulkan/texture.rs`
removed inner `sample_count`, Metal `draw_primitives`→inline). Gate gap closed: CI now compile-gates the
Noop-buildable tiled code (`.github/workflows/ci.yml`); backend-tiled gates added to the review workflow
(see memory `feedback-run-feature-gated-hal-tests`).
