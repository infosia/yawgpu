# Block 65 — `dual-source-blending` optional feature

Status: **Slices 1–2 DONE, real-GPU verified** (Metal M2 + MoltenVK). Slice 3
(docs + Phase Review) in progress. Owner: Dawn-parity feature backfill.

The WebGPU `dual-source-blending` optional feature
(`WGPUFeatureName_DualSourceBlending = 0x11`) lets a fragment shader emit a
**second color output** at `@location(0) @blend_src(1)` and use the
`src1`/`one-minus-src1`/`src1-alpha`/`one-minus-src1-alpha` blend factors, so the
blend equation can reference a second source color (classic dual-source blending,
e.g. subpixel font AA).

Much of the machinery already exists in the tree:
- `BlendFactor::{Src1, OneMinusSrc1, Src1Alpha, OneMinusSrc1Alpha}` are defined
  (`render_pipeline.rs:339`) and map to `HalBlendFactor::Src1*`.
- The HAL blend-factor mapping is **complete on both backends**: Metal
  `MTLBlendFactor::Source1Color/Alpha` (`metal/pipeline.rs:355`), Vulkan
  `vk::BlendFactor::SRC1_COLOR/ALPHA` (`vulkan/pipeline.rs:954`).
- Tint already knows the WGSL surface: `Extension::kDualSourceBlending`
  (`enable dual_source_blending;` + `@blend_src(0|1)`), and its MSL/SPIR-V
  writers emit the `[[color(0), index(1)]]` / SPIR-V `Index` decoration
  automatically — **no yawgpu reflection of `@blend_src` is required**.

What is missing: the `yawgpu-core` feature, the Tint-extension gate, the Vulkan
`dualSrcBlend` device-feature enable, and the core validation that the `Src1*`
factors / the WGSL extension require the feature. Today `Src1*` blend factors are
**silently accepted** in core and would hit a Vulkan VUID (`dualSrcBlend` not
enabled), so the feature is effectively broken — this block makes it correct and
gated.

## Public API surface

No new C entry points:
- **`wgpuAdapterGetFeatures` / `wgpuAdapterHasFeature` / device equivalents**
  report `WGPUFeatureName_DualSourceBlending` when supported / requested.
- A render pipeline may use `Src1*` blend factors and a shader with
  `enable dual_source_blending;` iff the device enabled the feature.

### `yawgpu-core::Feature`

Add a `DualSourceBlending` variant (not `cfg`-gated); map it C↔Rust in
`yawgpu/src/conv/feature.rs`; advertise from `Adapter::features()` via
`add_dual_source_blending_feature` consulting
`HalAdapter::supports_dual_source_blending()`.

## Behaviour contract

### Advertisement (HAL capability query) — Dawn parity

`HalAdapter::supports_dual_source_blending() -> bool`, static-enum-dispatched:

- **Metal** — always `true` (Dawn enables it unconditionally,
  `PhysicalDeviceMTL.mm:716`; `MTLBlendFactor::Source1*` is universally available).
- **Vulkan** — `VkPhysicalDeviceFeatures::dualSrcBlend == VK_TRUE`
  (`get_physical_device_features`). Dawn gates the same way.
- **Noop** — `true` (keeps the accept path unit-testable; Noop does not blend).
- **GLES** — `false` (Tier 2).

### Vulkan device-feature enable

In `VulkanAdapter::create_device`, enable `enabled_features.dual_src_blend =
vk::TRUE` whenever `supported_features.dual_src_blend == vk::TRUE` (mirrors how
`depth_clamp` / `independent_blend` are enabled). Without this, a pipeline using
`SRC1_*` blend factors is a VUID violation on Vulkan.

### Tint-extension gate (shader compilation)

Thread a `dual_source_blending: bool` from the device's enabled features into the
shader-compile path exactly like `shader_f16` / `subgroups`:
`create_shader_module` → `from_wgsl` → `parse_and_validate_wgsl_gated` →
`Program::parse` → the shim's `yawgpu_tint_program_create`, which does:
```cpp
if (dual_source_blending) {
    options.allowed_features.extensions.insert(tint::wgsl::Extension::kDualSourceBlending);
}
```
Without the feature, `enable dual_source_blending;` (and thus `@blend_src`) is
rejected at `createShaderModule`, identically on every backend.

### Pipeline validation (Tier-independent core rule)

In render-pipeline validation:
- If any color-target blend factor is `Src1 | OneMinusSrc1 | Src1Alpha |
  OneMinusSrc1Alpha`, the device must have `Feature::DualSourceBlending` enabled
  — else a validation error to the error sink. Identical on every backend.
- A pipeline that uses dual-source blend factors must have **exactly one color
  target** (Dawn / the hardware limit `maxFragmentDualSrcAttachments == 1`).
  A `Src1*` factor on target index > 0, or with more than one color target, is
  rejected.

(Full fragment-output/`@blend_src` interface checking is performed by Tint on the
shader itself; the pipeline-level rules above are the core additions.)

## Slices

1. **Feature plumbing + Tint gate + validation (Noop + HAL cap).**
   `Feature::DualSourceBlending`, `add_dual_source_blending_feature`,
   `HalAdapter::supports_dual_source_blending` (4 backends), Vulkan
   `dualSrcBlend` device-feature enable, `conv/feature.rs`, the shader-compile
   `dual_source_blending` gate threaded to the shim (`kDualSourceBlending`), and
   the pipeline validation (Src1 factors + single-target require the feature).
   Inline unit tests. **Acceptance:** `cargo test --workspace` green on Noop —
   `enable dual_source_blending;` and `Src1` factors rejected without the
   feature, accepted with it.

2. **Real-GPU execution e2e (Metal + Vulkan).**
   `e2e_{metal,vulkan}_dual_source_blending.rs`: a fragment shader outputting
   `@location(0) @blend_src(0)` = c0 and `@blend_src(1)` = c1, a pipeline with
   `srcFactor=Src1` (so `result = c0*? + c1*...`), drawn over a known
   destination, read back to prove the second source contributed. **Acceptance:**
   both e2e suites pass on the M2 (Metal) and MoltenVK (Vulkan; MoltenVK supports
   `dualSrcBlend`) — or, if MoltenVK lacks it, self-skip with a documented note.

3. **Docs + Phase Review.** README capability note; Block 65 finalization; the
   mandatory no-context Phase Review.

Tracking: `specs/tracking/dual-source-blending.md`.
