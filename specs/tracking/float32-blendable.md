# Tracking — `float32-blendable` optional feature

Spec: [Block 64](../blocks/64-float32-blendable.md). Goal: Dawn parity for the
WebGPU `float32-blendable` optional feature
(`WGPUFeatureName_Float32Blendable = 0x0F`) on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + caps gate (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | TODO |
| 3 | Docs + Phase Review | TODO |

## Key facts (verified 2026-07-01)

- Template: `depth-clip-control` (`[[subgroups-feature]]` shape). Pure
  capability + validation; no shader / no HAL execution change.
- `TextureFormat::caps(features)` (`format.rs:272`) already feature-aware
  (Depth32FloatStencil8 gate at `:460`). float32 formats R32_FLOAT (`:344`),
  RG32_FLOAT (`:401`), RGBA32_FLOAT (`:442`) currently lack `.blendable()`.
- Render-pipeline blend check `render_pipeline.rs:2690` already reads
  `caps(features).is_blendable` — flipping the caps table gates it automatically.
- Dawn parity: Metal unconditional (`PhysicalDeviceMTL.mm:720`); Vulkan =
  R32/RG32/RGBA32_SFLOAT all have `COLOR_ATTACHMENT_BLEND_BIT`
  (`PhysicalDeviceVk.cpp:444`); caps add mirrors `Format.cpp:336`.
- Touch-points: `adapter.rs` Feature + add_*_feature + features();
  `yawgpu-hal/src/lib.rs` dispatch + 4 backends; `conv/feature.rs`;
  `format.rs` caps; `ffi/mod.rs` feature-count test. No shim / FFI-struct change.
