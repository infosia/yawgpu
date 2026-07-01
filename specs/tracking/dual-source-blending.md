# Tracking — `dual-source-blending` optional feature

Spec: [Block 65](../blocks/65-dual-source-blending.md). Goal: Dawn parity for
`WGPUFeatureName_DualSourceBlending = 0x11` on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + Tint gate + validation (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | TODO |
| 3 | Docs + Phase Review | TODO |

## Key facts (verified 2026-07-01)

- **Already in tree:** `BlendFactor::{Src1,OneMinusSrc1,Src1Alpha,OneMinusSrc1Alpha}`
  (`render_pipeline.rs:339`) → `HalBlendFactor::Src1*`; HAL blend mapping complete
  both backends (Metal `metal/pipeline.rs:355`, Vulkan `vulkan/pipeline.rs:954`).
  Tint `Extension::kDualSourceBlending` emits MSL/SPIR-V decorations
  automatically → **no `@blend_src` reflection needed** in yawgpu.
- **Gap:** no `Feature`; `Src1*` factors silently accepted in core; Vulkan
  `dualSrcBlend` device feature not enabled (→ VUID if used); Tint extension not
  allowed (so `enable dual_source_blending;` is rejected — baseline test at
  `yawgpu-tint/src/lib.rs:3416`).
- Dawn parity: Metal unconditional (`PhysicalDeviceMTL.mm:716`); Vulkan gates on
  `features.dualSrcBlend` (`PhysicalDeviceVk.cpp`).
- Shim signature currently `(wgsl, wgsl_len, shader_f16, subgroups,
  allow_framebuffer_fetch, lang_features, n_lang_features, err)` — add
  `dual_source_blending` after `subgroups` (keep .h / .cpp / Rust extern / parse
  call site in sync).
- Vulkan device-feature enable mirrors `independent_blend` (`mod.rs:568/594`).
- Template: `subgroups` shader gate + `float32-blendable` blend validation.
