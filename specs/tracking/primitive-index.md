# Tracking — `primitive-index` optional feature

Spec: [Block 69](../blocks/69-primitive-index.md). Goal: Dawn parity for
`WGPUFeatureName_PrimitiveIndex = 0x15` on Tier-1 Metal + Vulkan. **Last standard
optional-feature backfill.**

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + Tint gate (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU e2e + CTS | TODO |
| 3 | Docs + Phase Review | TODO |

## Key facts (verified 2026-07-01)

- **Reflection + inter-stage counting already exist:** shim reflects
  `primitive_index_used` (`tint_shim.cpp:398`); `input_inter_stage_builtin_count`
  counts it (`shader_tint.rs:934`). So NO inter-stage / reflection change — just
  advertise + Vulkan `geometryShader` enable + Tint `kPrimitiveIndex` gate.
- `@builtin(primitive_index)` needs `enable primitive_index;` (CTS
  `maxInterStageShaderVariables.spec.cpp:135`).
- Dawn parity: Metal `supportsFamily(MTLGPUFamilyApple7)` (`PhysicalDeviceMTL.mm:743`);
  Vulkan `VkPhysicalDeviceFeatures.geometryShader` (`PhysicalDeviceVk.cpp:345`).
  Metal pattern: mirror `supports_subgroups` (`metal/mod.rs`, uses
  `MTLGPUFamily::Apple6`). M2 is Apple8 → advertises.
- Shader-compile gate + shim mirror `clip_distances` (`Program::parse` now
  `(wgsl, shader_f16, subgroups, dual_source_blending, clip_distances,
  language_features)` → add `primitive_index`; keep .h/.cpp/Rust-extern/callsites
  in sync). Vulkan enable mirrors `shader_clip_distance` (`vulkan/mod.rs`).
- `features_validation.rs` "unsupported example" currently `PrimitiveIndex` (about
  to become supported) → swap to a feature the Noop adapter never advertises, e.g.
  `native::WGPUFeatureName_TextureCompressionBC` (Noop HAL reports no BC support).
- CTS trees: `shader,validation,shader_io,builtins` (primitive_index) +
  `capability_checks,limits,maxInterStageShaderVariables` (primitive_index items).
