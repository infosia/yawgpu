# Tracking — `clip-distances` optional feature

Spec: [Block 68](../blocks/68-clip-distances.md). Goal: Dawn parity for
`WGPUFeatureName_ClipDistances = 0x10` on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + Tint gate + inter-stage limit (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU e2e + CTS | TODO |
| 3 | Docs + Phase Review | TODO |

## Key facts (verified 2026-07-01)

- Dawn parity: Metal unconditional (`PhysicalDeviceMTL.mm:719`); Vulkan gates on
  `VkPhysicalDeviceFeatures.shaderClipDistance` (`PhysicalDeviceVk.cpp:339`).
- Tint `Extension::kClipDistances` ("clip_distances") lowers to MSL
  `[[clip_distance]]` / SPIR-V `ClipDistance` — no HAL codegen; only the Vulkan
  `shaderClipDistance` device-feature enable is needed.
- Tint reflects `inspector::EntryPoint::clip_distances_size`
  (`std::optional<uint32_t>`, `entry_point.h:221`). Shim entry-point struct
  `YawgpuTintEntryPoint` already carries `*_used` bools (tint_shim.h:67) — add
  `has_clip_distances` + `clip_distances_size` there (mirror the reflection of
  the other entry-point fields, filled in `fill_entry_point`, tint_shim.cpp:~386).
- Inter-stage: CTS
  `capability_checks,features,clip_distances:createRenderPipeline,at_over`
  formula — `vertex_outputs + ceil(clip_distances_size/4) + (pointList?1:0) ≤
  maxInterStageShaderVariables`. Wire into `validate_inter_stage_interface`
  (`render_pipeline.rs:2986`) + `validate_inter_stage_limits` (`:3070`, which
  already takes an `extra_builtins` count) + the point-list check (`:3006`).
- Shader-compile gate + shim thread mirror `subgroups` / `dual_source_blending`
  (`Program::parse` now `(wgsl, shader_f16, subgroups, dual_source_blending,
  language_features)` → add `clip_distances`; keep .h/.cpp/Rust-extern/callsites
  in sync). Vulkan enable mirrors `dual_src_blend` (`vulkan/mod.rs`).
- Biggest remaining backfill (shader gate + reflection + limit); PrimitiveIndex
  (0x15) is the only one left after.
