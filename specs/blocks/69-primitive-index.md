# Block 69 — `primitive-index` optional feature

Status: **IN PROGRESS** (Slice 1). Owner: Dawn-parity feature backfill.
**Last of the WebGPU standard optional-feature backfill.**

The WebGPU `primitive-index` optional feature
(`WGPUFeatureName_PrimitiveIndex = 0x15`) lets a fragment shader read
`@builtin(primitive_index) idx: u32` (via `enable primitive_index;`), the index
of the primitive that generated the fragment. Tint (`Extension::kPrimitiveIndex`)
lowers it to Metal `[[primitive_id]]` / SPIR-V `PrimitiveId` (which needs the
`Geometry` capability → `geometryShader` on Vulkan) — **no explicit HAL codegen**.

The reflection + inter-stage accounting **already exist**: the shim reflects
`primitive_index_used` (`tint_shim.cpp:398`) and
`input_inter_stage_builtin_count` already counts it toward
`maxInterStageShaderVariables` (`shader_tint.rs:934`). So this is the *simplest*
remaining shape: advertise + Vulkan `geometryShader` enable + the Tint-extension
gate. No new inter-stage / reflection logic.

## Public API surface
Standard feature surface. Add a `PrimitiveIndex` `Feature` variant (not
`cfg`-gated) + `conv/feature.rs` mapping + `add_primitive_index_feature`.

## Behaviour contract

### Advertisement (HAL capability query) — Dawn parity
`HalAdapter::supports_primitive_index() -> bool`:
- **Metal** — `device.supportsFamily(MTLGPUFamily::Apple7)` (Dawn gates on
  Apple7, `PhysicalDeviceMTL.mm:743`; `[[primitive_id]]` fragment input needs it).
- **Vulkan** — `VkPhysicalDeviceFeatures::geometryShader == VK_TRUE`
  (Dawn parity, `PhysicalDeviceVk.cpp:345`; the SPIR-V `PrimitiveId` builtin
  needs the `Geometry` capability).
- **Noop** — `true`. **GLES** — `false` (Tier 2).

### Vulkan device-feature enable
Enable `enabled_features.geometry_shader = vk::TRUE` whenever
`supported_features.geometry_shader == vk::TRUE` (mirror `dual_src_blend` /
`shader_clip_distance`). Required so the fragment `PrimitiveId` SPIR-V is valid.

### Tint-extension gate (shader compilation)
Thread a `primitive_index: bool` from the device's enabled features into the
shader-compile path exactly like `clip_distances` / `subgroups`
(create_shader_module → … → Program::parse → shim), where the shim inserts
`tint::wgsl::Extension::kPrimitiveIndex` when set. Without the feature,
`enable primitive_index;` is rejected at `createShaderModule`, identically on
every backend.

No inter-stage change: `primitive_index_used` is already counted toward
`maxInterStageShaderVariables` on the fragment-input side.

## Slices

1. **Feature plumbing + Tint gate (Noop + HAL cap).** `Feature::PrimitiveIndex`,
   `add_primitive_index_feature`, `HalAdapter::supports_primitive_index` (4
   backends; Metal Apple7, Vulkan geometryShader), the Vulkan `geometryShader`
   enable, `conv/feature.rs` + `conv/mod.rs` + `ffi/mod.rs` count 20→21, and the
   `primitive_index` shader-compile gate → shim (`kPrimitiveIndex`). Inline unit
   tests. **Acceptance:** `cargo test --workspace` green on Noop;
   `enable primitive_index;` rejected without the feature, accepted with it.

2. **Real-GPU + CTS verification.** An `e2e_metal_primitive_index.rs` drawing two
   triangles and reading back a per-primitive value derived from
   `@builtin(primitive_index)`; plus the CTS `shader,validation,shader_io,builtins`
   (primitive_index cases) + `capability_checks,limits,maxInterStageShaderVariables`
   (primitive_index items) green on real Metal. **Acceptance:** e2e passes on the
   M2; CTS 0 fail (MoltenVK self-skip / documented if it can't compile PrimitiveId).

3. **Docs + Phase Review.** README note; Block 69 finalization; the mandatory
   no-context Phase Review. **This completes the WebGPU standard optional-feature
   backfill** (all of 0x01–0x16 now advertised where the backend supports them).

Tracking: `specs/tracking/primitive-index.md`.
