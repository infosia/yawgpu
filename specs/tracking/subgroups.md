# Tracking — `subgroups` optional feature

Spec: [Block 62](../blocks/62-subgroups.md). Goal: Dawn parity for the WebGPU
`subgroups` optional feature (`WGPUFeatureName_Subgroups = 0x12`) on the Tier-1
Metal + Vulkan backends.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + validation gate (Noop + HAL caps) | **DONE** (2026-07-01) |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | TODO |
| 3 | WGSL language-feature reporting + subgroup uniformity | TODO |
| 4 | Docs + example + Phase Review | TODO |

## Key facts (verified 2026-07-01)

- FFI already has the canonical values: `WGPUFeatureName_Subgroups = 0x12`,
  `WGPUWGSLLanguageFeatureName_SubgroupId = 6`, `SubgroupUniformity = 8`, and
  `WGPUAdapterInfo::subgroupMinSize` / `subgroupMaxSize` fields
  (`yawgpu/ffi/webgpu-headers/webgpu.h`). No header regen for the feature enum.
- Tint enum names: `Extension::kSubgroups` (WGSL `enable subgroups;`),
  `LanguageFeature::kSubgroupId` ("subgroup_id"),
  `LanguageFeature::kSubgroupUniformity` ("subgroup_uniformity")
  (`third_party/dawn/src/tint/lang/wgsl/enums.h`). Tint MSL + SPIR-V writers
  ship subgroup codegen upstream — no fork transform needed.
- `shader-f16` is the end-to-end template. Touch-points to mirror:
  - `yawgpu-core/src/adapter.rs` — `Feature` enum (:154), `add_shader_float16_feature`
    (:241), `features()` aggregation (:75).
  - `yawgpu-core/src/device.rs:354` — feature→gate-bool derivation in
    `create_shader_module`.
  - `yawgpu-core/src/shader.rs` / `shader_tint.rs:35` — `from_wgsl` /
    `parse_and_validate_wgsl_gated` gate threading.
  - `yawgpu-hal/src/lib.rs:343` — `supports_shader_float16` dispatch; add
    `supports_subgroups` + size query beside it.
  - `yawgpu-hal/src/{metal,vulkan,noop,gles}/…` — per-backend cap impls.
  - `yawgpu/src/conv/feature.rs:23,53` — C↔Rust feature mapping.
  - `yawgpu/src/ffi/mod.rs:1885-1886` — AdapterInfo subgroupMin/MaxSize (currently
    hard-coded 0).
  - `yawgpu-tint/shim/tint_shim.{h,cpp}` — `yawgpu_tint_program_create` signature
    (+`bool subgroups`), `kSubgroups` insert, `to_tint_language_feature` cases 6/8.
  - `yawgpu-tint/src/lib.rs` — `Program::parse` signature threading.

## Slice 1 — landed

Mirrors `shader-f16` end to end. `WGPUFeatureName_Subgroups` advertised when the
HAL backend reports support; `WGPUAdapterInfo::subgroupMin/MaxSize` wired from
the HAL; `enable subgroups;` gated on the device feature via a `subgroups: bool`
threaded `create_shader_module` → `from_wgsl` → `parse_and_validate_wgsl_gated`
→ `Program::parse` → shim (`if (subgroups) allowed_features.extensions.insert(
Extension::kSubgroups)`). Shim `to_tint_language_feature` now maps 6/8 (mapping
completeness only; NOT added to `SUPPORTED_WGSL_LANGUAGE_FEATURES`).

HAL caps: Noop = supported, 4..4 nominal; Metal = Apple6 || Metal3, 32..32;
Vulkan = `VkPhysicalDeviceSubgroupProperties` requires BASIC|VOTE|ARITHMETIC|
BALLOT|SHUFFLE ops + COMPUTE stage, size from
`VkPhysicalDeviceSubgroupSizeControlProperties` (1.3/ext) else `subgroupSize`;
GLES = unsupported. Verified: `cargo test --workspace` green (Noop), `--features
tiled` build green, default clippy green, `-p yawgpu-hal --features metal` green.
Real-GPU advertisement/sizes on Metal + MoltenVK are confirmed in Slice 2.

## Notes / open questions

- **Vulkan device-creation enablement** (Slice 2): basic Vulkan 1.1 subgroup ops
  in compute need no feature toggle, but fragment-stage ops depend on
  `supportedStages`, and required/full subgroup size needs
  `VK_EXT_subgroup_size_control` (or Vulkan 1.3). Verify against MoltenVK during
  Slice 2; MoltenVK subgroup support is partial — expect to scope the e2e to the
  op subset MoltenVK reports (document any skips, never silently narrow).
- **Metal quad ops** need Apple6+/Metal3; the cap gate already encodes that.
- Slice 1 keeps `subgroup_id`/`subgroup_uniformity` OUT of
  `SUPPORTED_WGSL_LANGUAGE_FEATURES`; the canonical-values unit test still asserts
  their absence until Slice 3.
