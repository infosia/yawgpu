# Tracking — `subgroups` optional feature

Spec: [Block 62](../blocks/62-subgroups.md). Goal: Dawn parity for the WebGPU
`subgroups` optional feature (`WGPUFeatureName_Subgroups = 0x12`) on the Tier-1
Metal + Vulkan backends.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + validation gate (Noop + HAL caps) | **DONE** (2026-07-01) |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | **DONE** (2026-07-01) |
| 3 | WGSL language-feature reporting + subgroup uniformity | **DONE** (2026-07-01) |
| 4 | Docs + Phase Review | **DONE** (2026-07-01) |

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

## Slice 2 — landed

`yawgpu/tests/e2e_{metal,vulkan}_subgroups.rs` (3 tests each, `#[ignore]`
manual real-backend): adapter advertises `Subgroups` + non-zero size range; an
`enable subgroups;` compute reads `@builtin(subgroup_size)` and does
`subgroupAdd(1u)` over a full 64-lane workgroup — readback asserts the runtime
size is uniform, within the adapter's reported `subgroupMin/MaxSize`, and that
`subgroupAdd(1u) == subgroup_size` for every full subgroup; a device without the
feature rejects the shader to the error sink. **Verified green on real M2 Metal
(3/3) and MoltenVK (3/3).** No HAL device-creation changes were needed — basic
Vulkan 1.1 compute subgroup ops run on MoltenVK without a size-control toggle.
MSL `simd_sum` / SPIR-V `GroupNonUniformArithmetic` both exercised.

Pre-existing lint debt noted (out of scope): `yawgpu-hal/src/metal/mod.rs:34`
imports `HalRenderPipeline` unused (from tiled commit 5a7fb23); invisible to the
default clippy gate because `metal` is not a default feature.

## Slice 3 — landed

Dawn parity confirmed by source: `feature_status.cc` marks `kSubgroupId` /
`kSubgroupUniformity` as `kShippedWithKillswitch`, and `Instance.cpp`
(`GetAllowedWGSLLanguageFeatures`) exposes `kShipped` + `kShippedWithKillswitch`
unconditionally. So yawgpu now reports both `subgroup_id` (6) and
`subgroup_uniformity` (8) in `wgpuInstanceGetWGSLLanguageFeatures`:
`SUPPORTED_WGSL_LANGUAGE_FEATURES == [1..=10]`. The shim's `to_tint_language_
feature` 6/8 mapping (Slice 1) means both are also passed to Tint's
`allowed_features` on every parse — identical to Dawn's instance-level
`mTintLanguageFeatures`. Uniformity analysis is Tint-internal (same compiler as
the CTS oracle) → parity by construction. The `enable subgroups;` device-feature
gate is unchanged. Unit tests flipped from "absent/rejected" to
"present/accepted" in wgsl_language_features.rs, shader_tint.rs, ffi/instance.rs.

## Slice 4 — landed + Phase Review

README `Shaders` section documents the feature; Block 62 finalized. No dedicated
`examples/` demo (matches shader-f16, which is e2e-verified only). The mandatory
no-context Phase Review of `c64d033..HEAD` found:
- **MAJOR (resolved)** — Vulkan `subgroups_supported` dropped `QUAD` + `FRAGMENT`
  + `SHUFFLE_RELATIVE`, over-advertising vs Block 62 / README. Fixed to match
  Dawn's `hasBaseSubgroupSupport` (COMPUTE|FRAGMENT stages; BASIC|BALLOT|SHUFFLE|
  SHUFFLE_RELATIVE|ARITHMETIC|QUAD ops) + `[4,128]` size guard. **MoltenVK e2e
  re-verified green (3/3)** after tightening — Apple/MoltenVK reports the full
  set. Intentional deviation: size-control extension not required for
  advertisement (documented).
- **MINOR (documented)** — Metal size range hardcoded `(32,32)` (Apple M2 = 32;
  non-Apple Metal refinement deferred). Noted in Block 62.
- All other sections clean (gate tier-independence, FFI param-order threading,
  Slice-3 no-gate-bypass, e2e non-vacuous, unit-test coverage).

Phase is **COMPLETE** — no open CRITICAL/MAJOR.

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
