# Block 62 — `subgroups` optional feature

Status: **Slices 1–3 DONE, real-GPU verified** (Metal M2 + MoltenVK). Slice 4
(docs + Phase Review) in progress. Owner: subgroups initiative.

The WebGPU `subgroups` optional feature (`WGPUFeatureName_Subgroups = 0x12`)
exposes SIMD-lane ("subgroup" / "wave" / "SIMD-group") built-ins and collective
operations to WGSL. It is a **Tier-1 real-GPU feature**: like `shader-f16`
([Block 60](60-real-backends.md), `specs/tracking/shader-f16.md`) it advertises
only when the HAL backend supports it, is validation-gated in `yawgpu-core`
identically across all backends, and is execution-verified on real Metal +
Vulkan hardware. Tint (the shader frontend) already ships subgroup codegen for
the MSL and SPIR-V backends upstream — **no Tint fork transform is required**;
the shim only has to add `Extension::kSubgroups` to the allowed feature set when
the device enabled `subgroups`.

This block does **not** cover `subgroup_matrix`
(`chromium_experimental_subgroup_matrix`) — that is a separate, later feature.

## Motivation

The project goal is Dawn parity (`[[cts-dawn-parity-goal]]`). Subgroups is a
shipped Dawn optional feature; the CTS trees `basics,subgroups` and
`uniform_subgroup_ops` currently skip on yawgpu because we do not advertise the
feature. "Not executable on Noop" is **not** a reason to skip: real-GPU gating
is the established pattern (`shader-f16`, `tiled`, occlusion queries). The
validation surface (feature enumeration, the `enable subgroups;` accept/reject
gate) is fully Noop-testable; only execution needs a GPU.

## Public API surface

No new public C entry points. The feature flows through existing surfaces:

- **`wgpuAdapterGetFeatures` / `wgpuAdapterHasFeature`** report
  `WGPUFeatureName_Subgroups` when the adapter's backend supports subgroups.
- **`wgpuAdapterGetInfo`** populates `WGPUAdapterInfo::subgroupMinSize` and
  `subgroupMaxSize` from the HAL (currently hard-coded `0`). These are reported
  regardless of whether the feature is requested (they describe the hardware),
  matching Dawn. When the backend does not support subgroups both are `0`.
- **`wgpuDeviceGetFeatures` / `wgpuDeviceHasFeature`** report `Subgroups`
  when it was in `requiredFeatures` at device creation.
- **Device creation** validates `requiredFeatures` as usual: requesting
  `Subgroups` on an adapter that does not advertise it is a validation error
  (existing `resolve_features` path).

### `yawgpu-core::Feature`

Add a `Subgroups` variant to `yawgpu-core::adapter::Feature` (not `cfg`-gated —
subgroups ships in the default build, unlike `tiled`). Map it C↔Rust in
`yawgpu/src/conv/feature.rs` (`Subgroups ↔ WGPUFeatureName_Subgroups`). Advertise
it from `Adapter::features()` via a new `add_subgroups_feature(features, hal)`
that consults `HalAdapter::supports_subgroups()`.

## Behaviour contract

### Advertisement (HAL capability query)

`HalAdapter` gains `supports_subgroups() -> bool` and a subgroup-size query
(`subgroup_min_size()`/`subgroup_max_size()`, or a single
`subgroup_size_range() -> Option<(u32, u32)>`), static-enum-dispatched to each
backend:

- **Metal** — supported when the device is Apple-family GPU6+ **or** Metal3
  (`[MTLDevice supportsFamily:MTLGPUFamilyApple6]` /
  `MTLGPUFamilyMetal3`), matching Dawn's `IsGPUFamilyApple6OrNewer` gate for the
  full subgroup op set (ballot/broadcast/shuffle/reduce/quad). Size range: Apple
  GPUs have a fixed SIMD width of 32 (`threadExecutionWidth`), so
  min == max == 32; on Mac2 discrete GPUs read `threadExecutionWidth`.
  **Known simplification (Phase-Review MINOR):** the implementation currently
  reports a fixed `(32, 32)` for every supported Metal device rather than reading
  `threadExecutionWidth` — that width is a compute-pipeline-state property, not a
  cheap `MTLDevice` query, and the verified hardware (Apple M2) is 32. An AMD Mac
  (SIMD width 64) would under-report; refine when a non-Apple Metal GPU is in
  scope.
- **Vulkan** — supported when `VkPhysicalDeviceSubgroupProperties` reports the
  required operation groups (`BASIC | VOTE | ARITHMETIC | BALLOT | SHUFFLE |
  QUAD`) in `supportedOperations`, and both `COMPUTE` and `FRAGMENT` in
  `supportedStages`. Size range: `minSubgroupSize`/`maxSubgroupSize` from
  `VkPhysicalDeviceSubgroupSizeControlProperties` when available (Vulkan 1.3 /
  `VK_EXT_subgroup_size_control`), else `subgroupSize` for both, and rejected
  (unsupported) if it falls outside WebGPU's `[4, 128]` bounds (Dawn's
  `allowSubgroupSizeRanges`). The advertisement query matches Dawn's
  `hasBaseSubgroupSupport`: `COMPUTE | FRAGMENT` stages and `BASIC | BALLOT |
  SHUFFLE | SHUFFLE_RELATIVE | ARITHMETIC | QUAD` operations. **Deviation from
  Dawn:** yawgpu does not additionally require `VK_EXT_subgroup_size_control`
  for advertisement, because it does not yet create varying-subgroup-size
  pipelines (`ALLOW_VARYING_SUBGROUP_SIZE`); the op/stage set is the
  spec-meaningful requirement.
- **Noop** — reports **supported**, size range 4..=4 (nominal). This keeps the
  accept-path validation unit-testable with no GPU, mirroring `shader-f16`'s
  Noop=`true`. Noop never executes, so a nominal width is harmless.
- **GLES** — reports **not supported** (Tier 2; no clean GLES 3.1 mapping).
  Catalogue in [Block 67](67-gles-backend.md) if revisited.

### Validation gate (Tier-independent core rule)

`Device::create_shader_module` derives `let subgroups =
self.inner.features.contains(&Feature::Subgroups)` and threads it (alongside the
existing `shader_f16`) into `ShaderModule::from_wgsl` →
`parse_and_validate_wgsl_gated` → `yawgpu_tint::Program::parse` → the shim's
`yawgpu_tint_program_create`, which does:

```cpp
if (subgroups) {
    options.allowed_features.extensions.insert(tint::wgsl::Extension::kSubgroups);
}
```

Contract:

- Device **without** `Subgroups`: a WGSL module containing `enable subgroups;`
  (or a bare subgroup built-in) is **rejected** at `createShaderModule` and the
  message routes to the device error sink. **Identical on every backend**,
  including Noop and GLES (core validation is tier-independent — never relaxed
  for a backend).
- Device **with** `Subgroups`: the same module **compiles**. On Noop it compiles
  but does not execute; on Metal/Vulkan it executes (Slice 2).

### WGSL language features `subgroup_id` / `subgroup_uniformity`

The two WGSL *language* features (`WGPUWGSLLanguageFeatureName_SubgroupId = 6`,
`SubgroupUniformity = 8`; Tint `LanguageFeature::kSubgroupId` /
`kSubgroupUniformity`) are **distinct** from the `enable subgroups;` extension.
Slice 1 only adds their cases to the shim's `to_tint_language_feature` mapping so
the mapping is complete. **Instance-level reporting is deferred to Slice 3** —
adding them to `SUPPORTED_WGSL_LANGUAGE_FEATURES`
(`yawgpu-core/src/wgsl_language_features.rs`) requires verifying subgroup
uniformity-analysis parity with Dawn first, and the existing unit test
(`supported_wgsl_language_features_match_canonical_api_values`) asserts they are
absent (`!contains(&6)`, `!contains(&8)`) — that assertion stays true until
Slice 3 flips it deliberately.

## Slices

1. **Feature plumbing + validation gate (Noop + HAL caps, no execution).**
   `Feature::Subgroups`, `add_subgroups_feature`, `HalAdapter::supports_subgroups`
   + size query on all four backends, `conv/feature.rs`, AdapterInfo size wiring
   (`yawgpu/src/ffi/mod.rs`), the `create_shader_module` gate threaded to the
   shim (`bool subgroups` param + `kSubgroups` insert), shim
   `to_tint_language_feature` cases 6/8. Inline unit tests: feature enumeration,
   accept/reject gate on Noop, AdapterInfo size reporting. **Acceptance:**
   `cargo test --workspace` green on Noop; `enable subgroups;` rejected without
   the feature and accepted with it.

2. **Real-GPU execution e2e (Metal + Vulkan).** `e2e_metal_subgroups.rs` /
   `e2e_vulkan_subgroups.rs`: a compute shader using `subgroup_size`,
   `subgroupAdd`, and `subgroupBallot`, dispatched with buffer readback and
   value verification. Close any HAL device-creation gaps (Vulkan subgroup-size
   control / required-op enablement). **Acceptance:** both e2e suites pass on the
   M2 (Metal) and MoltenVK (Vulkan), verified by Claude.

3. **WGSL language-feature instance reporting + subgroup uniformity.** Add
   `subgroup_id` / `subgroup_uniformity` to `SUPPORTED_WGSL_LANGUAGE_FEATURES`,
   flip the canonical-values unit test, wire subgroup uniformity analysis, and
   confirm CTS `uniform_subgroup_ops` parity. **Acceptance:** instance reports
   both features; CTS subgroup-uniformity parity with Dawn.

4. **Docs + example + Phase Review.** README + Block 62 finalization, an
   optional `examples/` subgroup demo, and the mandatory no-context Phase Review.

Tracking: `specs/tracking/subgroups.md`.
