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

HAL caps: Noop = supported, 4..4 nominal; Metal = Apple6 || Mac2 (Block 141), 32..32 on Apple GPUs and 4..64 otherwise, range
reported even without subgroups;
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

## Block 108 — `subgroup-size-control` (`@subgroup_size`)

Spec: `specs/blocks/108-subgroup-size-control.md`.

| Slice | Content | Status |
|---|---|---|
| S1 | core `Feature::SubgroupSizeControl` (⇒ `Subgroups`), FFI `0x17`, Tint `kSubgroupSizeControl` gate + override-resolved `subgroup_size`, core rules 4 → 1 → 2 → 3, HAL `subgroup_size_control_caps()` (Noop `{4,4,64}`, GLES `None`; Vulkan per S2; Metal `{32,32,32}` on Apple GPUs since Block 140), `required_subgroup_size` HAL input | **DONE 2026-09-26** |
| S2 | Vulkan advertisement (`VK_EXT_subgroup_size_control` + `subgroupSizeControl` + `computeFullSubgroups`), device enable, `RequiredSubgroupSize` + `REQUIRE_FULL_SUBGROUPS` / `ALLOW_VARYING_SUBGROUP_SIZE`; predicate also requires `requiredSubgroupSizeStages ∋ COMPUTE` (VUID-02755, stricter than Dawn) | **DONE 2026-09-26** — NVIDIA caps `{32, 32, 32}`, 55 ignored HAL tests validation-clean |
| S3 | `e2e_vulkan_subgroup_size_control.rs` (Claude) | **DONE 2026-09-26** — 3/3 under the Khronos layer, 0 VUID; `e2e_vulkan_subgroups` 4/4 + `e2e_vulkan_compute` 3/3 regression |
| S4 | webgpu-native-cts port update + CTS runs | **DONE 2026-09-26** — CTS `c77b620`; targets 6/0 + 87/0 + 1/0; full sweep +94 pass / −94 skip, fail set unchanged |
| S5 | Phase Review | **DONE 2026-09-26**. 0 CRITICAL / 0 MAJOR / 4 MINOR; m1, m2, m4 fixed; m3 deferred (no fitting `HalError` variant, and adding one is an API change; unreachable after validation). Table in Block 108 |

S1 finding: Tint validates a const `@subgroup_size` (zero / non-power-of-two rejected at parse) but not an override-driven one after `SubstituteOverrides`; core enforces rule 4.

Owed on the Mac for Block 108: a `--features metal` compile check, and a MoltenVK run of `e2e_vulkan_subgroup_size_control` (is `computeFullSubgroups` exposed?).

Block 140 (2026-11-04): Metal reports Dawn's subgroup range (Apple 32/32, other vendors and iOS 4/64) and advertises `subgroup-size-control` when the range is fixed. `e2e_metal_subgroup_size_control` 3/3 on the M2.

## Block 170 — `chromium-experimental-subgroup-matrix`

Spec: `specs/blocks/170-subgroup-matrix.md`.

| Slice | Content | Status |
|---|---|---|
| S1 | core `Feature::ChromiumExperimentalSubgroupMatrix` (⇒ `Subgroups`), FFI `0x00050034`, `WGPUAdapterPropertiesSubgroupMatrixConfigs` (`0x0005003B`) on both adapter-info queries with the caller chain preserved, Tint `kChromiumExperimentalSubgroupMatrix` gate, `uses_subgroup_matrix` + `SubgroupMatrixInfo` reflection on the override-substituted IR, core rules R5 / R6, HAL `subgroup_matrix_configs()` (Noop one 8×8×8 f32; Metal / Vulkan / GLES empty) | **DONE 2026-11-28** |
| S2 | Metal advertisement (Apple7) + `e2e_metal_subgroup_matrix.rs` | **DONE 2026-11-28** — M2 reports 8×8×8 f32 and f16, range 32–32; e2e 7/0 at `7863494`, including the store-with-offset, out-of-bounds and constructor scenarios added in S3 (those three not checked under a mutation on Metal) |
| S3 | Vulkan advertisement, device enable, full subgroups + `e2e_vulkan_subgroup_matrix.rs` on native Windows / Linux | **DONE on Windows 2026-11-28** — RTX 5060 Ti (621.24): 12 configurations (16×16×16, 16×8×16, 16×8×8 f16 → f16 / f32; 16×16×32, 16×8×32 u8 → u32 / i32 and i8 → i32), range 32–32, `cooperativeMatrixRobustBufferAccess` false so Tint clamps; e2e 7/0 (store-with-offset, out-of-bounds load / store and constructor scenarios added to the shared body; each fails under a mutation) and the 4 ignored HAL subgroup tests under the Khronos layer, 0 VUID; 8-bit configurations are listed only with `shaderInt8` (R9, stricter than Dawn); `e2e_vulkan_subgroup_size_control` 3/0, `e2e_vulkan_subgroups` 3/0, `e2e_vulkan_compute` 4/0 regression. **DONE on Linux 2026-11-28** at `01cb53f` — native RTX 5060 Ti (615.71.09, Vulkan 1.4.351): 10 configurations (the same six f16 ones; u8 → u32 and i8 → i32 at 16×16×32 and 16×8×32 — the two u8 → i32 configurations of the Windows 621.24 driver are not listed), range 32–32; with `YAWGPU_REQUIRE_SUBGROUP_MATRIX=1` under the Khronos layer `e2e_vulkan_subgroup_matrix` 7/0, `e2e_vulkan_subgroup_size_control` 3/0, `e2e_vulkan_subgroups` 3/0, `e2e_vulkan_compute` 4/0, 0 VUID |
| S4 | docs + Phase Review | **DONE 2026-11-28** — `README` section, Block 67 matrix row, Phase Review below |

### Block 170 Phase Review (2026-11-28)

Fresh no-context review of `6cff8cb..717bee0` against R1–R10 and the Dawn sources. **0 CRITICAL, 0 MAJOR, 10 MINOR.**

| # | Finding | Decision |
|---|---|---|
| m1 | `yawgpu/src/ffi/mod.rs` `write_adapter_info` held a `#[cfg(test)]` thread-local to model an adapter without the feature | Fixed: the writer takes the configuration slice; the test passes an empty one |
| m2 | Both adapter-info entry points read the caller's `nextInChain`, so `*info` must be initialized; R2 did not say so | Fixed in the spec (R2) and in the `yawgpu.h` comment. In-repo callers all initialize. `webgpu-native-cts` and the JS binding are outside the repo and were not checked |
| m3 | R6 is scoped to the pipeline's entry point; Dawn's Vulkan backend gathers from the whole module. The Vulkan list is de-duplicated; Dawn's is not | Documented in R6 and R9. Behaviour kept: it matches Dawn's Metal backend and R4 |
| m4 | Metal predicate is Apple7 **and** subgroups; R8 said Apple7 | Spec R8 updated. Equivalent on real devices |
| m5 | `VulkanDeviceInner::cooperative_matrix` is read only by `Debug` and an ignored real-device test | Deferred: it is the record that R10's enable took effect, which that test asserts on native drivers; removing it loses the check |
| m6 | No core test for a matrix shader with an override-driven workgroup size | Fixed: `subgroup_matrix_pipeline_validates_overridden_workgroup_size` (`x = 6` rejected, `x = 8` accepted) |
| m7 | `subgroup_matrix_full_subgroups_reaches_hal_descriptor` asserted only the resolved flag | Fixed: renamed `subgroup_matrix_entry_point_requires_full_subgroups` |
| m8 | Shared e2e body: only the advertisement test honoured `required`; `subgroup_matrix_validation_errors` had no non-empty assertion | Fixed: every scenario takes `required`, and the validation scenario asserts a non-empty configuration list. The out-of-bounds scenario still skips when every configuration is 8-bit: Dawn's shader binds `array<ComponentType>`, which has no 8-bit form |
| m9 | The Dawn-named block of `yawgpu.h` had no documentation | Fixed: Doxygen group with the shape convention, the error condition and the ownership / free contract |
| m10 | `yawgpu-tint` builds `#[repr(u32)]` enums from shim-written memory | Deferred: the shim's `static_assert`s pin every value and range against Tint's header, so a Dawn bump that changes them fails the build |

Gate after the fixes (Apple M2): `cargo fmt --check` clean; workspace 1,532/0; clippy clean on default, `metal` and `vulkan`; `e2e_metal_subgroup_matrix` 7/0. Before the fixes at `717bee0`: HAL lib 219/0 (`metal`) and 393/0 (`vulkan`). The e2e change (m8) ran on a native Windows Vulkan driver afterwards: `e2e_vulkan_subgroup_matrix` 7/0 under the Khronos layer (`specs/tracking/dawn-compat.md`, "Windows / Vulkan verification").

The bindgen input is `yawgpu.h`, so the Dawn-named extension items are generated into `yawgpu::native`.
