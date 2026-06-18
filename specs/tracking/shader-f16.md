# shader-f16 feature — implementation plan & behaviour contract

Status: **CODE COMPLETE + REAL-GPU VERIFIED** (single large slice;
Noop+Metal+Vulkan, full feature). Pending: Phase Review + commit; optional
external webgpu-native-cts f16 re-confirm.

**Real-GPU results (Claude, M2):**
- `yawgpu/tests/e2e_metal_f16.rs` — 3/3 PASS on Metal: adapter advertises
  shader-f16; f16 compute (u32→`half` arithmetic→f16 storage out) readback
  exact; f16 shader rejected without the feature (S12 gate on a real backend).
- `yawgpu/tests/e2e_vulkan_f16.rs` — 3/3 PASS on Vulkan/MoltenVK: same three,
  exercising the `VK_KHR_shader_float16_int8` (`shaderFloat16`) +
  `VK_KHR_16bit_storage` (`storageBuffer16BitAccess`) device enablement with
  **zero** validation errors. MoltenVK advertises `shaderFloat16`.
- Noop workspace test (63/63 bins) + feature-gated HAL tests
  (`-p yawgpu-hal --features metal,vulkan --lib`, 126 pass) green.

Resolves the long-standing Phase-5 divergence **m4**
(`specs/tracking/phase-5-review.md:26`, `specs/blocks/40-pipeline.md:70`):
the naga `Validator` enabled `SHADER_FLOAT16` unconditionally, so `enable
f16;` WGSL validated even when the device had not requested the WebGPU
`shader-f16` feature. This work introduces the canonical `ShaderF16`
feature-gating path and makes f16 a real, backend-honoured feature.

## Goal

Implement the WebGPU `shader-f16` optional feature end-to-end:

- advertise it on adapters whose HAL backend supports f16 (Noop, Metal,
  Vulkan-when-`shaderFloat16`), reject on GLES (Tier 2);
- gate WGSL `f16` validation on the device having requested the feature
  (request absent ⇒ `enable f16;` / f16-usage shaders are a validation
  error routed to the device error sink);
- make f16 actually run on Metal (`half`, already proven by F-119) and
  Vulkan (enable `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` so
  f16 works in arithmetic **and** in storage/uniform buffers and inter-stage
  IO — "full feature", not arithmetic-only).

`SHADER_FLOAT16_IN_FLOAT32` stays **unconditional / baseline** (F-119:
`pack2x16float` / `unpack2x16float` / `quantizeToF16` are core WGSL builtins
with internal-only f16 and require no `shader-f16` feature).

## Current state (anchors)

- `Feature` enum: `yawgpu-core/src/adapter.rs:164` — no `ShaderF16`.
- `supported_features()`: `adapter.rs:222`; per-backend feature assembly in
  `Adapter::features()` `adapter.rs:77` (`add_texture_compression_features`
  pattern at `adapter.rs:237`).
- FFI map: `yawgpu/src/conv/feature.rs:6` (`map_feature`) /:39
  (`map_feature_to_native`) — `WGPUFeatureName_ShaderF16` (0x0000000B,
  `webgpu.h:652`) currently falls through to `Feature::Other(11)`.
- naga validation: `yawgpu-core/src/shader_naga.rs:536` `validate_module`
  (capabilities hardcoded). Production callers that must be gated:
  `ShaderModule::from_wgsl` `shader.rs:134` and `from_spirv`/`reflect_spirv`
  `shader.rs:145` / `shader_naga.rs:528`. `create_shader_module`
  (`device.rs:385`) has `self.inner.features`. The ~40 other
  `parse_and_validate_wgsl` callers are `#[cfg(test)]`.
- HAL dispatch for capabilities: `yawgpu-hal/src/lib.rs:309-365`
  (`supports_texture_compression_*` pattern). Vulkan device feature chain
  template: `yawgpu-hal/src/vulkan/mod.rs` robustness2 block (~360-435),
  cached in `VulkanDeviceInner` (`vulkan/device.rs:4-29`). Metal `half` is
  native (no device flag); GLES has none.
- naga SPIR-V backend `spv::Options { capabilities: None, .. }`
  (`shader_naga.rs:589`) already emits Float16 / 16-bit-storage SPIR-V
  capabilities on demand — no codegen change needed; the Vulkan **device**
  just has to enable the matching VK features.

## Behaviour contract (rules)

- **F16-1 (advertise, HAL-gated).** A new `HalAdapter::supports_shader_float16()`
  dispatches per backend: **Noop ⇒ true**, **Metal ⇒ true**, **Vulkan ⇒
  true iff `VK_KHR_shader_float16_int8`/`shaderFloat16` is present**, **GLES
  ⇒ false**. `Adapter::features()` inserts `Feature::ShaderF16` when it
  returns true (new `add_shader_float16_feature` helper, mirroring
  `add_texture_compression_features`).
- **F16-2 (FFI round-trip).** `Feature::ShaderF16` ⇄
  `WGPUFeatureName_ShaderF16` in both `map_feature` and
  `map_feature_to_native` (no longer `Other(11)`).
- **F16-3 (request honoured).** `Adapter::create_device` already validates
  requested features against `features()`; once advertised, `requiredFeatures
  = [ShaderF16]` resolves and `device.HasFeature(ShaderF16)` is true.
- **F16-4 (validation gate).** WGSL/SPIR-V shader-module creation enables the
  naga `SHADER_FLOAT16` capability **iff the device has `ShaderF16`**.
  Baseline capabilities (incl. `SHADER_FLOAT16_IN_FLOAT32`,
  `CUBE_ARRAY_TEXTURES`, `MULTISAMPLED_SHADING`,
  `STORAGE_TEXTURE_16BIT_NORM_FORMATS`, `TEXTURE_EXTERNAL`) stay
  unconditional. Without the feature, an f16-using shader fails naga
  validation ⇒ device error + error shader-module handle (existing
  `create_shader_module` error path).
  - Threading: add a gated entry (e.g.
    `parse_and_validate_wgsl_gated(src, shader_f16: bool)` +
    `validate_module(module, shader_f16: bool)`); keep the existing
    `parse_and_validate_wgsl(src)` as a thin wrapper that passes
    `shader_f16 = true` so the ~40 test callers are untouched. Production
    `from_wgsl`/`from_spirv` take `shader_f16: bool` (passed from
    `device.create_shader_module` via `self.inner.features.contains(
    &Feature::ShaderF16)`).
  - Known nuance (document, acceptable): naga gates on f16 **usage**, not on
    the bare `enable f16;` directive — a shader that declares `enable f16;`
    but uses no f16 type may validate without the feature. Record as a
    naga≠Tint divergence (block 30 convention: assert error-vs-success on
    real usage, not the directive alone).
- **F16-5 (Vulkan device enablement, full feature).** When the adapter
  supports it, Vulkan device creation enables `VK_KHR_shader_float16_int8`
  (`shaderFloat16`) **and** `VK_KHR_16bit_storage`
  (`storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`, and
  `storageInputOutput16` / `storagePushConstant16` where the physical device
  reports them) so naga-emitted f16-in-buffer / f16-IO SPIR-V is loadable.
  Query via `get_physical_device_features2` pNext chain + extension presence
  (robustness2 pattern); enable only the sub-features the device reports.
  Mirror the equivalent wgpu-hal/vulkan path in `../wgpu`.
- **F16-6 (Metal).** Advertise unconditionally; naga MSL already emits
  `half`/`half2` and compiles (F-119). No device flag. (If a GPU-family gate
  proves necessary on older HW, add it; M-series is fine.)
- **F16-7 (GLES, Tier 2).** `supports_shader_float16()` ⇒ false; no f16. A
  validated f16 shader cannot reach GLES because the feature is never
  advertised there (core gate, not a HAL `HalError`).

## Tests

- **Unit (Noop, principle 1):**
  - `adapter.rs`: Noop adapter advertises `ShaderF16`; `create_device`
    resolves `requiredFeatures=[ShaderF16]`; device `HasFeature(ShaderF16)`.
  - `conv/feature.rs`: `map_feature`/`map_feature_to_native` round-trip
    `ShaderF16 ⇄ WGPUFeatureName_ShaderF16`.
  - `shader_naga.rs` / `shader.rs`: an `enable f16;` + f16-usage shader
    **fails** validation when `shader_f16=false` and **passes** when true;
    `pack2x16float` still passes with `shader_f16=false` (F-119 unchanged).
  - HAL: `supports_shader_float16()` per-backend dispatch unit tests
    (Noop=true; Metal/Vulkan behind `--features`, Vulkan honest about
    query path; GLES=false).
- **Integration (Noop):** a C-FFI test — request `shader-f16`, create an
  f16 WGSL module (success); without it, creation raises a device error.
  Plus the inverse on an adapter that does advertise it.
- **Real-GPU (Claude runs, post-handoff):**
  - `e2e_metal_*`: f16 compute (storage buffer in/out, half arithmetic)
    readback correctness.
  - `e2e_vulkan_*`: same on Vulkan/MoltenVK (exercises the
    `shaderFloat16`+`16bit_storage` enablement).
- **CTS (Claude runs):** un-defer the 2 `overrides.spec.ts` f16 cases; run
  webgpu-native-cts f16 coverage (Metal + MoltenVK), update the ledger.

## Integration checklist (Claude, after the agent's diff is reviewed)

- [x] `cargo test --workspace` + `clippy -D warnings` (default + `--features
      tiled`) green on Noop.
- [x] `cargo test -p yawgpu-hal --features metal,vulkan --lib` (run, not just
      compile — see [[feedback-run-feature-gated-hal-tests]]). 126 pass.
- [x] Real-GPU e2e (Metal + Vulkan/MoltenVK) authored & green on the M2.
- [x] Spec updates: `00-foundation.md` advertised-feature list (+`ShaderF16`
      / 0xB); `30-shader-binding.md` **S12** gate rule; `40-pipeline.md` m4
      note → **resolved**; `cts-coverage.md` overrides f16 cases un-deferred.
- [x] Phase Review (fresh subagent) on the cumulative diff — 0 CRITICAL /
      0 MAJOR; 2 MINOR fixed, 3 deferred (see "Phase Review" below).
- [x] Commit.
- [ ] (Optional, sibling project) external webgpu-native-cts f16 query
      re-confirm on Metal + MoltenVK; record in the suite's `docs/FINDINGS.md`.

## Phase Review (fresh no-context subagent on the cumulative diff)

**Verdict: 0 CRITICAL, 0 MAJOR — commit-ready.** Confirmed: Vulkan ash
pNext enable-structs do not dangle (locals in the `create_device` scope);
query vs enable structs cleanly separated; `Feature` enum is match-mapped
(not ordinal) so the mid-enum insert is ABI-safe; the validation gate is
un-bypassable (only `#[cfg(test)]` callers hit the f16-on default helpers);
`SHADER_FLOAT16_IN_FLOAT32` stays unconditional. 5 MINOR:

- **#2 FIXED** — Vulkan `enabled_16bit_storage_features` now sets
  `enabled=true` only when ≥1 device-reported sub-feature is TRUE (else the
  `VK_KHR_16bit_storage` extension is not pushed). +unit-test case.
- **#4 FIXED** — added a direct `reflect_spirv_gated` unit test
  (`shader-passthrough`). Real f16-SPIR-V in-test was impractical: naga's
  strict SPIR-V frontend rejects `StorageInputOutput16` before
  `validate_module`, so the test asserts a plain module round-trips both gate
  states; the f16 rejection is covered transitively by the WGSL gate tests.
- **#1 DEFERRED** (rationale) — the five new `VulkanDeviceInner` f16/16-bit
  flags are write-only (populated + `Debug`-printed, not yet read). Kept for
  parity with `robust_buffer_access2` and as a record of what was enabled,
  for a future HAL consumer; harmless.
- **#3 DEFERRED** — `supports_shader_float16` (adapter advertise) and
  `create_device` each query the physical-device feature once; distinct call
  sites, negligible cost.
- **#5** — no action; all new `pub*` items carry `///` docs (private helpers
  exempt).

Post-fix gates re-run by Claude: workspace test 65/65 bins EXIT=0; clippy
`-D warnings` default + `--features tiled` EXIT=0; HAL `--features
metal,vulkan --lib` 126 pass; Metal e2e 3/3; Vulkan/MoltenVK e2e 3/3.

## Out of scope

- D3D backends (permanent). GLES f16 (Tier 2 reject is the contract).
- f16 as a vertex-buffer *format* — already supported
  (`metal/format.rs` `Float16`→`MTLVertexFormat::Half`); unrelated to the
  WGSL `shader-f16` feature.
