# Tracking — real hardware-queried adapter limits

Spec: [Block 92](../blocks/92-adapter-limits.md). Goal: `Adapter::limits()` returns
the true Metal/Vulkan hardware limits (Dawn parity) instead of `Limits::DEFAULT`,
closing the ~2679-case `capability_checks,limits` + ~170-case `,features` at-limit
CTS gap vs the Dawn oracle on this M2.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | HAL seam (`HalLimits` + `HalAdapter::limits()`) + Noop/GLES DEFAULT + full Metal query + core map/clamp + unit tests | **DONE** (2026-07-01) — Metal CTS-verified |
| 2 | Vulkan query | **DONE** (2026-07-01) — MoltenVK CTS-verified; **native Vulkan CTS-verified 2026-07-02** (RTX 5060 Ti, whole tree 9149/0) |
| 3 | Phase Review + re-diff vs Dawn | **DONE** (2026-07-02) — review clean, capability_checks at Dawn parity |

## Slice 1 result (real Metal, M2, webgpu-native-cts)

`api,validation,capability_checks,limits`: **pass 6601→9290, skip 4484→1795,
fail 0** (Dawn oracle 9280/1805 — yawgpu now matches/slightly exceeds).
`capability_checks,features`: pass 2254→2382, fail 0 (limit un-skips leaked in).
Regression: `api,validation,render_pipeline` 60445 pass / 0 fail (limit-sensitive:
interStage 31, colorAttachments 8/64). Metal HAL real-device limits unit test green.

Three CTS-driven fixes on top of the initial Metal query (all root-caused vs the
direct-MSL-binding model, NOT argument buffers):
1. Texture counts: report yawgpu's real budget — sampled 31 (MAX_TEXTURE_SLOT+1),
   storage 8 (Metal read_write cap) — not Dawn's argument-buffer 70/58.
2. Alignment floor: `from_hal` clamps min offset alignment to `[32,256]` (Apple
   raw 4 → 32; WebGPU floor is 32, `greaterThanOrEqualTo32`). Core-wide fix.
3. Storage-limit spec invariant (Dawn `EnforceLimitSpecInvariants`, core level) in
   `validate_required_limits`: auto-upgrade per-shader-stage from the requested
   in-vertex/in-fragment storage limits, then pin in-stage == per-stage. Fixed the
   `too many storage {textures,buffers}` false-rejects + `auto_upgrade*` cases.

## Slice 2 result (MoltenVK, per-limit)

`VulkanAdapter::limits()` maps `VkPhysicalDeviceLimits` (Dawn `PhysicalDeviceVk.cpp`
mapping) with `max_buffer_size` from Maintenance3; NVIDIA storage cap +
`maxFragmentCombinedOutputResources` redistribution deferred (MoltenVK N/A).
Every `capability_checks,limits` limit run **individually** on MoltenVK passes
0-fail (e.g. maxTextureDimension2D 10/0, minUniformBufferOffsetAlignment 22/0 —
the [32,256] alignment fix un-skips + passes on Vulkan too, maxSampledTextures
1020/0, maxStorageTextures 1860/0, maxInterStageShaderVariables 1280/0). One
CTS-found fix: dynamic-buffer ceilings were 16; MoltenVK's large
`maxDescriptorSet*Dynamic` made us advertise 16 > maxUniformBuffersPerShaderStage
(12) → 7 `maxDynamicUniformBuffersPerPipelineLayout` fails; lowered to Dawn's
advertised tier maxima (uniform 10, storage 8, `Limits.cpp:74-75`) → 80/0.
**Caveat (MoltenVK only):** running the WHOLE limits tree in one MoltenVK process
aborts (resource exhaustion, no output) — a MoltenVK harness limit, not a yawgpu
defect; each limit passes when run separately. Confirmed a harness artifact by the
native-Vulkan run below, which completes all 9149 cases in a single process.

## Slice 2 result (native Vulkan, RTX 5060 Ti, webgpu-native-cts, 2026-07-02)

The deferred native-Vulkan verification, run from HEAD `3a30443` (Block 92 COMPLETE)
with fresh `yawgpu.dll`/`tint_shim.dll`:

| tree | native Vulkan | ref: Metal M2 |
|---|---|---|
| `capability_checks,limits`   | pass 9149 / skip 1936 / fail 0 | 9290 / 1795 / 0 |
| `capability_checks,features` | pass 2240 / skip  272 / fail 0 | 2424 /   88 / 0 |

**fail=0 / crash=0 / xpass=0** on both trees — the `VkPhysicalDeviceLimits` mapping
+ the Phase-Review in-stage-storage pin (`fddb0da`) are clean on real Vulkan. The
whole `limits` tree completes in one process (no MoltenVK abort). The higher skip
count vs Metal is hardware (limits equal to DEFAULT on this GPU, and Vulkan-unsupported
features, land in CTS `==default` / unsupported skips) — no yawgpu-skips-but-Dawn-runs
gap (zero fail/xpass).

## Phase Review (2026-07-02) — clean

No-context review of the cumulative diff found 1 MAJOR + 3 MINOR. MAJOR (Vulkan
in-stage storage left at DEFAULT → native-Vulkan device>adapter) fixed by pinning
`in_stage == per_stage` in `from_hal` (commit fddb0da). MINOR: Vulkan inter-stage
`saturating_sub(2)`; removed redundant `clamp_metal_maximums`. MINOR (Vulkan
`max_color_attachment_bytes_per_sample` not queried) kept as the documented
in-code deferral. Field mapping / Metal family table / invariant verified clean.

## Final CTS parity (real Metal M2, vs Dawn oracle, 2026-07-02)

| tree | yawgpu | Dawn | verdict |
|---|---|---|---|
| `capability_checks,limits`   | pass 9290 / skip 1795 / fail 0 | 9280 / 1805 | yawgpu ≥ Dawn |
| `capability_checks,features` | pass 2424 / skip 88 / fail 0   | 2424 / 88   | **byte-identical** |

Zero "yawgpu-skips-but-Dawn-runs" cases remain in either tree. **Block 92 COMPLETE.**

## Follow-on that landed with this work (features parity, commit d2803dc)

The features re-diff surfaced a separate gap (not limits): Metal did not advertise
`texture-compression-astc-sliced-3d` / `-bc-sliced-3d` (hardcoded false), and
`apply_feature_implications` wrongly auto-added the base BC/ASTC feature from the
sliced-3d feature. Advertised both sliced-3d features (Dawn PhysicalDeviceMTL.mm
613-637: BC/BCSliced3D on `supportsBCTextureCompression`, ASTC/ASTCSliced3D on
Apple3; plain ASTC moved Apple2→Apple3) and dropped the sliced-3d→base
implications → features tree reached exact Dawn parity (astc/bc_sliced_3d
byte-identical: 56/56, 28/28).

## Baseline (webgpu-native-cts, same M2, 2026-07-01)

| tree | Dawn pass/skip | yawgpu pass/skip |
|---|---|---|
| `capability_checks,features` | 2424 / 88 | 2254 / 258 |
| `capability_checks,limits`   | 9280 / 1805 | 6601 / 4484 |

yawgpu skips ~3040 at-limit/betterValue cases Dawn runs, because
`Adapter::limits() == Limits::DEFAULT` → CTS skips "limit==default".

## Key facts

- Core seam is minimal: `validate_required_limits` already uses `self.limits()`
  as the request ceiling (`limits.rs:131` maximum!, `:157` alignment!), so only
  `Adapter::limits()` (`adapter.rs:61`) must change to return real HAL limits.
- HAL cannot return `core::Limits` (no core dep) → new `HalLimits` mirror in
  `yawgpu-hal/src/lib.rs` + `HalAdapter::limits()`.
- Raising a limit commits the HAL to honour it: CTS at-limit *operation* cases
  create resources at the reported value. Triage each on real Metal.
- Dawn Metal source: `third_party/dawn/src/dawn/native/metal/PhysicalDeviceMTL.mm`
  `InitializeSupportedLimitsImpl` (~:815-973) + `kMTLLimits` table + `Limits.cpp`.
  Apple8/M2 targets: tex 16384/16384/2048, arrayLayers 2048, colorAttachments 8,
  bytesPerSample 64, threadgroupMem 32768, threads/workgroupSize 1024,
  interStage 31, bufferOffsetAlign 4, sampled/storageTex 70/58, storageBuf 10,
  dynamic uniform/storage buffers 11, vertexAttributes 248, maxBufferSize =
  [device maxBufferLength].
- Dawn Vulkan source: `.../vulkan/PhysicalDeviceVk.cpp` (~:744-918).

## Log

- 2026-07-01: measured the Dawn-vs-yawgpu skip gap with `build-dawn/cts` (Dawn
  oracle, prebuilt `libwebgpu_dawn.dylib` at
  `/Users/kiguchi/Documents/workspace/C/dawn/out/Release`); root-caused to
  default-limit reporting. Wrote Block 92. Slice 1 handoff dispatched.
- 2026-07-02: ran the deferred native-Vulkan CTS verification (RTX 5060 Ti, fresh
  DLL from HEAD `3a30443`): `capability_checks,limits` 9149/1936/0 (whole tree, one
  process) and `capability_checks,features` 2240/272/0 — both fail=0. See "Slice 2
  result (native Vulkan …)".
