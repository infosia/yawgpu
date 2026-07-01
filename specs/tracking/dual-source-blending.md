# Tracking — `dual-source-blending` optional feature

Spec: [Block 65](../blocks/65-dual-source-blending.md). Goal: Dawn parity for
`WGPUFeatureName_DualSourceBlending = 0x11` on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + Tint gate + validation (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | **DONE** (2026-07-01) |
| 3 | Docs + Phase Review | **DONE** (2026-07-01) |

**Phase COMPLETE** — no-context Phase Review of `d89b0bf..HEAD` returned no
CRITICAL/MAJOR. The C-shim `dual_source_blending` param sits in the identical
5th position across tint_shim.h / .cpp / the Rust `extern "C"` decl (verified
char-by-char — no wrong-bool UB); all 40+ `Program::parse` call sites updated to
the new arity. Tint gate rejects/accepts `enable dual_source_blending;` on the
device feature; validation checks both color+alpha and src+dst factors
(tier-independent) + single-target rule; Vulkan `dualSrcBlend` enabled
when-supported; e2e C0·C1 distinguishes from Src-misread (C0·C0). Two MINORs, no
action (rustfmt diff noise; single-target keys off Src1 factors per stated scope).

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

## CTS finding (post-COMPLETE) — RESOLVED

External webgpu-native-cts (Dawn oracle, real Metal) found 8 fails in
`api,validation,render_pipeline,fragment_state:dual_source_blending,use_blend_src`
(`useBlendSrc1=false`): a pipeline using a `src1*` blend factor whose fragment
shader does NOT write `@location(0) @blend_src(1)` must be a pipeline-creation
ValidationError (CTS success = `!isDualSourceBlendFactor || useBlendSrc1`), but
yawgpu let it reach the Metal compiler → uncaptured error. Fixed: reflect
`StageVariable.attributes.blend_src` through the shim → Rust →
`ReflectedModule::fragment_writes_blend_src_1`, and reject in core when a
dual-source factor is used without a `@blend_src(1)` output. **CTS re-verified:
`fragment_state:*` pass=3729 fail=0 on real Metal** (was 3721/8). Our own e2e
missed it (it used a well-formed dual-source shader); the pipeline
shader-output/factor interface check is core's job, not Tint's. Shows the value
of running the external CTS against every new feature.
