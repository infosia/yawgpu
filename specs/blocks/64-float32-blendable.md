# Block 64 — `float32-blendable` optional feature

Status: **COMPLETE** — all slices done, real-GPU verified (Metal M2 + MoltenVK),
Phase Review clean (no CRITICAL/MAJOR). Owner: Dawn-parity feature backfill.

The WebGPU `float32-blendable` optional feature
(`WGPUFeatureName_Float32Blendable = 0x0F`) allows a render pipeline to attach a
**blend state to 32-bit-float color targets** (`r32float`, `rg32float`,
`rgba32float`). Without it, blending on those formats is a validation error
(they are renderable but not blendable by default).

This is a pure **capability + validation** feature — no shader changes, no HAL
execution changes (the HAL already programs whatever blend state the pipeline
carries). It mirrors `depth-clip-control` / `subgroups` in shape.

## Public API surface

No new C entry points:
- **`wgpuAdapterGetFeatures` / `wgpuAdapterHasFeature`** report
  `WGPUFeatureName_Float32Blendable` when the backend supports it.
- **`wgpuDeviceGetFeatures` / `wgpuDeviceHasFeature`** report it when requested.
- A render pipeline whose color target is a float32 format may carry a
  `blend` state iff the device enabled the feature.

### `yawgpu-core::Feature`

Add a `Float32Blendable` variant (not `cfg`-gated); map it C↔Rust in
`yawgpu/src/conv/feature.rs`; advertise from `Adapter::features()` via
`add_float32_blendable_feature` consulting `HalAdapter::supports_float32_blendable()`.

## Behaviour contract

### Advertisement (HAL capability query) — Dawn parity

`HalAdapter::supports_float32_blendable() -> bool`, static-enum-dispatched:

- **Metal** — always `true`. Dawn enables it unconditionally on Metal
  (`PhysicalDeviceMTL.mm:720`).
- **Vulkan** — `true` iff `VK_FORMAT_R32_SFLOAT`, `VK_FORMAT_R32G32_SFLOAT`,
  and `VK_FORMAT_R32G32B32A32_SFLOAT` all report
  `VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT` in `optimalTilingFeatures`
  (`get_physical_device_format_properties`). Exact match to Dawn
  `PhysicalDeviceVk.cpp:444-447`.
- **Noop** — `true` (keeps the accept path unit-testable; Noop does not blend).
- **GLES** — `false` (Tier 2).

### Validation gate (Tier-independent core rule)

`TextureFormat::caps(features)` (`yawgpu-core/src/format.rs:272`) already takes
the device's enabled `FeatureSet`. When `features` contains
`Feature::Float32Blendable`, add `.blendable()` to the caps of `R32_FLOAT`,
`RG32_FLOAT`, `RGBA32_FLOAT` (mirrors Dawn `Format.cpp:336` adding `Cap::Blendable`).

The existing render-pipeline check at `render_pipeline.rs:2690`
(`if target.blend.is_some() && !caps.is_blendable → Err`) then automatically:
- **rejects** a blend state on a float32 target **without** the feature
  (float32 stays non-blendable), identically on every backend; and
- **accepts** it **with** the feature.

No change to the blend-validation code itself is needed — only the caps table.

## Slices

1. **Feature plumbing + caps gate (Noop + HAL cap).** `Feature::Float32Blendable`,
   `add_float32_blendable_feature`, `HalAdapter::supports_float32_blendable` on
   all four backends, `conv/feature.rs`, and the `caps()` blendable-when-enabled
   block. Inline unit tests: Noop advertises it; `caps` marks float32 blendable
   iff the feature is present; a render pipeline with a blend on `rgba32float` is
   rejected without the feature and accepted with it; C↔Rust round-trip; FFI
   feature count. **Acceptance:** `cargo test --workspace` green on Noop.

2. **Real-GPU execution e2e (Metal + Vulkan).**
   `e2e_{metal,vulkan}_float32_blendable.rs`: render into an `rgba32float` target
   with a blend state (e.g. additive `src=One, dst=One`), draw two overlapping
   full-screen quads of known float colors, read back, and assert the sum — a
   result only produced if blending actually ran on the float32 target.
   **Acceptance:** both e2e suites pass on the M2 (Metal) and MoltenVK (Vulkan).

3. **Docs + Phase Review.** README capability note; Block 64 finalization; the
   mandatory no-context Phase Review.

Tracking: `specs/tracking/float32-blendable.md`.
