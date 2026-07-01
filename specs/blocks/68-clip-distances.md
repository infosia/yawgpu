# Block 68 — `clip-distances` optional feature

Status: **Slices 1–2 DONE, real-GPU + CTS verified** on Metal (M2 e2e + CTS
clip_distances 240/0 + 4/0; MoltenVK advertises but can't compile clip_distances —
its limitation, Vulkan execution deferred to native HW). Slice 3 (docs + Phase
Review) in progress. Owner: Dawn-parity feature backfill.

The WebGPU `clip-distances` optional feature
(`WGPUFeatureName_ClipDistances = 0x10`) lets a vertex shader emit
`@builtin(clip_distances) clip_distances: array<f32, N>` (N ≤ 8) via
`enable clip_distances;`, adding user-defined clip planes (a fragment is culled
where any clip distance is negative). Tint (`Extension::kClipDistances`) lowers
it to Metal `[[clip_distance]]` / SPIR-V `ClipDistance` (needs `shaderClipDistance`
on Vulkan) — **no explicit HAL clip code**, only the device-feature enable.

## Public API surface

No new C entry points; flows through the standard feature surface
(`wgpuAdapterGetFeatures` / `HasFeature` / device equivalents report
`WGPUFeatureName_ClipDistances`). Add a `ClipDistances` `Feature` variant
(not `cfg`-gated) + `conv/feature.rs` mapping + `add_clip_distances_feature`.

## Behaviour contract

### Advertisement (HAL capability query) — Dawn parity
`HalAdapter::supports_clip_distances() -> bool`:
- **Metal** — always `true` (Dawn unconditional, `PhysicalDeviceMTL.mm:719`).
- **Vulkan** — `VkPhysicalDeviceFeatures::shaderClipDistance == VK_TRUE`
  (Dawn parity, `PhysicalDeviceVk.cpp:339`).
- **Noop** — `true`. **GLES** — `false` (Tier 2).

### Vulkan device-feature enable
Enable `enabled_features.shader_clip_distance = vk::TRUE` whenever
`supported_features.shader_clip_distance == vk::TRUE` (mirror `depth_clamp` /
`dual_src_blend`). Required for the SPIR-V `ClipDistance` builtin to be valid.

### Tint-extension gate (shader compilation)
Thread a `clip_distances: bool` from the device's enabled features into the
shader-compile path exactly like `subgroups` / `dual_source_blending`
(create_shader_module → from_wgsl → parse_and_validate_wgsl_gated →
Program::parse → shim), where the shim inserts
`tint::wgsl::Extension::kClipDistances` when set. Without the feature,
`enable clip_distances;` (and the `clip_distances` builtin) is rejected at
`createShaderModule`, identically on every backend.

### Inter-stage limit (Tier-independent core rule)
Clip distances consume `maxInterStageShaderVariables` slots:
`clipDistanceSlots = ceil(clip_distances_size / 4)` (`alignTo(N,4)/4`). Reflect
the vertex entry's `clip_distances_size` (Tint
`inspector::EntryPoint::clip_distances_size`, `std::optional<uint32_t>`) through
the shim → Rust → core, and in `validate_inter_stage_interface` count those slots
against the **vertex output** budget alongside the existing point-list reservation:
a valid pipeline needs `vertex_outputs + clipDistanceSlots + (pointList ? 1 : 0)
≤ maxInterStageShaderVariables` (CTS
`capability_checks,features,clip_distances:createRenderPipeline,at_over`). Also
reject `clip_distances_size > 8` (spec cap).

No other HAL / codegen change — Tint emits the clip-distance output.

## Slices

1. **Feature plumbing + Tint gate + inter-stage limit (Noop + HAL cap).**
   `Feature::ClipDistances`, `add_clip_distances_feature`,
   `HalAdapter::supports_clip_distances` (4 backends), Vulkan
   `shaderClipDistance` enable, `conv/feature.rs` + `conv/mod.rs` + `ffi/mod.rs`
   count 19→20, the `clip_distances` shader-compile gate → shim
   (`kClipDistances`), `clip_distances_size` reflection, and the inter-stage
   slot counting (+ `>8` reject). Inline unit tests. **Acceptance:**
   `cargo test --workspace` green on Noop; `enable clip_distances;` rejected
   without the feature, accepted with it; a vertex output count that overflows
   once clip-distance slots are added is rejected.

2. **Real-GPU + CTS verification.** An `e2e_{metal,vulkan}_clip_distances.rs`
   drawing a full-screen triangle with one clip distance set negative over half
   the viewport (culled → clear) vs positive (kept), readback-verified; plus the
   CTS `shader,validation,extension,clip_distances` +
   `capability_checks,features,clip_distances` trees green on real Metal.
   **Acceptance:** e2e passes on M2 (Metal) + MoltenVK (Vulkan if it advertises
   `shaderClipDistance`, else self-skip); CTS 0 fail.

3. **Docs + Phase Review.** README note; Block 68 finalization; the mandatory
   no-context Phase Review.

Tracking: `specs/tracking/clip-distances.md`.
