# Tracking — `texture-component-swizzle` optional feature

Spec: [Block 71](../blocks/71-texture-component-swizzle.md). Goal: Dawn parity for
`WGPUFeatureName_TextureComponentSwizzle = 0x16` on Tier-1 Metal + Vulkan. **Last
standard optional-feature backfill; the only texture-view (real HAL) one.**

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Advertise + validate + HAL threading & application | **DONE** (2026-07-01) |
| 2 | CTS verification (Metal) | **DONE** (2026-07-01) — validation 8/0 + operation 529/0 after 2 CTS-found fixes (depth R001 + use-site gate) |
| 3 | Docs + Phase Review | **DONE** (2026-07-01) — Phase Review clean (no CRITICAL/MAJOR); Vulkan depth-swizzle verification deferred to native HW |

## Key facts (verified 2026-07-01, HAL map via Explore)

- Already wired: FFI parses the swizzle chain (`conv/descriptors.rs:381`), core
  `ComponentSwizzle`/`TextureComponentSwizzle` (default identity) +
  `is_identity()` + `TextureView::swizzle()` (`texture_view.rs`), feature C↔Rust
  map (`conv/feature.rs`). Missing: advertise + validate + **HAL apply** (swizzle
  is stored but dropped before the HAL → silent no-op today).
- Dawn parity: Metal `supportsFamily(Mac2) || supportsFamily(Apple2)`
  (`UtilsMetal.mm:1011`); Vulkan unconditional (`PhysicalDeviceVk.cpp:638`).
- HAL model (no view object; views built lazily at bind time):
  - `HalBoundTexture` per-view descriptor: `yawgpu-hal/src/command.rs:311-346`
    (add a `swizzle` field).
  - Core builds `HalBoundTexture` in `queue.rs:1572-1609` (sampled + storage
    arms) — extract `texture_view.swizzle()` there.
  - Vulkan bind-time views: `vulkan/pipeline.rs:1558` (sampled) + `:1584`
    (storage) — add `.components(VkComponentMapping)`.
  - Metal bind-time view: `metal/encode.rs:1340` `metal_texture_view` uses the
    5-param `newTextureViewWithPixelFormat_textureType_levels_slices`; swap to the
    6-param `..._swizzle` overload with `MTLTextureSwizzleChannels`.
- Validation: `validate_texture_view_descriptor` (from `Texture::create_view`,
  `texture.rs:341`) has the texture's device `features` — reject non-identity
  swizzle without `Feature::TextureComponentSwizzle`.
- CTS: `api,operation,texture_view,texture_component_swizzle` (**529**, reads
  swizzled channels — needs HAL apply) + `capability_checks,features,
  texture_component_swizzle` (9, feature gate). (The `shader,{execution,validation},
  statement,swizzle` cases are WGSL vector `.xyz` swizzles — unrelated, already
  green.)
- Different shape from Blocks 62–69 (shader/pipeline features): real HAL view
  work on both backends, not Tint codegen.
