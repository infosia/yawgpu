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
| 4 | Native-Vulkan depth R001 verification (Block 106) | **DONE** (2026-09-22) — `e2e_vulkan_texture_component_swizzle.rs` 4/4 (colour remap, depth swizzle over the `(d, 0, 0, 1)` base, identity depth, feature gate) on Windows / NVIDIA under the validation layer (0 lines); native Vulkan CTS (this host, yawgpu `4d5bc52` release `--features vulkan`, CTS `2f0fb9f`, raw, 2026-09-22): `shader,execution,shader_io,vertex_builtins:outputs,clip_distances` 8/0, `capability_checks,features,clip_distances` 368/0, `shader,validation,extension,clip_distances` 4/0, `api,operation,texture_view,texture_component_swizzle` 32,832 pass / 19,494 skip (all skips = compressed formats this GPU does not expose; every depth/stencil-format subcase passes: depth16unorm 1,197, depth24plus 1,197, depth32float 1,197, depth24plus-stencil8 1,539, depth32float-stencil8 1,539, stencil8 342) / 0 fail, `capability_checks,features,texture_component_swizzle` 855/0, `encoding,programmable,pipeline_immediate` 181/0, `encoding,cmds,setImmediates` 378/0; summary pass=34,626 skip=19,495 fail=0 crash=0 |

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
