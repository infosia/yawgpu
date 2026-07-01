# Block 71 — `texture-component-swizzle` optional feature

Status: **Slices 1–2 DONE, CTS-verified** on Metal (validation 8/0 + operation
529/0, after 2 CTS-found fixes: depth R001 base + use-site gate). Slice 3 (docs +
Phase Review) in progress. Owner: Dawn-parity feature backfill.
**Last of the WebGPU standard optional-feature backfill; the only one that is a
texture-view feature (real HAL view work), not a shader/pipeline feature.**

The WebGPU `texture-component-swizzle` optional feature
(`WGPUFeatureName_TextureComponentSwizzle = 0x16`) lets a texture *view* remap
its `r/g/b/a` components (each → one of `r/g/b/a/0/1`) via a
`WGPUTextureComponentSwizzleDescriptor` chained on the view descriptor. Reads
through the view (`textureSample`/`textureLoad`/`textureGather`) see the swizzled
channels. Metal `MTLTextureSwizzleChannels`, Vulkan `VkComponentMapping`.

Much is **already wired**: the FFI parses the swizzle chain
(`yawgpu/src/conv/descriptors.rs:381` `texture_component_swizzle`), core has the
`ComponentSwizzle` enum + `TextureComponentSwizzle` struct (default = identity
`r,g,b,a`) with an `is_identity()` helper and a `TextureView::swizzle()` accessor
(`yawgpu-core/src/texture_view.rs`), and the feature enum maps C↔Rust
(`conv/feature.rs`). What is **missing**: advertisement, the validation gate, and
— the real work — **threading the swizzle to the HAL and applying it** on both
backends (today it is parsed + stored but dropped before the HAL, so it is a
silent no-op).

## Behaviour contract

### Advertisement (HAL capability query) — Dawn parity
`HalAdapter::supports_texture_component_swizzle() -> bool`:
- **Metal** — `device.supportsFamily(MTLGPUFamily::Mac2) ||
  device.supportsFamily(MTLGPUFamily::Apple2)` (Dawn `UtilsMetal.mm:1011`
  `SupportTextureComponentSwizzle`; essentially all modern Metal GPUs).
- **Vulkan** — `true` (Dawn enables it unconditionally, `PhysicalDeviceVk.cpp:638`;
  `VkComponentMapping` is core Vulkan 1.0).
- **Noop** — `true`. **GLES** — `false` (Tier 2).

### Validation gate (Tier-independent core rule)
In `validate_texture_view_descriptor` (reached from `Texture::create_view`,
which has the texture's device `features`): if the resolved swizzle is
**not identity** and the device did not enable
`Feature::TextureComponentSwizzle`, reject with a validation error. An identity
swizzle is always allowed. Identical on every backend.

### HAL application (the real work — both backends)
Thread the swizzle from the core `TextureView` into the HAL's per-binding view
creation (the HAL has no view object; it builds views lazily at bind time from
`HalBoundTexture`, `yawgpu-hal/src/command.rs:311`):
- **HAL descriptor** — add a `swizzle` field (an `HalComponentSwizzle` r/g/b/a,
  new enum in `yawgpu-hal/src/descriptors.rs`) to `HalBoundTexture`.
- **Core → HAL** — in `queue.rs:1572-1609` (both the sampled- and storage-texture
  match arms that build `HalBoundTexture`) set
  `swizzle: hal_texture_component_swizzle(texture_view.swizzle())` (a new
  conv fn mapping `core::ComponentSwizzle` → `HalComponentSwizzle`).
- **Vulkan** — set `.components(component_mapping(bound.swizzle))` on the
  bind-time `vk::ImageViewCreateInfo` chains at `vulkan/pipeline.rs:1558` (sampled)
  and `:1584` (storage), mapping each channel to `vk::ComponentSwizzle`
  (`IDENTITY`/`ZERO`/`ONE`/`R`/`G`/`B`/`A`).
- **Metal** — replace the 5-param
  `newTextureViewWithPixelFormat_textureType_levels_slices` at
  `metal/encode.rs:1340` with the 6-param
  `newTextureViewWithPixelFormat_textureType_levels_slices_swizzle`, passing an
  `MTLTextureSwizzleChannels` built from the swizzle (each channel →
  `MTLTextureSwizzle`). When the swizzle is identity, the plain 5-param overload
  may be kept (identity `MTLTextureSwizzleChannels` is also fine); pick one and be
  consistent.

`storage-texture` bindings: a non-identity swizzle on a storage binding is
disallowed by WebGPU — the gate + storage validation should keep storage views
identity (verify against CTS); wire the swizzle on the sampled path first.

## Slices

1. **Advertise + validate + HAL threading & application.**
   `Feature::TextureComponentSwizzle` advertisement + `add_*_feature` +
   `HalAdapter::supports_texture_component_swizzle` (4 backends), the
   `validate_texture_view_descriptor` gate, the `HalComponentSwizzle` HAL type +
   `HalBoundTexture.swizzle` + the `queue.rs` extraction, and the Vulkan
   `.components()` + Metal 6-param-overload application. Inline unit tests
   (advertise; non-identity-without-feature rejected, identity + with-feature
   accepted; conv round-trips). **Acceptance:** `cargo test --workspace` green on
   Noop.

2. **Real-GPU + CTS verification.** An `e2e_{metal,vulkan}_texture_component_swizzle.rs`
   creating a view with e.g. `r→g, g→r` (or `→0/→1`) over a known texture,
   sampling it in a shader, and reading back the remapped channels; plus the CTS
   `api,operation,texture_view,texture_component_swizzle` (529) +
   `api,validation,capability_checks,features,texture_component_swizzle` (9) trees
   on real Metal + MoltenVK. **Acceptance:** e2e passes on M2 + MoltenVK; CTS 0 fail.

3. **Docs + Phase Review.** README note; Block 71 finalization; the mandatory
   no-context Phase Review. **Completes the WebGPU standard optional-feature
   backfill (0x01–0x16).**

Tracking: `specs/tracking/texture-component-swizzle.md`.
