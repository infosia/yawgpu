# Block 72 — `texture-formats-tier2` advertisement is a device query

Status: **Slice 1 (Metal) DONE — real-GPU verified (M2), CTS byte-identical**. **Slice 2 (Vulkan) DONE — MoltenVK-verified, CTS byte-identical** (see "Slice 2" below). Owner: external-report follow-up
(`specs/tracking/external-reports.md`, 2026-08-24 entry).

`texture-formats-tier2` (`WGPUFeatureName_TextureFormatsTier2 = 0x14`) grants
`read-write` storage access to a fixed set of formats
(`yawgpu-core/src/format.rs` → `caps()`, the `TextureFormatsTier2` arm:
`r8unorm/uint/sint`, `rgba8unorm/uint/sint`, `r16uint/sint/float`,
`rgba16uint/sint/float`, `rgba32uint/sint/float`). Until this block the Metal
HAL advertised the feature unconditionally. Dawn gates it on a device query
(`PhysicalDeviceMTL.mm`: `EnableFeature(TextureFormatsTier2)` only when
`[device readWriteTextureSupport] == MTLReadWriteTextureTier2`). On a Metal
device below read-write texture tier 2 the unconditional advertisement let a
`read-write` bind group layout on e.g. `rgba8unorm` validate, then handed Tint
MSL to a compiler whose device cannot execute it — the failure surfaced late
(Metal compiler / wrong results) instead of at the feature query.

## Behaviour contract

### Advertisement (Metal HAL)

- `MetalAdapter::new` queries `MTLDevice.readWriteTextureSupport` **once** and
  caches the `MTLReadWriteTextureTier` on the adapter.
- `MetalAdapter::supports_texture_formats_tier2()` returns
  `tier == MTLReadWriteTextureTier::Tier2`. Nothing else changes: tier 1
  advertisement, `HalAdapter` dispatch, and `Adapter::features()` are untouched
  (`TextureFormatsTier2 ⇒ TextureFormatsTier1` implication stays in core).
- Noop and GLES advertisement is unchanged. Vulkan is Slice 2 (below).

### Rejection below the tier (core, backend-independent)

Core validation is identical on every backend. A device that does not hold
`TextureFormatsTier2` (never advertised, or advertised but not requested)
rejects a bind group layout storage-texture entry with `read-write` access on
a format that needs the feature. The rejection is the existing
`read_write_storage_capable` check in `yawgpu-core/src/bind_group_layout.rs`;
this block changes only the **message**:

- When the format *would* be read-write capable with `TextureFormatsTier2`
  added to the device's feature set (compute by evaluating `caps()` against
  `features ∪ {TextureFormatsTier2}` — no second format table), the message
  names the format and the tier:
  `storage texture binding format <name> supports read-write storage access only with the texture-formats-tier2 feature, which this device does not have`
  where `<name>` is the WebGPU name (e.g. `rgba8unorm`).
- Otherwise (the format never supports read-write storage, e.g. `rg32float`)
  the existing message
  `storage texture binding format must support read-write storage access`
  is kept verbatim.

### `TextureFormat::name()`

`yawgpu-core::TextureFormat` gains `pub fn name(self) -> &'static str` returning
the WebGPU IDL / `webgpu.h` name (`"rgba8unorm"`, `"depth24plus-stencil8"`,
`"astc-4x4-unorm-srgb"`, …; `"undefined"` for `UNDEFINED`, `"unknown"` for
any raw value outside the enum). It exists so validation messages can name
formats; it must cover every `WGPUTextureFormat_*` constant of the pinned
header. The `Debug` derive stays as is.

## Tests

- **HAL unit (Metal, real GPU, `#[ignore]`)**: `supports_texture_formats_tier2()`
  equals `device.readWriteTextureSupport() == Tier2` on the first adapter
  (true on the M2 that runs the suite).
- **Core unit**: `TextureFormat::name()` round-trips a sample across the enum
  (incl. `undefined`, compressed, depth/stencil, and an out-of-range raw); the
  BGL message names format + tier for `rgba8unorm` without the feature, keeps
  the generic message for `rg32float` with the feature, and is absent for
  `rgba8unorm` with the feature.
- **Noop integration** (`yawgpu/tests/bind_group_layout_validation.rs` or the
  existing storage-texture BGL test file): device created *without*
  `TextureFormatsTier2` → `read-write` + `rgba8unorm` entry errors with a
  message containing `rgba8unorm` and `texture-formats-tier2`; device created
  *with* it → no error.
- **Real-Metal e2e** (Claude-authored, `yawgpu/tests/e2e_metal_texture_formats_tier2.rs`):
  adapter advertises the feature iff the HAL query says tier 2; a device
  without the feature rejects the `rgba8unorm` `read-write` layout with the
  tier-naming message; a device with the feature runs a compute pass that
  reads and writes a `texture_storage_2d<rgba8unorm, read_write>` and the
  readback matches — the proof that an advertised tier 2 actually executes.
- **CTS**: `capability_checks,features` + storage-texture validation trees
  must be byte-identical before/after on the M2 (which reports tier 2).

## Slice 2 — Vulkan advertisement + device feature enablement

`VulkanAdapter::supports_texture_formats_tier1()` / `_tier2()` were both an
unconditional `true`, and `VulkanAdapter::create_device` never enabled
`shaderStorageImageExtendedFormats` — yet the formats tier 1/2 add to the
storage set (`r8unorm/uint/sint`, `r16uint/sint/float`, `rg8*`, `rg16*`,
`rgb10a2*`, `rg11b10ufloat`, …) are Vulkan *extended* storage-image formats,
which are only legal on a logical device with that feature enabled.

### Advertisement (Dawn `PhysicalDeviceVk.cpp`, verbatim)

Tier 1 and tier 2 are advertised **together** (both true or both false):

- `VkPhysicalDeviceFeatures::shaderStorageImageExtendedFormats == VK_TRUE`, and
- every format in `{R16_UNORM, R16_SNORM, R16G16_UNORM, R16G16_SNORM,
  R16G16B16A16_UNORM, R16G16B16A16_SNORM, R8_SNORM, R8G8_SNORM,
  R8G8B8A8_SNORM, B10G11R11_UFLOAT_PACK32}` has
  `COLOR_ATTACHMENT | COLOR_ATTACHMENT_BLEND` in
  `VkFormatProperties::optimalTilingFeatures`.

Implemented as one private `supports_texture_formats_tiers()` query (same
`get_physical_device_format_properties` pattern as
`supports_float32_blendable`) that both public fns return. `HalAdapter`
dispatch and core are untouched (core already applies `Tier2 ⇒ Tier1`).

### Device creation

`VulkanAdapter::create_device` enables
`enabled_features.shader_storage_image_extended_formats` whenever the
physical device supports it — yawgpu's HAL device creation does not receive
the requested feature set, so the existing enable-if-supported pattern
(`dual_src_blend`, `shader_clip_distance`, …) applies; Dawn enables it when
`TextureFormatsTier1` is requested, and enabling it unconditionally-when-
supported is a harmless superset.

### Tests

- **HAL unit (Vulkan, real GPU, `#[ignore]`)**: recompute Dawn's rule in the
  test from raw `ash` queries on the first adapter and assert
  `supports_texture_formats_tier1() == supports_texture_formats_tier2() == rule`.
- **Real-Vulkan e2e** (Claude-authored,
  `yawgpu/tests/e2e_vulkan_texture_formats_tier2.rs`, MoltenVK on the M2):
  device without the feature rejects the `rgba8unorm` `read-write` layout
  with the tier-naming message; device with the feature executes
  `texture_storage_2d<rgba8unorm, read_write>` and an extended-format
  `texture_storage_2d<r8unorm, read_write>` compute pass with matching
  readback (the second proves the device-level feature enablement).
- **CTS (Vulkan/MoltenVK)**: the same tree set as Slice 1 before/after.
