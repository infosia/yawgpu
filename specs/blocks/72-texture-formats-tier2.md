# Block 72 — `texture-formats-tier2` advertisement is a device query

Status: **Slice 1 (Metal) DONE — real-GPU verified (M2), CTS byte-identical**. Follow-up slice: Vulkan advertisement (see below). Owner: external-report follow-up
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
- Noop, Vulkan and GLES advertisement is unchanged by this slice. The Vulkan
  analogue (Dawn gates tier 1+2 on `shaderStorageImageExtendedFormats` plus
  per-format colour-attachment properties) is a known follow-up, not part of
  this block's Slice 1.

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
