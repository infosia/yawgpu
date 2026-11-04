# Block 141 — Metal subgroups gate; `texture-compression-unaligned` and `atomic-vec2u-min-max`

Status: **IN PROGRESS**.
Source: backlog A26 and B8, started by user request on 2026-11-05.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`metal/PhysicalDeviceMTL.mm` (subgroup sizes at lines 322-330, feature list at lines
750-813), `vulkan/PhysicalDeviceVk.cpp` (lines 289 and 533-542), `vulkan/DeviceVk.cpp`
(lines 760-765), `Texture.cpp` (lines 430-439), `Device.cpp` (`SetWGSLExtensionAllowList`),
`Adapter.cpp` (`APIGetInfo`).

## Problem

### A26 — Metal subgroups gate and size range

- yawgpu advertises `subgroups` on Metal when the device supports `Apple6` or `Metal3`
  (`yawgpu-hal/src/metal/mod.rs`, `supports_subgroups`). Dawn's condition is `Apple6` or
  `Mac2`. Intel and AMD Macs that support `Mac2` but not `Metal3` get `subgroups` in Dawn
  and not in yawgpu.
- Dawn sets the Metal subgroup size range in the physical device for every device (32/32
  for Apple, 4/64 otherwise) and `wgpuAdapterGetInfo` reports it whether or not
  `subgroups` is supported. yawgpu reports 0/0 when `subgroups` is unsupported. On Apple
  Silicon nothing changes, because every Apple Silicon GPU supports `Apple6`.

### B8 — two standard features

A feature-list probe on the M2 (Block 140) found two standard features that Dawn
advertises and yawgpu does not.

- `texture-compression-unaligned` (`WGPUFeatureName_TextureCompressionUnaligned`,
  `0x18`). Dawn enables it unconditionally on Metal and Vulkan, not on OpenGL ES. With the
  feature enabled, `createTexture` no longer requires a compressed texture's width and
  height to be multiples of the block size (`Texture.cpp:430`). Everything that uses a
  mip level's physical size, that is, the size rounded up to whole blocks, is unchanged.
  yawgpu always rejects such a texture with "compressed texture size must be
  block-aligned" (`yawgpu-core/src/texture.rs`).
- `atomic-vec2u-min-max` (`WGPUFeatureName_AtomicVec2uMinMax`, `0x19`). With it enabled,
  WGSL `enable atomic_vec2u_min_max;` allows `atomicStoreMax` and `atomicStoreMin` on
  `atomic<vec2<u32>>`, a 64-bit atomic. Dawn advertises it on Metal for `Apple9`, or for
  `Mac2` together with `Apple8` (the M2). On Vulkan it advertises it when the device has
  `VK_KHR_shader_atomic_int64` (core in 1.2), `shaderBufferInt64Atomics`, and
  `shaderInt64`, and enables `shaderInt64` and `shaderBufferInt64Atomics` at device
  creation when the device has the feature. Dawn adds the WGSL extension to Tint's allow
  list only when the device has the feature.

## Rules

R1. **Metal `subgroups` gate.** Metal advertises `subgroups` when the device supports
`MTLGPUFamilyApple6` or `MTLGPUFamilyMac2`, as Dawn does. `Metal3` is no longer part of
the condition. `subgroup-size-control` still requires `subgroups` (Block 140 R4).

R2. **Metal subgroup size range is always reported.** `WGPUAdapterInfo.subgroupMinSize`
and `subgroupMaxSize` on Metal follow Block 140 R3 (32/32 for vendor `0x106b`, 4/64
otherwise) whether or not `subgroups` is supported. Feature advertisement and the
`subgroup-size-control` caps still require `subgroups`. Vulkan, GLES, and Noop reporting
does not change.

R3. **`texture-compression-unaligned`.**
- Metal, Vulkan, and Noop advertise it. GLES does not.
- With the feature enabled on the device, texture creation does not reject a compressed
  format whose width or height is not a multiple of the block size. Without it, the
  rejection and its message stay as they are.
- Copies, `writeTexture`, lazy zero-initialization, views, and the HAL resource creation
  work on such a texture. Each uses the physical size of the mip level where Dawn does.
  Any HAL path that cannot take an unaligned level-0 size is a defect of this block.

R4. **`atomic-vec2u-min-max`.**
- Metal advertises it when the device supports `MTLGPUFamilyApple9`, or both
  `MTLGPUFamilyMac2` and `MTLGPUFamilyApple8`.
- Vulkan advertises it when the device exposes `VK_KHR_shader_atomic_int64` (or is
  Vulkan 1.2 or newer), and reports `shaderBufferInt64Atomics` and `shaderInt64`. Device
  creation enables `shaderInt64` and `shaderBufferInt64Atomics` (and the extension when
  the device is older than 1.2) only when the feature is requested. Query the feature
  bits once, as Block 140 did for float16.
- Noop advertises it. GLES does not.
- The WGSL extension `atomic_vec2u_min_max` is allowed in shader modules only on a
  device with the feature enabled. Without it, the shader module fails with Tint's
  extension error, as for `shader-f16`.

R5. **C API.** `WGPUFeatureName_TextureCompressionUnaligned` and
`WGPUFeatureName_AtomicVec2uMinMax` map to new `Feature` variants both ways:
`wgpuAdapterGetFeatures`, `wgpuAdapterHasFeature`, `wgpuDeviceGetFeatures`,
`wgpuDeviceHasFeature`, and `requiredFeatures` in `wgpuAdapterRequestDevice`.

R6. **Unchanged.** Every other feature and limit; GLES; the Vulkan subgroup range.

## Tests

- Unit (HAL, Metal, pure helpers where the family check needs a device): the
  `subgroups` decision for `Apple6`, `Mac2` without `Metal3`, and neither; the range is
  reported without `subgroups`; the `atomic-vec2u-min-max` decision for `Apple9`,
  `Mac2 && Apple8`, `Mac2` alone, and `Apple8` alone.
- Unit (HAL, Vulkan, pure): the `atomic-vec2u-min-max` decision from the extension,
  version, and feature bits; the device-creation chain enables the two bits only when
  requested.
- Unit (core, Noop): an unaligned BC texture is rejected without the feature and created
  with it; the physical size of level 0 rounds up; a `writeTexture` and a
  texture-to-buffer copy of the whole level validate. A shader with
  `enable atomic_vec2u_min_max;` fails without the feature and compiles with it.
- Unit (FFI, Noop): the two enum values round-trip, and both features are advertised
  on Noop and accepted as `requiredFeatures`.
- Metal and Vulkan `#[ignore]` (Claude): the adapter advertises both features on the M2
  (Vulkan: as far as MoltenVK reports the atomic bits).
- e2e (Claude, shared scenario, Metal and Vulkan): `atomicStoreMax` and
  `atomicStoreMin` on `atomic<vec2u>` from many invocations, read back as the 64-bit
  max and min; an unaligned BC texture (for example 5×7 `bc1-rgba-unorm`) is written
  with `writeTexture`, copied to a buffer, and read back.
- Claude: the feature-list probe against Dawn lists the same standard features on the
  M2. CTS on Metal:
  `webgpu:api,validation,capability_checks,features,texture_compression_unaligned:*`,
  `webgpu:api,validation,createTexture:*`,
  `webgpu:api,operation,command_buffer,image_copy:*`,
  `webgpu:api,operation,command_buffer,copyTextureToTexture:*`,
  `webgpu:shader,execution,expression,call,builtin,atomics,atomicStoreMax:*`,
  `webgpu:shader,execution,expression,call,builtin,atomics,atomicStoreMin:*`,
  `webgpu:api,operation,adapter,requestDevice:*`, the subgroup queries; then
  `webgpu:api,*` and `webgpu:shader,execution,*` must not regress.

## Out of scope

- The Vulkan subgroup range when the device has no `VK_EXT_subgroup_size_control`
  properties.
- Intel Gen9 subgroup gating (A21).
- `atomic-vec2u-min-max` on GLES.

## Slices

- **S1 (coding agent):** R1-R6 with the unit tests.
- **S2 (Claude):** ignored tests, e2e, probe, CTS, Phase Review.
