# Block 73 — Vulkan compressed-texture completion

Status: **COMPLETE** — Phase Review clean (0 CRITICAL / 0 MAJOR; 6 of 8 MINOR
fixed, 2 deferred with rationale). Verified VUID-clean on a native Windows
Vulkan driver for BC (E1–E7, E10). Known deferrals: ETC2 / ASTC / ASTC
sliced-3d probes (E8, E9) self-skip on desktop GPUs and are unverified on real
hardware; external webgpu-native-cts re-confirmation of the compressed-format
trees on Vulkan is pending. Owner: Dawn-parity backfill.

Compressed textures (BC1–BC7, ETC2/EAC, ASTC LDR, + sRGB variants) are already
implemented end to end for 2D / 2D-array textures: the core `FormatCaps` table,
`HalTextureFormat` variants, the Vulkan + Metal native mappings, per-device
feature gating (`add_texture_compression_features`, `yawgpu-core/src/adapter.rs`),
and block-sized B2T/T2B copies. See `specs/tracking/format-completeness-audit.md`
(Audit C, finding #2).

This block closes the **remaining Vulkan gaps**:

1. `texture-compression-bc-sliced-3d` / `texture-compression-astc-sliced-3d` are
   hardcoded `false` on Vulkan, so 3D compressed textures cannot be created.
2. `vkCreateDevice` does not enable `textureCompressionBC` / `ETC2` / `ASTC_LDR`
   (works today because Vulkan format support is independent of the feature
   bit, but diverges from Dawn and from the "enable what you use" convention of
   the neighbouring features).
3. Real-GPU coverage is one 4x4 block create → writeTexture → T2B round-trip per
   family (+ one 2-layer BC1). No sampling, mip chain, T2T, 3D, or BC2–BC7.

No new public C API. No core validation change is expected: the sliced-3d core
gates already exist (`yawgpu-core/src/texture.rs` `validate_texture_descriptor`,
"3D BC compressed textures require texture-compression-bc-sliced-3d" etc.) and
are Tier-independent. **Do not relax any core rule.**

## Behaviour contract

### R1 — Vulkan sliced-3d advertisement (Dawn parity)

Dawn `PhysicalDeviceVk.cpp:270-303`:

- `VulkanAdapter::supports_texture_compression_bc_sliced_3d()` returns
  `supports_texture_compression_bc()` (Dawn enables `TextureCompressionBCSliced3D`
  unconditionally with `textureCompressionBC`).
- `VulkanAdapter::supports_texture_compression_astc_sliced_3d()` returns true iff
  `textureCompressionASTC_LDR` is supported **and**, for **every** ASTC LDR
  `vk::Format` yawgpu maps (all 14 block sizes × {UNORM, SRGB} = 28 formats),
  `vkGetPhysicalDeviceImageFormatProperties(format, TYPE_3D, OPTIMAL,
  SAMPLED, flags = empty)` returns `VK_SUCCESS` (Dawn
  `IsTextureCompressionASTCSliced3DSupported`). The `vk::Format` values must be
  obtained by mapping the ASTC `HalTextureFormat` variants through
  `vulkan/format.rs` `map_texture_format` (no second hand-written `vk::Format`
  list that can drift); enumerating the 28 `HalTextureFormat` variants by hand
  is acceptable (the ASTC LDR set is closed) provided a unit test pins the list
  to 28 unique entries that are all ASTC `*_BLOCK` formats. The probe result is
  computed at most once per adapter (cached), since feature enumeration calls it
  repeatedly.
- Core `add_texture_compression_features` is unchanged (sliced-3d is only
  inserted when the base family is supported — already the case).
- Metal / GLES / Noop unchanged.

### R2 — enable compression features at `vkCreateDevice`

In the `enabled_features` construction (`yawgpu-hal/src/vulkan/mod.rs`, the
`vk::PhysicalDeviceFeatures::default()` block), set
`texture_compression_bc`, `texture_compression_etc2`,
`texture_compression_astc_ldr` to `vk::TRUE` iff the physical device supports
each — same "enable whenever supported" policy as the neighbouring features
(HAL has no requested-feature set). Extend the existing inline unit test that
mirrors this block (`mod.rs` ~1448, "leaves each unavailable feature FALSE") to
cover the three new bits in both the supported and unsupported cases.

### R3 — 3D compressed textures work on the Vulkan HAL

With R1 advertised and the feature enabled on the device, a 3D BC texture
(block-aligned width/height, arbitrary depth) must:

- create (`VK_IMAGE_TYPE_3D`, compressed format),
- accept `wgpuQueueWriteTexture` and B2T for a depth range (`origin.z`,
  `copy_size.depth` > 1) with block-sized `bytesPerRow` / `rowsPerImage`
  (`rowsPerImage` is in **block rows**; `bufferImageHeight` in **texels** — the
  existing Round-2 block→texel conversion must hold for 3D depth slices the
  same way it does for array layers),
- read back via T2B per depth slice,
- be sampled through a `3d` view.

Any HAL defect found is fixed in `yawgpu-hal/src/vulkan/**` with an inline unit
test. If a case has no clean mapping it returns `HalError` — never a panic.

### R3b — compressed copy extents at mip edges (found in slice 2)

WebGPU copy sizes on a compressed mip are in **physical** (block-rounded)
texels; Vulkan requires `imageExtent` to end at the **logical** mip size for a
partial edge block (`VUID-vkCmdCopyImage-srcOffset-00144`-family, 07971/07972
for buffer copies). The HAL converts the extent per image
(`compressed_copy_extent`), keeping buffer pitches in block units.

For texture-to-texture copies the converted source and destination extents can
differ (Dawn's example: source = mip 0 of a 16x16 BC texture, destination =
mip 2 of a 60x60 one — neither 16x16 nor 15x15 is a legal `vkCmdCopyImage`
extent). This is a **valid WebGPU operation and must succeed on Tier 1**: when
the extents differ, the Vulkan HAL performs the copy as texture→temporary
buffer→texture (Dawn `CommandBufferVk.cpp` `RecordCopyImageWithTemporaryBuffer`,
toggle `UseTemporaryBufferInCompressedTextureToTextureCopy`, default on for
Vulkan). The temporary buffer lives until the submission completes. Returning
`HalError` for this case is not acceptable.

### R4 — real-GPU e2e coverage

New file `yawgpu/tests/e2e_vulkan_texture_compression.rs` (`#![cfg(feature =
"vulkan")]`, every test `#[ignore = "manual real-backend test"]`, each test
self-skips with an `eprintln!` when the adapter lacks the needed feature — and
the skip message must name the missing feature so a run log shows what was NOT
exercised). Required probes:

| # | Probe | Asserts |
|---|---|---|
| E1 | every BC format (BC1–BC7, unorm/snorm/ufloat/sfloat + srgb variants — all 14 WebGPU BC formats) 8x8 (4 blocks), writeTexture → T2B | bytes identical; block size 8 or 16 per format |
| E2 | BC1 mip chain, 12x12 base, 4 mips (12, 6, 3, 1 logical → 12, 8, 4, 4 physical): write every mip, T2B every mip | bytes identical per mip, using per-byte-distinct data (so swapped blocks/rows or a wrong `bufferRowLength` are detected); validates physical-size copy bounds on real HW |
| E3 | BC1 T2T: copy a 2x2-block region between two textures, then T2B the destination | destination bytes == source bytes; untouched blocks stay zero |
| E4 | BC1 **sampled render**: 4x4 texture with one solid-colour block (`color0 = color1 = 0xF800`, indices 0 → opaque red), sample into an `rgba8unorm` target with a fullscreen triangle, read back | every pixel == (255, 0, 0, 255) |
| E5 | BC1 sRGB format: the same block in a texture created as `bc1-rgba-unorm-srgb` (not a `viewFormats` reinterpretation) | red channel 255, alpha 255 (endpoints are 0/1 so sRGB decode is exact) |
| E6 | 3D BC1 (sliced-3d): 4x4x3, distinct block per slice; write slices 0..3 in one writeTexture; T2B each slice | per-slice bytes identical |
| E7 | 3D BC1 sampled through a `3d` view at slice centres | each slice's solid colour reads back |
| E8 | ETC2 (`etc2-rgb8unorm`, `eac-r11unorm`) and ASTC (`astc-4x4-unorm`, `astc-8x8-unorm`, `astc-12x12-unorm`) multi-block round-trip | bytes identical (self-skip when the family is absent — expected on desktop NVIDIA/AMD) |
| E10 | BC1 T2T with mismatched logical edges (R3b): source mip 0 of 16x16 → destination mip 2 of 60x60 (physical 16x16, logical 15x15), copy size 16x16; T2B the destination mip | destination blocks == source blocks; no device error |
| E9 | 3D ASTC round-trip | self-skip unless `texture-compression-astc-sliced-3d` |

The run must be clean under `VK_LAYER_KHRONOS_validation` (no VUID output).
The pre-existing probes in `e2e_vulkan_threading_audit.rs` stay as they are.

### R5 — docs

- `specs/tracking/cts-coverage.md`: the "Compressed formats … remain
  `Unsupported` — deferred" note and the "compressed-format subcases deferred:
  Noop lacks feature" table cells are stale (formats are implemented; Noop
  advertises all five features; the in-repo CTS ports were removed in favour of
  webgpu-native-cts). Correct them.
- `specs/tracking/format-completeness-audit.md`: "Sliced-3d … deferred" →
  resolved for Vulkan by this block.
- `specs/blocks/60-real-backends.md` / README feature table if they list
  per-backend compression support.

## Slices

1. **HAL: R1 + R2** with inline unit tests. Acceptance: `cargo test --workspace`
   green on Noop; `cargo clippy --workspace --all-targets --features vulkan --
   -D warnings` clean; on this machine `wgpuAdapterHasFeature` reports
   `texture-compression-bc-sliced-3d` on the native Vulkan adapter.
2. **e2e: R3 + R4** (+ any HAL fix the probes expose, each with an inline unit
   test). Acceptance: all non-skipped probes pass on the native Windows Vulkan
   driver with validation layers enabled and no VUID output.
3. **Docs (R5) + Phase Review.** External CTS re-confirmation of the
   compressed-format trees (`api,validation,image_copy,*`,
   `api,operation,command_buffer,copyTextureToTexture` compressed subcases,
   `createTexture` sliced-3d cases) on Vulkan is run from webgpu-native-cts and
   logged in the tracking doc.

Tracking: `specs/tracking/texture-compression-vulkan.md`.
