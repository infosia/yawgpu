# Block 134 — Vulkan adapter limits: NVIDIA storage cap and Dawn's remaining mappings

Status: **PLANNED (2026-11-03)**. Starts after Block 133, which edits the same file.
Source: backlog E6, by user request on 2026-11-03 ("the remaining NVIDIA items").
Oracle: Dawn at the pinned `third_party/dawn`: `native/vulkan/PhysicalDeviceVk.cpp`
(`InitializeSupportedLimitsImpl`: the NVIDIA `k2GbMinus4` cap,
`maxColorAttachmentBytesPerSample`, the `maxFragmentCombinedOutputResources`
redistribution), `native/Adapter.cpp` (`UpdateLimits`: tiers apply only when
`mUseTieredLimits`, which native Dawn leaves off).

## Problem

`hal_limits_from_vk` (`yawgpu-hal/src/vulkan/mod.rs`) says it ports Dawn's mapping, but
three of Dawn's adjustments are marked deferred or missing. On the Windows NVIDIA host
(RTX 5060 Ti, driver 617.14) two of them change what the adapter reports:

| Limit | yawgpu | Dawn |
|---|---|---|
| `maxStorageBufferBindingSize` | 4 GiB - 1 (`maxStorageBufferRange`) | 2 GiB - 4 |
| `maxColorAttachmentBytesPerSample` | 32 (the WebGPU default; never queried) | 256 (32 x `maxColorAttachments` 8) |

1. **NVIDIA storage cap (correctness).** On some NVIDIA devices `OpArrayLength` returns 0
   for buffers of 2 GiB or more (crbug.com/435684920). Dawn caps the binding size at
   2 GiB - 4 on NVIDIA. yawgpu advertises 4 GiB - 1, so an application may bind a large
   storage buffer whose `arrayLength` reads 0: silent wrong results.
2. **`maxColorAttachmentBytesPerSample`.** Vulkan has no such limit. Dawn reports
   32 x `maxColorAttachments` so the limit never binds. yawgpu reports the default 32, so
   it rejects render pipelines Dawn accepts, for example four `rgba32float` targets
   (64 bytes per sample).
3. **`maxFragmentCombinedOutputResources`.** When the Vulkan limit is meaningful and
   smaller than the sum of storage buffers, storage textures, and color attachments
   per stage, Dawn lowers those three limits to fit. yawgpu ignores it. NVIDIA reports
   `u32::MAX`, so this does not fire here; it matters on mobile drivers.

## Rules

R1. **NVIDIA storage cap.** On vendor `0x10DE`, `max_storage_buffer_binding_size` is
`min(maxStorageBufferRange, max_buffer_size, 2^31 - 4)`. Dawn sets
`min(maxBufferSize, 2^31 - 4)` and drops `maxStorageBufferRange`. yawgpu keeps the Vulkan
range in the minimum as well, because binding beyond it is invalid usage. On real
NVIDIA drivers the range is `u32::MAX`, so the result is the same. Other vendors are
unchanged.

R2. **Color bytes per sample.** `max_color_attachment_bytes_per_sample` is
`32 * VkPhysicalDeviceLimits::maxColorAttachments`, as Dawn. Core still clamps it from
below by the WebGPU default, as it does every limit.

R3. **Fragment combined output resources.** Port Dawn's redistribution exactly, on
yawgpu's already-clamped per-stage values:
- `min` = the WebGPU defaults for storage buffers per stage, storage textures per stage,
  and color attachments, summed.
- `max` = the adapter's values for the same three, summed.
- If `maxFragmentCombinedOutputResources > min` and `< max`:
  - `extra = maxFragmentCombinedOutputResources - min`
  - color attachments = `min(default + extra / 3, vk maxColorAttachments)`; subtract
    what was added from `extra`
  - storage textures = `min(default + extra / 2, vk maxPerStageDescriptorStorageImages)`;
    subtract what was added
  - storage buffers = `min(default + extra, vk maxPerStageDescriptorStorageBuffers)`
- Otherwise the three are unchanged.

The vertex- and fragment-stage storage limits core derives from the per-stage values
follow, as today. R2 uses the Vulkan `maxColorAttachments`, as Dawn does, not the
redistributed value.

R4. Every rule is a pure function of the Vulkan limits and the vendor id, unit-tested
without a GPU: NVIDIA and non-NVIDIA for R1; a `max_buffer_size` below 2 GiB on NVIDIA;
R3 not firing (equal to `min`, at or above `max`, `u32::MAX`) and firing with each
clamp reached. Remove the "Deferred" comment in `hal_limits_from_vk`.

R5. Metal, GLES, and Noop are unchanged.

## Tests

- Unit (HAL): R4.
- Unit (core): the adapter limits built from such HAL limits keep the capped storage size
  and the 256-byte color limit; a device requesting more than the cap fails.
- Vulkan HAL `#[ignore]`: logs the three values on the real device and asserts R1-R3 on
  them.
- e2e (Claude): a render pipeline with four `rgba32float` targets is accepted on Windows
  native Vulkan.
- CTS (Claude): `api,validation,capability_checks,limits,*` and `webgpu:*` on Windows
  native Vulkan.

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests, after Block 133 is committed.
- **S2 (Claude):** e2e, CTS, Phase Review.
