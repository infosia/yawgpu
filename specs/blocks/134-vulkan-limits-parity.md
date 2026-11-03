# Block 134 — Vulkan adapter limits: NVIDIA storage cap and Dawn's remaining mappings

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 2 MAJOR, 5 MINOR; all fixed, see
below.
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

R1. **Storage and uniform binding sizes.**
- On vendor `0x10DE`, `max_storage_buffer_binding_size` is
  `min(maxStorageBufferRange, max_buffer_size, 2^31 - 4)`. Dawn sets
  `min(maxBufferSize, 2^31 - 4)` and drops `maxStorageBufferRange`. yawgpu keeps the
  Vulkan range in the minimum as well, because binding beyond it is invalid usage. On real
  NVIDIA drivers the range is `u32::MAX`, so the result is the same.
- On every vendor, as Dawn's `NormalizeLimits` (`native/Limits.cpp`) does afterwards:
  `max_storage_buffer_binding_size` is then `min(.., max_buffer_size)` aligned down to a
  multiple of 4, and `max_uniform_buffer_binding_size` is `min(.., max_buffer_size)`.

R2. **Color bytes per sample.** `max_color_attachment_bytes_per_sample` is
`32 * VkPhysicalDeviceLimits::maxColorAttachments`, as Dawn, computed with a saturating
multiply: a driver value large enough to overflow must not panic. Core still clamps it from
below by the WebGPU default, as it does every limit.

R3. **Fragment combined output resources.** Port Dawn's redistribution, on yawgpu's
already-clamped per-stage values, then clamp as Dawn's `NormalizeLimits` does:
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
- Then, in every case, color attachments are clamped to 8, storage textures per stage to
  8, and storage buffers per stage to 16 (`kMaxColorAttachments`,
  `kMaxStorageTexturesPerShaderStage`, `kMaxStorageBuffersPerShaderStage`).

Two differences from Dawn, both deliberate. First, `max` sums yawgpu's clamped values,
while Dawn sums the raw Vulkan values, so the redistribution fires in fewer cases; when it
does not fire, the clamped values already fit the budget. Second, the redistributed values
are clamped by the raw Vulkan limits, as in Dawn, before the final clamp.

The vertex- and fragment-stage storage limits core derives from the per-stage values
follow, as today. R2 uses the Vulkan `maxColorAttachments`, as Dawn does, not the
redistributed value.

R4. Every rule is a pure function of the Vulkan limits and the vendor id, unit-tested
without a GPU: NVIDIA and non-NVIDIA for R1; a `max_buffer_size` below 2 GiB on NVIDIA,
including one that is not a multiple of 4; the uniform size above `max_buffer_size`; R2
with `maxColorAttachments = u32::MAX`; R3 not firing (equal to `min`, at or above `max`,
`u32::MAX`) and firing with each clamp reached, including the final 8 / 8 / 16 clamp.
Remove the "Deferred" comment in `hal_limits_from_vk` and correct its stale Dawn line
reference.

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

## Verification (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351. The
Khronos validation layer was on for the e2e and HAL runs. The HAL test logs
`storageBindingSize=2147483644, colorBytesPerSample=256, combinedOutputs=4294967295
(colors=8, storageTextures=8, storageBuffers=16)`: R1 and R2 change the reported values on
this host, R3 does not fire.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,367 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default (workspace) and `vulkan,tiled` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`), all targets | clean |
| Vulkan HAL lib tests | 366 passed, 0 failed |
| Vulkan HAL ignored tests | 80 passed, 0 failed, 0 VUIDs |
| e2e `four_rgba32float_targets_are_accepted` | before: fails, the adapter reports `maxColorAttachmentBytesPerSample=32` and `maxStorageBufferBindingSize=4294967295`; after: passes. A device with the adapter's limits accepts four `rgba32float` targets; a device with default limits rejects them with one validation error |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored` | 163 passed before the e2e's control was added; the e2e then passes on its own; 0 VUIDs |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, `expectations/yawgpu-vulkan.txt` | `fail=2 crash=0` |

| Area | pass | skip | fail | Block 133 run pass |
|---|---:|---:|---:|---:|
| `api,validation` | 246,666 | 111,101 | 0 | 246,484 |
| `api,operation` | 180,779 | 152,819 | 0 | 180,779 |
| `shader,execution` | 562,548 | 320,680 | 0 | 562,523 |
| `shader,validation` | 662,535 | 20,299 | 2 | 662,535 |

`api,validation` gains 182 passes and loses 182 skips: cases that need more than 32
color bytes per sample now run, and pass. The 2 failures are the Block 119 R9 cases. The
F-165 crash did not fire in this run.

The CTS and the full e2e ran on the library before the Phase Review fixes. On this host
those fixes leave every reported value unchanged (the storage size is already a multiple
of 4 and below `maxBufferSize`, the uniform range is below `maxBufferSize`, R3 does not
fire, and the final clamps do not bind), so they were not rerun; the gates, HAL tests and
the Block 134 e2e above ran after the fixes.

Found on the way: `wgpuAdapterGetInfo` reports `vendorID` and `deviceID` as 0 on every
backend (`yawgpu/src/ffi/mod.rs`, `adapter_info_from_core`); Dawn reports the PCI ids.
The e2e reads the vendor from the device name instead. Backlog A23.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| M1 | `32 * maxColorAttachments` could overflow and panic in debug | Saturating multiply; `u32::MAX` unit case (R2) |
| M2 | R3 could raise limits above 8 / 8 / 16; Dawn clamps afterwards in `NormalizeLimits` | Final clamp added (R3); test expectations corrected |
| m1 | The spec said R3 was an exact port | R3 states both differences from Dawn |
| m2 | Dawn's `NormalizeLimits` also aligns the storage size down to 4 and caps both binding sizes by `maxBufferSize` | Added for every vendor (R1) |
| m3 | Stale Dawn line reference in the code comment | Cites function names |
| m4 | The ignored test mirrored the algorithm with unchecked subtraction | Saturating arithmetic |
| m5 | The e2e had no negative control | A default-limits device must reject the same pipeline |
