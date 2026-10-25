# Block 114 — bug-class audit: Vulkan

Status: **COMPLETE (2026-10-25)** — S1 `af83e6f`, S2 `da96e94`, Phase Review
fixes (R-PR1..3, m1–m4) in the closing commit. Gate: `cargo test --workspace`,
strict clippy + fmt; MoltenVK all `e2e_vulkan_*` with `vulkan,tiled` under
sync validation: 0 `SYNC-HAZARD` (incl. the three tiled deferred tests, R-PR3),
only the pre-existing MoltenVK 3D-attachment fail; HAL ignored 58/58, 0 VUID.
Source: `specs/tracking/class-audit-2026-10.md` (Vulkan rows). Oracle: Dawn
`third_party/dawn/src/dawn/native/vulkan/` (`BufferVk.cpp`, `TextureVk.cpp`,
`SwapChainVk.cpp`, `CommandBufferVk.cpp`, `PhysicalDeviceVk.cpp`).

| Slice | Rows | Files | Wave |
|---|---|---|---|
| S1 sync + copies (HAL only) | S1, S2, S3, S4, S6, S7, S9, F1, T7 | `yawgpu-hal/src/vulkan/**` | A |
| S2 cross-layer | F2, T2, T1, T3, T4 | `yawgpu-hal` descriptors + vulkan, `yawgpu-core` texture HAL descriptor, `yawgpu/src/ffi/surface.rs`, `yawgpu-tint` shim | B |
| S3 | Phase Review | | |

## S1 rules

- **S2** every barrier whose destination covers shader reads also includes
  `UNIFORM_READ` (copy → *, query resolve → *, post-render, pre-dispatch),
  matching Dawn's Uniform → `VK_ACCESS_UNIFORM_READ_BIT` mapping.
- **S1** read → write ordering for buffers: before a copy / fill / clear /
  copy-to-buffer / compute write / render storage write into a buffer that
  earlier work in the same or a previous submission read (vertex input,
  index, indirect, uniform, shader read, transfer read), record an
  execution dependency from the read stages (Dawn `BufferVk.cpp`
  `TransitionUsageNow`: tracks the last read usage per buffer). A
  conservative global barrier is acceptable if per-buffer tracking is too
  invasive; state which in the report.
- **S3** the first queue submit after an acquire waits on the acquire
  semaphore whatever it touches (Dawn `SwapChainVk.cpp`), incl. compute,
  lazy-init clears, bind-group use and tiled subpasses.
- **S6** `pending.consumed` is set only after `vkQueueSubmit` succeeds.
- **S4** `subpass_dependencies` incoming dependency has the attachment
  write source access (mirror Block 109 R9); fix its unit test to assert
  the source side.
- **S7** buffers that can be mapped for reading get a `HOST_READ` barrier
  after GPU writes (Dawn `TransitionMappableBuffersEagerly`).
- **S9** the present-ready semaphore is indexed by swapchain image (or a
  fresh one per present), not by sync slot.
- **F1** 2D-array ↔ 3D image copies use `extent.depth = copy depth` when
  either side is 3D (Dawn `CommandBufferVk.cpp`).
- **T7** swapchain `preTransform = IDENTITY` when supported (Dawn).

## S2 rules

- **F2** texture `viewFormats` reach the HAL descriptor; Vulkan sets
  `MUTABLE_FORMAT` + `VkImageFormatListCreateInfo` whenever view formats
  differ from the texture format (Dawn `TextureVk.cpp`).
- **T2** surface `viewFormats` reach the HAL; the swapchain uses
  `VK_KHR_swapchain_mutable_format` when needed (advertise / reject
  otherwise, as Dawn).
- **T1** the swapchain image size equals the configured size when it is
  within `[minImageExtent, maxImageExtent]`; otherwise render into an
  intermediate texture of the configured size and blit at present (Dawn
  `SwapChainVk.cpp` blit path). The core texture extent always equals the
  HAL image extent the app renders to.
- **T3** acquire / present map `VK_SUBOPTIMAL_KHR` → `SuccessSuboptimal`,
  `VK_ERROR_OUT_OF_DATE_KHR` → `Outdated`, surface lost → `Lost`, through
  HAL error variants and the FFI status mapping.
- **T4** advertise `shader-f16` only with `shaderFloat16 && shaderInt16 &&
  storageBuffer16BitAccess` (Dawn `PhysicalDeviceVk.cpp`); pass
  `use_storage_input_output_16 = storageInputOutput16` to the Tint SPIR-V
  writer.

## Verification (Claude)

Before the fixes, new `e2e_vulkan_*` cases reproduce S1 (draw reading a
vertex buffer, then `writeBuffer` into it), S2 (`writeBuffer(uniform)` →
draw / dispatch), S3 (compute writing the swapchain image as storage), F1
(2D-array → 3D round trip) and F2 / T2 (srgb view of a unorm texture /
surface) under the Khronos layer with synchronization validation;
after the fixes they pass with 0 VUID / `SYNC-HAZARD` lines, and the whole
`e2e_vulkan_*` set stays at 0.

## S3 — Phase Review (2026-10-25)

Fresh reviewer over `af83e6f`, `da96e94`: **0 CRITICAL / 1 MAJOR / 4 MINOR**
plus a cost note. Gates before review: `cargo test --workspace`, strict
clippy (all backend combos) + fmt clean; Metal e2e 128/0; MoltenVK all
`e2e_vulkan_*` under sync validation 122 pass / 1 pre-existing MoltenVK
fail, 0 `SYNC-HAZARD`; HAL ignored tests 58/58, 0 VUID; the F2 / T1 / T2
e2e cases fail on the pre-fix HAL and pass after.

| ID | Finding | Disposition |
|---|---|---|
| M1 | T1: a fixed `currentExtent` always wins, so an in-range configured size still goes through the blit path (extra scaled blit per frame, intermediate images, forced TRANSFER_DST + BLIT/FILTER_LINEAR format requirements) — Dawn uses the configured size whenever it is inside `[minImageExtent, maxImageExtent]` | fix (R-PR1) + regression test with fixed `currentExtent`, wide range, in-range config |
| m1 | blit path still requires `VK_KHR_swapchain_mutable_format` for view formats (they apply to the intermediate) | fix: skip mutable flags / format list on the swapchain when blitting |
| m2 | unsupported swapchain usage errors; Dawn blits (`SwapChainVk.cpp` usage subset check) | fix: blit when the requested usage is not supported by the surface |
| m3 | shader-f16 enabled without `uniformAndStorageBuffer16BitAccess` (Dawn refuses unless it decomposes uniforms; yawgpu cannot) | fix: require it for advertisement |
| m4 | `PhysicalDevice16BitStorageFeatures` chained unconditionally (invalid on a Vulkan 1.0 device without the extension) | fix: gate on device API ≥ 1.1 or the extension |
| cost | S1's conservative global read→write barrier runs before every top-level command (K writes + P passes + 1 per frame), draining all shader stages at every pass boundary | narrow (R-PR2): emit only before commands that write a buffer (buffer copy / clear, texture→buffer copy, query resolve, writeBuffer staging copies, passes with writable storage buffers); keep the e2e `vulkan_buffer_read_by_copy_then_overwritten_in_same_command_buffer` hazard-free |
| tiled | first run of `e2e_vulkan_tiled` under sync validation: `SYNC-HAZARD-WRITE-AFTER-WRITE` at the end-of-pass / next-subpass store of an attachment that an earlier subpass wrote and a later subpass reads as an input attachment (prior usage: the inter-subpass layout transition; the dependency only covers `INPUT_ATTACHMENT_READ`). Pre-existing (identical before Block 115 S3) | fix (R-PR3): inter-subpass dependencies' destination covers the later subpass's attachment writes / store as well as the input read; e2e_vulkan_tiled 0 hazards |
