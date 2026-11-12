# Block 155 — Vulkan images and devices support the cube views core accepts

Status: **PLANNED (2026-11-13)**.
Source: whole-code review at `92e38c7` (`specs/tracking/code-review-2026-11.md`), finding R07.
Oracle: pinned Dawn — `native/vulkan/TextureVk.cpp:1515-1517` (cube-compatible flag),
`native/vulkan/DeviceVk.cpp:575` (enable `imageCubeArray`),
`native/vulkan/PhysicalDeviceVk.cpp:271-272` (core requires `imageCubeArray`).

## Problem

`texture_image_flags` (`yawgpu-hal/src/vulkan/texture.rs` ~159) never sets
`VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT`, yet `vulkan/pipeline.rs` (~2121) lowers accepted
`cube` / `cube-array` views to `VK_IMAGE_VIEW_TYPE_CUBE` / `CUBE_ARRAY`. That violates
`VUID-VkImageViewCreateInfo-image-01003`. The logical device also never enables
`imageCubeArray`, required for `CUBE_ARRAY` views
(`VUID-VkImageViewCreateInfo-viewType-01004`). Drivers often tolerate both, so the CTS
without validation layers does not see it.

## Rules

R1. **Cube-compatible images.** A Vulkan image gets `CUBE_COMPATIBLE` when it is 2D, has at
least 6 array layers, and its width equals its height — Dawn's condition. (WebGPU
multisampled textures have one layer, so they never qualify.) Other flags are unchanged.

R2. **`imageCubeArray` is enabled.** Logical-device creation enables
`VkPhysicalDeviceFeatures::imageCubeArray` when the physical device supports it.
An adapter whose physical device lacks it is not exposed as a core WebGPU adapter, as in
Dawn; the agent reports how adapter enumeration currently filters unsupported devices
and applies the same mechanism.

R3. **Portability subset.** On MoltenVK the portability-subset features are queried and
chained at device creation (memory `vulkan-portability-subset-moltenvk`). The agent
confirms nothing in the portability subset forbids these cube views on MoltenVK, and
reports it.

## Slices

S1: R1–R3 in `yawgpu-hal/src/vulkan`, inline unit tests for `texture_image_flags` (6-layer
square 2D → flag; 6-layer non-square, 5-layer, 3D, 1D → no flag) and for the
enabled-feature builder.

## Verification

- Agent gates: `cargo test -p yawgpu-hal --features vulkan --lib` (non-ignored),
  `cargo clippy -p yawgpu-hal --features vulkan --all-targets -- -D warnings`, workspace
  clippy.
- Claude: workspace test; a new `e2e_vulkan_*` test that samples a 6-layer texture through
  a `cube` view and a 12-layer texture through a `cube-array` view and reads back the
  expected face colors, run under the Khronos validation layer on MoltenVK with 0 VUIDs;
  full `e2e_vulkan_*` regression under the layer.
