# Vulkan buffer/texture usage VUIDs — pre-existing, surfaced 2026-05-23

Status: **OPEN.** Not Phase-14-introduced (the Vulkan HAL buffer path has
hardcoded `TRANSFER_SRC | TRANSFER_DST` since the backend landed in Phase 7);
surfaced by running the `--features vulkan,tiled --ignored` suite on a
native Windows Vulkan driver with `VK_LAYER_KHRONOS_validation` enabled.
All affected tests still pass functionally (drivers tolerate the missing
flags), but the layer reports real spec violations.

## Findings

### F1 — Buffer usage hardcoded `TRANSFER_SRC | TRANSFER_DST`

`yawgpu-hal/src/vulkan/buffer.rs:116` creates every Vulkan buffer with:

```rust
.usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
```

The caller-supplied `WGPUBufferUsage` (vertex / index / uniform / storage /
indirect / query-resolve) is never read, so every buffer is missing every
non-transfer usage bit.

**VUIDs surfaced:**
- `VUID-VkWriteDescriptorSet-descriptorType-00330` — uniform buffer binding
  on a buffer without `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`
  (`e2e_vulkan_render::vulkan_render_uniform_color_triangle_readback`).
- `VUID-VkWriteDescriptorSet-descriptorType-00331` — storage buffer binding
  on a buffer without `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`
  (`e2e_vulkan_compute::*`).
- `VUID-vkCmdBindVertexBuffers-pBuffers-00627` — vertex bind on a buffer
  without `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`
  (`e2e_vulkan_render::*`).

**Fix sketch:** add a `map_buffer_usage(HalBufferUsage) -> vk::BufferUsageFlags`
analogous to `vulkan/format.rs::map_texture_usage`, called from
`create_buffer`. Always OR in `TRANSFER_SRC | TRANSFER_DST` (yawgpu uses
staging for `mapAtCreation` / `writeBuffer` paths). Index/indirect/query
bits too. Add a HAL unit test asserting the mapping.

### F2 — Image views over transfer-only images

`VUID-VkImageViewCreateInfo-image-04441` fires in
`e2e_vulkan_texture::vulkan_buffer_texture_buffer_round_trip` and
`vulkan_texture_texture_round_trip`: views are created on textures whose
only usage is `COPY_SRC | COPY_DST`. View-compatible bits
(`SAMPLED` / `STORAGE` / `*_ATTACHMENT` / `INPUT_ATTACHMENT`) are absent.

`vulkan/format.rs::map_texture_usage` already maps the WGPU texture usage
flags correctly, so this is likely a **test-design** issue (the tests
exercise copy-only round-trips and never sample / render the texture, yet
the view machinery is invoked) **or** a place in the HAL that creates a
view path without going through `map_texture_usage`. Investigate before
fixing — the underlying flag mapping looks right.

### F3 — SPIR-V env extension declaration

`VUID-VkShaderModuleCreateInfo-pCode-08742` — shaders declare
`SPV_KHR_storage_buffer_storage_class` but the device is targeted at
Vulkan 1.0 and `VK_KHR_storage_buffer_storage_class` is not enabled.

**Fix sketch:** either request Vulkan 1.1 in `vkCreateInstance`
(`api_version = vk::API_VERSION_1_1`) or enable
`VK_KHR_storage_buffer_storage_class` as a device extension. 1.1 is
trivially available on every driver we care about and removes the need to
list the KHR extension.

## Why none of this was caught earlier

- Validation layers were not enabled by default in CI / cargo-test loops.
- The Apple-Silicon dev host runs Vulkan via MoltenVK, whose conformance
  is laxer than a native driver in some areas and stricter in others;
  these particular VUIDs weren't surfaced there either.
- The Windows native driver became the standard verification target only
  in Phase 14 B5c (when `vulkan_two_subpass_draw_subpass_load_readback`
  was first run outside MoltenVK's self-skip). Until 2026-05-23 the
  `--features vulkan,tiled --ignored` run on Windows was not paired with
  validation layers.

## Scope decision

These are **not** Phase 14 regressions — every affected file
(`yawgpu-hal/src/vulkan/buffer.rs`, `format.rs`, instance-creation code)
existed before Phase 14 and was not touched by Phase 14 commits. The
tiled section runs validation-clean. Track these as a follow-up so the
non-tiled Vulkan path can be brought to validation-clean parity with the
tiled path; no Phase-14 status change.

## Verification command

```pwsh
$env:VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
cargo test -p yawgpu --features vulkan,tiled --no-fail-fast -- --ignored
```

A successful fix run should print zero `VUID-…` lines for the listed
test binaries.
