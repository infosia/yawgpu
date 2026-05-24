# Vulkan buffer/texture usage VUIDs — pre-existing, surfaced 2026-05-23

Status: **CLOSED 2026-05-24** (F1: `483f90f`, F2: `fe08b59`, F3: `c664c24`).
The `--features vulkan,tiled --ignored` suite under
`VK_LAYER_KHRONOS_validation` on a native Windows Vulkan driver now
prints zero VUID lines; the Vulkan backend is validation-clean.
Originally surfaced by running that suite on 2026-05-23 — not Phase-14-introduced.

## Findings

### F1 — Buffer usage hardcoded `TRANSFER_SRC | TRANSFER_DST` *(CLOSED 2026-05-24, commit `483f90f`)*

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

**Fix shipped (commit `483f90f`):** caller usage now threads through the HAL.
`HalDevice::create_buffer` currently takes `size: u64` only
(`yawgpu-core/src/device.rs:224`), so the WebGPU usage bits validated in
`yawgpu-core` are dropped at the HAL boundary. Fix is structural:

1. Introduce `HalBufferUsage` in `yawgpu-hal/src/format.rs`, mirroring
   `HalTextureUsage`'s `Debug + Clone + Copy` struct-of-bools shape, with
   fields `map_read`, `map_write`, `copy_src`, `copy_dst`, `index`,
   `vertex`, `uniform`, `storage`, `indirect`, `query_resolve`.
2. Extend `HalDevice::create_buffer(&self, size: u64, usage: HalBufferUsage)`
   across all three backends (Noop accepts and discards; Metal accepts and
   discards for now — `MTLBuffer` has no equivalent validation; Vulkan
   propagates).
3. Add `map_buffer_usage(HalBufferUsage) -> vk::BufferUsageFlags` in
   `yawgpu-hal/src/vulkan/format.rs`. Always OR in
   `TRANSFER_SRC | TRANSFER_DST` (yawgpu uses staging for `mapAtCreation` /
   `writeBuffer` paths). Map `index → INDEX_BUFFER`,
   `vertex → VERTEX_BUFFER`, `uniform → UNIFORM_BUFFER`,
   `storage → STORAGE_BUFFER`, `indirect → INDIRECT_BUFFER`,
   `query_resolve → TRANSFER_DST` (no dedicated flag; query copy is a
   transfer-dst write). `map_read` / `map_write` map to nothing (host
   visibility is on memory, not buffer usage).
4. `yawgpu-core/src/buffer.rs` adds a `hal_buffer_usage(BufferUsage) ->
   HalBufferUsage` conversion helper; `device.rs:224` calls
   `self.inner.hal.create_buffer(descriptor.size,
   hal_buffer_usage(descriptor.usage))`.
5. Inline unit tests per CLAUDE.md principle 1 for the new public/internal
   functions (struct construction, the mapping table, the core conversion).

Coding-agent handoff: V11.3 in `HANDOFF.md`.

### F2 — Image views over transfer-only images *(CLOSED 2026-05-24, commit `fe08b59`)*

`VUID-VkImageViewCreateInfo-image-04441` fires in
`e2e_vulkan_texture::vulkan_buffer_texture_buffer_round_trip` and
`vulkan_texture_texture_round_trip`: views are created on textures whose
only usage is `COPY_SRC | COPY_DST`. View-compatible bits
(`SAMPLED` / `STORAGE` / `*_ATTACHMENT` / `INPUT_ATTACHMENT`) are absent.

`vulkan/format.rs::map_texture_usage` maps the WGPU texture usage flags
correctly. The bug is in `vulkan/texture.rs:174-185`: `create_texture`
**unconditionally** calls `vkCreateImageView` for every texture, regardless
of whether the caller's usage permits a view. Copy-only textures (used
purely as staging sources/destinations) trip 04441 because the view they
get isn't usage-compatible.

**Fix shipped (commit `fe08b59`):** image view creation is now conditional.
Skip `vkCreateImageView` when the caller's `HalTextureUsage` has no
view-compatible bit set (`texture_binding`, `storage_binding`, or
`render_attachment` are all `false`); store `vk::ImageView::null()`
in that case. The Drop impl is naturally safe — the Vulkan spec defines
`vkDestroyImageView(VK_NULL_HANDLE, …)` as a no-op. Downstream consumers
in `vulkan/encode.rs` (`:803`, `:809`, `:1281`) only read the view for
render-pass attachment / bind-group resource paths; yawgpu-core
validation already rejects binding or attaching a texture without the
matching usage bit, so a copy-only texture never reaches those sites.
Coding-agent handoff: V11.2 in `HANDOFF.md`.

### F3 — SPIR-V env extension declaration *(CLOSED 2026-05-24, commit `c664c24`)*

`VUID-VkShaderModuleCreateInfo-pCode-08742` — shaders declare
`SPV_KHR_storage_buffer_storage_class` but the device is targeted at
Vulkan 1.0 and `VK_KHR_storage_buffer_storage_class` is not enabled.

**Fix shipped (commit `c664c24`):** `VulkanInstance::new` chains a
`pApplicationInfo` declaring `VK_API_VERSION_1_1` via the
`YAWGPU_VULKAN_API_VERSION` constant; `VulkanAdapter::new` drops physical
devices whose `properties.api_version` is below 1.1 through the
`is_supported_api_version` helper. The 08742 VUID no longer appears under
`VK_LAYER_KHRONOS_validation` on the native Windows Vulkan driver.
Block 60 § "Minimum Vulkan version" carries the authoritative rule.

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
