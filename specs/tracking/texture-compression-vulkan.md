# Vulkan compressed-texture completion — tracking

Spec: `specs/blocks/73-texture-compression-vulkan.md`.

## Baseline (2026-09-20, native Windows Vulkan driver)

`cargo test -p yawgpu --features vulkan --test e2e_vulkan_threading_audit
--no-fail-fast -- --ignored compressed --test-threads=1` → 4 passed
(`vulkan_bc1_compressed_roundtrip`, `vulkan_compressed_multilayer_roundtrip`
executed; `vulkan_etc2_*` / `vulkan_astc_*` self-skip when the adapter lacks the
family, so a pass there is not evidence of execution on a desktop GPU).
Validation layers were not enabled for this baseline run.

Gaps found by code reading:

- `VulkanAdapter::supports_texture_compression_{bc,astc}_sliced_3d` hardcoded
  `false` (Dawn: BC sliced-3d follows `textureCompressionBC`; ASTC sliced-3d is
  probed per format with `vkGetPhysicalDeviceImageFormatProperties(TYPE_3D)`).
- `vkCreateDevice` `enabled_features` never sets `textureCompressionBC` /
  `ETC2` / `ASTC_LDR` (Dawn `DeviceVk.cpp:567-581` does).
- e2e coverage limited to single-block 2D round-trips.
- `cts-coverage.md` still describes compressed formats as `Unsupported` and
  refers to in-repo CTS ports that no longer exist.

## Slice log

| Slice | Scope | Status |
|---|---|---|
| 1 | R1 sliced-3d advertisement + R2 device feature enable | **DONE** |
| 2 | R3 3D HAL path + R4 e2e probes (+ same-image 3D copy layout fix) | dispatched |
| 3 | R5 docs + Phase Review + external CTS re-confirmation | R5 `cts-coverage.md` notes done; rest pending |

### Slice 1 (2026-09-20)

`yawgpu-hal/src/vulkan/mod.rs`: `supports_texture_compression_bc_sliced_3d` →
base BC; `supports_texture_compression_astc_sliced_3d` →
`astc_sliced_3d_supported(base, probe)` over `astc_ldr_formats()` (28 formats,
mapped through `format::map_texture_format`, probed with
`vkGetPhysicalDeviceImageFormatProperties(TYPE_3D, OPTIMAL, SAMPLED)`);
`enabled_texture_compression_features` seeds `enabled_features` at
`vkCreateDevice`. 5 new inline unit tests + the mirroring feature-enable test
extended. Gate: `cargo test --workspace` green; `cargo clippy --workspace
--all-targets --features vulkan -- -D warnings` clean. Real GPU (native Windows
driver, `VK_LAYER_KHRONOS_validation`): `e2e_vulkan_threading_audit` 14/14,
`e2e_vulkan_texture` 4/4 — no regression.

**Pre-existing defect surfaced by the validation-layer run (not caused by this
slice):** `e2e_vulkan_texture::vulkan_same_3d_texture_disjoint_copy` passes but
emits `VUID-vkCmdCopyImage-srcImage-09460` (+ the follow-on
`srcImageLayout-00128`): a same-image copy between disjoint depth slices of a 3D
image shares one array layer, so src/dst layouts must both be `GENERAL`; the HAL
uses `TRANSFER_SRC_OPTIMAL` / `TRANSFER_DST_OPTIMAL`. Folded into slice 2 since
R4 requires a VUID-clean run and 3D compressed T2T shares the path.
