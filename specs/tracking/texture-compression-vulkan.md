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
| 2 | R3/R3b 3D + mip-edge HAL path, R4 e2e probes, same-image 3D copy layout fix | **DONE** |
| 3 | R5 docs + Phase Review + external CTS re-confirmation | docs + Phase Review **DONE**; external CTS re-confirmation pending |

### Phase Review (2026-09-20) — Clean Review of `1001859..f639ab2`

Fresh no-context reviewer: **0 CRITICAL / 0 MAJOR / 8 MINOR**. Confirmed sound
by code reading: tracked vs real image layout on every `encode_texture_to_texture`
path (different image; same image shared subresource; same image disjoint
mips/layers; temporary-buffer fallback incl. same image), barrier aspect masks
for depth+stencil, fallible work ordered before the first barrier,
`compressed_copy_extent` cannot newly reject a core-valid copy, temporary
buffer size/regions/lifetime/OOM, advertised vs enabled feature agreement, no
panics in added library code, E1–E10 present with feature-naming SKIP lines.

| # | Site | Finding | Triage |
|---|---|---|---|
| 1 | `vulkan/mod.rs` `astc_ldr_formats` | 28 `HalTextureFormat` variants enumerated by hand although R1 said "derived from the same table" | **Spec fix (Claude):** R1 reworded — the `vk::Format`s must come through `map_texture_format`; a hand enumeration of the closed ASTC LDR set is acceptable when pinned by the 28-unique-entries unit test |
| 2 | `vulkan/mod.rs:434,465` | changed `supports_*_sliced_3d` pub fns have no direct inline test (principle 1) | **FIX** — real-adapter inline test |
| 3 | `vulkan/mod.rs:465` | 28 format-property queries on every feature enumeration | **FIX** — cache per adapter; R1 updated to require it |
| 4 | e2e E2 | constant-byte mip data cannot detect swapped blocks/rows or wrong `bufferRowLength` | **FIX** — per-byte-distinct data; E2 row updated |
| 5 | e2e E5 | spec said "sRGB view", test creates an sRGB-format texture | **Spec fix (Claude):** E5 row reworded to the intended meaning (sRGB format texture, full-pixel assertion) |
| 6 | `vulkan/encode.rs` temp buffer | scratch buffer is host-visible mapped memory → mismatched-edge compressed T2T round-trips through system memory on discrete GPUs | **DEFERRED** — correct, affects only the rare Dawn-fallback path; a device-local scratch allocation path is a separate HAL allocator change |
| 7 | `vulkan/encode.rs` same-image path | up to 3+2 separate `vkCmdPipelineBarrier` calls could be batched | **DEFERRED** — redundant work only on the rare same-image T2T path; batching would complicate the tracker-restore logic for no measurable gain |
| 8 | `vulkan/mod.rs:432` | awkward doc comment | **FIX** |

**Fix round:** #2 (`vulkan_adapter_sliced_3d_compression_matches_base_support_and_is_cached`,
real-adapter inline test — executed here, not skipped), #3
(`Arc<OnceLock<bool>>` on `VulkanAdapter`, shared across clones), #4, #8 fixed;
#1, #5 resolved by spec rewording. **Final gate:** `cargo test --workspace`
green; clippy `-D warnings` clean with and without `--features vulkan`; under
`VK_LAYER_KHRONOS_validation` — compression e2e 10/10, `e2e_vulkan_texture` 5/5,
`e2e_vulkan_threading_audit` 14/14, zero VUID / Validation Error lines. No open
CRITICAL/MAJOR → **Block 73 COMPLETE**.

### External CTS re-confirmation (2026-09-20, webgpu-native-cts, native Vulkan, NVIDIA RTX 5060 Ti)

yawgpu rebuilt at `b77d531` (`--release --features vulkan`, `target-vulkan`),
DLL + matching `tint_shim.dll` dropped into the suite's `build-yawgpu/Release`;
run with `--isolate --workers 6 --expectations expectations/yawgpu-vulkan.txt`.

| Trees | Result |
|---|---|
| `api,operation,command_buffer,copyTextureToTexture:*` + `api,operation,command_buffer,image_copy:*` | pass=2846 skip=5 fail=0 crash=0 |
| `api,validation,image_copy,*` + `encoding,cmds,copyTextureToTexture:*` + `createTexture:*` + `capability_checks,features,texture_formats:*` | pass=8971 skip=2483 fail=0 crash=0 (1042 passing cases name a compressed format; 57 are 3D BC) |

- **Sliced-3d is now exercised by the CTS.** Against the suite's previous
  yawgpu DLL (2026-07-04 build, so a coarse baseline, not the immediate
  pre-change commit) `createTexture` + `texture_formats` went pass 2385→2456,
  skip 1009→938: exactly **71 cases flipped skip→pass, 0 regressions** —
  `texture_compression_bc_sliced_3d` (14), `texture_size,3d_texture,compressed_format`
  (14), `mipLevelCount,format` (14), `texture_usage` (14), `sampleCount,…` (14),
  `zero_size_and_usage` (1), all `supportsBC=true;supportsBCSliced3D=true` /
  3D BC variants.
- Skips are adapter-feature skips: ETC2 540, ASTC 1484 (absent on this GPU), BC
  364 (the `supportsBC=false` negative parameterisations), 95 other.
- **Coverage gap in the suite, not in yawgpu:** the compressed *operation*
  cases (`copyTextureToTexture:color_textures,compressed,{non_array,array}`) are
  still "deferred" (unported) in webgpu-native-cts, and no compressed-format case
  runs in the operation trees (0 of 2846). So R3b (mip-edge clamp + temporary
  buffer fallback) is verified only by the in-repo e2e probes E2/E3/E10, not by
  the CTS oracle comparison. Porting those two tests in webgpu-native-cts is the
  follow-up that would close this.

**Open follow-ups (outside this block's gate):**
- E8 / E9 (ETC2, ASTC, ASTC sliced-3d) need a run on hardware exposing those
  families (Android Vulkan, or MoltenVK on Apple Silicon).
- Port `copyTextureToTexture:color_textures,compressed,*` in webgpu-native-cts
  so R3b gets oracle (Dawn) comparison on real hardware.
- Deferred MINOR #6 (device-local scratch buffer) and #7 (barrier batching).

### Slice 2 (2026-09-20)

New `yawgpu/tests/e2e_vulkan_texture_compression.rs` (E1–E10) + shared
`yawgpu/tests/common/vulkan.rs` helpers; regression probe added to
`e2e_vulkan_texture.rs`. Vulkan HAL defects found and fixed
(`yawgpu-hal/src/vulkan/encode.rs`, `texture.rs`), each with an inline unit test:

| Defect | Fix | Unit test |
|---|---|---|
| Same-image T2T on a shared subresource (3D depth slices) used `TRANSFER_SRC`/`TRANSFER_DST` → `VUID-vkCmdCopyImage-srcImage-09460` (pre-existing) | both layouts `GENERAL` only when same image + same mip + overlapping layers; disjoint mips/layers keep transfer layouts and restore the tracked state | `texture_copy_layouts_require_general_only_for_shared_subresources` |
| `GENERAL` layout sync lacked transfer access/stage; consecutive same-image copies had no write→read dependency | transfer access + stage for `GENERAL`, barrier emitted even for `GENERAL→GENERAL` | `general_layout_synchronizes_same_image_transfer_reads_and_writes` |
| Compressed B2T/T2B/T2T passed the physical (block-rounded) mip extent → VUID 07971/07972 on non-block-aligned mips (E2) | `compressed_copy_extent` clamps `imageExtent` to the logical mip edge, buffer pitches stay in block units (R3b) | `compressed_copy_extent_clamps_mip_edges_and_preserves_depth_and_pitches`, `compressed_copy_extent_rejects_invalid_ranges_without_panicking` |
| Compressed T2T whose source/destination logical edges differ has no legal `vkCmdCopyImage` extent | **Revision round:** first implementation returned `HalError` — rejected in review (valid WebGPU op on Tier 1, CTS regression risk). Now texture→temporary buffer→texture, Dawn `RecordCopyImageWithTemporaryBuffer` parity; buffer retained via the existing `RetainedResource::Buffer` → `RetireRing` until fence completion | `compressed_copy_needs_temporary_buffer_only_for_mismatched_compressed_extents`, `compressed_temporary_copy_preserves_blocks_mip_edges_and_layers` |

Gate (Claude, native Windows driver, `VK_LAYER_KHRONOS_validation`):
`cargo test --workspace` green; clippy `-D warnings` clean with and without
`--features vulkan`; `e2e_vulkan_texture_compression` 10/10, `e2e_vulkan_texture`
5/5, `e2e_vulkan_threading_audit` 14/14; **zero VUID / Validation Error lines**.
Executed on this GPU: E1–E7, E10. Self-skipped (feature absent on this desktop
GPU): E8 (`texture-compression-etc2`, `texture-compression-astc`), E9
(`texture-compression-astc-sliced-3d`) — **ETC2/ASTC paths, including the ASTC
sliced-3d probe loop, remain unverified on real hardware** until run on a
device exposing them (Android / Apple Silicon via MoltenVK).

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
