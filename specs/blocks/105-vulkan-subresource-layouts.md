# Block 105 — Vulkan image layouts are tracked per subresource

Status: **COMPLETE (2026-09-22)** — S1 `a00daf9`, S2 `10a59de`, S3 `6d391aa`, Phase Review fixes in the S4 commit (1 MAJOR fixed, 1 MAJOR deferred to backlog A8 as pre-existing / out of scope, 11 MINOR: 5 fixed, rest deferred or documented); final gate on the Windows native NVIDIA host: workspace 1093/0, Vulkan e2e 112/0 + passthrough 2/0, HAL 54/0, 0 validation-layer lines; MoltenVK re-confirmation done (Mac, 2026-09-22): the 10 CTS trees 274,884 pass / 9 fail = identical to the pre-Block-105 baseline (documented MoltenVK artifacts only), HAL 54/0, e2e 106/0, 0 layout-related validation lines — ledger "CTS" section. Backlog item **A5**
(`specs/tracking/backlog.md`): "Vulkan image-layout tracking is
per-texture, not per-subresource" (L), "the `tiled` subpass path lacks
the sampled-texture layout transition" (M); the third A5 item
(aspect-narrowed copy barriers, S) was already fixed by
`barrier_aspect_mask` in `974a818` and is subsumed by R2 below. Ledger:
`specs/tracking/vulkan-subresource-layout.md`.

## Problem

`VulkanTextureInner` tracks the image layout in one `AtomicU8`
(`yawgpu-hal/src/vulkan/texture.rs`), and every transition barriers the
**whole image** (`transition_image` / `transition_image_aspect`, full
`image_subresource_range`). WebGPU lets a command use different
subresources of one texture in different roles at once, so the single
state is wrong whenever subresources diverge. Concrete failures, each
reproducible under `VK_LAYER_KHRONOS_validation` on a native driver:

1. **Sampling one mip while rendering to another.** `encode_render_pass_impl`
   skips the sampled-texture transition for any image that is also an
   attachment of the pass (`attachment_images.contains(..)`), so the
   sampled mip stays in whatever layout it had (e.g. `TRANSFER_DST` after
   `writeTexture`) while its descriptor declares
   `SHADER_READ_ONLY_OPTIMAL` (VUID-vkCmdDraw-None-09600 class; the read
   is undefined).
2. **Stale tracked state after a render pass.** The post-pass
   `layout.store(TRANSFER_SRC)` for colour/resolve attachments marks
   every mip/layer as `TRANSFER_SRC`, but the `VkRenderPass` `finalLayout`
   only applied to the attached subresource; the next transition of any
   other subresource passes a wrong `oldLayout`
   (VUID-VkImageMemoryBarrier-oldLayout-01197).
3. **Copy to one layer, sample another.** A `writeTexture` into layer 0
   and a compute pass sampling layer 1 move the *whole* image to
   `SHADER_READ_ONLY` — over-broad (layer 0 is barriered for a read it is
   not part of) but self-consistent, so a following `copyTextureToBuffer`
   of layer 0 is valid today. (Measured in S2: e2e case 2 was already
   validation-clean before R4; it is kept as the regression guard for the
   per-range form, where the sampled transition no longer touches layer 0.)
4. **Tiled subpass passes never transition their bound textures.**
   `encode_subpass_render_pass` transitions attachments only; a
   `draw.bind_textures` entry last written by a copy is sampled in
   `TRANSFER_DST_OPTIMAL` (`specs/tracking/cts-full-sweep-0704-native-vulkan.md`
   "Known related gaps" 2).
5. **Same-image texture copies** work around the whole-image tracker by
   forcing `GENERAL`, splitting the two subresources with ad-hoc
   barriers and restoring `GENERAL` afterwards
   (`transition_copy_subresource`); the tracker cannot represent the
   real end state.

Dawn tracks `TextureSyncInfo` per subresource
(`third_party/dawn/src/dawn/native/vulkan/TextureVk.h:176`
`SubresourceStorage<TextureSyncInfo> mSubresourceLastSyncInfos`;
`TextureVk.cpp:1018` `TransitionUsageForPassImpl`, `:1133`
`TransitionUsageAndGetResourceBarrierImpl`) and emits one
`VkImageMemoryBarrier` per range whose last state differs.

## Design

Keep yawgpu's **layout-state** model (the tracked state *is* the
`VkImageLayout`; access/stage masks derive from it via
`access_mask_for_layout` / `stage_mask_for_layout`) — no usage /
shader-stage tracking as in Dawn — but give it a **mip x layer** axis:

- A `SubresourceLayouts` tracker replaces the `AtomicU8`. Combined
  depth-stencil images keep **one** state per subresource for both
  aspects: without `separateDepthStencilLayouts` (which yawgpu does not
  enable) Vulkan requires both aspects to share a layout and every
  barrier to name both (VUID-VkImageMemoryBarrier-image-03320), so an
  aspect axis would be unrepresentable anyway. 3D images have one
  array layer; swapchain images are 1 x 1.
- Every transition site names the **exact subresource range** the
  command touches — the same `(mip, layer)` values it already puts into
  its `VkBufferImageCopy` / `VkImageCopy` / clear range / attachment view.
- The tracker returns the **coalesced runs** of subresources whose
  current state differs from the target, and the barrier helper emits
  one `VkImageMemoryBarrier` per run inside a single
  `vkCmdPipelineBarrier`.
- Barriers always name the image's full `aspect_flags` (R2), which
  retires the requested-aspect parameter and `barrier_aspect_mask`.

## Behaviour contract

### R1 — `SubresourceLayouts` tracker

New module `yawgpu-hal/src/vulkan/layout.rs` (pure, no `ash::Device`,
fully unit-testable with no GPU under `--features vulkan`):

```rust
/// A `(mip, layer)` rectangle of an image's subresources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SubresourceRange {
    pub(super) base_mip_level: u32,
    pub(super) mip_level_count: u32,
    pub(super) base_array_layer: u32,
    pub(super) array_layer_count: u32,
}

/// One coalesced run of subresources that shared an old layout state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LayoutRun {
    pub(super) range: SubresourceRange,
    pub(super) old_state: u8,
}

/// Per-subresource image layout state (Block 105 R1).
#[derive(Debug)]
pub(super) struct SubresourceLayouts {
    mip_level_count: u32,
    array_layers: u32,
    states: std::sync::Mutex<Vec<u8>>,   // index = mip * array_layers + layer
}

impl SubresourceLayouts {
    pub(super) fn new(mip_level_count: u32, array_layers: u32) -> Self; // all IMAGE_LAYOUT_UNDEFINED
    pub(super) fn whole(&self) -> SubresourceRange;
    /// Records `new_state` over `range` and returns the runs that need a
    /// barrier, in ascending (mip, layer) order.
    pub(super) fn transition(&self, range: SubresourceRange, new_state: u8) -> Result<Vec<LayoutRun>, HalError>;
    /// Records `state` over `range` with no barrier (a `VkRenderPass`
    /// `finalLayout` already performed the transition).
    pub(super) fn set(&self, range: SubresourceRange, state: u8) -> Result<(), HalError>;
    /// The tracked state of one subresource (tests, debug assertions).
    pub(super) fn state(&self, mip_level: u32, array_layer: u32) -> Result<u8, HalError>;
}
```

- `transition` / `set` / `state` return the existing
  `texture_error("subresource range exceeds texture")` for a range
  outside `mip_level_count x array_layers` or with a zero count — never a
  panic (CLAUDE.md principle 3). Every caller today validates the mip
  (`validate_mip_level`) / origin-extent before reaching the tracker, so
  the error is a defence-in-depth path.
- **Run coalescing**: a run is a maximal rectangle
  `[mip .. mip + n) x [layer .. layer + m)` of subresources sharing one
  old state: first merge adjacent layers within each mip, then merge
  adjacent mips whose layer-run lists are identical. A whole-image
  transition of a uniform image therefore returns exactly **one** run,
  and a 2-mip image whose mip 0 is split in two states returns three
  runs (two for mip 0, one for mip 1). Runs are returned in ascending
  (mip, layer) order.
- **Which runs need a barrier (R6)**: a run whose old state equals
  `new_state` is dropped when `new_state` is a read-only layout
  (`IMAGE_LAYOUT_TRANSFER_SRC`, `IMAGE_LAYOUT_SHADER_READ_ONLY`,
  `IMAGE_LAYOUT_PRESENT`) or an attachment layout
  (`IMAGE_LAYOUT_COLOR_ATTACHMENT`, `IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT`,
  ordered by the `VkRenderPass` external dependencies of
  `render_pass_dependencies`), and **kept** when it is a write layout
  (`IMAGE_LAYOUT_TRANSFER_DST`, `IMAGE_LAYOUT_GENERAL`), so consecutive
  writes / a write then a read-write on the same subresource get an
  execution + memory dependency (Dawn `CanReuseWithoutBarrier`,
  `TextureVk.cpp:955`: reuse without a barrier only for read-only ->
  same read-only).
- The `Mutex` is held only inside these three methods; the ordering of
  two threads encoding submissions concurrently is unchanged from the
  atomic (pre-existing single-queue model, `specs/tracking/threading-audit.md`).
- Memory: `mip_level_count x array_layers` bytes (at most 15 x 2048 =
  30 KiB for the largest cube-array); no compression needed.

`VulkanTextureInner.layout: AtomicU8` becomes
`layouts: SubresourceLayouts` (constructed with the image's
`mip_level_count` / `array_layers`; swapchain images `new(1, 1)`). The
`IMAGE_LAYOUT_*` constants and `image_layout` / `access_mask_for_layout`
/ `stage_mask_for_layout` are unchanged.

### R2 — `transition_image_range` emits one barrier per run

In `texture.rs`:

```rust
pub(super) fn transition_image_range(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    texture: &VulkanTextureInner,
    range: SubresourceRange,
    new_layout: vk::ImageLayout,
    new_state: u8,
) -> Result<(), HalError>;

/// Whole-image convenience: `transition_image_range(.., texture.layouts.whole(), ..)`.
pub(super) fn transition_image(..) -> Result<(), HalError>;
```

- Calls `layouts.transition(range, new_state)`; with no runs it records
  nothing.
- Builds one `VkImageMemoryBarrier` per run: `oldLayout =
  image_layout(run.old_state)`, `newLayout`, `subresourceRange` = the
  run with `aspectMask = texture.aspect_flags` (always the image's full
  aspect set — a single-aspect image's `aspect_flags` *is* its only
  aspect, and a combined depth-stencil barrier must name both;
  VUID 03320), `srcAccessMask = access_mask_for_layout(old)`,
  `dstAccessMask = access_mask_for_layout(new)`.
- One `cmd_pipeline_barrier` for all runs: `srcStageMask` = OR of
  `stage_mask_for_layout(old)` over the runs, `dstStageMask =
  stage_mask_for_layout(new)`.
- `transition_image_aspect` (the requested-aspect parameter) and
  `barrier_aspect_mask` are removed; every former caller passes a
  `SubresourceRange` instead. `transition_copy_subresource` is removed
  (R3).

### R3 — Copy, clear and present sites name their subresources

| Site (`encode.rs`) | Range |
|---|---|
| `transition_swapchain_image_to_present` | whole (1 x 1) |
| `encode_texture_clear` (colour and depth-stencil arms) | `clear.mip_level`, layers `clear.base_array_layer .. + array_layer_count` (3D: layer 0, count 1) — the values `texture_clear_subresource_range` already computes |
| `encode_compressed_texture_clear` | the clear's mip; layers as above |
| `encode_buffer_to_texture` / `encode_texture_to_buffer` | `copy.mip_level`; layers `copy.origin.z .. + extent.depth_or_array_layers` for 1D/2D, `0..1` for 3D — the values `texture_copy_subresource_layers` uses |
| `encode_texture_to_texture`, distinct images | source range -> `TRANSFER_SRC`, destination range -> `TRANSFER_DST` (same derivation) |
| `encode_texture_to_texture`, same image, `texture_copy_layouts` = transfer pair (disjoint subresources) | source range -> `TRANSFER_SRC`, destination range -> `TRANSFER_DST`; **no** `GENERAL` detour, no `transition_copy_subresource` |
| `encode_texture_to_texture`, same image, `texture_copy_layouts` = `GENERAL` (overlapping subresources) | the union rectangle of both ranges -> `GENERAL` (R6 keeps the write->write barrier between consecutive same-subresource copies) |
| compressed temporary-buffer path | source range -> `TRANSFER_SRC` before the image->buffer copy, destination range -> `TRANSFER_DST` before the buffer->image copy |

A pure helper per derivation (`clear_subresource_range(dimension, clear)`,
`copy_subresource_range(dimension, mip, z, depth_or_array_layers)`,
`attachment_subresource_range(mip, layer)`, `union_subresource_range(a, b)`)
is unit-tested with no GPU. `texture_clear_write_after_write_barrier`
(F-138) stays as is.

### R4 — Render passes transition attachments and bound views per subresource

In `encode_render_pass_impl`:

- Colour attachment: `transition_image_range` on
  (`target.mip_level`, layer = `array_layer` for 1D/2D or layer 0 for 3D
  — `attachment_subresource_range_of`; a 3D image has one array layer,
  so the barrier names layer 0 whatever `depth_slice` the view targets) to
  `COLOR_ATTACHMENT` / `GENERAL` (framebuffer fetch). Resolve:
  (`resolve_mip_level`, `resolve_array_layer`) -> `COLOR_ATTACHMENT`.
  Depth-stencil: (`mip_level`, `array_layer`) ->
  `DEPTH_STENCIL_ATTACHMENT`.
- Bound textures (`transition_sampled_textures` /
  `transition_storage_textures`, also used by the compute path): the
  range is the **view's** `base_mip_level .. + mip_level_count` x
  `base_array_layer .. + array_layer_count`. The image-level
  `attachment_images` skip becomes a **subresource-level exclusion**:
  the helper receives the list of `(vk::Image, SubresourceRange)`
  attachment subresources of this pass and transitions only the part of
  the bound range that does not intersect any of them (split the bound
  range into non-overlapping sub-rectangles; a range contained in an
  attachment range is skipped whole). WebGPU validation guarantees a
  subresource is never both sampled and written as an attachment in one
  pass, so the intersecting case is the read-only depth-stencil
  attachment sampled in the same pass — kept unchanged (skipped) and
  documented below. The compute path passes an empty exclusion list
  (unchanged semantics).
- After `cmd_end_render_pass`: `layouts.set(range, TRANSFER_SRC)` for
  each colour and resolve **attachment range only** (the `finalLayout`
  applied to that subresource alone); the `storeOp: Discard` epilogue
  clears use `transition_image_range` on the same attachment range
  (colour) / the depth-stencil attachment range.

### R5 — Tiled subpass passes (`tiled` feature)

In `encode_subpass_render_pass`:

- Colour attachments (`subpass_attachment_views` uses each texture's
  default whole-image view, so the attachment range is `layouts.whole()`):
  `transition_image_range` as today (`GENERAL` when the slot feeds an
  input attachment, else `COLOR_ATTACHMENT`), then `layouts.set(whole,
  subpass_color_tracked_layout(transient))` after the pass — unchanged
  semantics, new API.
- **Depth-stencil attachment** (Phase Review MAJOR 1, pre-existing): it
  must not take the colour path. Transition it to
  `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` / `IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT`
  before the pass and record **no** post-pass state (the tiled
  `VkRenderPass` declares `initialLayout = finalLayout =
  DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, so the tracked state already matches).
  The sentinel slot `u32::MAX` that routed it through the colour loop is
  gone; the attachment stays in the S3 exclusion list.
- **New**: before `cmd_begin_render_pass`, collect every
  `draw.bind_textures` of every subpass and run the R4 sampled + storage
  transitions with the attachment images' whole ranges as the exclusion
  list (barriers are illegal inside the render pass instance, exactly as
  `encode_render_pass_impl`). Closes "Known related gaps" 2 of the 0704
  sweep.

### R6 — Same-layout write barriers

Stated in R1: a `transition` into `TRANSFER_DST` or `GENERAL` emits a
barrier for runs already in that state (write-after-write / read-after-
write on the same subresource within one command buffer); a transition
into a read-only layout or an attachment layout does not.

### R7 — Unchanged

`IMAGE_LAYOUT_*` states and their access/stage mapping; `VkRenderPass`
`initialLayout` / `finalLayout` choices (`render_color_attachment_layout`,
`subpass_color_final_layout`); descriptor image layouts
(`SHADER_READ_ONLY_OPTIMAL` sampled, `GENERAL` storage / input
attachment); `texture_copy_layouts`; the Metal and GLES backends; core.

## Tests

- **HAL inline, no GPU (`cargo test -p yawgpu-hal --features vulkan`)**,
  `layout.rs`: `new` is all-`UNDEFINED`; a sub-range transition returns
  one run with the old state and leaves the rest untouched (`state`);
  a range over mixed states returns coalesced runs in ascending order
  with the right old states (2 mips x 4 layers fixture: mip 0 layers
  0..2 `TRANSFER_DST`, layers 2..4 `SHADER_READ_ONLY`, mip 1 uniform ->
  three runs); a uniform whole-image transition returns exactly one run
  covering all mips and layers; read-only same-state -> no run,
  `TRANSFER_DST` / `GENERAL` same-state -> run (R6), `COLOR_ATTACHMENT`
  same-state -> no run; `set` records without runs; out-of-range /
  zero-count -> `Err`; 3D-style `new(n, 1)` indexing. `encode.rs` /
  `texture.rs`: the range derivation helpers (R3 table incl. 3D ->
  layer 0..1, union) and the R4 exclusion split (bound
  `mips 0..2 x layers 0..1` minus attachment `(0, 0)` -> `[(1, 0)]`;
  contained -> empty; disjoint -> whole).
- **Real-GPU e2e, Vulkan (`yawgpu/tests/e2e_vulkan_layouts.rs`, C ABI,
  `#[ignore]`; the coding agent writes them, Claude runs them under
  `VK_LAYER_KHRONOS_validation` on the native NVIDIA host — the
  validation layer is the oracle for layout mismatches, so every case
  must run with **zero** layer messages)**:
  1. `vulkan_render_to_mip0_while_sampling_mip1_reads_sampled_color`
     — 2-mip `rgba8unorm` (`TEXTURE_BINDING | RENDER_ATTACHMENT |
     COPY_SRC | COPY_DST`); `writeTexture` fills mip 1 with a solid
     colour; a render pass targets a mip-0 view and its fragment shader
     `textureLoad`s a mip-1 view (`baseMipLevel: 1, mipLevelCount: 1`);
     `copyTextureToBuffer` of mip 0 reads the colour (problems 1 + 2).
  2. `vulkan_copy_to_layer0_then_sample_layer1_then_copy_layer0_round_trips`
     — 2-layer texture; `writeTexture` into both layers; a compute pass
     samples layer 1 into a storage buffer; `copyTextureToBuffer` from
     layer 0 returns the written bytes (problem 3).
  3. `vulkan_tiled_subpass_samples_texture_written_by_copy` (`tiled`
     feature, next to `e2e_vulkan_tiled.rs`'s helpers) — a one-subpass
     tiled pass whose draw samples a texture last written by
     `writeTexture`; the attachment readback equals the sampled colour
     (problem 4).
  4. Existing `vulkan_same_texture_copy_chains_preserve_layout_tracking`
     (`e2e_vulkan_texture.rs`) keeps passing (problem 5, now without the
     `GENERAL` detour).
- **Full Vulkan gate**: `cargo test -p yawgpu --features vulkan,tiled
  --no-fail-fast -- --ignored` (+ the `shader-passthrough` binary) on
  the native NVIDIA host under the validation layer: all green, zero
  VUID lines (baseline 106/106, 0 lines, 2026-09-22). `cargo test
  --workspace`, clippy `--features vulkan,tiled --all-targets -D
  warnings`, fmt.
- **CTS** (MoltenVK, next Mac session): `webgpu:api,operation,command_buffer,*`,
  `rendering,*`, `resource_init,*` — no regression vs the Block 104
  baseline. Not a gate for landing S1–S3 on Windows; logged in the
  ledger when run.

## Slices

- **S1 tracker + transfer sites** (R1, R2, R3, R6, R7): `layout.rs`,
  `VulkanTextureInner.layouts`, all copy / clear / present sites, removal
  of `transition_image_aspect` / `barrier_aspect_mask` /
  `transition_copy_subresource`; the render / subpass sites keep their
  behaviour through `transition_image` + `set(whole, ..)` so the tree
  stays green between slices. Unit tests above.
- **S2 render passes** (R4) + e2e 1, 2.
- **S3 tiled subpasses** (R5) + e2e 3.
- **S4** full gate, backlog A5 closure, Phase Review (fresh-context
  reviewer over the S1–S3 range), CTS on the Mac when next available.

## Known limitations (documented)

- A **read-only depth-stencil attachment that is also sampled in the
  same pass** stays in `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` while its
  sampled descriptor declares `SHADER_READ_ONLY_OPTIMAL` (pre-existing;
  the subresource-level exclusion in R4 preserves today's skip). Fixing
  it needs a `DEPTH_STENCIL_READ_ONLY_OPTIMAL` state, matching
  `VkRenderPass` initial / final layouts and a per-binding descriptor
  layout — a follow-up backlog item, not part of A5.
- A subresource bound both as a **sampled texture and a read-only
  storage texture in one pass** (allowed by WebGPU) ends in `GENERAL`
  ("storage follows sampled so `GENERAL` wins") while the sampled
  descriptor declares `SHADER_READ_ONLY_OPTIMAL` — the same class as the
  read-only depth-stencil case (pre-existing policy; a fix would declare
  `GENERAL` on the sampled descriptor when the same subresource is also
  storage-bound). Backlog A9.
- Layout state is not usage / stage tracked (Dawn's `shaderStages`
  reuse optimisation is not replicated); barriers are derived from the
  layout pair alone, as today.
- Cross-thread submission ordering is the single-queue model's
  (unchanged).
