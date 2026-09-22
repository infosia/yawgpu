# Block 107 — Vulkan: drop the eager `storeOp: Discard` clear (backlog A8) and give sampled + storage-bound subresources one layout (backlog A9)

Status: **IN PROGRESS (2026-09-23)** — S1 (A8) and S2 (A9) dispatched to
the coding agent as one handoff; e2e red-before-fix on the Windows native
NVIDIA host under `VK_LAYER_KHRONOS_validation`. Backlog items **A8** and
**A9** (`specs/tracking/backlog.md`), both raised by the Block 105 Phase
Review (MAJOR 2 → A8, m12 → A9). Host: Windows 11, NVIDIA RTX 5060 Ti,
native Vulkan driver.

## Problem

### A8 — the Vulkan Discard epilogue clears the wrong range on 3D images

`encode_render_pass_impl` (`yawgpu-hal/src/vulkan/encode.rs`) ends every
render pass with an **eager clear** of each colour attachment whose
`store == false` and of each discarded depth / stencil aspect:
transition the attachment range to `TRANSFER_DST_OPTIMAL`,
`vkCmdClearColorImage` / `vkCmdClearDepthStencilImage` to zero, and
transition back. It was added by F-107 (`08ba839`, 2026-06-15) when
yawgpu had **no subresource-initialization tracking**: WebGPU requires a
discarded attachment to read back as zero, and
`VK_ATTACHMENT_STORE_OP_DONT_CARE` on NVIDIA kept the rendered contents.

Since Block 104 (lazy zero-init Stage 2, R4) core tracks initialization
per `(texture, mip, layer, aspect)`: `storeOp: Discard` **un-marks** the
attachment aspect, and the next read (copy source, sampled / storage
binding, `loadOp: Load`, resolve source) triggers a `ClearTexture` or a
`Load → Clear` rewrite before the consumer runs. The eager HAL clear is
therefore redundant — Metal has no equivalent and passes the CTS
`api,operation,render_pass,storeOp*` trees through core's mechanism alone
— and, worse, it is **wrong for 3D colour attachments**: the clear range
comes from `color_attachment_subresource_range`, which uses
`target.depth_slice` as `baseArrayLayer`. A 3D image has one array layer,
so:

- `depthSlice > 0` → `baseArrayLayer >= arrayLayers`, invalid
  (`VUID-vkCmdClearColorImage-pRanges-01692` class); the driver may
  ignore it or corrupt memory;
- `depthSlice == 0` → the clear covers layer 0 of the mip, i.e. **every
  depth slice**, zeroing slices the pass never touched (data loss).

Both are reachable by valid WebGPU: a 3D texture rendered through
`depthSlice` with `storeOp: "discard"`.

### A9 — a subresource bound sampled **and** read-only storage in one pass

WebGPU allows one subresource to be bound as a sampled texture and as a
read-only storage texture in the same pass (both are `TextureAccess::Read`
in core's usage scope; `compatible_in_compute_scope` / `..render_scope`
accept `(Read, Read)`). Block 105 R4 transitions bound views before the
pass in two passes over `bind_textures`: `transition_sampled_textures`
puts every non-storage binding's range in `SHADER_READ_ONLY_OPTIMAL`,
then `transition_storage_textures` puts every storage binding's range in
`GENERAL`. "Storage follows sampled, so `GENERAL` wins" — but the sampled
descriptor written by `descriptor_info`
(`yawgpu-hal/src/vulkan/pipeline.rs`, `HalDescriptorBindingKind::Texture`)
unconditionally declares `SHADER_READ_ONLY_OPTIMAL`. The image is in
`GENERAL` when the shader reads it through that descriptor:
`VUID-VkDescriptorImageInfo-imageLayout-00344` (descriptor layout must
match the actual layout at the time of the access). Same class as the
read-only depth-stencil attachment sampled in the same pass (Block 105
"Known limitations", which this block does **not** address).

Dawn resolves this with usage-tracked layouts: a texture whose combined
usage in a pass includes storage is kept in `GENERAL` and every
descriptor that references it declares `GENERAL`
(`dawn/src/dawn/native/vulkan/TextureVk.cpp` `VulkanImageLayout`:
a usage set containing storage → `GENERAL`).

## Behaviour contract

### R1 — No eager Discard clear in the Vulkan render-pass epilogue (A8)

`encode_render_pass_impl` no longer clears discarded colour attachments
or discarded depth / stencil aspects after `cmd_end_render_pass`. The
attachment description keeps `VK_ATTACHMENT_STORE_OP_DONT_CARE` for
`store == false` (unchanged). Zero-on-next-read is guaranteed by Block 104
R4 (core un-marks the aspect) + R6 (`ClearTexture`), exactly as on Metal.

- The post-pass `layouts.set(attachment range, TRANSFER_SRC)` for colour /
  resolve attachments (Block 105 R4) stays; the depth-stencil attachment's
  tracked layout after the pass is what the render pass's `finalLayout`
  left it in (unchanged from the non-discard path today).
- `discarded_depth_stencil_aspects` and its unit test are removed if
  nothing else uses them (the `ClearTexture` path in
  `encode_texture_clear` builds its own aspect mask — verify before
  deleting; if it is shared, keep it).
- `color_attachment_subresource_range` remains the range used for the
  **attachment image view** (a `TYPE_2D` view of a 3D image at
  `baseArrayLayer = depth_slice` is correct there,
  `VUID-VkImageViewCreateInfo-image-06723` permits it) — it is no longer
  passed to any clear command.
- Block 105 R4's sentence "the `storeOp: Discard` epilogue clears use
  `transition_image_range` on the same attachment range" is superseded by
  this rule (spec 105 is annotated, not rewritten).

### R2 — A sampled binding whose subresources are also storage-bound in the same pass uses `GENERAL` (A9)

Define, per pass (compute pass: `pass.bind_textures`; render pass: the
stream-wide texture set `collect_vulkan_stream_textures` produces; tiled
subpass pass: the union of every subpass's `draw.bind_textures`, i.e. the
same slice the Block 105 R5 transitions consume):

> a sampled binding `b` (`storage_access.is_none()`) **shares** with
> storage iff some storage binding `s` (`storage_access.is_some()`) in the
> same pass references the same `VkImage` and
> `bound_view_subresource_range(b)` intersects
> `bound_view_subresource_range(s)` (mip **and** layer ranges overlap).

Then:

- `transition_sampled_textures` transitions the **whole** view range of a
  sharing sampled binding (minus the pass's attachment exclusions, as
  today) to `GENERAL` / `IMAGE_LAYOUT_GENERAL` instead of
  `SHADER_READ_ONLY_OPTIMAL`. Non-sharing sampled bindings are unchanged.
  Rationale: a descriptor declares one layout for its entire view, so a
  partially overlapping view (sampled mips 0..2, storage mip 1) must be
  uniformly `GENERAL`; `transition_storage_textures` then finds the
  overlapping part already in `GENERAL` (no barrier emitted for an
  identical layout — confirm `transition_image_range` short-circuits, or
  accept the redundant barrier if it does not; do not add a special case).
- `descriptor_info` for `HalDescriptorBindingKind::Texture` declares
  `GENERAL` for a sharing sampled binding and `SHADER_READ_ONLY_OPTIMAL`
  otherwise. **The decision must come from one function** shared by the
  transition and the descriptor write (e.g.
  `sampled_binding_shares_storage(bound, pass_textures) -> bool` in
  `encode.rs`, given the pass-wide texture slice), so the two can never
  disagree. `descriptor_info` needs the pass-wide slice (or a precomputed
  per-binding layout map) — it currently receives per-draw / per-dispatch
  bindings on some paths; plumb whichever is simpler, but the predicate
  must be evaluated against the **pass-wide** set, because the transitions
  are pass-wide.
- The tiled subpass path (`encode_subpass_render_pass`, `tiled` feature)
  follows the same rule through the same predicate.
- The Block 105 "Known limitations" bullet for A9 is removed; the
  read-only depth-stencil bullet stays.
- Layout policy stays layout-pair derived (no usage/stage tracking);
  `GENERAL`'s access / stage masks are those `vulkan/texture.rs` already
  maps.

### R3 — Unchanged

Core (no changes: usage-scope rules, Block 104 marks), Metal, GLES, Noop.
Non-3D Discard behaviour is observably unchanged (zero on next read via
core). No public API changes.

## Tests

### Unit (`#[cfg(test)]`, Noop / GPU-free)

- `encode.rs`: the sharing predicate — same image + overlapping mips and
  layers → true; same image, disjoint mips → false; same image, disjoint
  layers → false; different image → false; a storage binding never
  "shares" (only sampled bindings are queried); empty pass set → false.
- `encode.rs`: if `discarded_depth_stencil_aspects` is deleted, delete its
  test; if kept, its test stays.
- Existing Block 105 range tests unchanged.

### e2e (`yawgpu/tests/e2e_vulkan_layouts.rs`, `#[ignore]`, real Vulkan, run under `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`)

Add to the existing file (reuse its helpers; extend `create_texture`
for a 3D dimension or add a sibling helper):

- **e2e 4 (A8)** `vulkan_3d_attachment_discard_keeps_other_slices_and_zeroes_discarded_slice`:
  `rgba8unorm` 3D texture 8×8×3, mip 1, usage
  `RenderAttachment | CopySrc | CopyDst`; `writeTexture` a distinct
  colour per slice (slices 0, 1, 2); render pass with a `TYPE_2D`-view
  colour attachment on the whole texture with `depthSlice = 1`,
  `loadOp = Clear` (any clear colour), `storeOp = Discard`, one draw
  (any pipeline; the fullscreen triangle from e2e 1 is fine); then
  `copyTextureToBuffer` each slice (`origin.z = 0 / 1 / 2`). Expect slice
  0 and slice 2 **unchanged**, slice 1 **all zero**, no device error, and
  **no validation-layer output**. Before R1 this case emits
  `VUID-vkCmdClearColorImage-pRanges-01692` (or zeroes slices 0 / 2 when
  a second variant uses `depthSlice = 0` — cover both `depthSlice` values
  in one test body via a loop).
- **e2e 5 (A9)** `vulkan_compute_binds_one_texture_sampled_and_read_only_storage_in_one_pass`:
  `r32uint` 2D texture 4×4 (usage `TextureBinding | StorageBinding |
  CopyDst`), `writeTexture` a known texel; compute shader with
  `@binding(0) var t: texture_2d<u32>;` and
  `@binding(1) var s: texture_storage_2d<r32uint, read>;` writing
  `textureLoad(t, vec2i(0,0), 0).r + textureLoad(s, vec2i(0,0)).r` into a
  storage buffer; both bindings are views of the **same** texture (two
  `createView` calls or one view bound twice). Expect the sum, no device
  error, no validation output. Before R2 this case emits
  `VUID-VkDescriptorImageInfo-imageLayout-00344` (and the matching
  `VUID-vkCmdDispatch-None-08114`-class line).
- **e2e 6 (A9, render + partial overlap)**
  `vulkan_render_samples_mip_range_while_storage_reads_one_mip`: `r32uint`
  2D texture 4×4, 2 mips, usage `TextureBinding | StorageBinding |
  CopyDst`; render to a separate `rgba8unorm` target; fragment shader
  reads a `texture_2d<u32>` view covering mips 0..2
  (`textureLoad(t, .., 1)`) plus a `texture_storage_2d<r32uint, read>`
  view of mip 1 only; output encodes the two values; read back the colour
  target. Expect correct values, no validation output. Exercises the
  "whole sampled range goes `GENERAL`" clause. Read-only storage textures
  in the **fragment** stage are allowed by WebGPU; skip this case (with a
  comment) only if yawgpu core rejects it — report if so.

Each e2e must be shown **red before / green after** on this host (the
report records the VUID lines observed before the fix).

### Regression

- `cargo test --workspace` (Claude runs it), clippy default + `vulkan` +
  `vulkan,tiled` with `-D warnings`, fmt.
- All `e2e_vulkan_*` `--ignored` under the validation layer: 0 lines.
- CTS on this host (release `--features vulkan`, raw):
  `webgpu:api,operation,render_pass,storeOp:*`,
  `webgpu:api,operation,render_pass,storeop2:*`,
  `webgpu:api,operation,rendering,3d_texture_slices:*`,
  `webgpu:api,operation,resource_init,texture_zero_init:*`,
  `webgpu:api,operation,storage_texture,*`,
  `webgpu:api,validation,resource_usages,texture,in_pass_encoder:*`
  — fail 0 / crash 0.

## Slices

- **S1 (A8)** R1 + e2e 4.
- **S2 (A9)** R2 + unit tests + e2e 5, 6.

One coding-agent handoff covers both (small, same file); one commit per
slice on review.
