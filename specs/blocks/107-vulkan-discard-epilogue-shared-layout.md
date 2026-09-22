# Block 107 — Vulkan: drop the eager `storeOp: Discard` clear (backlog A8) and give sampled + storage-bound subresources one layout (backlog A9)

Status: **COMPLETE (2026-09-23)** — S1 + S2 `3f809b1` (one coding-agent
handoff, HAL + tests, committed together because the two slices touch the
same functions); Phase Review: 1 MAJOR (F1, pairwise → image-wide sharing
rule) + 5 MINOR, fixed in the S3 commit that follows `3f809b1` (F5
deferred with rationale, F2 documented) — table below. S3 gates: HAL
`--lib` 261/0 (`vulkan`) and 269/0 (`vulkan,tiled`), `e2e_vulkan_layouts`
7/7 under the layer with 0 VUID lines (e2e 7 was red before the fix:
`VUID-VkDescriptorImageInfo-imageLayout-00344` + `vkCmdDispatch-None-08114`
with `GENERAL` declared while mip 0 sat in `SHADER_READ_ONLY_OPTIMAL`),
`e2e_vulkan_lazy_init` 7/7, `e2e_vulkan_tiled` 4/4, `e2e_vulkan_compute`
3/3 under the layer, `cargo test --workspace` green, fmt + clippy default /
`vulkan` / `vulkan,tiled` clean. S1 + S2 gates on the Windows native NVIDIA
host: `cargo test --workspace` green; fmt + clippy default / `vulkan` /
`vulkan,tiled` clean; HAL `--ignored` 54/0 and every `e2e_vulkan_*` binary
green under `VK_LAYER_KHRONOS_validation` with **0 validation lines**
(`e2e_vulkan_layouts` 6/6 incl. the three new cases, red-before-fix VUIDs
recorded in the Tests section); CTS raw (release `--features vulkan`, CTS
`2f0fb9f`): `render_pass,storeOp` 26/0, `storeop2` 2/0,
`rendering,3d_texture_slices` 7/0, `storage_texture,*` 6/0,
`resource_usages,texture,in_pass_encoder` 1,578/0, `resource_init,*`
8/0, `command_buffer,*` 170,202/0 (35 skip), `rendering,*` 1,127/0
(8 suite-deferred skips) — fail 0 / crash 0 everywhere. Backlog items
**A8** and **A9** (`specs/tracking/backlog.md`), both raised by the Block
105 Phase Review (MAJOR 2 → A8, m12 → A9). Host: Windows 11, NVIDIA RTX
5060 Ti, native Vulkan driver.

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
  (observed on the Khronos layer as
  `VUID-vkCmdClearColorImage-baseArrayLayer-01472` +
  `VUID-vkCmdClearColorImage-pRanges-01693`); the driver may ignore it
  or corrupt memory;
- `depthSlice == 0` → the clear covers layer 0 of the mip, i.e. **every
  depth slice** — redundant with core's lazy clear (see below), not a
  divergence from the oracle.

Both are reachable by valid WebGPU: a 3D texture rendered through
`depthSlice` with `storeOp: "discard"`.

**Oracle note (found while executing, 2026-09-23).** Dawn tracks 3D
texture initialization **per mip level** (`CommandBuffer.cpp`
`LazyClearRenderPassAttachments`: "rendering to a single depthSlice marks
the entire mip level as initialized", and `storeOp: Discard` calls
`SetIsSubresourceContentInitialized(false, view range)` on that whole
mip). Discarding one depth slice therefore leaves the **whole mip**
uninitialized in Dawn, and the next read lazily zeroes every slice.
yawgpu core mirrors this (`tracked_init_layer` maps every D3 slice to
layer 0). The spec's first draft expected sibling slices to survive a
Discard; that is stricter than the oracle and was dropped — e2e 4 asserts
Dawn's behaviour (all slices zero, no validation output).

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
- **Present is the one consumer core does not clear for** (Phase Review
  F2): `wgpuSurfacePresent` hands the swapchain image to the HAL without
  an `InitOverlay` check, so a surface texture rendered with
  `storeOp: Discard` and presented without any further use now reaches
  the compositor with `STORE_OP_DONT_CARE` contents on Vulkan (it was
  zeroed by the removed epilogue). This matches Metal today and is not
  observable by any WebGPU read; Dawn does not clear on present either.
  Recorded as the only behaviour change of R1 outside the fixed VUIDs.

### R2 — A sampled binding whose subresources are also storage-bound in the same pass uses `GENERAL` (A9)

Define, per pass (compute pass: `pass.bind_textures`; render pass: the
stream-wide texture set `collect_vulkan_stream_textures` produces; tiled
subpass pass: the union of every subpass's `draw.bind_textures`, i.e. the
same slice the Block 105 R5 transitions consume):

> a sampled binding `b` (`storage_access.is_none()`) **shares** with
> storage iff some storage binding `s` (`storage_access.is_some()`) in the
> same pass references the same `VkImage`. The rule is **image-wide**:
> subresource ranges are deliberately *not* compared.

*(Amended after the Phase Review, F1. The first draft compared
subresource ranges pairwise; that is not transitive across sampled views
of one image — sampled V1 = mips 0..2, sampled V2 = mip 0, storage S =
mip 1: V1 shares, V2 does not, and mip 0 ends in whichever layout the
last-visited binding chose while the other binding's descriptor declares
the opposite (VUID-00344 again). Dawn's `VulkanImageLayout` decides per
texture from the pass's combined usage — any storage usage → `GENERAL`
for every sampled descriptor of that texture — so yawgpu now does the
same. Cost: a sampled view of mip 0 while storage reads mip 1 is
`GENERAL` instead of `SHADER_READ_ONLY_OPTIMAL`, which is what Dawn
does too.)*

Then:

- `transition_sampled_textures` transitions the **whole** view range of a
  sharing sampled binding (minus the pass's attachment exclusions, as
  today) to `GENERAL` / `IMAGE_LAYOUT_GENERAL` instead of
  `SHADER_READ_ONLY_OPTIMAL`. Non-sharing sampled bindings are unchanged.
  Because the rule is image-wide, every sampled binding of a
  storage-bound image reaches the same layout regardless of visiting
  order; `transition_storage_textures` then finds the storage range
  already in `GENERAL` (a redundant same-layout barrier is acceptable; do
  not add a special case).
- The predicate's pure core takes only `VkImage` handles
  (`image_shares_storage(image, storage_images)`-shape helper next to
  `SubresourceRange` in `layout.rs`, or equivalent) so it can be
  unit-tested without constructing a fake device; the
  `HalBoundTexture` adapter (`sampled_binding_shares_storage`) filters
  on `storage_access` and unwraps the Vulkan texture. The adapter
  returns `false` for a non-Vulkan texture or a failed `inner()` only
  because every caller fails on that same condition first — its doc
  comment says so (Phase Review F4).
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

- the pure image-level helper (plain `vk::Image` handles, no device):
  same image among the storage images → true; same image listed with
  disjoint subresource ranges → still true (image-wide rule, the F1
  scenario: two sampled views of one image must agree); different image →
  false; empty storage set → false.
- `encode.rs` adapter: a storage binding queried → false (only sampled
  bindings share). If an adapter test needs a Vulkan-backed
  `HalBoundTexture`, do **not** build a fake `VulkanDeviceInner` with
  inert dispatch tables (Phase Review F3: sound today but fragile); the
  pure helper carries the coverage.
- `encode.rs`: if `discarded_depth_stencil_aspects` is deleted, delete its
  test; if kept, its test stays.
- Existing Block 105 range tests unchanged.

### e2e (`yawgpu/tests/e2e_vulkan_layouts.rs`, `#[ignore]`, real Vulkan, run under `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`)

Add to the existing file (reuse its helpers; extend `create_texture`
for a 3D dimension or add a sibling helper):

- **e2e 4 (A8)** `vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip_without_validation_errors`:
  `rgba8unorm` 3D texture 8×8×3, mip 1, usage
  `RenderAttachment | CopySrc | CopyDst`; `writeTexture` a distinct
  colour per slice (slices 0, 1, 2); render pass with a colour attachment
  view of the whole texture at `depthSlice = 1` (second variant:
  `depthSlice = 0`, both in one test body), `loadOp = Clear`,
  `storeOp = Discard`, one fullscreen-triangle draw; then
  `copyTextureToBuffer` each slice (`origin.z = 0 / 1 / 2`). Expect
  **every slice all zero** (Dawn's per-mip 3D tracking — see the oracle
  note above), no device error, and **no validation-layer output**.
  Before R1 the `depthSlice = 1` variant emitted
  `VUID-vkCmdClearColorImage-baseArrayLayer-01472` and
  `VUID-vkCmdClearColorImage-pRanges-01693`.
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
- **e2e 7 (A9, F1 scenario)** `vulkan_compute_two_sampled_views_of_one_image_agree_when_storage_reads_another_mip`:
  `r32uint` 2D texture 4×4, 2 mips (usage `TextureBinding |
  StorageBinding | CopyDst`); compute shader binding `texture_2d<u32>`
  view A = mips 0..2, `texture_2d<u32>` view B = mip 0 only,
  `texture_storage_2d<r32uint, read>` view S = mip 1 only; writes
  `A(mip 0) + A(mip 1) + B(mip 0) + S` into a storage buffer. Expect the
  sum, no device error, no validation output. Under the pairwise rule B
  declared `SHADER_READ_ONLY_OPTIMAL` while A left mip 0 in `GENERAL`
  (or vice versa) → VUID-00344.
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
  `webgpu:api,operation,resource_init,*`,
  `webgpu:api,operation,storage_texture,*`,
  `webgpu:api,validation,resource_usages,texture,in_pass_encoder:*`,
  plus the Block 105 re-confirmation trees `command_buffer,*` and
  `rendering,*` — fail 0 / crash 0.

## Phase Review (2026-09-23, fresh-context reviewer over `aaef70c..3f809b1`)

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| F1 | MAJOR | The R2 predicate compared subresource ranges pairwise, which is not transitive across sampled views of one image (sampled mips 0..2 + sampled mip 0 + storage mip 1: mip 0's layout depends on visiting order, one descriptor declares the other layout — VUID-00344 again); unit tests never had a second sampled view | **Fixed (spec + code)** — R2 amended to Dawn's image-wide rule (any storage binding of the image in the pass → `GENERAL` for every sampled binding of that image); pure `vk::Image`-level helper in `layout.rs` + adapter; unit tests on the helper; new e2e 7 reproduces the scenario red-before-fix |
| F2 | MINOR | `wgpuSurfacePresent` is the one consumer with no `InitOverlay` check: a Discarded, then presented, surface texture was zeroed by the removed epilogue and now presents `DONT_CARE` contents on Vulkan (Metal already behaves so; not observable by WebGPU reads) | **Documented (spec R1)** — matches Metal and Dawn; no code change |
| F3 | MINOR | Predicate unit tests built a fake `VulkanDeviceInner` with inert `ash` dispatch tables — sound (ash substitutes panicking stubs, all reachable drops are supplied) but fragile and technically outside `load_with`'s documented contract | **Fixed** — fixture removed; the pure helper carries the coverage |
| F4 | MINOR | The adapter silently returns `false` for a non-Vulkan texture / failed `inner()`; safe only because every caller fails first | **Fixed (docs)** — stated in the adapter's doc comment |
| F5 | MINOR | `#[allow(clippy::too_many_arguments)]` on `update_render_descriptor_sets`; `bind_textures` (per-draw) and `pass_textures` (pass-wide) sit side by side with the same type | **Deferred with rationale** — same pre-existing style as `encode_vulkan_render_commands`; a `RenderBindings<'_>` struct is a refactor across the render-stream call chain, out of this block's scope; tracked with refactor-dedup deferrals (backlog E4) |
| F6 | MINOR | e2e 4 printed the backlog id (`eprintln!("A8 …")`); its name claimed "without validation errors" although the layer check is performed by the human running under `VK_LAYER_KHRONOS_validation` | **Fixed** — `eprintln!` removed, renamed, doc comment states how the layer check is run |

Reviewer confirmed (no finding): every core read path clears before the
consumer (copies, compute / render / bundle / tiled bindings, `Load →
Clear` rewrite, 3D whole-mip clear, Discard un-mark) — present is the sole
gap (F2); depth-stencil tracker consistent after the pass without the
epilogue (`initial == final == DEPTH_STENCIL_ATTACHMENT_OPTIMAL`); the
pass-wide slice is identical on the transition and descriptor sides for
compute, render incl. bundles, and tiled; interval arithmetic correct; a
`WriteOnly` / `ReadWrite` storage sharer is unreachable (core usage scope);
attachment exclusions never interact with the `GENERAL` choice; no dead
code left; docs / no-panic conventions met; e2e 5 / 6 not vacuous.

## Slices

- **S1 (A8)** R1 + e2e 4.
- **S2 (A9)** R2 + unit tests + e2e 5, 6.
- **S3 (Phase Review fixes)** F1 / F3 / F4 / F6 + e2e 7.

One coding-agent handoff covers both (small, same functions); landed as
one commit on review because the R1 removal and the R2 plumbing edit the
same `encode_render_pass_impl` / transition helpers and cannot be split
cleanly.
