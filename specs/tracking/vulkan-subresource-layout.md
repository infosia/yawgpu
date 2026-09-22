# Vulkan per-subresource layout tracking — Block 105 ledger (backlog A5)

Spec: `specs/blocks/105-vulkan-subresource-layouts.md`. Host: Windows 11,
native NVIDIA Vulkan driver, `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

## Baseline (2026-09-22, `97e1713`)

- `cargo test -p yawgpu --features vulkan,tiled --no-fail-fast -- --ignored`:
  104/104 (+ `shader-passthrough` 2/2), 0 validation-layer lines.
- Problems 1–4 of the spec are latent: no in-tree test exercises a
  divergent-subresource sequence, so the whole-image tracker never
  meets a mismatch. The Block 105 e2e cases are written to fail first
  under the validation layer (red-before-fix is logged per slice below).

## Slice log

| Slice | Commit | Gate | Notes |
|---|---|---|---|
| S1 | `a00daf9` | workspace 1093/0; clippy vulkan,tiled + fmt clean; Vulkan e2e `--ignored` 109/0, HAL `--ignored` 54/0, 0 validation lines | `layout.rs` tracker (11 unit tests), `transition_image_range` + pure `layout_barriers`, copy/clear/present sites carry exact ranges, same-image copies drop the `GENERAL` split/restore (`transition_copy_subresource` removed), `barrier_aspect_mask` / `transition_image_aspect` removed; render/subpass sites still whole-image via `transition_image` + `set(whole, ..)` (S2/S3). Review nits for the Phase Review: `mod.rs` still duplicates the `IMAGE_LAYOUT_*` constants (only `UNDEFINED` aliased); `_attachment` binding in the depth-stencil pre-pass transition (goes away in S2). |
| S2 | `10a59de` | workspace 1093/0; clippy vulkan,tiled + fmt clean; Vulkan e2e `--ignored` 111/0 (incl. the 2 new `e2e_vulkan_layouts` cases), HAL `--ignored` 54/0, 0 validation lines | R4: attachments (colour / resolve / depth-stencil) transition their own (mip, layer); bound views transition their view rectangle minus the pass's attachment subresources (`subtract_subresource_ranges`, 7 unit tests); post-pass `set` + Discard epilogue on the attachment range only. **Red-before-fix**: case 1 (render to mip 0 while sampling mip 1) produced 2 validation lines — `VUID-VkDescriptorImageInfo-imageLayout-00344`, `VUID-vkCmdDraw-None-08114` — before R4, 0 after; case 2 (copy layer 0 / sample layer 1 / copy layer 0) was already clean (the whole-image barrier was over-broad but self-consistent; spec problem 3 corrected), kept as the per-range regression guard. |
| S3 | (this commit) | workspace 1093/0; clippy default + vulkan,tiled + fmt clean; Vulkan e2e `--ignored` 112/0 (+ passthrough 2/0), HAL `--ignored` 54/0, 0 validation lines | R5: `encode_subpass_render_pass` collects every draw's bound textures across subpasses (`subpass_bound_textures`, 2 unit tests) and runs the S2 sampled → storage transitions before the pass, excluding the attachment / depth-stencil / resolve images' whole ranges. **Red-before-fix**: case 3 (tiled draw sampling a texture written by `writeTexture`) produced 2 validation errors (`VUID-VkDescriptorImageInfo-imageLayout-00344`, `VUID-vkCmdDraw-None-08114`: image in `TRANSFER_DST_OPTIMAL`, descriptor `SHADER_READ_ONLY_OPTIMAL`) before R5, 0 after. Closes 0704-sweep "Known related gaps" 2. |
| S4 | — | — | — |
