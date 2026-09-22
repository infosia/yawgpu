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
| S3 | `6d391aa` | workspace 1093/0; clippy default + vulkan,tiled + fmt clean; Vulkan e2e `--ignored` 112/0 (+ passthrough 2/0), HAL `--ignored` 54/0, 0 validation lines | R5: `encode_subpass_render_pass` collects every draw's bound textures across subpasses (`subpass_bound_textures`, 2 unit tests) and runs the S2 sampled → storage transitions before the pass, excluding the attachment / depth-stencil / resolve images' whole ranges. **Red-before-fix**: case 3 (tiled draw sampling a texture written by `writeTexture`) produced 2 validation errors (`VUID-VkDescriptorImageInfo-imageLayout-00344`, `VUID-vkCmdDraw-None-08114`: image in `TRANSFER_DST_OPTIMAL`, descriptor `SHADER_READ_ONLY_OPTIMAL`) before R5, 0 after. Closes 0704-sweep "Known related gaps" 2. |
| S4 | (this commit) | workspace 1093/0; clippy default / vulkan,tiled / +shader-passthrough + fmt clean; Vulkan e2e `--ignored` 112/0 (+ passthrough 2/0), HAL `--ignored` 54/0, 0 validation lines | Phase Review fixes (MAJOR 1, MINOR 3/4/9) + spec/backlog updates; Block 105 COMPLETE, backlog A5 closed. |

## Phase Review (2026-09-22, fresh-context reviewer over `3f007c0..6d391aa`)

Reviewer also ran `cargo test -p yawgpu-hal --features vulkan,tiled` (255/0) and clippy; no CRITICAL.

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| M1 | MAJOR | Tiled subpass depth-stencil attachment went through the colour loop (slot sentinel `u32::MAX`): `COLOR_ATTACHMENT_OPTIMAL` barrier on a depth image and a colour post-pass state while the `VkRenderPass` uses `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`; no tiled e2e set `depthStencilAttachment`, so the validation gate never saw it (pre-existing, inside this phase's exit criterion) | **Fixed** — `subpass_attachment_views` returns the depth-stencil texture separately; it transitions to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` and gets no post-pass `set`; e2e case 3 now carries a `Depth32Float` attachment + depth readback. **Red-before-fix**: 3 validation errors / 6 lines (`VUID-VkImageMemoryBarrier-oldLayout-01208`, `VUID-VkRenderPassBeginInfo-initialLayout-00900`, `VUID-vkCmdCopyImageToBuffer-srcImageLayout-00189`) → 0 after; spec R5 amended |
| M2 | MAJOR | 3D colour attachment `storeOp: Discard` epilogue clears with `color_attachment_subresource_range` (`depth_slice` as `baseArrayLayer`): invalid for `depth_slice > 0`, and for slice 0 zeroes every slice of the mip | **Deferred with rationale** — pre-existing since Block 104, outside Block 105's rules (the new layout barriers around the clear are correct; only the clear range is wrong); logged as backlog **A8** |
| m3 | MINOR | `mod.rs` duplicated the `IMAGE_LAYOUT_*` constants | **Fixed** — deleted; `texture.rs` is the single source |
| m4 | MINOR | Run-coalescing loop duplicated in `SubresourceLayouts::transition` and `subtract_subresource_ranges` | **Fixed** — shared `coalesce_subresources` in `layout.rs` (5 unit tests), both callers on top of it |
| m5 | MINOR | Per-transition `Vec` allocations on encode paths (the atomic path was allocation-free) | **Deferred** — common case is one run; measure first (Block 97 microbench) before a `SmallVec` |
| m6 | MINOR | One `vkCmdPipelineBarrier` per sub-rectangle / per attachment | **Deferred** — barrier count only; batch per image if profiling shows it |
| m7 | MINOR | Ledger S3 commit placeholder | **Fixed** |
| m8 | MINOR | Unrelated `Cargo.lock` hunk in the S1 commit (dev-deps from `97e1713`) | **Noted** — documented in the commit message |
| m9 | MINOR | `SubresourceLayouts::state` was `allow(dead_code)` outside tests | **Fixed** — `#[cfg(test)]` |
| m10 | MINOR | `transition_swapchain_image_to_present`: the new `?` between begin/end leaves a begun command buffer on `Err` | **Deferred** — unreachable (1 x 1 tracker, whole range; only lock poisoning fails) and mirrors the pre-existing early returns in that closure |
| m11 | MINOR | Spec R4 named the wrong helper for the 3D layer choice | **Fixed (spec)** |
| m12 | MINOR | Sampled + read-only storage binding of one subresource in one pass ends in `GENERAL` while the sampled descriptor declares `SHADER_READ_ONLY_OPTIMAL` (pre-existing policy, undocumented) | **Documented** — spec "Known limitations"; backlog **A9** |
| m13 | MINOR | e2e cases rely on the external validation layer as oracle (no in-process debug-utils sink) | **Noted** — the spec's stated method |

Reviewer confirmed (no finding): R1 tracker bounds/overflow/no-index-panics/lock scope; R6 policy exhaustive; R2 one barrier per run with full aspect mask and OR-ed source stages; every R3 site's range; range validity at every call site (core drops empty copies, HAL validates mip/origin, view ranges never exceed the tracker); R4 attachment ranges vs `finalLayout`, Discard epilogue, exclusion correctness; R5 collection order and exclusion; R7 untouched; barrier stage/access validity; no new panic paths; `Send + Sync`; tests vs the spec's list; CLAUDE.md conventions.

## CTS

Not run on this host (webgpu-native-cts oracle runs are Apple-side); re-confirm `command_buffer,*` / `rendering,*` / `resource_init,*` on MoltenVK at the next Mac session and log here.

**MoltenVK re-confirmation (Mac session, 2026-09-22, tree `6c8b417`, release `--features vulkan`, CTS `build-yawgpu-vulkan`, `--workers 8`):** `api,validation,render_pass,*` + `api,validation,encoding,*` + the 8 `api,operation` trees (resource_init, render_pass, rendering, command_buffer, texture_view, sampling, storage_texture, compute) — **274,884 pass / 9 fail / 0 crash**, byte-identical to the pre-Block-105 run at `4ecb8b2` (the 9 = `rendering,3d_texture_slices` ×7 + `depth_clip_clamp` ×2, documented MoltenVK artifacts F-139/F-140; `xfail 2 / xpass 4` = the expectations-file `index_buffer_format_dirtying` entries). Also on MoltenVK under the Khronos validation layer: HAL `--ignored` 54/0, every `e2e_vulkan_*` binary 106/0 (macOS skips: clip-distances execution, framebuffer fetch), 0 layout-related validation lines. The only validation output was two pre-existing portability-subset items unrelated to layouts (`imageView2DOn3DImage-04459` on 3D non-attachment images, `imageViewFormatSwizzle-04465` because the portability features were never chained at device creation), fixed in the follow-up commit.
