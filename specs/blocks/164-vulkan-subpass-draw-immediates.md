# Block 164 — Vulkan subpass draws push the immediates block

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 0 MAJOR.
Source: parameter-propagation audit at `b772306`
(`specs/tracking/threading-audit.md`, "Propagation audit 2026-11"), candidate 03.
Oracle: yawgpu's regular Vulkan render pass (`vulkan/encode.rs`
`prepare_vulkan_render_draw`, `specs/blocks/126-vulkan-frag-depth-clamp.md` R2 and R5,
`specs/blocks/94-immediates.md` "Immediates block layout"). The Metal subpass path already
delivers its block per draw (`metal/encode.rs` `encode_subpass_render_pass` →
`encode_render_immediates`).

## Problem

Core calls `configure_vulkan_render_immediates` for subpass pipelines, so a subpass
pipeline whose fragment shader needs the viewport depth range (frag-depth clamp,
pixel-center polyfill) or whose layout reserves user immediates gets a push-constant range
in its `VkPipelineLayout` and a shader that reads it. The regular draw path pushes the
block before every draw. `encode_subpass_draw` sets the viewport, binds descriptors and
draws without `vkCmdPushConstants`, so the shader reads push constants that were never
written in that command buffer: the frag-depth clamp uses undefined bounds.

## Rules

R1. **Push before every subpass draw.** When the subpass draw's pipeline has an
immediates block (`pipeline.inner.immediates` is `Some`), `encode_subpass_draw` pushes
the whole block over `VERTEX | FRAGMENT` at offset 0 after binding the pipeline and before
the draw call, for every draw.

R2. **Depth range follows the draw's viewport.** The depth-range pair is
`[min_depth, max_depth]` of the viewport that the same function passes to
`vkCmdSetViewport`: the draw's `HalSubpassDraw.viewport`, or `[0.0, 1.0]` when it is
`None`.

R3. **User prefix is zero.** The subpass encoder has no `setImmediates`; the user-immediate
prefix `[0, user_immediate_size)` of the block is all zeros.

R4. **One composition.** The block bytes come from
`crate::immediates::compose_immediates_block`, the function the regular path uses; the
regular and subpass paths share the code that maps `VulkanImmediates` + user bytes +
depth range to the pushed bytes. The regular path's bytes are unchanged.

## Slices

S1: R1–R4 in `yawgpu-hal/src/vulkan/encode.rs`, inline tests under
`--features vulkan,tiled`.

## Verification

- Inline HAL tests: the subpass block bytes for a pipeline with the depth-range pair and
  (a) no user immediates, (b) a nonzero user prefix — the prefix is zero and the pair
  equals the draw viewport's bounds for two different viewports and for `None`; a pipeline
  without immediates yields no block.
- Agent gates: `cargo test -p yawgpu-hal --features vulkan,tiled --lib`,
  `cargo test -p yawgpu-hal --lib`, clippy with `--features vulkan,tiled`.
- Claude: `e2e_vulkan_tiled` probe — a subpass pipeline that writes `frag_depth = 1.0`,
  drawn into two regions with viewport `max_depth` 0.25 and 0.75 in a fresh command
  buffer; depth readback equals the two maxima; the same draws in a regular pass as the
  control; run under the Khronos validation layer. Native Vulkan is the verdict; MoltenVK
  is reference only.

## Verification record

Shared gates for Blocks 163–167 (2026-11-28, M2): `cargo test --workspace` 1514/0, with
`yawgpu/vulkan,yawgpu/tiled` 1871/0; fmt clean; clippy clean for default,
`vulkan,tiled` and `metal,tiled`; `yawgpu-hal --lib` 395/0 (`vulkan,tiled`) and 218/0
(`metal,tiled`); all `e2e_metal_*` with `metal,tiled` 176/0; all `e2e_vulkan_*` with
`vulkan,tiled` on MoltenVK under the Khronos layer 174 passed, 1 failed (the known
`imageView2DOn3DImage` portability case; every VUID comes from that test).

`e2e_vulkan_tiled` 6/6 on MoltenVK with no VUID, including
`vulkan_tiled_frag_depth_clamps_to_draw_viewport` (regular-pass control and subpass both
read back [0.25, 0.75]). With the push removed the probe reads [0.0, 0.0] and the layer
reports `VUID-vkCmdDraw-maintenance4-08602` for each draw. The native Vulkan run is owed.

## Phase Review

Fresh-agent review of Blocks 163–164: 0 CRITICAL, 0 MAJOR, 1 MINOR for this block
(`compose_vulkan_render_immediates` had no doc comment — fixed).
