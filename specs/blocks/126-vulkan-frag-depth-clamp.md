# Block 126 — Vulkan: clamp `frag_depth` in the shader, as Dawn

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 5 MINOR; all resolved,
see below. Native Vulkan without `VK_EXT_depth_clip_enable` is owed.
Source: the Block 124 survey of Dawn's backend setup. Supersedes F-139's premise in
`specs/tracking/tint-migration-plan.md`.
Oracle: Dawn at the pinned `third_party/dawn`: `native/vulkan/RenderPipelineVk.cpp`
(`InitializeImpl`, the `clampFragDepth` immediates; `depthClampEnable =
HasUnclippedDepth()`), `native/vulkan/ShaderModuleVk.cpp` (`depth_range_offsets`),
and Tint `lang/spirv/writer/raise/shader_io.cc` (the frag-depth clamp).

## Problem

WebGPU clamps a fragment shader's `@builtin(frag_depth)` to the viewport's
`[minDepth, maxDepth]` unless the pipeline sets `unclippedDepth`. Dawn implements
this in the shader on Vulkan. When a pipeline writes `frag_depth` and
`unclippedDepth` is off, Dawn reserves the `clampFragDepth` immediates. It then sets
Tint's `depth_range_offsets`, so Tint clamps the output against the viewport range
pushed at draw time. Dawn's `depthClampEnable` is `unclippedDepth`.

yawgpu sets `depth_range_offsets` only for the pixel-center polyfill. Otherwise it
relies on fixed-function clamping:
- With `VK_EXT_depth_clip_enable`, `depthClampEnable` is always true. The hardware
  clamps to the viewport range, so results are correct.
- Without it, `depthClampEnable` equals `unclippedDepth`, which is false for a normal
  pipeline. Nothing clamps a shader-written depth to `[minDepth, maxDepth]`, which
  violates WebGPU.

F-139 assumed native Vulkan never needs the shader clamp. That holds only with the
extension. The CTS hosts (NVIDIA, and MoltenVK) have it, so the CTS has not seen
the failure. On MoltenVK, F-139's two `depth_clip_clamp` xfails come from the
missing shader clamp, so this block may also make them pass there.

## Rules

R1. On Vulkan, a render pipeline reserves the two depth-range immediates when
`(pixel-center polyfill || fragment writes frag_depth) && !unclippedDepth`, Dawn's exact
condition. The pixel-center polyfill option itself stays independent of
`unclippedDepth`, as in Dawn. SPIR-V generation
sets `depth_range_offsets` to them. This happens with or without the extension, as
Dawn does. The immediates sit in the same place the pixel-center polyfill uses: two
f32 values right after the layout's user-immediate region. The pixel-center polyfill
and the clamp share one pair of immediates, as they do in Dawn.

R2. The HAL declares a push-constant range that covers the immediates, and writes the
viewport's min and max depth at every draw that uses such a pipeline, including after
a viewport change. The existing pixel-center path already does this. R1 widens its
trigger.

R3. `depthClampEnable` and the depth-clip state are unchanged. With the extension,
the hardware clamp and the shader clamp agree, so applying both is harmless. Dawn's
rule (`depthClampEnable = unclippedDepth`, with clip state from the extension) is a
separate decision and out of scope.

R4. The trigger is part of the SPIR-V codegen key, as the pixel-center trigger is
(Block 95).

R5. Bundles, subpass (tiled) pipelines, and indirect draws follow the same rules as
the pixel-center path.

## Tests

- Core and shim: a fragment shader writing `frag_depth` with `unclippedDepth` false
  gets `depth_range_offsets`. With `unclippedDepth` true it does not. A shader that
  does not write `frag_depth` does not.
- Vulkan HAL unit: the push-constant range and the per-draw writes cover the clamp
  case.
- Real GPU e2e (Claude), Vulkan: a fragment shader writes `frag_depth = 0.9` under a
  viewport with `maxDepth = 0.5`. The stored depth is 0.5. Viewports are set both
  before and after the pipeline is bound. Run on MoltenVK under the Khronos layer,
  with 0 VUIDs. Running it on a native host without `VK_EXT_depth_clip_enable` is
  owed.
- CTS on MoltenVK: `api,operation,rendering,depth_clip_clamp:*`. Record whether
  F-139's two xfails now pass. On Metal, `webgpu:*` gives `fail=2 crash=0`.

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests.
- **S2 (Claude):** e2e, MoltenVK CTS, Phase Review.

R6. When the depth range is reserved, `minimum_immediate_size` is
`user_immediate_size + 8`, as Dawn's `mask.count() * 4`. Without it, Tint pads the
immediate struct (a `vec4` user prefix plus two f32 gives 32 bytes), and the block
exceeds the 24-byte push-constant range: VUID-VkGraphicsPipelineCreateInfo-layout-07987,
found by `vulkan_frag_depth_from_immediate_composes_with_clamp` under the layer. This
also fixes the same latent mismatch on the pixel-center path. Pipelines that reserve
no range are unchanged.

## Verification (2026-11-03)

Run on Apple M2, macOS 27.2; Vulkan is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,321 passed, 0 failed |
| fmt, clippy on 8 feature sets | clean |
| Vulkan HAL ignored tests | 74 passed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled` | 147 passed, 1 failed (known `imageView2DOn3DImage`); no other VUID |
| `frag_depth_is_clamped_to_viewport` on MoltenVK | before: stored 0.9, expected 0.5 (fails); after: passes, including the bundle draw |
| Metal e2e | 154 passed, 0 failed |
| webgpu-native-cts, MoltenVK, `depth_clip_clamp:*` | before (Block 125 library) `fail=2`; after `pass=4 fail=0` |
| webgpu-native-cts, Metal, `webgpu:*` | `fail=2 crash=0`; subcase counts unchanged |

F-139's two MoltenVK xfails now pass. `expectations/yawgpu-vulkan.txt` in
webgpu-native-cts still lists them, and will report them as xpass until the suite
drops them.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | The trigger kept pixel-center outside the `!unclippedDepth` gate | Changed to Dawn's `(pixel_center \|\| frag_depth) && !unclippedDepth`; Tint accepts pixel-center without the range |
| m2 | The e2e cannot tell on hosts with `VK_EXT_depth_clip_enable`, and had no bundle | Documented; second draw now comes from a render bundle |
| m3 | Vacuous R3 assertion | Removed |
| m4 | Spec, F-139, and CTS records | Updated |
| m5 | Vertex SPIR-V also carries the depth-range options | Correct, as in Dawn; informational |

