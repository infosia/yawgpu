# Tracking — `depth-clip-control` optional feature

Spec: [Block 63](../blocks/63-depth-clip-control.md). Goal: Dawn parity for the
WebGPU `depth-clip-control` optional feature
(`WGPUFeatureName_DepthClipControl = 0x02`) on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + gate flip (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | **DONE** (2026-07-01) |
| 3 | Docs + Phase Review | **DONE** (2026-07-01) |

**Phase COMPLETE** — no-context Phase Review of `eef10e6^..HEAD` returned no
CRITICAL/MAJOR. Gate flip tier-independent + single funnel (no bypass); Vulkan
`depthClamp`-only advertisement confirmed exact Dawn parity + VUID-sound
(depthClamp enabled whenever supported); e2e A/B non-vacuous. Three MINOR doc
nits fixed (e2e header + tracking note precision; metal unused-import removal was
already folded in).

## Key facts (verified 2026-07-01)

- **HAL already fully implements `unclippedDepth`** (threading audit group E,
  `[[threading-audit-silently-wrong]]`): Vulkan `mod.rs:538` /
  `pipeline.rs:665-749` (`VK_EXT_depth_clip_enable` +
  `VkPipelineRasterizationDepthClipStateCreateInfoEXT`), Metal `encode.rs:948`
  (`setDepthClipMode`). The `HalRenderPipelineDescriptor.unclipped_depth` field
  and the `conv/pipeline.rs:297` parse both exist. **No HAL work needed for
  Slice 1.**
- Core currently **hard-rejects** `unclippedDepth` at
  `render_pipeline.rs:2496-2497` — the only real gap.
- `resolve` (`render_pipeline.rs:2145`) already has `features: &FeatureSet` and
  passes it to `validate_depth_stencil_aspects` / `validate_fragment_depth_output`
  — thread it into `validate_primitive_state` the same way.
- Vulkan adapter cap must mirror the device predicate exactly:
  `depthClamp` feature `&&` `VK_EXT_depth_clip_enable` extension present.
- Template: `shader-f16` / `subgroups` (`[[subgroups-feature]]`). Touch-points:
  `adapter.rs` Feature enum + `add_*_feature` + `features()`,
  `yawgpu-hal/src/lib.rs` dispatch + 4 backend impls,
  `yawgpu/src/conv/feature.rs`. No shim / FFI-struct changes.

## Slice 2 — landed (+ HAL fix the e2e uncovered)

`e2e_{metal,vulkan}_depth_clip_control.rs` (2 tests each): adapter advertises the
feature; a full-screen triangle emitted entirely beyond the far plane (clip-space
z=1.5, w=1) is **clipped → clear black** with `unclippedDepth=false` and
**clamped → red** with `unclippedDepth=true`, proving the A/B behavior at
readback. **Metal 2/2 + MoltenVK 2/2 on real GPU.**

The MoltenVK run exposed a HAL over-restriction (Noop-invisible): the Vulkan cap
query + device-feature enablement required `VK_EXT_depth_clip_enable`, which
MoltenVK lacks — so it did not advertise the feature and `unclippedDepth`
aborted. Fix (matches Dawn `PhysicalDeviceVk.cpp:368`): advertise on core
`depthClamp` alone (unclippedDepth ⇒ `depthClampEnable`, which implicitly
disables clipping; the extension is only an optional independent-clip
enhancement), and enable the `depthClamp` device feature whenever supported (not
gated on the extension) so the no-extension `depth_clamp_and_clip` path is
VUID-valid. `depth_clamp_and_clip` (pipeline.rs) was already correct for both
paths.

## Notes

- Slice 2 e2e must actually differentiate the two paths (draw a primitive with
  depth outside [0,1]; clipped→absent vs clamped→present) — not just "no error".
  No depth attachment is needed: near/far clipping happens in the rasterizer, so a
  beyond-far-plane triangle is absent (clear color) when clipped and present when
  clamped — observable at color readback.
