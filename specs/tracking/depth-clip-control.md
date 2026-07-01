# Tracking — `depth-clip-control` optional feature

Spec: [Block 63](../blocks/63-depth-clip-control.md). Goal: Dawn parity for the
WebGPU `depth-clip-control` optional feature
(`WGPUFeatureName_DepthClipControl = 0x02`) on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + gate flip (Noop + HAL cap) | IN PROGRESS |
| 2 | Real-GPU execution e2e (Metal + Vulkan) | TODO |
| 3 | Docs + Phase Review | TODO |

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

## Notes

- Slice 2 e2e must actually differentiate the two paths (draw a primitive with
  depth outside [0,1]; clipped→absent vs clamped→present) — not just "no error".
  A depth attachment + depth test makes the difference observable at readback.
