# Block 63 — `depth-clip-control` optional feature

Status: **COMPLETE** — all slices done, real-GPU verified (Metal M2 + MoltenVK),
Phase Review clean (no CRITICAL/MAJOR). Owner: Dawn-parity feature backfill.

The WebGPU `depth-clip-control` optional feature
(`WGPUFeatureName_DepthClipControl = 0x02`) lets a render pipeline set
`primitive.unclippedDepth = true`, which **disables near/far depth-plane
clipping** (fragments outside the `[0, 1]` NDC depth range are kept and their
depth is clamped instead of the primitive being clipped). It is the standard
WebGPU spelling of Vulkan `VK_EXT_depth_clip_enable` / Metal
`MTLDepthClipMode.clamp`.

**The HAL already implements this end to end** — it was wired during the
threading audit (`[[threading-audit-silently-wrong]]`, group E):
- Vulkan (`yawgpu-hal/src/vulkan/mod.rs:538`, `pipeline.rs:665`): device
  creation enables `VK_EXT_depth_clip_enable` + `depthClamp` when the physical
  device supports them (`VulkanDevice::depth_clip_control`); the pipeline sets
  `VkPipelineRasterizationDepthClipStateCreateInfoEXT.depthClipEnable =
  !unclipped_depth` with `depthClampEnable` accordingly.
- Metal (`yawgpu-hal/src/metal/encode.rs:948`): `setDepthClipMode(.Clamp)` when
  `unclipped_depth`, else `.Clip`.

What is missing is **only the `yawgpu-core` feature**: `unclippedDepth` is
currently **hard-rejected** for every pipeline at
`render_pipeline.rs:2496` ("render pipeline unclippedDepth is not supported"),
and `DepthClipControl` is not a `Feature` variant, so the working HAL path is
unreachable. This block advertises the feature and flips the hard-reject into a
feature gate.

## Public API surface

No new C entry points. Flows through existing surfaces (mirrors `shader-f16` /
`subgroups`):

- **`wgpuAdapterGetFeatures` / `wgpuAdapterHasFeature`** report
  `WGPUFeatureName_DepthClipControl` when the backend supports it.
- **`wgpuDeviceGetFeatures` / `wgpuDeviceHasFeature`** report it when requested
  at device creation. Requesting it on an adapter that does not advertise it is
  a validation error (existing `resolve_features` path).
- **`WGPURenderPipelineDescriptor.primitive.unclippedDepth`** — already parsed
  (`yawgpu/src/conv/pipeline.rs:297`); accepted iff the device enabled the
  feature.

### `yawgpu-core::Feature`

Add a `DepthClipControl` variant (not `cfg`-gated). Map it C↔Rust in
`yawgpu/src/conv/feature.rs` (`DepthClipControl ↔ WGPUFeatureName_DepthClipControl`).
Advertise from `Adapter::features()` via `add_depth_clip_control_feature`
consulting `HalAdapter::supports_depth_clip_control()`.

## Behaviour contract

### Advertisement (HAL capability query)

`HalAdapter::supports_depth_clip_control() -> bool`, static-enum-dispatched:

- **Vulkan** — `depthClamp` (`VkPhysicalDeviceFeatures.depthClamp`) **and** the
  `VK_EXT_depth_clip_enable` extension present. This mirrors exactly the
  device-creation predicate `depth_clamp && depth_clip_enable_extension`
  (`mod.rs:538`), so advertisement and the actual enabled capability agree.
- **Metal** — always `true` (`MTLDepthClipMode.clamp` is universally available).
- **Noop** — `true` (keeps the accept-path unit-testable; Noop never rasterizes).
- **GLES** — `false` (Tier 2; `GL_DEPTH_CLAMP` is not in GLES 3.1 core). Catalogue
  in [Block 67](67-gles-backend.md) if revisited.

### Validation gate (Tier-independent core rule)

Thread the device's `FeatureSet` into `validate_primitive_state`. The check at
`render_pipeline.rs:2496` becomes:

- `primitive.unclippedDepth == true` **without** `DepthClipControl` enabled →
  rejected at `createRenderPipeline` with a validation error routed to the
  device error sink. **Identical on every backend** (including Noop / GLES).
- `unclippedDepth == true` **with** the feature → accepted; the descriptor's
  `unclipped_depth` flows to the HAL (already wired), which disables depth
  clipping. `unclippedDepth == false` is always accepted (default).

The existing `resolve` already receives `features: &FeatureSet` and passes it to
`validate_depth_stencil_aspects` / `validate_fragment_depth_output`; pass it to
`validate_primitive_state` the same way.

## Slices

1. **Feature plumbing + gate flip (Noop + HAL cap, no execution).**
   `Feature::DepthClipControl`, `add_depth_clip_control_feature`,
   `HalAdapter::supports_depth_clip_control` on all four backends,
   `conv/feature.rs`, and `validate_primitive_state(primitive, features)` gating
   the reject. Inline unit tests: Noop advertises the feature; `unclippedDepth`
   rejected without it and accepted with it; C↔Rust feature round-trip.
   **Acceptance:** `cargo test --workspace` green on Noop.

2. **Real-GPU execution e2e (Metal + Vulkan).**
   `e2e_{metal,vulkan}_depth_clip_control.rs`: render a triangle whose vertices
   have depth outside `[0, 1]` into a small depth+color target, once with
   `unclippedDepth=false` (clipped → pixel not drawn / depth-fail) and once with
   `unclippedDepth=true` (clamped → pixel drawn), and read back the color to
   prove the two paths differ as specified. **Acceptance:** both e2e suites pass
   on the M2 (Metal) and MoltenVK (Vulkan), verified by Claude.

3. **Docs + Phase Review.** README capability note; Block 63 finalization; the
   mandatory no-context Phase Review of the cumulative diff.

Tracking: `specs/tracking/depth-clip-control.md`.
