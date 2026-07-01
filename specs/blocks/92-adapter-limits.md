# Block 92 — real hardware-queried adapter limits (Dawn/CTS parity)

Status: **COMPLETE** (2026-07-02). All 3 slices done + Phase Review clean. Real
Metal + MoltenVK hardware-queried limits. Final CTS on real Metal vs Dawn oracle:
`capability_checks,limits` 9290/1795/0 (Dawn 9280/1805, yawgpu ≥ Dawn);
`capability_checks,features` 2424/88/0 (byte-identical to Dawn). Owner:
CTS-Dawn-parity. Commits 5dda1f6, 21a6860, d2803dc, fddb0da.

## Motivation

`Adapter::limits()` (`yawgpu-core/src/adapter.rs:61`) unconditionally returns
`Limits::DEFAULT` on **every** backend — the WebGPU spec *minimum* limits. This
is spec-conformant but far below what the real Metal/Vulkan hardware supports and
below what **Dawn** advertises on the same GPU.

Consequence, measured with webgpu-native-cts on this M2 (`build-dawn/cts` = Dawn
oracle vs `build-yawgpu/cts` = yawgpu), same machine:

| CTS tree | Dawn pass/skip | yawgpu pass/skip | Dawn runs extra |
|---|---|---|---|
| `api,validation,capability_checks,features` | 2424 / 88 | 2254 / 258 | **+170** |
| `api,validation,capability_checks,limits`  | 9280 / 1805 | 6601 / 4484 | **+2679** |

The CTS "at-limit" / "betterValue" cases **skip when the reported limit equals the
default** (nothing above default to test). yawgpu reports defaults → skips ~3040
cases that Dawn (reporting the true hardware limits) runs. This is the single
largest CTS-coverage gap vs Dawn and the reason we are not at CTS parity.

**Reporting a higher limit is a correctness commitment, not just a query change:**
the CTS at-limit *operation* cases create resources at the reported value and
expect them to work. So each raised limit must be genuinely honoured by
validation + the HAL. Slices are therefore ordered by risk and every slice is
CTS-verified on real hardware.

## Design — HAL seam

`yawgpu-hal` does **not** depend on `yawgpu-core`, so the HAL cannot return
`core::Limits`. Introduce a mirror in the HAL:

- **`yawgpu-hal/src/lib.rs`** — a new `pub struct HalLimits` with the same 38
  fields as `core::Limits` (u32/u64), `#[non_exhaustive]`, `Copy`, `Debug`, plus
  a `HalLimits::DEFAULT` const equal to the WebGPU spec defaults (copy the values
  from `core::Limits::DEFAULT`). Add `HalAdapter::limits(&self) -> HalLimits`
  dispatching (static enum) to each backend adapter's `limits()`:
  - **Noop** — `HalLimits::DEFAULT`.
  - **GLES** — `HalLimits::DEFAULT` (Tier 2; no query for now).
  - **Metal / Vulkan** — real query (below).
- **`yawgpu-core/src/adapter.rs`** — `Adapter::limits()` returns
  `hal_limits_to_core(self.inner.hal.limits())`, where `hal_limits_to_core`
  (new, in `adapter.rs` or `limits.rs`) maps field-by-field and **clamps**:
  - every *maximum* field → `max(hal, DEFAULT)` (WebGPU requires an adapter
    support at least the spec defaults),
  - every *alignment* field (`min_*_offset_alignment`) → `min(hal, DEFAULT)`
    (smaller alignment is "better"; hardware values are powers of two, keep them
    but never advertise worse than the 256 default),
  - `max_immediate_size` → pass through (0 default).
  The existing `validate_required_limits` already uses `self.limits()` as the
  request ceiling, so no other core change is needed for the request path.

Keep `HalLimits` field order identical to `core::Limits` for a trivial mapping.

## Metal query — mirror Dawn `PhysicalDeviceMTL.mm::InitializeSupportedLimitsImpl`

Add `MetalAdapter::limits()` (`yawgpu-hal/src/metal/mod.rs`). Port Dawn's
`kMTLLimits` feature-set table + logic. Pick the tier by probing
`device.supportsFamily(MTLGPUFamily::AppleN / MacN)` (highest supported wins);
Dawn indexes a 15-entry table. The fields we must set from Metal (Apple8/M2
target values in parentheses, but compute them from the family, don't hardcode
M2):

- `max_texture_dimension_1d/2d` = family `max1D/2DTextureSize` (16384),
  `_3d` = 2048, `max_texture_array_layers` = 2048.
- `max_color_attachments` = family `maxColorRenderTargets` (8),
  `max_color_attachment_bytes_per_sample` = family `maxTotalRenderTargetSize` (64).
- `max_compute_workgroup_storage_size` = `maxTotalThreadgroupMemory` (32768);
  `max_compute_invocations_per_workgroup` = `maxThreadsPerThreadgroup` (1024);
  `max_compute_workgroup_size_x/y/z` = `maxThreadsPerThreadgroup` (1024/1024/1024).
- `max_inter_stage_shader_variables` = Apple: `min(maxFragmentInputs,
  maxFragmentInputComponents/4)` (=31); non-Apple: `maxFragmentInputs - 4`.
- `min_uniform_buffer_offset_alignment` = `min_storage_..._alignment` =
  family `minBufferOffsetAlignment` (Apple 4, Mac 256).
- `max_buffer_size` = `max_uniform_buffer_binding_size` =
  `max_storage_buffer_binding_size` = `[device maxBufferLength]`
  (clamp binding sizes to `u32::MAX` like Dawn; `max_buffer_size` stays u64).
- `max_vertex_attributes` = `maxVertexBuffers(8) * maxVertexAttribsPerDescriptor`
  (248) — but see the note below; if honouring 248 attributes risks HAL breakage,
  keep at Dawn's value and let CTS confirm.
- Binding-count splits (Dawn `PhysicalDeviceMTL.mm:881-905`):
  `max_sampled_textures_per_shader_stage`,
  `max_storage_textures_per_shader_stage` (from `maxTextureArgumentEntriesPerFunc`
  split, base+remainder → 70 / 58 on Apple8), `max_samplers_per_shader_stage`
  (16), `max_storage_buffers_per_shader_stage`,
  `max_uniform_buffers_per_shader_stage` (from `maxBufferArgumentEntriesPerFunc`),
  `max_dynamic_uniform/storage_buffers_per_pipeline_layout` (11 hardcoded).
  Mirror the exact split arithmetic from Dawn.
- Left at DEFAULT (Dawn does not query): `max_bind_groups` (4),
  `max_bind_groups_plus_vertex_buffers` (24), `max_bindings_per_bind_group`
  (1000), `max_vertex_buffers` (8), `max_vertex_buffer_array_stride` (2048),
  `max_compute_workgroups_per_dimension` (65535).
- Compat per-stage (`max_storage_{buffers,textures}_in_{vertex,fragment}_stage`)
  mirror the per-shader-stage values.

Authoritative source with line refs:
`third_party/dawn/src/dawn/native/metal/PhysicalDeviceMTL.mm:815-973`
(+ `native/Limits.cpp`, `common/Constants.h`).

## Vulkan query — mirror Dawn `PhysicalDeviceVk.cpp::InitializeSupportedLimitsImpl`

Add `VulkanAdapter::limits()` (`yawgpu-hal/src/vulkan/mod.rs`) reading the cached
`VkPhysicalDeviceLimits` (query via `vkGetPhysicalDeviceProperties2`). Mapping
(Dawn `PhysicalDeviceVk.cpp:744-918`, clamps to Dawn `kMax*` constants):
`maxImageDimension1D/2D(min of several)/3D`, `maxImageArrayLayers`,
`maxBoundDescriptorSets`→bindGroups, `maxPerStageDescriptor{SampledImages,
Samplers,StorageBuffers,StorageImages,UniformBuffers}`, `maxDescriptorSet*Dynamic`,
`maxUniformBufferRange`(round down to 16), `maxStorageBufferRange`(NVIDIA 2GB-4
cap), `min{Uniform,Storage}BufferOffsetAlignment`, `maxVertexInputBindings/
Attributes`, `maxVertexInputBindingStride`, `maxColorAttachments`,
`maxComputeSharedMemorySize`, `maxComputeWorkGroupInvocations`,
`maxComputeWorkGroupSize[0..2]`, `maxComputeWorkGroupCount(min)`,
`maxInterStageShaderVariables = min(maxVertexOutputComponents,
maxFragmentInputComponents)/4 - 2`, `maxBufferSize` from Maintenance4/3 or 2GB.
Apply the `maxFragmentCombinedOutputResources` redistribution (`:796-825`).

## Slices

1. **HAL seam + Noop/GLES DEFAULT + full Metal query + core map/clamp + unit
   tests.** Inline `#[cfg(test)]` tests: `HalLimits::DEFAULT` round-trips; core
   `hal_limits_to_core` clamps a below-default max up and an above-default
   alignment down; Metal adapter `limits()` returns ≥ default (run under
   `--features metal`). **Acceptance:** `cargo test --workspace` +
   `cargo test -p yawgpu-hal --features metal --lib` green on Noop.
   *Claude then:* rebuild `libyawgpu.dylib`, run CTS `capability_checks,limits`
   + `,features` on **real Metal**, triage every new at-limit failure (fix HAL
   or dial the offending limit back to what the HAL honours), re-verify 0 fail.

2. **Vulkan query.** `VulkanAdapter::limits()` mirroring Dawn; unit test under
   `--features vulkan`. *Claude then:* CTS `capability_checks,{limits,features}`
   on **MoltenVK**, triage, re-verify. (MoltenVK reports MoltenVK's own limits;
   verify against the Vulkan `build-dawn` isn't apples-to-apples — compare yawgpu
   Vulkan skip-set shrink, and native-Vulkan is deferred.)

3. **Docs + Phase Review.** README limits note; Block 92 finalize; no-context
   Phase Review; re-diff yawgpu-Metal vs Dawn-Metal skip sets to confirm the
   `capability_checks` gap is closed.

Tracking: `specs/tracking/adapter-limits.md`.
