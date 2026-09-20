# Block 92 — real hardware-queried adapter limits (Dawn/CTS parity)

Status: **COMPLETE** (2026-07-02). All 3 slices done + Phase Review clean. Real
Metal + MoltenVK + native-Vulkan hardware-queried limits. Final CTS on real Metal
vs Dawn oracle: `capability_checks,limits` 9290/1795/0 (Dawn 9280/1805, yawgpu ≥
Dawn); `capability_checks,features` 2424/88/0 (byte-identical to Dawn). Native
Vulkan (RTX 5060 Ti, 2026-07-02): `limits` 9149/1936/0, `features` 2240/272/0 —
fail=0, whole tree in one process. Owner: CTS-Dawn-parity. Commits 5dda1f6,
21a6860, d2803dc, fddb0da.

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
   Vulkan skip-set shrink.) **Native-Vulkan verification DONE 2026-07-02**
   (RTX 5060 Ti): `limits` 9149/1936/0, `features` 2240/272/0 — fail=0.

3. **Docs + Phase Review.** README limits note; Block 92 finalize; no-context
   Phase Review; re-diff yawgpu-Metal vs Dawn-Metal skip sets to confirm the
   `capability_checks` gap is closed.

Tracking: `specs/tracking/adapter-limits.md`.

---

## Post-COMPLETE slice — P92.4: `maxVertexInputAttributeOffset` overflow aborts on RADV (filed 2026-09-20)

**Severity: CRITICAL.** yawgpu is unusable on any AMD RADV GPU in a build with
overflow checks on: `wgpuAdapterRequestDevice` aborts the process.

Found on a Linux clean-install bring-up while trying to exercise the
ETC2/ASTC paths of `e2e_vulkan_texture_compression` through RADV's
`vk_require_etc2` / `vk_require_astc` emulation. The bug is unrelated to
compression — it fires on plain device creation.

### Problem

`yawgpu-hal/src/vulkan/mod.rs:375-378`:

```rust
max_vertex_buffer_array_stride: vk
    .max_vertex_input_binding_stride
    .min(vk.max_vertex_input_attribute_offset + 1)
    .min(2048),
```

`VkPhysicalDeviceLimits::maxVertexInputAttributeOffset` is a `uint32_t` with no
spec-mandated upper bound, and **RADV reports `0xFFFFFFFF`** (`u32::MAX`). The
unchecked `+ 1` therefore overflows. Measured on this host:

| device | `maxVertexInputAttributeOffset` |
|---|---|
| NVIDIA GeForce RTX 5060 Ti (proprietary 595.91.07) | 2047 |
| llvmpipe (Mesa 26.0.8) | 2047 |
| **AMD Raphael iGPU (RADV, Mesa 26.0.8)** | **4294967295** |

Only RADV trips it, which is why the whole `e2e_vulkan_*` suite is green here:
adapter selection picks the discrete NVIDIA GPU, so RADV is never exercised.

### Observed

- **Overflow checks on** (`cargo` dev/test profiles): panic `attempt to add with
  overflow` inside `VulkanAdapter::limits`, reached from
  `wgpuAdapterRequestDevice`. Because that is an `extern "C"` frame the panic
  cannot unwind, so it escalates to `panic in a function that cannot unwind` →
  **`SIGABRT`**. Verified from C: the `device_info` example against RADV dies
  with `fatal runtime error: failed to initiate panic, error 5, aborting`,
  exit status 134. This is both a CLAUDE.md principle-3 violation (no panics in
  library code) and a hard crash of any consuming C application.
- **Overflow checks off** (release): the add wraps to `0`, so the HAL hands core
  a `max_vertex_buffer_array_stride` of `0`. The wrong value is then masked by
  `yawgpu-core/src/limits.rs:194-196`, which floors every advertised limit at the
  WebGPU default (`.max(default)`, 2048). Verified: a release `device_info`
  linked against `--features vulkan` reports `maxVertexBufferArrayStride 2048` on
  RADV and does not crash. **Correct by coincidence, not by construction** — the
  HAL's computed value is still wrong, and the coincidence only holds while the
  clamp target equals the default.

### Rules

- **R1 — Saturating arithmetic.** The `+ 1` becomes `saturating_add(1)`, so a
  driver-reported `u32::MAX` yields `u32::MAX` and the following `.min(2048)`
  produces the intended 2048. `saturating_sub` is already the idiom two fields
  below (`max_inter_stage_shader_variables`), so this matches local style.

- **R2 — No other unchecked arithmetic on driver-reported values.** A scan of
  `VulkanAdapter::limits` found this as the only unchecked add/sub/mul on a `vk.*`
  field (`max_uniform_buffer_range / 16 * 16` cannot overflow: it divides first).
  Re-check the whole function; if another is found, fix it under this rule rather
  than filing separately.

- **R3 — Value correctness, not just absence of panic.** The fix is verified by
  the *advertised limit on RADV*, not by the process surviving: core's
  `.max(default)` floor hides a wrong HAL value in release, so a test that only
  asserts "no crash" would pass against the unfixed code in release.

### Unit test (principle 1)

`VulkanAdapter::limits` takes a `VkPhysicalDeviceLimits`, so the conversion is
testable without a device. Add a `#[cfg(test)]` test under `--features vulkan`
that feeds a synthetic `VkPhysicalDeviceLimits` with
`max_vertex_input_attribute_offset: u32::MAX` (and
`max_vertex_input_binding_stride: 2048`) and asserts the resulting
`max_vertex_buffer_array_stride == 2048`. This must fail on the unfixed code
under `cargo test` (overflow checks on) — confirm that before fixing, so the test
is known to bite. If the limits conversion is not reachable as a pure function,
extract the `VkPhysicalDeviceLimits -> Limits` mapping so that it is.

### Verification

- The new unit test passes; confirmed failing (panicking) against the unfixed code.
- Real RADV, overflow checks on: `VK_DRIVER_FILES=.../radeon_icd.json` with the
  `e2e_vulkan_basic` suite and the C `device_info` example both succeed, and
  `device_info` reports `maxVertexBufferArrayStride 2048`.
- NVIDIA path unchanged: the full `e2e_vulkan_*` suite still passes (78 tests
  across 20 suites at the time of filing).
- Noop gate and clippy gate unchanged and green.

### Note

Adapter selection prefers the discrete GPU, so a multi-GPU host hides this.
Worth considering a `YAWGPU_VULKAN_DEVICE_INDEX`-style override (or reusing an
existing one, if present) so every installed ICD can be swept — but that is a
separate item, not this slice. `VK_DRIVER_FILES` already provides the
per-process escape used above.
