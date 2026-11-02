# Block 127 — Vulkan subgroup setup: Dawn parity

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 5 MINOR; all resolved,
see below. Native Vulkan ran on 2026-11-03 and took the maximal reconvergence path;
the default compute subgroup size still needs a device with varying sizes.
Source: the Block 124 survey, items 4 and 6, under the user's 2026-11-02 request
to fix similar items as they are found.
Oracle: Dawn at the pinned `third_party/dawn`: `native/vulkan/PhysicalDeviceVk.cpp`
(reconvergence toggles, `FindDefaultComputeSubgroupSize`, the
`maxComputeInvocationsPerWorkgroup` clamp), `native/vulkan/DeviceVk.cpp` (feature
enables), `native/vulkan/ShaderModuleVk.cpp` (Tint options),
`native/vulkan/ComputePipelineVk.cpp` (required subgroup size), and the Tint SPIR-V
printer (`MaximallyReconvergesKHR`, `SubgroupUniformControlFlowKHR`).

## Problem

1. **Reconvergence.** Dawn sets Tint's `use_maximal_reconvergence` when
   `VK_KHR_shader_maximal_reconvergence` and its feature are present, outside
   Android/Imagination. Otherwise it sets `use_subgroup_uniform_control_flow` when
   `VK_KHR_shader_subgroup_uniform_control_flow` and its feature are present and
   subgroups are supported in both the compute and fragment stages. It enables the
   matching device feature. yawgpu sets neither. Without them, how subgroup operations
   reconverge in divergent control flow is up to the driver, so results can differ
   from Dawn on the same device.
2. **Default compute subgroup size.** When subgroup size control is available and the
   minimum and maximum sizes differ (Intel today), Dawn requires a default size of
   `min * 2`, or `min` when that exceeds `max`, for every compute pipeline without
   `@subgroup_size`. It also clamps `maxComputeInvocationsPerWorkgroup` to
   `maxComputeWorkgroupSubgroups * default`. yawgpu sets only `ALLOW_VARYING`, so
   the size, and therefore `subgroup_size` and performance, differ from Dawn on those
   devices.

## Rules

R1. Device creation enables `shaderMaximalReconvergence` under Dawn's condition, or
otherwise `shaderSubgroupUniformControlFlow` under Dawn's condition, including the
Android/Imagination exclusion and the stage check. Never both. SPIR-V generation sets
the matching Tint option. Each decision is a pure function of device facts, with a unit
test.

R2. The default compute subgroup size follows `FindDefaultComputeSubgroupSize`
exactly. When it is set, a compute pipeline without an explicit `@subgroup_size`
chains `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` with that size. It does
so in place of the `ALLOW_VARYING` path, as Dawn does. An explicit `@subgroup_size`
still wins.

R3. When a default size exists, the adapter's `maxComputeInvocationsPerWorkgroup` is
clamped to `maxComputeWorkgroupSubgroups * default`, as Dawn does. Core validation
uses the clamped limit like any other adapter limit.

R4. Every flag that changes codegen is in the SPIR-V codegen key (Block 95).

## Tests

- Unit: R1 decisions for each branch, including the Imagination exclusion and the
  missing-stage case. R2 for equal min/max, `min * 2 <= max`, and `min * 2 > max`.
  The R3 clamp.
- Shim: each option adds its execution mode to the SPIR-V.
- Vulkan HAL `#[ignore]`: device creation reports the enabled reconvergence feature,
  if any. A compute pipeline without `@subgroup_size` is created with no validation
  errors.
- e2e: the existing Vulkan subgroup suites under the Khronos layer, with 0 VUIDs.
  Native Vulkan is owed, because MoltenVK exposes neither reconvergence extension and
  no stage in `requiredSubgroupSizeStages`.

## Slices

- **S1 (coding agent):** R1-R4 with unit tests.
- **S2 (Claude):** real-GPU runs, CTS, Phase Review.

## Deviations from Dawn

- **Default size needs compute in `requiredSubgroupSizeStages`.** Dawn's
  `FindDefaultComputeSubgroupSize` checks only the extension and `min != max`.
  MoltenVK reports a size range of 4 to 32 with `requiredSubgroupSizeStages` empty. On
  such a device, requiring a size is invalid (VUID-VkPipelineShaderStageCreateInfo-pNext-02755,
  225 times across the e2e suite in the first version). yawgpu returns no default size
  unless the `subgroupSizeControl` feature is enabled and the stages include compute.
- **No default size below 256.** Core raises `maxComputeInvocationsPerWorkgroup` to at
  least 256 (`yawgpu-core/src/limits.rs`). When `maxComputeWorkgroupSubgroups * default`
  is below 256, yawgpu returns no default size and no clamp, which avoids
  VUID-02756.
- **Size control needs WGSL subgroups support.** This is the existing Block 108 rule,
  so a device with varying sizes but no subgroup support gets neither the default size
  nor the clamp.

Each reconvergence device feature is enabled whenever it is present, as Dawn does,
so `shader-passthrough` SPIR-V can declare either execution mode. Dawn's rule picks
the Tint option, which is never both.

## Verification (2026-11-03)

Run on Apple M2, macOS 27.2; Vulkan is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,322 passed, 0 failed |
| fmt, clippy on 8 feature sets | clean |
| Vulkan HAL ignored tests | 75 passed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled` | 147 passed, 1 failed (known `imageView2DOn3DImage`); no other VUID |
| Metal e2e | 154 passed, 0 failed |

On MoltenVK neither reconvergence extension is present, and the default size is
None. So these runs show that no invalid usage is produced, but they do not exercise
the new paths. Metal CTS was not rerun: the change touches only SPIR-V options and the
Vulkan HAL. A native Vulkan run, ideally on an Intel GPU with varying subgroup sizes,
is owed. The ignored test
`vulkan_device_reconvergence_and_default_compute_pipeline` checks itself there.

### Windows native Vulkan (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`d7ff142`. The Khronos validation layer was on for the e2e and HAL runs.

`vulkan_device_reconvergence_and_default_compute_pipeline` passed and logged:

```text
reconvergence extensions: maximal=true, subgroup_uniform_control_flow=true; enabled features=ReconvergenceFeatures { maximal: true, uniform_control_flow: true }
Tint reconvergence: maximal=true, subgroup_uniform_control_flow=false; default compute subgroup size=None
subgroupSizeControl enabled=true; requiredSubgroupSizeStages=VERTEX | ... | FRAGMENT | COMPUTE | ...
```

Both device features are enabled and Tint gets maximal reconvergence only, as R1 and
"Deviations from Dawn" describe. The subgroup size is fixed at 32, so R2 and R3
correctly select no default size and no clamp. Those two rules still need a device
whose minimum and maximum sizes differ, such as Intel.

| Gate | Result |
|---|---|
| Vulkan HAL ignored tests | 75 passed, 0 failed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, including the subgroup suites | 157 passed, 0 failed, 0 VUIDs |
| webgpu-native-cts, `webgpu:*` | `fail=2 crash=0`; subgroup and quad queries have no failure |

The full CTS table is in Block 125's "Windows native Vulkan" section.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Only one reconvergence feature was enabled | Each present feature is enabled, as Dawn does; the Tint option stays exclusive |
| m2 | Default size depends on WGSL subgroup support | Recorded under Deviations (Block 108 rule) |
| m3 | Core can raise the clamped limit back to 256 | No default size below 256, with boundary tests |
| m4 | GPU test did not check the clamp or the decision | Asserts both, and logs extension presence |
| m5 | Stale status | Updated |

