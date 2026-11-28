# Block 170 — `chromium-experimental-subgroup-matrix`

Status: **S1 DONE (2026-11-28)** — feature, configuration query, WGSL gate,
reflection and both pipeline rules on Noop; workspace 1,526/0, HAL lib tests
with `metal` 215/0 and `vulkan` 381/0, Metal subgroup / subgroup-size-control /
compute e2e 11/0 on the M2. **S2 DONE (2026-11-28)** — Metal
advertises `{F32, F32}` and `{F16, F16}` 8×8×8 on Apple7; on the Apple M2
`e2e_metal_subgroup_matrix` is 4/0 (configuration query, multiply and
multiply-accumulate in both layouts, scalar arithmetic, the two pipeline
validation errors), each arithmetic test confirmed to fail under a mutated
CPU reference. The store-with-offset, out-of-bounds and fill-constructor
scenarios are owed in the shared e2e body. S3 (Vulkan on Windows / Linux) and
S4 (docs + Phase Review) not started.
Ledger: `specs/tracking/subgroups.md` ("Block 170" section).

Dawn's `ChromiumExperimentalSubgroupMatrix` feature enables the WGSL extension
`enable chromium_experimental_subgroup_matrix;`: the abstract matrix types
`subgroup_matrix_left<T, K, M>`, `subgroup_matrix_right<T, N, K>` and
`subgroup_matrix_result<T, N, M>`, whose storage is distributed over the
invocations of a subgroup, plus `subgroupMatrixLoad` / `subgroupMatrixStore` /
`subgroupMatrixMultiply` / `subgroupMatrixMultiplyAccumulate` and the scalar
arithmetic on them. It maps to MSL `simdgroup_matrix` and to SPIR-V
`SPV_KHR_cooperative_matrix` (`VK_KHR_cooperative_matrix`).

Tint ships the types, the builtins, both code generators and the usage analysis
upstream. This block adds what surrounds them: the feature, the adapter's list
of supported matrix configurations, the validation, and the backend enablement.

Oracle: `third_party/dawn/docs/dawn/features/subgroup_matrix.md`,
`src/dawn/native/{Adapter,Device,ShaderModule}.cpp`,
`src/dawn/native/metal/PhysicalDeviceMTL.mm`,
`src/dawn/native/vulkan/{PhysicalDeviceVk,DeviceVk,ShaderModuleVk,ComputePipelineVk}.cpp`,
`src/dawn/tests/end2end/SubgroupMatrixTests.cpp`.

## Scope

| Backend | Support |
|---|---|
| Vulkan (Windows, Linux, Android) | Supported when the driver meets R9. One code path for every OS: the predicate reads only Vulkan features and properties, never the host OS. |
| Metal | Supported on `MTLGPUFamilyApple7` and later (R8). |
| Noop | Advertised with one nominal configuration (R7) so that R1–R6 are testable without a GPU. |
| GLES | Not supported. Tint's GLSL writer rejects the type. Catalogued in [Block 67](67-gles-backend.md). |

MoltenVK does not expose `VK_KHR_cooperative_matrix`, so the Vulkan path is
verified on native Windows and Linux drivers only.

Out of scope: Metal 4 tensors (Dawn's `metal_enable_tensors` toggle, off by
default in Dawn), fragment-stage use, D3D12.

The WGSL surface is experimental in Dawn and can change at a Dawn pin bump.
This feature carries no SemVer guarantee.

## C ABI

The declarations live in `yawgpu/ffi/webgpu-headers/yawgpu.h`. They use Dawn's
names and Dawn's numeric values, so a program written against Dawn's
`webgpu.h` reads the same bytes. Dawn extension enums are `0x0005_0000 + n`
(`third_party/dawn/generator/dawn_json_parsing.py`).

```c
#define WGPUFeatureName_ChromiumExperimentalSubgroupMatrix ((WGPUFeatureName)0x00050034u)
#define WGPUSType_AdapterPropertiesSubgroupMatrixConfigs   ((WGPUSType)0x0005003Bu)

typedef enum WGPUSubgroupMatrixComponentType {
    WGPUSubgroupMatrixComponentType_F32 = 0x00000001,
    WGPUSubgroupMatrixComponentType_F16 = 0x00000002,
    WGPUSubgroupMatrixComponentType_U32 = 0x00000003,
    WGPUSubgroupMatrixComponentType_I32 = 0x00000004,
    WGPUSubgroupMatrixComponentType_U8  = 0x00000005,
    WGPUSubgroupMatrixComponentType_I8  = 0x00000006,
    WGPUSubgroupMatrixComponentType_Force32 = 0x7FFFFFFF
} WGPUSubgroupMatrixComponentType;

typedef struct WGPUSubgroupMatrixConfig {
    WGPUSubgroupMatrixComponentType componentType;
    WGPUSubgroupMatrixComponentType resultComponentType;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t minSubgroupSize;
    uint32_t maxSubgroupSize;
} WGPUSubgroupMatrixConfig;

/* Chained out-struct on WGPUAdapterInfo. */
typedef struct WGPUAdapterPropertiesSubgroupMatrixConfigs {
    WGPUChainedStruct chain;
    size_t configCount;
    const WGPUSubgroupMatrixConfig* configs;
} WGPUAdapterPropertiesSubgroupMatrixConfigs;

WGPU_EXPORT void wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers(
    WGPUAdapterPropertiesSubgroupMatrixConfigs value);
```

A multiply has the shape `Result(M rows × N columns) = Left(M × K) * Right(K × N)`.
`componentType` is the element type of the left and right operands;
`resultComponentType` is the element type of the result.

## Rules

Rules R1–R6 are `yawgpu-core` / FFI rules and hold identically on every
backend. R7–R10 are HAL rules.

### R1 — feature and implication

`yawgpu-core::Feature::ChromiumExperimentalSubgroupMatrix` maps to
`WGPUFeatureName_ChromiumExperimentalSubgroupMatrix` in both directions. The
adapter advertises it exactly when `HalAdapter::subgroup_matrix_configs()` is
non-empty. A device created with it also has `Feature::Subgroups`
(`apply_feature_implications`; Dawn `Device.cpp` `ApplyFeatures`).

### R2 — configuration query

`wgpuAdapterGetInfo` and `wgpuDeviceGetAdapterInfo` walk `info->nextInChain`:

- A `WGPUAdapterPropertiesSubgroupMatrixConfigs` node on an adapter that
  advertises the feature is filled with a heap array of every configuration
  and its count. The function returns `WGPUStatus_Success`.
- The same node on an adapter that does not advertise the feature makes the
  function return `WGPUStatus_Error`, and `*info` is left untouched (Dawn
  `Adapter.cpp` `ValidateAndUnpackAdapterInfo`). For
  `wgpuDeviceGetAdapterInfo` the test is the adapter's support, not whether
  the device enabled the feature.
- The caller's `nextInChain` pointer is preserved in the returned
  `WGPUAdapterInfo`, with or without a node on it.

`wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers` frees the array. A
null `configs` is accepted. The symbol is registered with `wgpuGetProcAddress`.

Each configuration's `minSubgroupSize` / `maxSubgroupSize` equal the adapter's
subgroup size range (`WGPUAdapterInfo::subgroupMinSize` / `subgroupMaxSize`).

### R3 — WGSL gate

`Device::create_shader_module` threads
`features.contains(Feature::ChromiumExperimentalSubgroupMatrix)` to the shim,
which inserts `tint::wgsl::Extension::kChromiumExperimentalSubgroupMatrix`
into the allowed extensions. Without the feature, a module containing
`enable chromium_experimental_subgroup_matrix;` fails `createShaderModule`
with a validation error.

### R4 — reflection

The shim reports, per compute entry point:

- `uses_subgroup_matrix` — `tint::inspector::EntryPoint::uses_subgroup_matrix`.
- The entry point's `tint::SubgroupMatrixInfo` (the `configs` and `multiplies`
  sets of `src/tint/api/common/subgroup_matrix.h`), gathered with
  `tint::core::ir::analysis::GatherSubgroupMatrixInfo` on the same
  single-entry-point, override-substituted IR that
  `yawgpu_tint_resolved_workgroup_size` builds, before any writer raise. The
  result is independent of the target language.

The shim returns both sets in a deterministic order (sorted), because Tint
stores them in unordered sets.

### R5 — workgroup size

At `createComputePipeline`, for an entry point with `uses_subgroup_matrix`
and no `@subgroup_size`: `workgroup_size.x` must be a multiple of the
adapter's maximum subgroup size (Dawn `ShaderModule.cpp`
`ValidateComputeStageWorkgroupSize`). An entry point with `@subgroup_size`
is covered by the existing `validate_explicit_subgroup_size` rule instead.

The device captures the adapter's subgroup size range and configuration list
at creation.

### R6 — configuration validation

At `createComputePipeline`, the entry point's `SubgroupMatrixInfo` is checked
against the adapter's configurations (Dawn `ShaderModule.cpp`
`ValidateSubgroupMatrixConfiguration`):

- Every used matrix type must match at least one configuration `c`:
  - result: `c.resultComponentType == type && c.M == M && c.N == N`
  - left: `c.componentType == type && c.M == M && c.K == K`
  - right: `c.componentType == type && c.N == N && c.K == K`
- Every multiply must match one configuration on all of `M`, `N`, `K`,
  `componentType == input_type` and `resultComponentType == output_type`.

A mismatch is a validation error naming the offending shape and type.

Dawn runs this check in its Vulkan backend and relies on Tint's MSL
`ValidateSubgroupMatrix` transform on Metal. With the Metal configuration list
of R8 the two accept the same shaders, so yawgpu runs the one core check on
every backend.

### R7 — Noop

`subgroup_matrix_configs()` returns one configuration:
`{F32, F32, M 8, N 8, K 8}` with the Noop subgroup size range.

### R8 — Metal (S2)

Advertised when `[MTLDevice supportsFamily:MTLGPUFamilyApple7]`. Two
configurations, both `M = N = K = 8`: `{F32, F32}` and `{F16, F16}` (Dawn
`PhysicalDeviceMTL.mm`). Tint's MSL writer runs with `enable_tensors = false`.

### R9 — Vulkan advertisement (S3)

`subgroup_matrix_configs()` is non-empty only when all of these hold (Dawn
`PhysicalDeviceVk.cpp`):

1. The adapter advertises `Feature::Subgroups`.
2. `VK_KHR_cooperative_matrix` is present, `cooperativeMatrix` is true, and
   `cooperativeMatrixSupportedStages` contains `COMPUTE`.
3. `vulkanMemoryModel` and `vulkanMemoryModelDeviceScope` are true.
4. `subgroupSizeControl` and `computeFullSubgroups` are true.
5. The driver is not Intel Mesa older than 25.2.

The list is `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` filtered to
entries with `AType == BType`, `CType == ResultType`,
`scope == VK_SCOPE_SUBGROUP_KHR`, no `saturatingAccumulation`, component types
among the six of `WGPUSubgroupMatrixComponentType`, and `F16` only when the
adapter advertises `shader-f16`.

Deviation from Dawn: Dawn advertises on conditions 1–5 alone and can return an
empty list. yawgpu also requires the filtered list to be non-empty, so an
advertised feature is always usable.

### R10 — Vulkan device and pipeline (S3)

- `HalDeviceFeatures` gains a `subgroup_matrix` request. When set, the device
  enables `VK_KHR_cooperative_matrix` with `cooperativeMatrix`, and `shaderInt8`
  when the driver has it (8-bit matrix components need it).
- When the driver has `cooperativeMatrixRobustBufferAccess` and the device
  enabled robust buffer access 2, the device enables it and Tint's SPIR-V
  writer runs with `disable_storage_subgroup_matrix_clamping = true`;
  otherwise Tint clamps.
- A compute pipeline whose entry point has `uses_subgroup_matrix` is created
  with `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`, with or
  without `@subgroup_size` (Dawn `ComputePipelineVk.cpp`).
- Tint's SPIR-V writer already receives `use_vulkan_memory_model`; cooperative
  matrix SPIR-V is rejected by Tint without it, and R9.3 guarantees it.

## Slices

1. **S1 — feature, query, gate, validation (Noop).** R1–R7. Metal, Vulkan and
   GLES return an empty configuration list.
2. **S2 — Metal.** R8 and `e2e_metal_subgroup_matrix.rs`.
3. **S3 — Vulkan.** R9, R10, a configuration-dump probe, and
   `e2e_vulkan_subgroup_matrix.rs`, run on native Windows and Linux drivers
   under `VK_LAYER_KHRONOS_validation`.
4. **S4 — docs and Phase Review.** Tracking ledger, Block 67 matrix row,
   `README` feature table.

## Verification

The CTS has no cases for this feature. The oracle is Dawn's
`SubgroupMatrixTests.cpp`.

S1, inline unit tests on Noop:

- R1: feature round-trip in `conv/feature.rs`; Noop adapter advertises it;
  a device created with only this feature also has `Subgroups`.
- R2: query with the node succeeds and returns the R7 configuration with the
  adapter's subgroup range; `nextInChain` is preserved; the node on an
  adapter without the feature returns `WGPUStatus_Error`; the same three
  through `wgpuDeviceGetAdapterInfo`; `FreeMembers` accepts a filled and a
  zeroed struct.
- R3: the `enable` directive is rejected without the feature and accepted
  with it.
- R4: an entry point using an 8×8 f32 left, right and result plus one
  multiply reports `uses_subgroup_matrix` and the three configs and one
  multiply in sorted order; a sibling entry point that uses no matrix reports
  `false` and empty sets.
- R5: `@workgroup_size(4)` with the Noop maximum subgroup size 4 is accepted;
  `@workgroup_size(6)` is rejected; an entry point without a matrix and
  `@workgroup_size(6)` is accepted.
- R6: the 8×8 f32 shader is accepted; a 4×4 f32 matrix, an 8×8 `i32` matrix,
  and a multiply whose result type differs from the configuration are each
  rejected.

S2 / S3, real GPU (ported from the Dawn tests): configuration query validity,
every advertised configuration compiles and runs, matrix × matrix multiply and
multiply-accumulate against a CPU reference in both memory layouts, matrix ×
scalar arithmetic, store with offset, out-of-bounds load / store leaving
neighbouring data intact, and the fill constructor.
