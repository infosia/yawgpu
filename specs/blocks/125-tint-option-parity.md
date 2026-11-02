# Block 125 — Tint writer and pipeline flags: Dawn parity, part 1

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 7 MINOR;
m1-m4 and m6 fixed, m5 and m7 recorded below. Native Vulkan ran on 2026-11-03;
see "Windows native Vulkan" below.
Source: user request (2026-11-02): "fix similar easy items as they are found". These
come from a read-only survey of Dawn's backend setup, run during Block 124. Every
item below was checked against the source.
Oracle: Dawn at the pinned `third_party/dawn`: `native/vulkan/ShaderModuleVk.cpp`,
`native/vulkan/PhysicalDeviceVk.cpp`, `native/vulkan/RenderPipelineVk.cpp`,
`native/vulkan/DeviceVk.cpp`, `native/metal/ShaderModuleMTL.mm`,
`native/metal/PhysicalDeviceMTL.mm`, and Tint `lang/spirv/writer/common/options.h`
and `lang/msl/writer/common/options.h`.

## Problem

Each item is a Tint option or pipeline flag that Dawn sets from the device and yawgpu
leaves at its default. No CTS host sees any of them.

| # | Backend | yawgpu | Dawn | Effect |
|---|---|---|---|---|
| 1 | Vulkan | `extensions.use_demote_to_helper_invocation` unset (false), though the HAL enables `shaderDemoteToHelperInvocation`. Tint emulates `discard` with a flag, masked stores, and `OpKill`. | true when the extension is enabled (`VulkanUseDemoteToHelperInvocationExtension`) | Discarded fragments run to the end as non-helper invocations. Slower, and subgroup operations after `discard` see them. HAL comments (F-129) claim the native path. |
| 2 | Vulkan | `extensions.dot_4x8_packed` unset (false); `shaderIntegerDotProduct` never enabled | Polyfills when the feature is missing (`PolyFillPacked4x8DotProduct`), otherwise enables the feature | `dot4I8Packed` / `dot4U8Packed` emit `OpSDot` / `OpUDot` with capabilities the device never enabled: invalid SPIR-V everywhere. |
| 3 | Vulkan | `extensions.use_vulkan_memory_model` follows `vulkanMemoryModel` alone | Requires `vulkanMemoryModel` and `vulkanMemoryModelDeviceScope` | Tint declares `VulkanMemoryModelDeviceScopeKHR` whenever the model is on: invalid SPIR-V on a device without device scope. |
| 4 | Vulkan | `emit_vertex_point_size` unset (true): every vertex shader writes `PointSize` | Only for point-list pipelines | Extra output on every vertex shader. Results unchanged. |
| 5 | Vulkan | Fragment stage has no create flags | `ALLOW_VARYING_SUBGROUP_SIZE_BIT` on the fragment stage when `subgroupSizeControl` is enabled | `subgroup_size` in a fragment shader may report the nominal size, not the actual one. |
| 6 | Vulkan | `extensions.use_zero_initialize_workgroup_memory` unset; extension never enabled | Enabled when `VK_KHR_zero_initialize_workgroup_memory` is supported, except on ARM and Imagination | Tint zero-fills workgroup memory with a loop and barrier. Slower only. |
| 7 | Metal | `disable_demote_to_helper` unset (false) | true on non-Apple GPUs | The demote transform runs on Intel and AMD Macs. Slower only. Apple GPUs are unaffected. |

## Rules

R1. Vulkan SPIR-V generation sets `extensions.use_demote_to_helper_invocation` to
whether the device enabled `shaderDemoteToHelperInvocation`. The HAL comments then
describe the real path.

R2. Vulkan: when `shaderIntegerDotProduct` is supported (Vulkan 1.3 core, or
`VK_KHR_shader_integer_dot_product`), the HAL enables it and SPIR-V uses the native
instructions. Otherwise SPIR-V generation sets `extensions.dot_4x8_packed = true`, so
Tint polyfills. Follow Dawn's toggle and feature logic in `PhysicalDeviceVk.cpp` and
`DeviceVk.cpp`.

R3. Vulkan: the memory model is enabled, and `use_vulkan_memory_model` is set, only
when both `vulkanMemoryModel` and `vulkanMemoryModelDeviceScope` are supported.

R4. Vulkan: `emit_vertex_point_size` is true only for pipelines with point-list
topology, as Dawn and as the Metal path already does. The flag becomes part of the
SPIR-V codegen key (Block 95), as it is for MSL.

R5. Vulkan: when `subgroupSizeControl` is enabled, the fragment stage gets
`VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT`. Tiled subpass
fragment stages follow the same rule.

R6. Vulkan: when `VK_KHR_zero_initialize_workgroup_memory` (or Vulkan 1.3
`shaderZeroInitializeWorkgroupMemory`) is supported and the vendor is neither ARM
(`0x13B5`) nor Imagination (`0x1010`), the HAL enables it and SPIR-V generation sets
`extensions.use_zero_initialize_workgroup_memory = true`. Follow Dawn's vendor gate.

R7. Metal: MSL generation sets `disable_demote_to_helper = true` on non-Apple GPUs,
using the same Apple test as Block 120.

R8. Every flag is a pure function of device facts, unit-tested without a GPU. Every
flag that changes codegen output is in the matching Block 95 codegen key.

## Tests

- Unit: each R1-R7 decision, on and off. Codegen keys differ when a flag differs.
- Shim: each new option changes the SPIR-V or MSL in the expected way. For example,
  `OpDemoteToHelperInvocation` appears with R1 on, and no `PointSize` without point
  list.
- Vulkan HAL `#[ignore]` (MoltenVK here, native later): device creation reports the
  enabled features, and a `discard` + derivative fragment shader and a
  `dot4U8Packed` compute shader run with no validation errors.
- e2e: the existing Vulkan and Metal suites under the Khronos layer, with 0 VUIDs.
- CTS on Metal, `webgpu:*`: `fail=2 crash=0`. Native Vulkan is owed to a native host:
  MoltenVK is reference-only.

## Out of scope (survey items for later blocks)

- `frag_depth` is not clamped on Vulkan devices without `VK_EXT_depth_clip_enable`
  (Block 126).
- Maximal reconvergence and subgroup uniform control flow.
- Default compute subgroup size (Intel heuristic).
- robustness2 and pipeline robustness, SPIR-V 1.4, extended dynamic state,
  `VkPipelineCache` (E3), `strip_all_names`.

## Slices

- **S1 (coding agent):** R1-R8 with the unit tests and the HAL ignored tests.
- **S2 (Claude):** real-GPU runs, CTS, Phase Review.

## Verification (2026-11-03)

Run on Apple M2, macOS 27.2; Vulkan is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,317 passed, 0 failed |
| fmt, clippy on default, `metal`, `metal,tiled`, `metal,shader-passthrough`, `vulkan`, `vulkan,tiled`, `vulkan,shader-passthrough`, `gles` | clean |
| Metal HAL ignored tests | 63 passed |
| Vulkan HAL ignored tests | 74 passed, 0 VUIDs, including feature report, `discard` + derivative readback, `dot4U8Packed` |
| Metal e2e | 154 passed, 0 failed |
| Vulkan e2e | all pass except the known `imageView2DOn3DImage` case and Block 126's `frag_depth_is_clamped_to_viewport`, which is that block's red check |
| `packed_dot_products` e2e (`yawgpu/tests/common/shader_parity.rs`) | passes on Metal and MoltenVK |
| webgpu-native-cts, Metal, `webgpu:*` | `fail=2 crash=0`; subcase counts unchanged |

Native Vulkan was owed at the time of this run; see the next section.

### Windows native Vulkan (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`d7ff142`. The Khronos validation layer was on for the e2e and HAL runs. This host takes the
feature-present side of every Vulkan rule: `shaderDemoteToHelperInvocation`,
`shaderIntegerDotProduct`, `vulkanMemoryModel` with `vulkanMemoryModelDeviceScope`,
`shaderZeroInitializeWorkgroupMemory`, and `subgroupSizeControl` are all present, and the
vendor is NVIDIA. So R1-R6 ran their native paths, not the polyfills. The same run covers
Blocks 121, 126, and 127.

| Gate | Result |
|---|---|
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored` | 157 passed, 0 failed, 0 VUIDs |
| Vulkan HAL ignored tests | 75 passed, 0 failed, 0 VUIDs |
| `packed_dot_products` e2e | passes |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, `expectations/yawgpu-vulkan.txt` | `fail=2 crash=0` |

The CTS ran as four per-area runs with `--workers 8`, as the 2026-11-01 Windows
baseline did, and took 35 minutes.

| Area | pass | skip | fail | xfail | baseline pass |
|---|---:|---:|---:|---:|---:|
| `api,validation` | 246,484 | 111,283 | 0 | 4 | 246,484 |
| `api,operation` | 180,779 | 152,819 | 0 | 0 | 180,779 |
| `shader,execution` | 562,548 | 320,680 | 0 | 25 | 562,543 |
| `shader,validation` | 662,535 | 20,299 | 2 | 0 | 662,535 |
| **total** | **1,652,346** | **605,081** | **2** | **29** | **1,652,341** |

The 2 failures are the Block 119 R9 cases (`have_all_language_features` for
`fragment_depth` and `unrestricted_aliasing`). The 29 xfails are the documented
non-defects in the expectations file. The 93 xpasses are passing subcases inside those
case-level entries (F-085 `sample_mask`, `index_buffer_format_dirtying`) and the
host-dependent F-152 `corr` entry. `shader,execution` gains 5 passes and loses the
baseline's 1 crash: that is the intermittent F-165 driver crash, which did not fire.
Every other area matches the baseline exactly.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Default flag meant native dot product with no feature | Inverted to `native_dot_4x8_packed`; default polyfills |
| m2 | Writer flags assembled inline | `vulkan_writer_flags`, unit-tested |
| m3 | Tautological decision tests; dormant Vulkan 1.3 branch | Explicit cases; comment on the dormant branch |
| m4 | R6 gates the feature enable too, unlike Dawn | Comment; no behavioural difference |
| m5 | With R1, MoltenVK lowers demote to `discard_fragment()`, which Dawn avoids on Apple GPUs for per-sample shading (crbug.com/562093713) | Recorded; MoltenVK is reference-only |
| m6 | `discard` test had no readback | Reads both sides of the boundary |
| m7 | `YAWGPU_FORCE_SHADER_WORKAROUNDS` cannot force the new flags | Kept: they are derived from device features, not driver workarounds |

