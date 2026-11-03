# Block 131 — Dawn's vendor-gated Tint workarounds

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 5 MINOR; all fixed,
see below. The forced native NVIDIA CTS ran on 2026-11-03: 11 of the 12 SPIR-V options
change no result; `texture_sample_compare_2d_polyfill` fails 7,820 sampling subcases
when forced, see "Windows native Vulkan" below.
Source: backlog A13 (Block 120 "Out of scope"), started by user request on 2026-11-03.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/`):
- `native/metal/PhysicalDeviceMTL.mm`: vendor detection (lines 70-159) and toggle defaults
  (lines 482-505, 610-612)
- `native/vulkan/PhysicalDeviceVk.cpp`: vendor predicates (lines 1520-1610), driver-version
  decode (lines 64-89), and toggle defaults (lines 1076-1220)
- `native/metal/ShaderModuleMTL.mm` (lines 333-349) and `native/vulkan/ShaderModuleVk.cpp`
  (lines 276-369): mapping from toggle to Tint option
- `common/GPUInfo.cpp`: vendor ids and the driver-version comparison

## Problem

Block 120 wired the Tint driver workarounds that Dawn added at the `v20261030` bump.
Dawn sets about twenty more Tint options from vendor and driver checks. yawgpu's Tint
shim does not expose them, and `HalShaderWorkarounds` has no gate for them. So on the
affected hardware, yawgpu emits the shader that each workaround exists to avoid. That
means wrong results or a driver crash on AMD Mesa below 25.3, Qualcomm, ARM,
Imagination, Intel Mesa, AMD and Intel Macs, and Apple9 (M3 and later) GPUs.

None of these gates fires on the hosts we have: Apple M2 Metal, MoltenVK, and native
NVIDIA on Windows and Linux. A survey of every row confirmed it. Our hosts can show only
two things: that each forced option keeps shaders working, and that the gates choose
correctly from device facts in unit tests.

## Rules

R1. **Shim.** The shim exposes each Tint option below and passes it into the MSL or
SPIR-V writer options. Every option is in the codegen cache key, as Block 95 requires.
The two generate functions already take many positional flags. Group the new ones, and
the existing workaround flags, into one `#[repr(C)]` struct per writer. Guard each
struct with the existing ABI layout assertions.

R2. **MSL gates.** `HalShaderWorkarounds::for_metal` takes two new inputs: `is_amd` and
`supports_apple9`. It sets the options below. `is_intel` already exists.

| Tint MSL option | Dawn gate |
|---|---|
| `disable_module_constant_f16` | Intel |
| `polyfill_unpack_2x16_snorm` | AMD, or `MTLGPUFamilyApple9` |
| `polyfill_unpack_2x16_unorm` | `MTLGPUFamilyApple9` |
| `polyfill_tanh_f16` | AMD |
| `polyfill_clamp_float` | AMD |
| `collapse_subgroup_min_max` | AMD |
| `replace_workgroup_bool_with_u32` | AMD or Intel |
| `scalarize_max_min_clamp` | no Metal gate; force only |

- **Vendor detection:** Dawn reads the IOKit PCI `vendor-id`. When that fails, it
  matches the device name against "AMD", "Apple", and "Intel". yawgpu keeps its
  name-based detection and adds AMD the same way, matching "AMD" or "Radeon".
- **Apple9:** comes from `MetalGpuFamily::Apple9`, which the HAL already queries.
- **Platforms:** none of these gates has a platform guard in Dawn, so the Apple9 gate
  also applies on iOS.

R3. **Vulkan device facts.** The Vulkan HAL queries `VkPhysicalDeviceDriverProperties`
only when the device advertises `VK_KHR_driver_properties`. Dawn's
`DeviceExt::DriverProperties` is set by the extension name alone: the "Promoted in 1.2"
group in `VulkanExtensions.cpp` gets no core-version promotion. That is the same finding
as Block 128 for descriptor indexing. A Vulkan 1.2 device that does not list the
extension therefore has no driverID, and it counts as "may be proprietary". The HAL
passes the `driverID`, or none, to `for_vulkan` together with the device id. These predicates are
pure functions with unit tests, and they follow Dawn's definitions:
- `is_amd_mesa`: driverID is `MESA_RADV`.
- `is_intel_mesa`: driverID is `INTEL_OPEN_SOURCE_MESA`.
- `may_be_{qualcomm,arm,imagination}_proprietary`: the vendor id matches (0x5143,
  0x13B5, 0x1010), and either there is no driverID or it is that vendor's proprietary
  driverID.
- `is_windows_amd`: a Windows build and vendor 0x1002.
- `is_android_qualcomm`: an Android build and vendor 0x5143.
- `is_swiftshader`: Google's vendor id and SwiftShader's device id, as in Dawn's
  `gpu_info`.

The Mesa version comparison uses the standard Vulkan decode. It keeps Dawn's
lexicographic quirk, which Block 120 already copies for Samsung: the decoded 25.3.0 is
below the gate `{25, 3, 0, 0}`.

R4. **SPIR-V gates.**

| Tint SPIR-V option | Dawn gate |
|---|---|
| `polyfill_float_negation`, `polyfill_float_abs`, `polyfill_length_scalar_float`, `polyfill_distance_scalar_float` | AMD Mesa with a driver below `{25, 3, 0, 0}` |
| `polyfill_case_switch` | may be Imagination proprietary |
| `texture_sample_compare_2d_polyfill` | may be Qualcomm proprietary |
| `scalarize_max_min_clamp` | may be Qualcomm proprietary |
| `polyfill_pack_unpack_4x8_norm` | may be ARM proprietary |
| `dva_transform_handle` | may be Qualcomm proprietary, may be ARM proprietary, or SwiftShader |
| `polyfill_saturate_as_min_max_f16` | Intel Mesa |
| `collapse_subgroup_min_max` | Windows AMD |
| `pass_matrix_by_pointer` | Android Qualcomm (set directly in `ShaderModuleVk.cpp`, not a toggle) |

Dawn's four AMD Mesa polyfills come from two toggles: `polyfill_float_negation`, and
`polyfill_float_abs`, which sets the other three. Both toggles share one gate.

R5. **Forcing.** `YAWGPU_FORCE_SHADER_WORKAROUNDS` accepts each new option by its Tint
name, as Block 120's flags already do. Each name forces the option for the writer that
has it. `scalarize_max_min_clamp` and `collapse_subgroup_min_max` exist in both writers,
so their names force both.

R6. **Unchanged.**
- Default output on our hosts: every gate is off there.
- GLSL: Dawn sets none of these for GLES.
- MoltenVK: it gets only the Vulkan gates, which do not fire for Apple.

## Tests

- Unit (HAL):
  - Every gate in R2 and R4: on for its vendor, off for the others.
  - Every R3 predicate, including the cases with no driverID.
  - The Mesa version boundary: 25.2.x and 25.3.0 are bad, 25.3.1 is good.
- Unit (`parse_forced`): every new name.
- Unit (shim): each option changes the generated code for a shader that uses the
  affected construct. For example, `tanh` on f16 for `polyfill_tanh_f16`, or a `switch`
  for `polyfill_case_switch`.
- Unit (core): the codegen key differs when any new flag differs.
- e2e (Claude), on the M2 Metal and on MoltenVK, under the Khronos layer:
  - The existing suites with every new option forced.
  - The Vulkan HAL test logs the driverID and the predicates.
- CTS (Claude): `webgpu:shader,execution,*` on Metal with every new MSL option forced. It
  must equal the unforced baseline.
- Native NVIDIA (owed to the Windows host): `webgpu:*` with every new SPIR-V option
  forced, and the HAL log of the driverID.

## Out of scope

- `polyfill_subgroup_broadcast_f16` for MSL and SPIR-V, and Dawn's Intel Gen9 subgroups
  gate. Dawn has no default for the option, and the gate needs the PCI device id on both
  backends. Backlog A21.
- `texture_sample_compare_depth_cube_array` (32-bit Qualcomm builds only; yawgpu ships
  no 32-bit Android target).
- `cooperative_matrix_stride_is_matrix_elements` (yawgpu has no subgroup-matrix feature).
- SPIR-V 1.4 (Dawn's `UseSpirv14`). It is a Block 124 survey item, not a driver
  workaround, and it does change output on our hosts. It gets its own block.

## Slices

- **S1 (coding agent):** R1-R6 with the unit tests.
- **S2 (Claude):** forced e2e, the Metal CTS, and the Phase Review.

## Verification (2026-11-03)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only. No gate fires on this host. MoltenVK reports vendor `0x106b` and
driverID 14 (MoltenVK), and the HAL test logs every predicate as false. So every
workaround path below ran forced, through `YAWGPU_FORCE_SHADER_WORKAROUNDS`.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,356 passed, 0 failed |
| fmt, clippy on 8 feature sets | clean |
| Shim unit tests, real Tint: each of the 20 options changes the generated code | 20 of 20 pass |
| Metal e2e, `metal,tiled`, the 8 MSL options forced | 161 passed, 0 failed |
| Vulkan HAL ignored tests, the 12 SPIR-V options forced | 78 passed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled`, the 12 SPIR-V options forced | 159 passed, 1 failed (the known `imageView2DOn3DImage`). Every VUID comes from that test and its leaked objects |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*`, unforced baseline (Block 130 library) | pass 853,787, skip 29,466, fail 0, crash 0 |
| The same query, the 8 MSL options forced (this block) | pass 853,787, skip 29,466, fail 0, crash 0; identical to the baseline |

The forced CTS library was built before the Phase Review fixes. Those fixes change only
vendor detection, a helper, and tests, not what a forced option emits.

Owed to the Windows NVIDIA host: `webgpu:*` with the 12 SPIR-V options forced, and the
HAL log of the driverID. NVIDIA takes no gate by default either, so its default output
is unchanged.

### Windows native Vulkan (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`f3a63e9`, CTS `a5cbf67`. The Khronos validation layer was on for the e2e and HAL runs.
The HAL test logs `vendorID=0x10de, deviceID=0x2d04, driverID=Some(4)` (NVIDIA
proprietary) and every predicate false, so no gate fires by default. The forced runs
set all 12 SPIR-V options through `YAWGPU_FORCE_SHADER_WORKAROUNDS`.

| Gate | Result |
|---|---|
| Vulkan HAL ignored tests, default and forced | 78 passed each, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, default and forced | 162 passed each, 0 failed, 0 VUIDs |
| webgpu-native-cts `webgpu:*`, 12 options forced, 4 per-area runs, `--workers 8`, 32 minutes | `fail=8,254 crash=0`; attributed below |

| Area | pass | fail | cause of the failures |
|---|---:|---:|---|
| `api,validation` | 246,052 | 432 | not this block: Block 130, backlog A22; they fail unforced too |
| `api,operation` | 180,159 | 620 | `texture_sample_compare_2d_polyfill`: `texture_component_swizzle:read_swizzle` with `textureSampleCompare(Level)` |
| `shader,execution` | 555,348 | 7,200 | `texture_sample_compare_2d_polyfill`: `textureSampleCompare:2d_coords` and `arrayed_2d_coords` |
| `shader,validation` | 662,535 | 2 | the Block 119 R9 cases |

The `textureSampleCompare` and `read_swizzle` queries isolate the option. With no option
forced, they give 0 failures. With `texture_sample_compare_2d_polyfill` alone, they give
7,820, the full forced count. With the other 11 options forced, they give 0. The failures
are split evenly between `nearest` and `linear` samplers. The polyfill replaces the
sample with `OpImageDrefGather` plus manual bilinear weights computed from the level-0
size (`spirv/writer/raise/builtin_polyfill.cc`). yawgpu only sets the option, so the
generated code is Tint's. Whether Dawn with `vulkan_sample_compare_2d_workaround` fails
the same cases is not verified: the CTS harness has no hook for Dawn toggles. The option
fires only on Qualcomm proprietary drivers, which no host here has.

The other 11 SPIR-V options forced change nothing on this host. In every area, pass
plus the failures attributed above equals the pass count of the unforced `019bd3b` run
(Block 129), and skips are unchanged.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| — | R3 asked for the driverID query on Vulkan 1.2 or the extension. Dawn checks only the extension name. Found in review before the Phase Review | Spec and code are now extension-only |
| m1 | `driver_properties_available` was the identity function, and its test was tautological | Removed. `query_driver_id` checks the extension and says why |
| m2 | Metal vendor matching was inline, lowercase, and untested | `metal_vendor_flags` now copies Dawn's case-sensitive trademark table in Dawn's order, with a unit test |
| m3 | No test case for Qualcomm ACPI `0x4D4F4351` or Samsung | Both added to the vendor matrix; neither fires a Qualcomm gate |
| m4 | The Samsung comparison comment was separated from its code | Moved back; `bad_amd_mesa` has its own comment |
| m5 | No test caught a HAL-to-Tint field mismatch in core | `hal_workarounds_map_only_to_same_named_tint_flags` sets one field at a time |
