# Block 133 — Vulkan SPIR-V 1.4, as Dawn

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 3 MINOR; all fixed,
see below. On Windows NVIDIA, SPIR-V 1.4 makes the F-165 driver crash much more frequent;
kept as Dawn by user decision, see Verification.
Source: the Block 124 survey (backlog "Recommended order" 1), carried as Block 131 "Out
of scope". Started by user request on 2026-11-03.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/`):
- `native/vulkan/PhysicalDeviceVk.cpp` (`UseSpirv14` default, `IsAndroidARM`,
  `IsAdreno7xx`)
- `native/vulkan/ShaderModuleVk.cpp` (`tintOptions.spirv_version`)
- `native/vulkan/VulkanExtensions.cpp` (`Spirv14` = `VK_KHR_spirv_1_4`, which depends on
  `ShaderFloatControls` = `VK_KHR_shader_float_controls`)
- `platform/DawnPlatform.cpp` (`kWebGPUUseSpirv14` is `true` on the default platform)
- `gpu_info.json` (Adreno 7xx device ids)

## Problem

Dawn's default platform enables `kWebGPUUseSpirv14`. So on every Vulkan device that has
`VK_KHR_spirv_1_4` with its `VK_KHR_shader_float_controls` dependency, Dawn sets Tint's
`spirv_version` to SPIR-V 1.4, except on Android ARM and Adreno 7xx. yawgpu always emits
SPIR-V 1.3 and does not enable either extension.

So on those devices yawgpu and the CTS oracle compile different SPIR-V from the same WGSL.
The Windows NVIDIA host (RTX 5060 Ti, driver 617.14) has both extensions, so Dawn uses
SPIR-V 1.4 there and yawgpu does not. A driver difference between the two versions is
invisible to the CTS: Dawn passes on 1.4 and yawgpu passes on 1.3.

yawgpu creates its instance and devices at Vulkan 1.1 (`YAWGPU_VULKAN_API_VERSION`), so
SPIR-V 1.4 needs both extensions enabled. Dawn enables every supported known extension
by name (`DeviceVk.cpp`, `usedKnobs.extensions`), including these two, independently of
the SPIR-V version it emits.

## Rules

R1. **Decision.** The Vulkan HAL computes `use_spirv_1_4` once per device. It is true
when all of these hold:
- the device reports `VK_KHR_spirv_1_4`
- the device reports `VK_KHR_shader_float_controls`
- not Android ARM: target is Android and the vendor is ARM (`0x13B5`)
- not Adreno 7xx, by Dawn's `gpu_info.json` tables:
  - Qualcomm PCI vendor `0x5143`, `deviceId & 0xFF000000` is one of `0x07000000`,
    `0x43000000`, `0x36000000`, `0x37000000`
  - Qualcomm ACPI vendor `0x4D4F4351`, `deviceId & 0xFFFFFF00` is one of `0x37314400`,
    `0x36334300`

The decision is a pure function of those facts, with a unit test for each branch. Reuse
the Block 131 vendor constants and helpers where they exist.

R2. **Extensions.** Device creation enables `VK_KHR_spirv_1_4` and
`VK_KHR_shader_float_controls` whenever the device reports both, as Dawn does, including
on Android ARM and Adreno 7xx where the R1 decision is false. Neither has a feature struct
to chain. The extension condition is the first two bullets of R1; the exclusions apply
only to the SPIR-V version. So a `shader-passthrough` SPIR-V 1.4 module is accepted on
those devices, as under Dawn.

R3. **Codegen.** `use_spirv_1_4` reaches core through `HalShaderWorkarounds`, as
Block 125's writer features do. Every SPIR-V generation sets Tint's
`options.spirv_version` to `SpvVersion::kSpv14` when it is true, and to `kSpv13` (today's
default) otherwise. This includes the tiled subpass shaders. The shim exposes the option.

R4. **Codegen key.** The flag is part of `SpirvCodegenKey` (Block 95).

R5. **Unchanged.**
- Metal, GLES, and Noop.
- `shader-passthrough` SPIR-V is passed through as given. With R2 a 1.4 module can now
  be accepted on such devices; nothing else changes.
- MoltenVK takes the same rule. It is reference-only.

## Tests

- Unit (HAL): R1 on and off for each condition: either extension missing, Android ARM,
  each Adreno 7xx PCI and ACPI id, an Adreno 6xx and 8xx id (not excluded), NVIDIA.
- Unit (HAL): R2: the extension-list helper adds both names exactly when both are
  reported, independent of the R1 exclusions. The real device-creation list is checked
  only by the ignored test below.
- Unit (shim, real Tint): with the option on, the module header's version word is
  `0x00010400`; off, `0x00010300`.
- Unit (core): the codegen key differs when the flag differs, and the flag reaches the
  SPIR-V request.
- Vulkan HAL `#[ignore]` (real GPU): device creation logs the decision and the two
  extension names; a compute pipeline and a render pipeline are created. The decision it
  asserts is computed with the R1 function, so it proves the wiring, not the decision;
  the table test covers the decision. VUIDs are not captured by the test: run it under
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` and check the output.
- e2e and CTS (Claude): the Vulkan e2e under the Khronos layer, and `webgpu:*` on
  Windows native Vulkan, where the decision is true. Compare with the 1.3 run at
  `ba0b289` (Block 132).

## Out of scope

- Raising `YAWGPU_VULKAN_API_VERSION`.
- SPIR-V validation (`spirv-val`) inside yawgpu, which Dawn runs only under a toggle.
- Other Block 124 survey items: extended dynamic state, `VkPipelineCache`.

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests.
- **S2 (Claude):** Vulkan e2e, CTS on Windows native Vulkan, Phase Review.

## Verification (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351. The
decision is true on this host: the HAL test logs `vendorID=0x10de, deviceID=0x2d04,
decision=true, VK_KHR_spirv_1_4 enabled=true, VK_KHR_shader_float_controls enabled=true`.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,366 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default (workspace) and `vulkan,tiled` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, `yawgpu-tint`), all targets | clean |
| Shim, real Tint: `spirv_version_option_sets_module_header` | passes (`0x00010400` on, `0x00010300` off) |
| Vulkan HAL ignored tests, Khronos layer | 79 passed, 0 failed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored`, Khronos layer | 163 passed, 0 failed, 0 VUIDs |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, `expectations/yawgpu-vulkan.txt`, before the m1 fix | `fail=2 crash=5` |

The m1 fix changes only which devices enable the two extensions. On this host both are
enabled either way, so the CTS was not rerun after it; the gates, HAL tests, and e2e above
ran after it.

| Area | pass | skip | fail | crash | Block 132 run (SPIR-V 1.3) pass |
|---|---:|---:|---:|---:|---:|
| `api,validation` | 246,484 | 111,283 | 0 | 0 | 246,484 |
| `api,operation` | 180,779 | 152,819 | 0 | 0 | 180,779 |
| `shader,execution` | 562,523 | 320,680 | 0 | 5 | 562,547 |
| `shader,validation` | 662,535 | 20,299 | 2 | 0 | 662,535 |

The 2 failures are the Block 119 R9 cases. Every other difference from the SPIR-V 1.3 run
is the 5 crashes and the passes they take with them (the Block 132 run's one
`memory_model,weak` failure did not recur).

**F-165 is much more frequent with SPIR-V 1.4.** All 5 crashes are shard workers exiting
`0xC0000005` in `bufferArrayView:mixed_types_aliasing` and `bufferView:mixed_types_aliasing`
with `buffer="workgroup"`, `aliased=true`, and an f16 pair: the shape of F-165, the NVIDIA
shader compiler crash recorded in the CTS suite's `docs/FINDINGS.md`. Those two queries run
3 times each way:

| Library | crashes per run |
|---|---|
| This block, SPIR-V 1.4 | 11, 4, 12 |
| Before this block (working tree stashed), SPIR-V 1.3 | 0, 0, 0 |

Dawn also emits SPIR-V 1.4 on this host, but the Dawn CTS build does not list these tests,
because Dawn does not report the WGSL language feature they need. So the comparison with
Dawn is not available. By user decision (2026-11-03) yawgpu keeps Dawn's rule on NVIDIA and
records the effect here instead of excluding NVIDIA.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Dawn enables both extensions whenever they are reported; yawgpu enabled them only when the decision was true, so a passthrough SPIR-V 1.4 module was rejected on Android ARM and Adreno 7xx | Fixed: extensions follow the two extension facts, codegen follows R1. R2 rewritten |
| m2 | The ignored test computes its expectation with the R1 function, and VUIDs are not captured | Stated in Tests; run under the Khronos layer |
| m3 | The R2 unit test checks the helper, not the real device-creation list | Stated in Tests |
