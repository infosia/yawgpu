# Block 120 — Tint driver workarounds

Status: **COMPLETE (2026-11-02)**. Phase Review: 0 CRITICAL, 3 MAJOR, 6 MINOR; every
MAJOR and m1-m4 fixed, m5 and m6 recorded below. Native Vulkan ran on 2026-11-02;
see "Windows native Vulkan" below.
Source: Block 118 "Follow-ups". Dawn sets five driver-bug workarounds in Tint's
writer options; the yawgpu shim sets none. Every option defaults to off, so a device
that needs one silently runs the unmitigated shader. The CTS cannot see this on the
development hosts, because none of them matches a gate.
Oracle: Dawn at the pinned `third_party/dawn`: `native/metal/PhysicalDeviceMTL.mm`,
`native/metal/ShaderModuleMTL.mm`, `native/metal/ComputePipelineMTL.mm`,
`native/metal/RenderPipelineMTL.mm`, `native/vulkan/PhysicalDeviceVk.cpp`,
`native/vulkan/ShaderModuleVk.cpp`.

## Gates

| Tint option | Writer | Dawn gate | crbug |
|---|---|---|---|
| `fix_u32_div_mod` | MSL | macOS, Apple GPU, OS below 26.6.1 | 517225032 |
| `polyfill_bool_vec_dynamic_store` | MSL | macOS, Intel GPU | 540789158 |
| `polyfill_bool_vec_dynamic_store` | SPIR-V | Windows, Intel vendor | 556809584 |
| `replace_workgroup_atomic_store_with_exchange` | SPIR-V | Android, Samsung vendor | 487773864 |
| `replace_unsigned_compare_zero` | SPIR-V | Android, Samsung vendor, driver below 25.0.0.0 | 543420711 |

## Rules

R1. Each gate above is a pure function of its inputs, unit-tested at each boundary
without a GPU:
- Metal inputs: target is macOS, the GPU is Apple, the GPU is Intel, the OS is at
  least 26.6.1. iOS and other non-macOS targets get no Metal workaround, as in Dawn,
  whose block is inside `DAWN_PLATFORM_IS(MACOS)`.
- Vulkan inputs: `VkPhysicalDeviceProperties::vendorID`, `driverVersion`, target is
  Android, target is Windows. Intel is `0x8086`, Samsung is `0x144d`. The Samsung
  driver version decodes as Dawn's default arm: `(raw >> 22, (raw >> 12) & 0x3ff,
  raw & 0xfff)`. Dawn compares that 3-element version against the 4-element
  `{25, 0, 0, 0}` with `lexicographical_compare_three_way`, so a shorter equal prefix
  orders as less: 25.0.0 is below the gate and 25.0.1 is not. yawgpu matches this, so
  the workaround is on for a decoded version `<= (25, 0, 0)`.

R2. Metal GPU identity:
- Apple: `MetalGpuFamily::is_apple()` on the device's detected family.
- Intel: the `MTLDevice` name contains `intel`, case-insensitive. This differs from
  Dawn, which reads the PCI vendor through IOKit. yawgpu holds no PCI ids on Metal,
  and the gate only applies to macOS Intel GPUs, whose names carry the vendor.

R3. The OS check uses `NSProcessInfo isOperatingSystemAtLeastVersion:` with
`{26, 6, 1}`, as Dawn's `IsMacOSVersionAtLeast`.

R4. The HAL decides. `yawgpu-hal` exposes a `Copy` `#[non_exhaustive]`
`HalShaderWorkarounds` through `HalDevice::shader_workarounds()`. Noop and GLES return
the all-false default. Metal and Vulkan compute it once per device. `yawgpu-core`
passes it to Tint for every MSL and SPIR-V generation.

R5. The two codegen memoization keys (`MslCodegenKey`, `SpirvCodegenKey`) include the
workaround flags their writer consumes. Block 95 memoizes per request, so a flag that
is not in the key could return output generated for a different flag set.

R6. With all flags off, generated MSL and SPIR-V are byte-identical to the output
before this block.

R7. **The non-constant zero must be delivered.** `fix_u32_div_mod` rewrites each u32
divide and modulo to read `tint_non_constant_zero` from the immediate block. When the
flag is on, the shim reports `uses_immediates` for every entry point, even one with no
user immediates, no frag-depth clamp, and no buffer sizes. Without this the HAL binds
no block and the shader reads an unbound buffer. Dawn adds `nonConstantZero` to the
immediate mask of every Metal pipeline. `compose_metal_immediates_block` already
leaves that word zero. With the flag off, `uses_immediates` is unchanged, per R6.

R8. `YAWGPU_FORCE_SHADER_WORKAROUNDS` is a comma-separated list of the four flag names,
read once at device creation, and ORed over the gate result on Metal and Vulkan. Unknown
names and an unset variable force nothing. It serves the role of Dawn's force-enabled
toggles: no development host matches a gate, and the real-GPU tests need the
workaround code paths. The parser is a pure function with a unit test. Neither Noop nor
GLES reads it.

## Tests

- Gate boundaries (R1): each row on and off, including a driver of 25.0.0 vs 25.0.1,
  each vendor on the wrong platform, and iOS. The `{26, 6, 1}` literal is a named
  constant asserted by a unit test.
- Shim (R6): flags off leaves no workaround signature in the output of a shader that
  would trigger it, and each flag on adds its own signature. A golden hash is not used,
  because every Dawn bump would invalidate it.
- Core (R4): `select_{compute,render}_shader_source` with `fix_u32_div_mod` on produces
  immediates for every Metal stage, and MSL that reads `tint_non_constant_zero`.
- HAL (R4, R8): a direct test of `HalDevice::shader_workarounds()` on Noop and GLES,
  and `#[ignore]` real-device tests that the environment reaches a Metal and a Vulkan
  device.
- Shim (R7): flag on sets `uses_immediates` and a nonzero block size for an entry with
  no other immediate need, and the MSL contains `tint_non_constant_zero`; flag off
  leaves `uses_immediates` false for the same entry.
- Core (R5): the same request with a differing flag is a codegen miss.
- Real GPU: `yawgpu/tests/e2e_metal_shader_workarounds.rs` and the Vulkan twin run
  u32 divide and modulo, a boolean vector dynamic store, and a workgroup atomic store
  through the C ABI with every applicable flag forced, in a child process, with results
  read back.

## Out of scope

Dawn sets more device-gated Tint workarounds that predate Block 118 and that the shim
also leaves off. They are not part of this block and need their own survey. For example:
- SPIR-V: `polyfill_float_negation`, `polyfill_float_abs`,
  `polyfill_length_scalar_float`, `polyfill_distance_scalar_float` (AMD Mesa below
  25.3), `polyfill_case_switch` (Imagination), `texture_sample_compare_2d_polyfill`,
  `polyfill_pack_unpack_4x8_norm`, `scalarize_max_min_clamp`,
  `polyfill_saturate_as_min_max_f16`, `dva_transform_handle`,
  `polyfill_subgroup_broadcast_f16`, `collapse_subgroup_min_max`
- MSL: `scalarize_max_min_clamp`, `disable_module_constant_f16`,
  `polyfill_subgroup_broadcast_f16`, `polyfill_clamp_float`,
  `polyfill_unpack_2x16_snorm`, `polyfill_unpack_2x16_unorm`, `polyfill_tanh_f16`,
  `replace_workgroup_bool_with_u32`, `collapse_subgroup_min_max`

GLSL has none of the Block 118 options. MoltenVK gets the Vulkan gates only.

## Slices

- **S1 (coding agent):** R1-R8, with the shim, core, and HAL unit tests.
- **S2 (Claude):** real-GPU e2e with forced flags; Metal, MoltenVK, and SPIR-V checks;
  CTS on Metal.
- **S3:** Phase Review.

## Verification (2026-11-02)

Run on Apple M2, macOS 27.2, so `fix_u32_div_mod` is off by its gate here and every
workaround path below ran through `YAWGPU_FORCE_SHADER_WORKAROUNDS`.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,298 passed, 0 failed |
| fmt, clippy on 8 feature sets | clean |
| Metal HAL ignored tests | 59 passed |
| Metal e2e, `metal,tiled,shader-passthrough` | 147 passed, 0 failed |
| Vulkan HAL ignored tests, MoltenVK under the Khronos layer | 71 passed, 0 VUIDs |
| Vulkan e2e, MoltenVK under the Khronos layer | 135 passed, 1 failed (known `imageView2DOn3DImage`) |
| webgpu-native-cts, Metal, `webgpu:*`, `fix_u32_div_mod` and `polyfill_bool_vec_dynamic_store` forced | `fail=2 crash=0`, 0 regressions |

The CTS subcase counts equal Block 119's exactly: pass 2,102,837, skip 154,617, fail 2.
The 2 failures are the Block 119 R9 cases. The run used plain `--workers 6`, which took
22 minutes. `--isolate` is per-case and took over 2 hours without finishing.

R7 has a red check: with the `uses_immediates` change reverted, the Metal debug layer
reports `missing Buffer binding at index 0 for tint_immediate_data[0]` and the render
e2e fails.

Not exercised:
- Native Vulkan, at the time of this run. It ran later; see the next section.
- A device that matches a gate. The gates are proven by unit tests at each boundary.
- The tiled subpass path with a forced flag. Delivery was checked by reading the code.

### Windows native Vulkan (2026-11-02)

Run on Windows 11 with an NVIDIA RTX 5060 Ti, driver 617.14, at yawgpu `2b3f4a7`. The
code under test is the same as at `36f2719`. No Vulkan gate matches this host, so the
three SPIR-V workarounds ran through `YAWGPU_FORCE_SHADER_WORKAROUNDS`.

| Gate | Result |
|---|---|
| Vulkan e2e (`vulkan,tiled,shader-passthrough`, `--ignored`) under the Khronos layer | 152 passed, 0 failed, 0 VUIDs |
| `e2e_vulkan_shader_workarounds`, all three SPIR-V flags forced in the child | 2 passed |
| webgpu-native-cts (CTS `a5cbf67`), related queries, default | pass 6,872, skip 1,898, fail 1, xfail 1, xpass 1 |
| The same queries, all three SPIR-V flags forced | pass 6,872, skip 1,898, fail 1, xfail 1, xpass 1 |

The CTS run covered only the queries the three flags can reach, not `webgpu:*`:
`shader,execution,expression,call,builtin,atomics,*`, `shader,execution,memory_model,*`,
`shader,execution,expression,binary,u32_comparison:*`, `shader,execution,flow_control,*`,
`shader,execution,expression,access,vector,*`, `shader,execution,statement,*`,
`shader,execution,expression,binary,bool_logical:*`,
`shader,execution,expression,unary,bool_conversion:*`, and `api,operation,compute,*`. It
used `--workers 8` and `expectations/yawgpu-vulkan.txt`. The two runs together took 2
minutes.

The xfail and the xpass are the F-141 and F-152 `memory_model,coherence:corr` cases.
Which of them fires depends on the host. The single fail was a different
`memory_model,weak` case in each run: `store` (`atomic_storage`, 2 weak results) in the
default run, and `read` (`atomic_storage`, 1 weak result) in the forced run. Every
other subcase gave the same status in both runs. `memory_model,weak:*` then passed
12/12 three times with default flags and three times with the flags forced. These are
statistical tests on storage atomics. The atomic-store workaround rewrites only
workgroup atomic stores, so the flags do not cause these failures. They are
intermittent behaviour of this driver.

## Phase Review findings left open

- m5: `MslWorkarounds` and `SpirvWorkarounds` are not `#[non_exhaustive]`. They are
  `yawgpu-tint` FFI-facing structs that core builds with literals, and that crate is
  not one of the crates the code-convention rules name.
- m6: on Apple Silicon below macOS 26.6.1 every pipeline now binds the immediate block,
  as on Dawn. This is correct, and its per-draw cost has not been measured.

With the flag on, a stage that has no free slot for the block fails generation with a
clean "no free slot" error. Whether real pipelines can reach this is unconfirmed.
