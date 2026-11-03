# Block 135 — `WGPUAdapterInfo` reports the GPU, as Dawn

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 1 MAJOR, 3 MINOR; all fixed, see
below. Metal verified on the M2 at `d3e574b`.
Source: backlog A23, found in Block 134. Started by user request on 2026-11-03.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/`):
- `native/Adapter.cpp` (`AdapterBase::APIGetInfo`: which physical-device value fills which
  field)
- `native/PhysicalDevice.cpp` (`mVendorName`, `mArchitectureName` from `gpu_info`)
- `native/vulkan/PhysicalDeviceVk.cpp` (`InitializeImpl`: ids, adapter type, driver
  description)
- `native/metal/PhysicalDeviceMTL.mm` (constructor; `GetVendorIdFromVendors` trademark
  table; architecture fallback when the device id is 0)
- `native/opengl/PhysicalDeviceGL.cpp` (`InitializeImpl`; `kVendors`,
  `GetVendorIdFromVendors`, `GetDeviceIdFromRender`)
- `gpu_info.json` and `generator/templates/dawn/common/GPUInfo.cpp` (`GetVendorName`,
  `GetArchitectureName`), `generator/dawn_gpu_info_generator.py` (`js_enum_case`)

## Problem

`adapter_info_from_core` (`yawgpu/src/ffi/mod.rs`) fills `WGPUAdapterInfo` with constants:
`vendor` is "yawgpu", `architecture` and `description` are empty, `vendorID` and
`deviceID` are 0, and `adapterType` is `Unknown` on every real GPU. Only `device` (the
adapter name), `backendType` and the subgroup sizes are real. Dawn reports the GPU. An
application cannot tell an NVIDIA adapter from an Intel one through the API, and a test
cannot gate on the vendor (Block 134's e2e reads the device name instead).

## Rules

R1. **HAL adapter info.** `HalAdapter` exposes one adapter-info value per adapter, a
`#[non_exhaustive]` struct with: vendor id, device id, adapter type (an enum: discrete,
integrated, CPU, unknown), driver description, vendor name, architecture name. Core and
the FFI build `WGPUAdapterInfo` from it: `vendor` = vendor name, `architecture` =
architecture name, `device` = the adapter name as today, `description` = driver
description, `adapterType`, `vendorID`, `deviceID`. `backendType` and the subgroup sizes
are unchanged. Fallback (CPU) adapters keep `adapterType` CPU, as today.

R2. **Vulkan**, from `VkPhysicalDeviceProperties` and, where available,
`VkPhysicalDeviceDriverProperties` (already queried by Block 131):
- ids: `vendorID`, `deviceID`
- type: integrated → integrated; discrete and virtual → discrete; CPU → CPU; else
  unknown
- description, as Dawn: with driver properties, `driverName`, then `": " + driverInfo`
  when `driverInfo` is not empty, then `" " + <driver version string>` when that string
  is not already contained; without them, `"Vulkan driver version " + <version string>`.
  The version string is Dawn's `DecodeVulkanDriverVersion` joined with `.`:
  - NVIDIA: `raw >> 22 & 0x3FF`, `raw >> 14 & 0xFF`, `raw >> 6 & 0xFF`, `raw & 0x3F`
  - Intel on Windows: `raw >> 14`, `raw & 0x3FFF`, each truncated to 16 bits as Dawn's
    `uint16_t` parts are
  - every other case: `raw >> 22`, `raw >> 12 & 0x3FF`, `raw & 0xFFF`

R3. **Metal:**
- vendor id, macOS only, from the device name with Dawn's trademark table, in Dawn's
  order:
  `AMD`, `Apple`, `Radeon`, `Intel`, `Geforce`, `Quadro` (case-sensitive) → AMD `0x1002`,
  Apple `0x106B`, AMD, Intel `0x8086`, NVIDIA `0x10DE`, NVIDIA. No match → 0. Share the
  table with Block 131's `metal_vendor_flags`. On iOS the vendor id is 0, as Dawn's iOS
  `GetDevicePCIInfo` returns `{0, 0}`; the Block 131 Metal gates keep their behaviour.
- device id 0. Dawn first reads the PCI ids from the IO registry and falls back to the
  table. On Apple Silicon that gives the same result; on Intel and AMD Macs Dawn reports
  the PCI device id and yawgpu reports 0. Recorded difference; the IOKit lookup is out of
  scope.
- type: macOS `hasUnifiedMemory` → integrated, else discrete; iOS → integrated
- description: `"Metal driver on macOS "` (or `"iOS "`) +
  `NSProcessInfo.operatingSystemVersionString`
- architecture, because the device id is 0, as Dawn's nesting: on macOS 13 / iOS 16 or
  later, `"metal-3"` when the device supports `MTLGPUFamilyMetal3`, else empty; on older
  OS versions, the highest of `"common-3"`, `"common-2"`, `"common-1"` it supports, else
  empty

R4. **GLES (Tier 2):**
- vendor id from `GL_VENDOR` with Dawn's GL table, first substring match in order:
  `ATI` → AMD, `ARM` → ARM `0x13B5`, `Imagination` → `0x1010`, `Intel`, `NVIDIA`,
  `Qualcomm` → `0x5143`, `Samsung` → `0x144D`. No match → 0.
- device id: when `GL_RENDERER` starts with `ANGLE`, the hex number after the first
  `(0x` or `(0X`, as Dawn's `GetDeviceIdFromRender`; else 0. Dawn's `std::stoul` throws on
  overflow; yawgpu parses at most into a `u64`, truncates to `u32` as the LP64 build of
  Dawn does, and returns 0 beyond 16 hex digits instead of failing
- type: CPU when `GL_RENDERER` contains `SwiftShader`, else unknown (Dawn's default)
- description: `"OpenGL version " + GL_VERSION`

R5. **Vendor and architecture names** come from a Rust port of Dawn's `gpu_info.json`:
- vendor name: the vendor's `name_override` if present, else its name, in Dawn's
  `js_enum_case` (lower case, words joined with `-` except after a digit): for example
  `nvidia`, `qualcomm`, `img-tec`. Unknown vendor → empty.
- architecture name: the first public (non-`internal`) device set of the vendor whose
  `deviceId & mask` equals one of an architecture's ids → that architecture name in
  `js_enum_case`, for example `adreno-7xx`. No match → empty. On Metal, R3's architecture
  rule replaces this because the device id is 0.
- The table lives in yawgpu-hal as checked-in Rust, generated by a small script under
  `tools/` from `third_party/dawn/src/dawn/gpu_info.json`. A unit test parses the JSON at
  the pinned submodule (when it is present) and asserts the Rust table matches it, so a
  Dawn pin bump that changes the JSON fails the test until the table is regenerated.
- Block 131's and Block 133's vendor constants may move onto this table; behaviour must
  not change.

R6. **Noop**: ids 0, empty vendor, architecture and description, type CPU as today. Dawn's
null backend reports type `Unknown`; yawgpu keeps CPU because the Noop adapter is
yawgpu's fallback adapter, and `adapterType` CPU is how a fallback adapter is reported.

## Tests

- Unit (HAL): R2 type mapping and description assembly (with and without driver
  properties, `driverInfo` empty, version already contained); R3 trademark table, type,
  architecture choice from the supported families; R4 vendor table order, the ANGLE
  device-id parse (`(0x`, `(0X`, no hex digit, no prefix, not starting with ANGLE);
  R5 names for a sample of vendors and devices, including a masked match and unknown ids;
  the JSON parity test. The pinned JSON has no internal device set on a real vendor, so
  internal-set exclusion is tested on a synthetic JSON through the generator.
- Unit (FFI): `wgpuAdapterGetInfo` and `wgpuDeviceGetAdapterInfo` copy every field from
  the HAL value; Noop values per R6. Update the existing tests that assert `vendorID == 0`.
- Vulkan HAL `#[ignore]` (real GPU): log the info and assert ids equal
  `VkPhysicalDeviceProperties`. Metal `#[ignore]`: log the info; Apple GPU → vendor
  `0x106B`, name `apple`, architecture `metal-3` or `common-*`.
- e2e (Claude): Windows native Vulkan reports vendor `0x10DE`, `nvidia`, a non-empty
  architecture and description, discrete type. Block 134's e2e switches back to the
  vendor id.

## Out of scope

- The IOKit PCI lookup on Intel and AMD Macs (R3).
- `DawnAdapterPropertiesPowerPreference` and other chained structs.

## Slices

- **S1 (coding agent):** R1-R6 with the unit tests.
- **S2 (Claude):** e2e on Windows native Vulkan, the Metal test on a Mac host, Phase
  Review.

## Verification (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351. The
Khronos validation layer was on for the e2e and HAL runs.

`wgpuAdapterGetInfo` on this host now reports:

```text
vendor="nvidia" architecture="blackwell" device="NVIDIA GeForce RTX 5060 Ti"
description="NVIDIA: 617.14 617.14.0.0" vendorID=0x10de deviceID=0x2d04
adapterType=DiscreteGPU backendType=Vulkan
```

The description follows Dawn: `driverName` "NVIDIA", `": "` and `driverInfo` "617.14",
then the NVIDIA-decoded version "617.14.0.0", appended because "617.14" does not contain it.
Before this block the same call gave vendor "yawgpu", empty architecture and description,
ids 0, and type Unknown (seen in the Block 134 e2e log).

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,375 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default (workspace), `vulkan,tiled` and `gles` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`), all targets | clean |
| HAL lib tests, `vulkan,tiled` / `gles` | 373 / 253 passed, 0 failed |
| FFI lib tests, `vulkan` | 277 passed, 0 failed |
| `python -I tools/gen_gpu_info.py --check`; `python -I tools/test_gen_gpu_info.py` | table current; 1 test OK |
| Vulkan HAL ignored tests | 81 passed, 0 failed, 0 VUIDs, including `adapter_info_matches_physical_device_properties` |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored` | 165 passed, 0 failed, 0 VUIDs |
| e2e `adapter_info_reports_the_gpu` and `four_rgba32float_targets_are_accepted` after the Phase Review fixes | pass; the Block 134 e2e gates on `vendorID == 0x10de` again |
| webgpu-native-cts, `api,operation,adapter,*` and `api,validation,capability_checks,*`, before the fixes | pass 11,956, skip 2,967, fail 0 |

The full HAL ignored and e2e runs and the CTS used the library before the Phase Review
fixes. The fixes change Metal, iOS, the Intel-on-Windows version decode, a GLES storage
detail, comments, and tests; none changes a value reported on this host, and the two
e2e above ran after them.

Not exercised: GLES (R4) on a real driver; it is covered by unit tests only.

### Mac host (2026-11-03)

Run on Apple M2, macOS 27.2, at yawgpu `d3e574b`. The Metal code in this block was
first compiled here.

| Gate | Result |
|---|---|
| clippy `-D warnings`, `metal,tiled` (`yawgpu-hal`, `yawgpu-core`, `yawgpu`), all targets | clean |
| `cargo test --workspace` | 1,375 passed, 0 failed |
| HAL lib tests, `metal,tiled`, `--include-ignored` | 254 passed, 0 failed; `adapter_info_reports_apple_identity` logs vendor `0x106B` `apple`, device 0, integrated, architecture `metal-3`, `"Metal driver on macOS Version 27.2 (Build 26B123)"` |
| Every ignored `yawgpu` test, `metal,tiled` | 162 passed, 0 failed |
| C probe calling `wgpuAdapterGetInfo` on Metal, linked once against Dawn `v20261030` and once against yawgpu | identical output in every field: vendor, architecture, device, description, backend type, adapter type, vendor and device ids, subgroup sizes |

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| M1 | Metal used `common-N` on macOS 13 / iOS 16 or later when Metal3 was missing, and reported vendor 0x106B on iOS; Dawn reports "" and vendor 0 there | Fixed to Dawn's nesting and iOS ids; R3 corrected |
| m1 | The internal-device-set exclusion was untested: the example vendor is skipped wholesale | Python test on a synthetic JSON (`tools/test_gen_gpu_info.py`) |
| m2 | Intel-on-Windows version parts were not truncated to 16 bits | Truncated; unit case with raw >= 2^30; R2 states it |
| m3 | The ANGLE device-id overflow behaviour differs from Dawn's `std::stoul` | Comment in code; R4 states it |
| — | clippy `--features gles`: `large_enum_variant` on `HalAdapter` once the GLES adapter held the info | The GLES adapter boxes it |
