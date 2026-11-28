# Block 171 — Dawn compatibility layer

Status: **spec only.** No slice started.

Programs written against Dawn's native API — the generated `dawn/webgpu.h`,
the `webgpu_cpp.h` C++ wrapper, `DawnProcTable`, and the `dawn::native` C++
classes — cannot link against yawgpu today, even though yawgpu implements the
same `webgpu.h` semantics. This block adds an opt-in layer that presents
Dawn's native surface on top of yawgpu, so such a program builds and runs with
yawgpu in place of Dawn's monolithic `webgpu_dawn` library.

The motivating consumer is ONNX Runtime's native WebGPU execution provider,
which targets Dawn. Its source is not available to this block (the upstream
repository returned 404 on 2026-11-28), so the surface below is derived from
Dawn alone and the consumer-specific part is deferred to S5.

## Measured starting point

Comparison of the pinned yawgpu `webgpu.h` with Dawn's generated
`dawn/webgpu.h` (Dawn build `v20261030`), by normalized text:

| Item | Dawn | yawgpu | Result |
|---|---|---|---|
| C functions | 276 | 202 | all 202 exist in Dawn under the same name |
| Structs | 196 | 92 | all 92 have identical member lists |
| Enum values | — | — | no name has two different values |

yawgpu's C ABI is a subset of Dawn's. The comparison did not check `sizeof` /
`offsetof` with a compiler; R1 does.

The 74 Dawn-only functions fall into these groups:

| Group | Functions | Plan |
|---|---|---|
| Device / adapter plumbing | `wgpuDeviceTick`, `wgpuDeviceGetAdapter`, `wgpuAdapterGetInstance`, `wgpuAdapterCreateDevice`, `wgpuDeviceSetLoggingCallback` | Implemented (R2) |
| Subgroup matrix | `wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers` | Implemented by Block 170 |
| External texture | `wgpuDeviceCreateExternalTexture`, `wgpuExternalTexture{Destroy,Expire,Refresh}` | Stub (yawgpu's own API is `yawgpuDeviceCreateExternalTexture`) |
| In-pass timestamps | `wgpu{Compute,Render}PassEncoderWriteTimestamp` | Stub |
| Shared memory / fences | `wgpuShared*`, `wgpuDeviceImportShared*` (26) | Stub |
| Resource tables, texel buffer views | `wgpuResourceTable*`, `wgpuTexelBufferView*`, `wgpu*SetResourceTable`, `wgpuBufferCreateTexelView` (15) | Stub |
| Error injection / testing | `wgpuDeviceCreateError*`, `wgpuDeviceInjectError`, `wgpuDeviceForceLoss`, `wgpuCommandEncoderInjectValidationError`, `wgpuTextureCreateErrorView`, `wgpuDeviceValidateTextureDescriptor` | Stub |
| Other | multi-draw indirect, pixel local storage, browser copies, format capabilities, memory heaps, DRM, AHardwareBuffer, memory dump | Stub |

## Scope

- Opt-in. Nothing in this block is part of the default cargo build or the
  default `libyawgpu` export set beyond R2's five functions.
- Experimental: the layer tracks the pinned Dawn revision and carries no
  SemVer guarantee.
- Backends: every yawgpu backend. The layer contains no backend code.

Out of scope: Dawn's wire, Dawn's D3D / OpenGL backend headers
(`D3D12Backend.h` and siblings), `dawn::platform`, Chromium shared-memory
interop, and reproducing Dawn toggles' behaviour.

## Rules

### R1 — ABI subset guard

A test certifies, for the pinned Dawn revision, that yawgpu's C ABI is a subset
of Dawn's:

- Every `wgpu*` function yawgpu declares exists in Dawn's generated header with
  the same signature.
- Every struct in both headers has the same `sizeof`, and every member the same
  `offsetof`, measured by a compiler on the host.
- Every enum constant in both headers has the same value.

Dawn's header is generated from the `third_party/dawn` submodule by Dawn's own
generator (`generator/dawn_json_generator.py`, target `headers`), not copied
from another checkout. The test fails at a Dawn pin bump that breaks the
subset, and prints the offending names.

The `yawgpu.h` items that reuse Dawn names (Block 170) are covered by the same
check.

### R2 — Dawn plumbing entry points

Five functions are added to yawgpu's C ABI with Dawn's signatures, declared in
`yawgpu.h`, registered with `wgpuGetProcAddress`:

- `wgpuDeviceTick(device)` — processes the pending events of the device's
  instance, like `wgpuInstanceProcessEvents`.
- `wgpuDeviceGetAdapter(device)` — returns a new reference to the adapter the
  device was created from.
- `wgpuAdapterGetInstance(adapter)` — returns a new reference to the adapter's
  instance.
- `wgpuAdapterCreateDevice(adapter, descriptor)` — synchronous device
  creation. Returns null on failure; the failure message goes to the logging
  callback when one is set, else to stderr.
- `wgpuDeviceSetLoggingCallback(device, callbackInfo)` — installs a callback
  that receives the device's log messages (`WGPULoggingType` + message).

`WGPULoggingType`, `WGPULoggingCallback` and `WGPULoggingCallbackInfo` are
declared with Dawn's values.

### R3 — Dawn chained structs

Descriptor chains accept these Dawn `sType`s without error:

| `sType` | Where | Behaviour |
|---|---|---|
| `WGPUSType_DawnTogglesDescriptor` (`0x0005000A`) | instance, adapter-request and device descriptors | Accepted. The toggle names are recorded and readable through R6's `GetTogglesUsed`. No toggle changes yawgpu's behaviour in this block. |
| `WGPUSType_DawnCacheDeviceDescriptor` (`0x00050007`) | device descriptor | Accepted and ignored. |

A toggle that asks Dawn for less validation or robustness (`skip_validation`,
`disable_robustness`) is therefore not honoured: yawgpu keeps validating. Dawn
features that Dawn hides behind `allow_unsafe_apis` are advertised by yawgpu
whenever the adapter supports them.

Any other `sType` keeps its current handling.

### R4 — compat library

A new directory `yawgpu-dawn/` builds one library, `webgpu_dawn`, that a
Dawn consumer links instead of Dawn's library of the same name. It contains
the yawgpu static library plus a C++ shim, and exports:

- every `wgpu*` symbol of Dawn's generated `webgpu.h` (276 at the measured
  revision): yawgpu's own, R2's, and a stub for each remaining one (R5);
- `dawnProcSetProcs` and `dawnProcGetVersion` (`dawn/dawn_proc.h`).
  `dawnProcSetProcs` stores the table; the exported `wgpu*` symbols always
  call yawgpu directly;
- the `dawn::native` symbols of R6.

It installs Dawn's generated `dawn/webgpu.h`, `dawn/webgpu_cpp.h`,
`dawn/webgpu_cpp_print.h`, `dawn/dawn_proc_table.h`, the `webgpu/` forwarding
headers, `dawn/dawn_proc.h`, and the shim's own `dawn/native/DawnNative.h`.

It ships a CMake package so that `find_package(Dawn)` followed by
`target_link_libraries(app dawn::webgpu_dawn)` works unchanged.

The stub list is generated from the difference between Dawn's header and
yawgpu's exports, not written by hand, so a Dawn bump or a new yawgpu function
updates it.

### R5 — stubs

A stub for a Dawn entry point yawgpu does not implement:

- writes one line to stderr the first time it is called, naming the function;
- returns null for a handle, `WGPUStatus_Error` for a status, zero for other
  values, and does nothing for `void`;
- never crashes and never reports success.

`wgpuGetProcAddress` returns null for a stubbed name. `DawnProcTable` (R6)
holds the stub, so a table call is safe.

Setting the environment variable `YAWGPU_DAWN_STRICT=1` makes a stub abort
after printing, to find the first unsupported call in a consumer.

Feature and limit queries need no stub: a Dawn-only `WGPUFeatureName` is not
advertised, and requesting it fails device creation as for any unsupported
feature.

### R6 — `dawn::native` shim

`dawn/native/DawnNative.h` declares, with Dawn's signatures:

- `dawn::native::GetProcs()` — a `DawnProcTable` with Dawn's `version` bytes
  for the pinned revision and every member filled (yawgpu function or stub).
- `dawn::native::Instance` — constructors from `WGPUInstanceDescriptor*` and
  `wgpu::InstanceDescriptor*`, `Get()`, `EnumerateAdapters(options)`,
  `GetToggleInfo(name)` (returns null),
  `SetBackendValidationLevel(level)` (ignored).
- `dawn::native::Adapter` — default and copy construction, `operator bool`,
  `Get()`, `CreateDevice(descriptor)` (both overloads),
  `SetUseTieredLimits(bool)` (ignored).
- `dawn::native::DawnInstanceDescriptor` — accepted on the instance chain;
  its `loggingCallbackInfo` is installed, the other members are ignored.
- `dawn::native::GetTogglesUsed(device)` — the names recorded by R3.
- `dawn::native::DeviceTick(device)`,
  `dawn::native::InstanceProcessEvents(instance)`.

`Instance::EnumerateAdapters` returns yawgpu's adapters for the backend chosen
by `options->backendType`; `Undefined` uses yawgpu's default selection.

Classes and functions of `DawnNative.h` not listed here are not declared, so
a consumer that needs one fails at compile time rather than at run time.

### R7 — consumer bring-up (S5)

For each real consumer, starting with ONNX Runtime:

1. Build it against R4's package with `YAWGPU_DAWN_STRICT` unset.
2. Compile and link errors list the `dawn::native` items and symbols the
   consumer needs beyond R6.
3. Run its tests with `YAWGPU_DAWN_STRICT=1`; each abort names a stub the
   consumer actually calls.
4. Each finding is recorded in `specs/tracking/dawn-compat.md` and resolved by
   implementing the entry point in yawgpu (with its inline unit test) or by a
   documented decision not to support it.

The consumer's required and optional `WGPUFeatureName`s and toggles are
recorded in the same ledger with yawgpu's support state.

## Slices

1. **S1 — ABI subset guard.** R1. Generates Dawn's headers from the submodule
   and adds the test.
2. **S2 — plumbing entry points and chains.** R2, R3, in `yawgpu` /
   `yawgpu-core`, with inline unit tests on Noop.
3. **S3 — compat library.** R4, R5, R6.
4. **S4 — C++ consumer test.** A small C++ program using only
   `webgpu_cpp.h` and `dawn::native` — create an instance, enumerate
   adapters, create a device, run a compute pass, read the result back — built
   through the CMake package. Runs on Noop in CI and on Metal on the M2.
5. **S5 — ONNX Runtime bring-up.** R7. Blocked on access to the ONNX Runtime
   source.
6. **S6 — docs and Phase Review.**

## Verification

- S1: the guard passes at the current pin; it fails when a struct member in a
  scratch copy of yawgpu's header is reordered, and when an enum value is
  changed.
- S2: inline tests — `wgpuDeviceGetAdapter` / `wgpuAdapterGetInstance` return
  the originating objects and add a reference; `wgpuAdapterCreateDevice`
  succeeds on Noop and returns null for an unsupported required feature;
  `wgpuDeviceTick` completes a pending map callback; the logging callback
  receives a message; device creation with a `DawnTogglesDescriptor` succeeds
  and records the names.
- S3: `nm` on the built library shows every `wgpu*` name of Dawn's header,
  `dawnProcSetProcs`, and the R6 symbols; a stub prints once and returns the
  R5 value; `YAWGPU_DAWN_STRICT=1` aborts.
- S4: the consumer program's compute result matches on Noop (validation path)
  and on Metal (execution).
- S5: the ledger has no unresolved finding for the consumer's default code
  path.

## Open questions

- Whether a consumer needs Dawn's `webgpu_cpp.h` C++ module target
  (`dawn::cpp_module`); it is not provided unless S5 finds a use.
- Whether to honour a small set of toggles (for example
  `disable_symbol_renaming`) once a consumer depends on their effect.
