# Block 171 — Dawn compatibility layer

Status: **COMPLETE on macOS / Metal (2026-11-28).** S1–S6 done; ONNX Runtime
1.32.0 runs SqueezeNet and MNIST on yawgpu with CPU fallback disabled, matching
the CPU provider. Phase Review: 0 CRITICAL, 2 MAJOR and 9 MINOR, all fixed or
deferred with a written rationale. Owed on other hosts: the Vulkan concurrency
run, ThreadSanitizer, and the ONNX Runtime bring-up on Linux / Windows Vulkan.
Ledger: `specs/tracking/dawn-compat.md`.

A program written against Dawn's native API can run on yawgpu when it reaches
the implementation only through Dawn's generated headers and a
`DawnProcTable`. This block provides that surface: the proc table filled with
yawgpu's functions, and the "Dawn API package" (headers and dispatch sources)
a consumer compiles against.

The motivating consumer is ONNX Runtime's native WebGPU execution provider.
Its *external Dawn* mode is exactly this contract, so the block targets it
directly. `$ORT` below is an ONNX Runtime checkout.

## Consumer contract (ONNX Runtime 1.32.0)

Measured from `$ORT/docs/WebGPU_External_Dawn.md`,
`$ORT/cmake/external/onnxruntime_external_deps.cmake` and
`$ORT/onnxruntime/core/providers/webgpu/`.

Built with `onnxruntime_USE_EXTERNAL_DAWN` and
`onnxruntime_DAWN_PREBUILT_DIR=<package>`, ONNX Runtime neither builds nor
links Dawn. It compiles `<package>/src/dawn_proc.cpp` and
`dawn_thread_dispatch_proc.cpp` with its own toolchain, includes
`webgpu/webgpu_cpp.h`, and uses no `dawn::native` or `dawn::platform` symbol.
The host passes the address of a `DawnProcTable` in the session option
`ep.webgpuexecutionprovider.dawnProcTable`; ONNX Runtime calls
`dawnProcSetProcs` with it, which aborts unless the table's `version` bytes
equal the package's `dawn/dawn_version.h`.

What ONNX Runtime then asks of the implementation:

| Item | Use | yawgpu today |
|---|---|---|
| Instance feature `TimedWaitAny` | required at `CreateInstance` | supported |
| `WGPUFeatureName_ImplicitDeviceSynchronization` (`0x00050004`) | **required** on every native device, ORT-created or host-supplied; a device without it is rejected | not advertised (R3) |
| `WGPUDawnTogglesDescriptor` (`0x0005000A`) | chained on `WGPURequestAdapterOptions` and on `WGPUDeviceDescriptor`, always | R2 |
| `WGPUDawnCacheDeviceDescriptor` (`0x00050007`) | chained after the device toggles when the host supplies a cache interface | R2 |
| `ChromiumExperimentalSubgroupMatrix`, `ShaderF16`, `Subgroups`, `SubgroupSizeControl`, `TimestampQuery` | requested when the adapter has them | supported (Blocks 62, 108, 170) |
| `WGPUAdapterPropertiesSubgroupMatrixConfigs` | chained on `GetAdapterInfo` when the device has the feature | supported (Block 170) |
| `ChromiumExperimentalTimestampQueryInsidePasses` (`0x00050003`) | requested when present; enables `wgpuComputePassEncoderWriteTimestamp` | not advertised, so not called |
| `BufferMapExtendedUsages` (`0x00050013`), `AdapterPropertiesMemoryHeaps` (`0x00050014`) | requested / queried when present | not advertised |
| `SharedBufferMemory*`, `SharedFence*`, `RequestAdapterOptionsLUID`, `dawn::native::d3d12` | Windows D3D12 weight loading and PIX only | out of scope |

The toggles ONNX Runtime passes are `use_dxc`, `allow_unsafe_apis`,
`decompose_uniform_buffers`, `use_vulkan_memory_model`,
`vulkan_enable_f16_on_nvidia` (adapter); `skip_validation`,
`d3d_disable_ieee_strictness`, `disable_robustness`,
`lazy_clear_resource_on_first_use` (device, enabled); `timestamp_quantization`
(device, disabled).

A host may instead create the instance and device itself and pass them through
`webgpuInstance` / `webgpuDevice` / `deviceId`; the same requirements apply to
that device.

ONNX Runtime is written against Dawn `v20261013`; the package is generated
from yawgpu's pinned Dawn. Source-level drift between the two is found by S5.

## Measured starting point

Comparison of the pinned yawgpu headers with Dawn's generated `dawn/webgpu.h`
(Dawn build `v20261030`), by normalized text:

| Item | Dawn | yawgpu | Result |
|---|---|---|---|
| C functions | 276 | 203 | all 203 exist in Dawn under the same name |
| Structs | 196 | 92 + Block 170's | identical member lists |
| Enum values | — | — | no name has two different values |

The comparison did not check `sizeof` / `offsetof` with a compiler; R1 does.

Of the 73 Dawn-only functions, ONNX Runtime's non-Windows code can reach one,
`wgpuComputePassEncoderWriteTimestamp`, and only behind a feature yawgpu does
not advertise. All 73 are stubs (R5).

## Scope

- Opt-in. Nothing in this block is part of the default cargo build or the
  default `libyawgpu` export set, except R2's chain handling and R3's feature.
- Experimental: the layer tracks the pinned Dawn revision and carries no
  SemVer guarantee.
- Backends: every yawgpu backend. The layer contains no backend code.

Out of scope: a `dawn::native` C++ shim (`dawn::native::Instance`, `Adapter`,
`GetProcs`), a drop-in `webgpu_dawn` library with a `find_package(Dawn)` CMake
package, `dawn::platform`, Dawn's wire, D3D / shared-memory interop, PIX, and
reproducing Dawn toggles' behaviour. A consumer that needs one of these is a
new block.

## Rules

### R1 — ABI subset guard

A test certifies, for the pinned Dawn revision, that yawgpu's C ABI is a subset
of Dawn's:

- Every `wgpu*` function yawgpu declares exists in Dawn's generated header with
  the same signature.
- Every struct in both headers has the same `sizeof`, and every member the same
  `offsetof` and the same type with typedefs resolved, decided by a compiler on
  the host. Equal size and offset are not enough: an integer member that became
  a floating-point one changes how a struct is passed and returned.
- Every enum constant in both headers has the same value.

Dawn's header is generated from the `third_party/dawn` submodule by Dawn's own
generator (`generator/dawn_json_generator.py`), not copied from another
checkout. The guard is opt-in (`cargo test -p yawgpu-dawn --features
dawn-compat`); the Dawn bump procedure in `specs/reference/dependencies.md`
runs it. The test fails at a Dawn pin bump that breaks the subset, and prints
the offending names. The `yawgpu.h` items that reuse Dawn names (Block 170) are
covered by the same check.

### R2 — Dawn chained structs

These Dawn `sType`s are accepted without error:

| `sType` | Where | Behaviour |
|---|---|---|
| `WGPUSType_DawnTogglesDescriptor` (`0x0005000A`) | instance descriptor, request-adapter options, device descriptor | Accepted. No toggle changes yawgpu's behaviour. |
| `WGPUSType_DawnCacheDeviceDescriptor` (`0x00050007`) | device descriptor, including behind a toggles node | Accepted. The load / store callbacks are never called. |

Consequences, stated so a consumer is not surprised:

- `skip_validation` and `disable_robustness` are not honoured: yawgpu keeps
  validating and keeps robustness.
- Dawn features that Dawn hides behind `allow_unsafe_apis` are advertised by
  yawgpu whenever the adapter supports them.

Any other `sType` keeps its current handling.

### R3 — `ImplicitDeviceSynchronization`

`Feature::ImplicitDeviceSynchronization` (`0x00050004`, Dawn's value) is
advertised by every Noop, Metal and Vulkan adapter and may be required at
device creation. GLES does not advertise it: its context is bound to the
calling thread (Block 67 matrix).

Contract (Dawn `docs/dawn/features/implicit_device_synchronization.md`): every
public entry point of the device and of its child objects may be called from
several threads at once, except command encoding — one `WGPUCommandEncoder`,
pass encoder or render bundle encoder is used by one thread at a time.

yawgpu's synchronization does not depend on the feature: every entry point is
safe with or without it, and enabling it selects no different code path. The
audit in `specs/tracking/dawn-compat.md` records, per object type, what guards
the state, and every `unsafe impl Send / Sync` with its justification. The
rules the audit established:

- `tint::Initialize()` runs once per process.
- Vulkan calls `vkDeviceWaitIdle` under the queue lock.
- Several threads may wait on one future (`wgpuInstanceWaitAny`,
  `wgpuInstanceProcessEvents`). A waiter is told the future completed only
  after its callback has returned.
- A queue write racing `wgpuBufferDestroy` / `wgpuTextureDestroy` reports a
  validation error, as the same call after the destroy does.
- Every `unsafe impl Send / Sync` in `yawgpu-hal` carries a `// SAFETY:`
  comment.

Deviations and limits:

- Error scopes are per thread: `wgpuDevicePopErrorScope` pops a scope pushed
  by the calling thread, and an error is routed to the scopes of the thread
  that reports it. Dawn keeps one stack per device.
- Surface entry points keep the window system's thread rules (on Metal the
  `CAMetalLayer` setters); the feature does not make them callable from any
  thread.
- The `tiled` subpass pass holds a live HAL pass while encoding; it follows
  the command-encoding exception.
- While a future's callback runs, other threads see that future as pending.
  Two callbacks that each wait, without a timeout, on the future the other
  thread is delivering therefore never return; so does a callback blocked on a
  user lock held by a thread waiting on that callback's future.

Concurrent tests on Noop — threads sharing one device that create and drop
resources, write buffers, submit, map and wait, pop error scopes and process
events — and the same workload on Metal and Vulkan as real-GPU e2e.

### R4 — proc table

A new directory `yawgpu-dawn/` builds one static library, opt-in, that exports

```c
const DawnProcTable* yawgpuDawnGetProcTable(void);
```

The returned table has Dawn's layout for the pinned revision
(`dawn/dawn_proc_table.h`), `version` equal to that revision's
`dawn::kDawnVersion`, and every member non-null: yawgpu's function where
yawgpu implements the entry point, else a stub (R5). The pointer is valid for
the life of the process.

The member list is generated from Dawn's header and yawgpu's exports, not
written by hand, so a Dawn bump or a new yawgpu function updates it.

The table's `createInstance` differs from `wgpuCreateInstance` in one way. A
Dawn consumer creates its instance without `yawgpu.h` and expects real
adapters, while `wgpuCreateInstance` without a `YaWGPUInstanceBackendSelect`
gives the Noop backend. So when the descriptor chains no
`YaWGPUInstanceBackendSelect`, the table's entry selects the default backend:
Metal on Apple platforms and Vulkan elsewhere when that backend is compiled
in, else Noop. A descriptor that chains a selector is passed through
unchanged.
Nothing else changes: a null descriptor keeps `wgpuCreateInstance(NULL)`'s
instance features (`TimedWaitAny` enabled), and an empty descriptor keeps an
empty descriptor's.

```c
void yawgpuDawnSetDefaultBackend(uint32_t backend);  /* YAWGPU_INSTANCE_BACKEND_* */
```

overrides that default for instances created afterwards.

The table's `getProcAddress` returns this `createInstance` entry for the name
`wgpuCreateInstance`, and itself for `wgpuGetProcAddress`, so a consumer that
resolves procs by name — directly or through the looked-up lookup function —
gets the same behaviour.

Building the table runs R1's guard first and fails when the subset does not
hold, so a table is never produced for an ABI-incompatible yawgpu.

### R5 — stubs

A stub for a Dawn entry point yawgpu does not implement:

- writes one line to stderr the first time it is called, naming the function;
- returns null for a handle, `WGPUStatus_Error` for a status, zero for other
  values, and does nothing for `void`;
- never crashes and never reports success.

`wgpuGetProcAddress` returns null for a stubbed name.

Setting the environment variable `YAWGPU_DAWN_STRICT=1` makes a stub abort
after printing, to find the first unsupported call in a consumer.

Feature and limit queries need no stub: a Dawn-only `WGPUFeatureName` is not
advertised, and requesting it fails device creation as for any unsupported
feature.

### R6 — Dawn API package

A script produces a directory in the layout ONNX Runtime's
`onnxruntime_DAWN_PREBUILT_DIR` expects:

```text
<package>/
  include/dawn/      Dawn public and generated headers, including
                     dawn_proc.h, dawn_proc_table.h, dawn_version.h,
                     dawn_thread_dispatch_proc.h, webgpu.h, webgpu_cpp.h
  include/webgpu/    webgpu.h, webgpu_cpp.h and their companions
  src/dawn_proc.cpp
  src/dawn_thread_dispatch_proc.cpp
```

- The headers are generated from the `third_party/dawn` submodule, so they
  match R4's table and R1's guard by construction.
- The two sources compile using only the package headers and the C++ standard
  library. Dawn's generated `dawn_proc.cpp` includes private headers
  (`src/dawn/common/Compiler.h`, `src/utils/assert.h`, `src/utils/log.h`), so
  the package carries a standalone equivalent.
- `dawnProcSetProcs` keeps the version check: a table whose `version` differs
  from `dawn_version.h` aborts.

The package contains no yawgpu-specific header; a host that wants `yawgpu.h`
(backend selection) includes it from yawgpu.

### R7 — ONNX Runtime bring-up

1. Build ONNX Runtime with `--use_webgpu --use_external_dawn` and R6's package.
   Compile errors are source drift between ONNX Runtime's Dawn and yawgpu's pin
   (see "Consumer contract").
2. A host program links ONNX Runtime and yawgpu, passes
   `yawgpuDawnGetProcTable()` as `dawnProcTable`, and runs a model with CPU
   fallback disabled, in both modes: ONNX Runtime creates the device, and the
   host supplies `webgpuInstance` / `webgpuDevice`.
3. Run with `YAWGPU_DAWN_STRICT=1`; each abort names a stub the consumer
   actually calls.
4. Each finding is recorded in `specs/tracking/dawn-compat.md` and resolved by
   implementing the entry point in yawgpu (with its inline unit test) or by a
   documented decision not to support it.

The ledger also records, per ONNX Runtime feature and toggle, yawgpu's support
state, and which optional features (`BufferMapExtendedUsages`,
`ChromiumExperimentalTimestampQueryInsidePasses`) are worth a follow-up block.

## Slices

1. **S1 — ABI subset guard.** R1. Generates Dawn's headers from the submodule
   and adds the test.
2. **S2 — Dawn chains.** R2, in `yawgpu`, with inline unit tests on Noop.
3. **S3 — implicit device synchronization.** R3: the audit, the fixes it
   finds, the concurrent tests, then the feature.
4. **S4 — proc table and package.** R4, R5, R6, and a C++ smoke test that uses
   only the package: compile its two sources, `dawnProcSetProcs` with
   `yawgpuDawnGetProcTable()`, then through `webgpu_cpp.h` create an instance,
   request an adapter and a device with a `DawnTogglesDescriptor`, run a
   compute pass and read the result back. Noop in CI, Metal on the M2.
5. **S5 — ONNX Runtime bring-up.** R7, on macOS / Metal first.
6. **S6 — docs and Phase Review.**

S2, S3 and S4 are independent of each other; S5 needs all three.

## Verification

- S1: the guard passes at the current pin; it fails when a struct member in a
  scratch copy of yawgpu's header is reordered, and when an enum value is
  changed.
- S2: inline tests — adapter request and device creation succeed with a
  `DawnTogglesDescriptor` carrying ONNX Runtime's toggle names; device creation
  succeeds with toggles followed by a `DawnCacheDeviceDescriptor`, and neither
  callback is called.
- S3: inline tests for each rule above (two waiters on one future both observe
  the mapped buffer; destroy racing a write); the concurrent tests pass on
  Noop repeatedly and under ThreadSanitizer where the toolchain supports it,
  and on Metal and Vulkan; the Noop, Metal and Vulkan adapters report the
  feature and GLES does not.
- S4: the table has no null member and its `version` equals
  `dawn::kDawnVersion`; a stub prints once and returns the R5 value;
  `YAWGPU_DAWN_STRICT=1` aborts; a table with one changed `version` byte makes
  `dawnProcSetProcs` abort; the smoke test's compute result matches on Noop
  (validation path) and on Metal (execution).
- S5: a model runs on Metal through ONNX Runtime with CPU fallback disabled in
  both device modes, and the ledger has no unresolved finding for that path.

## Open questions

- Whether `skip_validation` should be honoured later for performance. ONNX
  Runtime enables it by default in release builds; yawgpu validating every call
  is correct but slower. Measured in S5 before deciding.
- Whether a second consumer needs the `dawn::native` shim or the
  `webgpu_dawn` CMake package excluded above.
