# Dawn compatibility layer — tracking

Spec: `specs/blocks/171-dawn-compat-layer.md`.

| Slice | Content | Status |
|---|---|---|
| S1 | ABI subset guard | **DONE 2026-11-28** — `tools/dawn_compat/{gen_dawn_headers,abi_guard}.py`, crate `yawgpu-dawn` (feature `dawn-compat`), `tests/abi_guard.rs` 4/0. At Dawn `9bba943462` on arm64 macOS: 203 functions, 98 structs, 459 members, 525 constants, no mismatch |
| S2 | Dawn toggles / cache chained structs | **DONE 2026-11-28** — yawgpu already skipped unrelated chain nodes on the instance and adapter-request chains and never read the device descriptor's chain, so no runtime change; `yawgpu.h` declares the two descriptors and three Noop tests pin the behaviour (`yawgpu --lib` 344/0) |
| S3 | `ImplicitDeviceSynchronization` | **DONE 2026-11-28** — F1, F3, F5, F7, F9 fixed; feature advertised on Noop / Metal / Vulkan, not GLES; `concurrent_device` (4 threads × 32 iterations) on Noop, and on the Apple M2 through `e2e_metal_concurrent_device` 5 runs / 0 failures. Metal e2e regression 188/0 (54 binaries). Not run: ThreadSanitizer (the nightly `-Zbuild-std` build needs a crate that is not in the local cache), the Vulkan wrapper `e2e_vulkan_concurrent_device` on a native driver |
| S4 | proc table, stubs, Dawn API package | **DONE 2026-11-28** — the table's `createInstance` selects the default real backend when no selector is chained, `yawgpuDawnSetDefaultBackend()` overrides it `yawgpuDawnGetProcTable()`: 276 members, 203 yawgpu + 73 stubs; `tools/dawn_compat/make_dawn_api_package.py` (12 headers, 2 standalone sources); C++ smoke test through the package on Noop, and on the Apple M2 (`smoke_computes_on_metal`). the table's `createInstance` selects the default real backend when no selector is chained and `yawgpuDawnSetDefaultBackend()` overrides it. `yawgpu-dawn --features dawn-compat` 17/0; its two Metal tests pass on the M2 |
| S5 | ONNX Runtime bring-up | **DONE on macOS / Metal 2026-11-28** — see "ONNX Runtime bring-up" below |
| S6 | docs + Phase Review | **DONE 2026-11-28** — `README` section, `examples/ort_webgpu/README.md`, Phase Review below |

## Thread-safety audit (R3)

Source reading at `afe5dd0`; nothing was executed. Two structural facts carry
most of the result:

- All HAL command recording happens at submit, under core's
  `QueueInner::submission` mutex. No Vulkan command buffer or Metal encoder
  exists while a `WGPUCommandEncoder` is alive, so encoders on several threads
  share no backend state. The exception is the `tiled` subpass pass.
- Bind groups and shader modules have no HAL object; descriptor sets are built
  per pass at submit.

No deadlock and no lock-order inversion was found on Noop, Metal or Vulkan.

### Findings

| # | Finding | Severity | Decision |
|---|---|---|---|
| F1 | `tint::Initialize()` is called unguarded from every `Program::parse` and writes a static (`yawgpu-tint/shim/tint_shim.cpp`) | C++ data race, benign in practice | Fix: once per process |
| F2 | GLES leaves the EGL / WGL context current on the calling thread; a second thread gets `eglMakeCurrent failed` | spurious device errors | GLES does not advertise the feature (Block 67 matrix) |
| F3 | Vulkan `vkDeviceWaitIdle` in surface unconfigure / drop runs without `queue_access` while another thread may be in `vkQueueSubmit` | Vulkan external-synchronization violation | Fix: take the queue lock |
| F4 | `MetalSurface` is `Send + Sync` although `CAMetalLayer` setters are main-thread-only | thread affinity, no Rust-level race | Documented limit: surfaces keep the window system's thread rules |
| F5 | Two waiters on one future: the second is told "completed" before the first has run the callback, so `GetMappedRange` returns null | wrong result | Fix: a waiter observes completion only after the callback returned |
| F6 | Error scopes are per thread (`ErrorSink.scopes` keyed by `ThreadId`); Dawn has one stack per device | divergence, memory-safe | Kept and documented in R3 |
| F7 | `wgpuBufferDestroy` racing a queue write reports an Internal error ("has no HAL buffer") instead of a validation error; same for textures | wrong error type | Fix |
| F8 | Latency only: core `submission` held across the HAL submit; codegen cache mutexes held across a Tint compile; Vulkan / Metal waits under their queue locks | none | No change |
| F9 | `unsafe impl Send / Sync` without a `// SAFETY:` comment on most Metal impls and all three Vulkan impls | documentation | Fix |

Suspicions, not acted on:

- A user uncaptured-error callback re-entering `dispatch_indirect_pipeline` /
  `timestamp_conversion_pipeline` initialization would deadlock on the
  `OnceLock`; it needs yawgpu's own internal WGSL to fail.
- `clear_uncaptured_error_callback` waits for other threads' running callbacks;
  a user mutex held across `wgpuDeviceDestroy` and inside the callback
  deadlocks through user code. No yawgpu lock is held at that point.
- Metal `wait_for_submission` assumes in-order terminal status.

### Guards per object type

| Object / subsystem | Guard | Verdict |
|---|---|---|
| FFI handles | `Arc::into_raw`; atomic `external_refs` / `in_use_refs` | safe |
| Device (core) | immutable limits / features; mutexes for `error_sink`, `lost`, `label`; atomics; `OnceLock` internal pipelines | safe |
| Uncaptured-error and device-lost callbacks | cloned / removed under the lock, invoked with no lock held | safe |
| Queue (core) | `submission` → `pending_writes` → resource state, one order everywhere | safe |
| Buffer (core) | `state` → `hal`; map start, resolve, unmap, destroy under `state`; a map serial rejects stale completions | safe |
| Texture, query set (core) | per-object mutexes; init / availability commit at submit under `submission` | safe |
| Sampler, view, layouts, bind group, pipelines, shader module (core) | immutable after creation | safe |
| FFI object caches | one `Mutex<HashMap<_, Weak>>` each; two threads may both compile, one result wins | safe |
| Command encoder, pass, bundle encoder | per-encoder mutex; no shared device state | safe (one thread each) |
| Futures registry, pending callbacks | one mutex each; callbacks fired after removal with no lock held | safe; F5 |
| Surface (FFI) | `current_texture` held for the whole operation | safe at FFI; F3, F4 below it |
| Vulkan queue, command pools, descriptor pools, fences, image layouts | `queue_access`; pools owned by one submit; `submissions` / `retire` mutexes | safe |
| Vulkan pipeline cache, allocator | not used | safe |
| Metal queue | `submission_lock` → `submissions`; waits clone the command buffer and wait unlocked | safe |
| Noop | atomics; never-nested mutexes | safe |
| GLES | `current_lock`, context never unbound | not safe across threads (F2) |
| Tint program / codegen / reflection | per-call state; Inspector under the shim mutex; no shim globals | safe; F1 |
| Rust shader caches | `OnceLock`s and one mutex per cache | safe |
| Process globals | atomics and `OnceLock`s; no `static mut`; no background threads | safe |

### F5 design

A poll claims one ready future under the registry mutex, records the executing
thread, and releases the mutex before delivering. The registry entry is removed
only when the callback has left the stack. Meanwhile other threads see the
future as pending (a zero-timeout wait times out, a timed wait retries); the
executing thread sees its own future as complete, so a callback may wait on it.

### Linking note for consumers

The package's `dawn_proc.cpp` defines the `wgpu*` symbols as dispatchers, and
yawgpu defines the same names as implementations. They cannot be linked
statically into one image. A consumer keeps the dispatch sources in its own
shared library (ONNX Runtime does) and the host links yawgpu; the smoke test
renames the consumer side with a pre-include instead.

### Not verified by the audit

- No concurrent test, ThreadSanitizer run or Vulkan thread-safety layer run
  (S3's tests cover this).
- The `tiled` subpass path.
- The bodies of `vulkan/encode.rs`, `metal/encode.rs` and `gles/queue.rs`
  beyond their locking skeleton; Tint writer / transform internals.
- External textures and render bundles beyond their lock fields.

## ONNX Runtime bring-up (R7)

ONNX Runtime 1.32.0 on the Apple M2, Metal backend, yawgpu at Dawn pin
`9bba943462`.

Build: `tools/ci_build/build.py --config Release --use_webgpu
--use_external_dawn --build_shared_lib --skip_tests --skip_submodule_sync
--compile_no_warning_as_error --parallel --target onnxruntime
--cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR=<package>` with the
package from `tools/dawn_compat/make_dawn_api_package.py`. It compiled with no
source change: ONNX Runtime's code written for Dawn `v20261013` builds against
the headers of yawgpu's pin. `build.py` needs Python 3.10 or later.

Host: `examples/ort_webgpu` links `libyawgpu_dawn.a` and
`libonnxruntime.dylib` and passes `yawgpuDawnGetProcTable()` as
`dawnProcTable`. Every run below had `session.disable_cpu_ep_fallback=1` and
`YAWGPU_DAWN_STRICT=1`; no stub was called.

| Run | ONNX Runtime-owned device | Host-owned device |
|---|---|---|
| Abs (ONNX Runtime's own external-Dawn test model) | pass | pass |
| `squeezenet/model.onnx` vs the CPU provider | 1000 elements, worst relative error 4.3e-7 | same |
| `mnist.onnx` vs the CPU provider | 10 elements, 4.4e-7 | same |
| `matmul_1.onnx`, `conv_autopad.onnx`, `mul_1.onnx` vs the CPU provider | exact | exact |
| Noop backend | session runs, nothing computed | — |

Findings:

| # | Finding | Resolution |
|---|---|---|
| O1 | The instance ONNX Runtime creates carries no `YaWGPUInstanceBackendSelect`, so it was a Noop instance | R4: the table's `createInstance` selects the default real backend; `yawgpuDawnSetDefaultBackend()` |
| O2 | The package's dispatch sources and yawgpu both define `wgpu*` | No change: ONNX Runtime keeps the dispatchers inside `libonnxruntime`, the host links yawgpu ("Linking note for consumers") |
| O3 | `libyawgpu_dawn.a` needs `libtint_shim.dylib` next to the executable (`@loader_path`) | `examples/ort_webgpu/build.sh` copies it |

ONNX Runtime features and toggles against yawgpu:

| Item | yawgpu |
|---|---|
| `TimedWaitAny`, `ImplicitDeviceSynchronization` (required) | supported |
| `ShaderF16`, `Subgroups`, `SubgroupSizeControl`, `TimestampQuery`, `ChromiumExperimentalSubgroupMatrix` | supported where the adapter has them |
| `ChromiumExperimentalTimestampQueryInsidePasses`, `BufferMapExtendedUsages`, `AdapterPropertiesMemoryHeaps` | not advertised; ONNX Runtime takes its fallback path |
| Toggles (`skip_validation`, `disable_robustness`, …) | accepted, no effect |

Optional features worth a follow-up block, by what ONNX Runtime gains:
`ChromiumExperimentalTimestampQueryInsidePasses` (per-kernel profiling inside
one pass; yawgpu's Metal backend already samples at pass boundaries), then
`BufferMapExtendedUsages` (maps storage buffers directly instead of copying
through a staging buffer on upload).

Not done:

- Linux / Windows Vulkan bring-up.
- ONNX Runtime's own test suites (`onnxruntime_webgpu_external_dawn_test`
  uses `dawn::native`, which is out of scope) and larger models, including
  one that takes the subgroup-matrix MatMul path.
- A performance comparison with Dawn; in particular the cost of yawgpu
  validating although ONNX Runtime asks for `skip_validation`.
- The host-supplied Dawn cache interface (accepted and unused).

## Windows / Vulkan verification (2026-11-28)

Native Vulkan driver (NVIDIA), MSVC 2022, yawgpu at Dawn pin `9bba943462`.

| Run | Result |
|---|---|
| `concurrent_device` (Noop) | 1/0 |
| `e2e_vulkan_concurrent_device` | 1/0, then 5 more runs / 0 failures; clean under `VK_LAYER_KHRONOS_validation` |
| `e2e_vulkan_subgroup_matrix` (Block 170) | 7/0, 12 configurations advertised; clean under the validation layer |
| `yawgpu-dawn --features dawn-compat,vulkan` | 20/0 including `smoke_computes_on_vulkan` and `default_off_apple_with_vulkan_is_vulkan` (both clean under the validation layer); guard: 203 functions, 98 structs, 459 members, 526 constants, no mismatch |
| ONNX Runtime 1.32.0, Abs | pass, ONNX Runtime-owned and host-owned device |
| `squeezenet/model.onnx` vs the CPU provider | 1000 elements, worst relative error 2.3e-7, both modes; clean under the validation layer |
| `mnist.onnx`, `matmul_1.onnx`, `conv_autopad.onnx`, `mul_1.onnx` vs the CPU provider | exact, both modes |

Every ONNX Runtime run had `session.disable_cpu_ep_fallback=1` and
`YAWGPU_DAWN_STRICT=1`; no stub was called. ONNX Runtime compiled with no
source change, built with `onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON` and
`onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF`.

Portability fixes this needed:

| # | Finding | Fix |
|---|---|---|
| W1 | `tests/common/concurrent_device.rs` stored `WGPUMapAsyncStatus` in an `AtomicU32`; the enum is `i32` on MSVC | cast to `u32` |
| W2 | `build.rs` and `tests/abi_guard.rs` ran `python3`, the Store stub on Windows | `PYTHON`, else `python` on Windows; `PYTHONUTF8=1` (the generators failed decoding as cp932) |
| W3 | The guard matched preprocessor line markers by raw path; Clang on Windows escapes backslashes | paths normalized before the comparison |
| W4 | The guard took its compiler from `CC`, which the `cc` crate also reads and needs to be MSVC | `YAWGPU_ABI_GUARD_CC` takes precedence |
| W5 | `examples/ort_webgpu/main.cc` passed a `char` path to `Ort::Session` (`wchar_t` on Windows); no Windows build script | `ORTCHAR_T` path; `examples/ort_webgpu/build.ps1` |

Limit: `ort_webgpu --noop` fails on Windows. ONNX Runtime's Windows build
requests the backend type chosen at its build time (Vulkan here) and the Noop
instance has no such adapter.

Still open: ThreadSanitizer on the concurrent workload; ONNX Runtime on Linux
Vulkan.

## Phase Review (2026-11-28)

Fresh no-context review of `6370d11..9347b53` against R1–R7. **0 CRITICAL,
2 MAJOR, 9 MINOR.** The future-delivery protocol and the Vulkan teardown lock
order were read in full and found clean.

| # | Finding | Decision |
|---|---|---|
| M1 | The proc-table build did not run R1's guard, so a Dawn bump could produce a table pointing at ABI-incompatible functions | Fixed: the table generator runs the guard first and fails naming the mismatch; `table_generator_rejects_an_incompatible_signature` |
| M2 | Nothing told whoever bumps Dawn to run the guard | Fixed: step added to the bump procedure in `specs/reference/dependencies.md` |
| m1 | While a callback runs, other threads see its future as pending; two callbacks waiting on each other's future never return | Documented as a limit in R3 |
| m2 | `FutureRegistry::register`, `Poll::ExternalCallback`, `WaitAnyResult::callbacks_to_fire` were test-only and could drop a callback | Fixed: removed, tests ported to `register_with_callback` |
| m3 | Stale Vulkan lock-order comment in `vulkan/encode.rs` | Fixed |
| m4 | `getProcAddress("wgpuCreateInstance")` bypassed the default-backend wrapper | Fixed: the table's lookup returns its own entry; R4 updated |
| m5 | Stub tests covered only `WGPUStatus`; the generator had no guard for a return type whose zero value may mean success | Fixed: every stub return kind classified, unknown kinds fail generation, one stub of each kind tested |
| m6 | Tests that can pass without their property: `initialize_supports_concurrent_parsers` (only ThreadSanitizer sees the race), `shared_hal_handles_are_send_and_sync` (covers compiled-in backends), `smoke_runs_on_noop` (no data on Noop), the guard's pass test (no minimum count) | Guard minimum counts fixed. The others deferred: the Send / Sync test runs under each backend's HAL gate; Noop computes nothing by design; the Tint race needs ThreadSanitizer, listed as not run |
| m7 | The standalone `dawn_proc.cpp` warning lost its trailing newline | Fixed |
| m8 | Ledger gave two test counts in one row; `yawgpu-dawn/README.md` carried session-local paths; no list of features worth a follow-up | Fixed |
| m9 | Redundant explicit `concurrent_device` test target | Fixed: removed |

Gate after the fixes (Apple M2): `cargo fmt --check` clean; workspace 1,544/0;
`yawgpu-dawn --features dawn-compat` 17/0 and its Metal tests 2/0; HAL lib
221/0 (`metal`) and 395/0 (`vulkan`); clippy clean on default, `metal`,
`vulkan`, `gles` and `dawn-compat,metal`; `e2e_metal_concurrent_device` 1/0;
Metal e2e 188/0 (54 binaries); the ONNX Runtime host re-run in both device
modes (Abs, SqueezeNet, MNIST).

Open after the review, all needing a host or tool this machine lacks:
ThreadSanitizer on the concurrent workload; `e2e_vulkan_concurrent_device` and
the F3 fix on a native Vulkan driver; ONNX Runtime on Linux / Windows Vulkan.

## Branch review (2026-11-28)

An independent review of `block-171-dawn-compat` at `2040a95`
(`REPORT-dawn.md`, a working file) requested four changes and made one
suggestion. Each was confirmed against the source and fixed.

| # | Finding | Fix |
|---|---|---|
| 1 | `concurrent_device` failed in the Tint-stub build (no Dawn submodule): it always encoded a WGSL compute pass | The workload probes for Tint through a validation scope and skips the shader / pipeline part on the stub; the rest runs in both builds |
| 2 | The ABI guard compared struct sizes and member offsets but not member types (`WGPUFuture.id` as `double` passed) | The compiler checks each shared member's type with typedefs resolved; regression tests for the guard and the table generator (R1 updated) |
| 3 | The table's `createInstance` turned a null descriptor into an empty one, dropping `TimedWaitAny` | A null descriptor keeps `wgpuCreateInstance(NULL)`'s features; tests for null, empty and explicit-feature descriptors (R4 updated) |
| 4 | `examples/ort_webgpu/README.md` expanded `$PWD` after `cd "$ORT"` | The package directory is set once in the yawgpu checkout |
| 5 | Looking up `wgpuGetProcAddress` and then `wgpuCreateInstance` through it bypassed the wrapper | The table's lookup returns itself for `wgpuGetProcAddress`; two-step lookup tested (R4 updated) |

Checking finding 1 on the whole workspace found one more test of this block
that needed Tint, `wgpuAdapterRequestDevice_ignores_dawn_cache_callbacks`; it
now uses the same probe.

Tint-stub build, `cargo test --workspace --no-fail-fast`: 318 failures at the
block's base commit `6370d11` and the same 318 after this block — the block
adds none. Those 318 are tests that compile WGSL (256 of them in
`yawgpu-core`); they predate this block and are not addressed here.

Gate (Apple M2, full Tint): workspace 1,544/0; `yawgpu-dawn --features
dawn-compat` 19/0 and its Metal tests 2/0; clippy clean on default, `metal`,
`vulkan` and `dawn-compat,metal`; `e2e_metal_concurrent_device` 1/0; the ONNX
Runtime host re-run in both device modes (Abs, SqueezeNet, MNIST).
