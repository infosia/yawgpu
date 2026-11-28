# Dawn compatibility layer — tracking

Spec: `specs/blocks/171-dawn-compat-layer.md`.

| Slice | Content | Status |
|---|---|---|
| S1 | ABI subset guard | **DONE 2026-11-28** — `tools/dawn_compat/{gen_dawn_headers,abi_guard}.py`, crate `yawgpu-dawn` (feature `dawn-compat`), `tests/abi_guard.rs` 4/0. At Dawn `9bba943462` on arm64 macOS: 203 functions, 98 structs, 459 members, 525 constants, no mismatch |
| S2 | Dawn toggles / cache chained structs | **DONE 2026-11-28** — yawgpu already skipped unrelated chain nodes on the instance and adapter-request chains and never read the device descriptor's chain, so no runtime change; `yawgpu.h` declares the two descriptors and three Noop tests pin the behaviour (`yawgpu --lib` 344/0) |
| S3 | `ImplicitDeviceSynchronization` | **DONE 2026-11-28** — F1, F3, F5, F7, F9 fixed; feature advertised on Noop / Metal / Vulkan, not GLES; `concurrent_device` (4 threads × 32 iterations) on Noop, and on the Apple M2 through `e2e_metal_concurrent_device` 5 runs / 0 failures. Metal e2e regression 188/0 (54 binaries). Not run: ThreadSanitizer (the nightly `-Zbuild-std` build needs a crate that is not in the local cache), the Vulkan wrapper `e2e_vulkan_concurrent_device` on a native driver |
| S4 | proc table, stubs, Dawn API package | **DONE 2026-11-28** — the table's `createInstance` selects the default real backend when no selector is chained, `yawgpuDawnSetDefaultBackend()` overrides it (`yawgpu-dawn --features dawn-compat` 13/0; the two Metal tests pass on the M2); `yawgpuDawnGetProcTable()`: 276 members, 203 yawgpu + 73 stubs; `tools/dawn_compat/make_dawn_api_package.py` (12 headers, 2 standalone sources); C++ smoke test through the package on Noop, and on the Apple M2 (`smoke_computes_on_metal`). `yawgpu-dawn --features dawn-compat` 11/0 |
| S5 | ONNX Runtime bring-up | **DONE on macOS / Metal 2026-11-28** — see "ONNX Runtime bring-up" below |
| S6 | docs + Phase Review | not started |

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

Not done:

- Linux / Windows Vulkan bring-up.
- ONNX Runtime's own test suites (`onnxruntime_webgpu_external_dawn_test`
  uses `dawn::native`, which is out of scope) and larger models, including
  one that takes the subgroup-matrix MatMul path.
- A performance comparison with Dawn; in particular the cost of yawgpu
  validating although ONNX Runtime asks for `skip_validation`.
- The host-supplied Dawn cache interface (accepted and unused).
