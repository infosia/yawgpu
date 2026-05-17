# yawgpu

Yet Another wgpu — a from-scratch Rust implementation of the WebGPU C API
(`webgpu.h`), strongly inspired by [wgpu-native].

Unlike wgpu-native (which is a C ABI shim over gfx-rs `wgpu-core`), yawgpu
implements WebGPU semantics, validation, resource management, and the GPU
backend (HAL) itself. The only third-party engine dependency is [naga] for
WGSL compilation.

## What we reuse vs. build

| Layer | Source | yawgpu approach |
|---|---|---|
| C ABI header | `webgpu-headers/webgpu.h` | Bind via `bindgen`; hand-write `extern "C"` fns |
| C ABI structure | `wgpu-native` (inspiration) | Arc-wrapped opaque handles, `conv` module |
| WebGPU semantics / validation | `dawn` (spec + tests) | Reimplemented in `yawgpu-core` |
| GPU backend (HAL) | `mgpu` (structural template) | `yawgpu-hal`, enum dispatch, Noop→Vulkan→Metal (no GL/D3D) |
| WGSL compiler | `wgpu/naga` (path dep) | Reused as-is |
| Project conventions | `mgpu` | Mirrored (see `CLAUDE.md`) |

## Development model

Test-driven, with [Dawn]'s `unittests/validation` suite treated as the
executable specification. Tests are ported to Rust integration tests and run
against the **Noop** backend so CI needs no GPU. See `specs/SPEC.md` for the
phased roadmap and `specs/reference/dawn-test-mapping.md` for the test port
plan.

Implementation is carried out by a **separate coding agent**. Claude owns
planning, review, and integration (git). Role split and the per-slice
loop: `specs/reference/workflow.md`.

## Status

Planning. Phase 0 (workspace scaffold + FFI + test harness) not yet started.
See `specs/tracking/phase-0.md`.

## Layout (planned)

```
yawgpu/        C ABI crate (cdylib + staticlib), bindgen, extern "C" fns
yawgpu-core/   WebGPU semantics, validation, resource lifetimes
yawgpu-hal/    enum-dispatch HAL: Noop | Vulkan | Metal (feature-gated; no GL/D3D)
yawgpu-test/   Rust port of Dawn's ValidationTest base + assert_device_error!
examples/      hello_triangle, hello_compute, ...
specs/         SPEC.md, blocks/, reference/, tracking/
```

[wgpu-native]: https://github.com/gfx-rs/wgpu-native
[naga]: https://github.com/gfx-rs/wgpu/tree/trunk/naga
[Dawn]: https://dawn.googlesource.com/dawn
