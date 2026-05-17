# DESIGN.md — yawgpu

## Goal

A drop-in `webgpu.h` implementation (C ABI compatible with the canonical
WebGPU-Native header), implemented entirely in Rust with its own GPU backend.
Inspiration: wgpu-native's C ABI structure; mgpu's HAL + project layout;
Dawn's tests as the spec.

## Layer model

```
  C consumer  ──►  yawgpu (C ABI)  ──►  yawgpu-core  ──►  yawgpu-hal  ──►  GPU
                   bindgen types        WebGPU            enum dispatch    Noop /
                   extern "C" fns        semantics,        Noop|Vk|Metal   Vulkan /
                   Arc handles           validation,                       Metal
                   conv.rs               resource hub
```

### yawgpu (C ABI crate)

- `crate-type = ["cdylib", "staticlib"]`.
- `build.rs`: `bindgen` over `webgpu-headers/webgpu.h`.
  Apply wgpu-native's opaque-handle rename trick: each `WGPUXxx` typedef is
  blocklisted and re-emitted as `pub type WGPUXxx = *const crate::WGPUXxxImpl;`
  so handles map to our Arc-backed structs. `.ignore_functions()` — every
  `wgpu*` function is hand-written.
- Each object: `pub struct WGPUXxxImpl { core: Arc<...>, ... }` with `Drop`.
- `conv.rs`: macro-generated enum maps + `WGPUStringView`↔`&str`, descriptor
  → core descriptor conversions.
- Errors: per-device error sink (uncaptured error callback + error scopes),
  mirrors Dawn semantics so ported `ASSERT_DEVICE_ERROR` tests pass.

### yawgpu-core

Reimplements the parts of Dawn's `native/` we need:

- Instance / Adapter / Device / Queue lifetime, ID + Arc resource hub.
- Per-object descriptor validation (the bulk of ported Dawn tests).
- Async model: `WGPUFuture`, callback-info structs, `InstanceWaitAny`,
  instance/device poll. Spec callback modes (AllowProcessEvents /
  AllowSpontaneous / WaitAnyOnly).
- Command encoder state machine, error scopes, device-lost.

### yawgpu-hal

- mgpu-hal–style **static enum dispatch**. One enum per resource:
  `HalInstance/HalAdapter/HalDevice/HalBuffer/...` with variants
  `Noop | Vulkan | Metal`, backends behind Cargo features.
- **Noop** backend implemented first: synthetic, allocation-tracking, no GPU.
  It is the CI substrate and the TDD substrate for all validation phases.
- **Primary platforms: Vulkan (ash) and Metal (objc2).** These are the only
  real backends targeted. Both added at Phase 7 for end2end.
- **OpenGL / OpenGL ES and DirectX (D3D11/D3D12) are explicitly out of scope
  for the initial implementation.** The `HalXxx` enum stays open to adding
  variants later, but no D3D/GL variant is planned or stubbed for now.

### yawgpu-test

Rust port of Dawn `ValidationTest.h`:

- `ValidationTest`: creates a Noop instance/adapter/device with a captured
  uncaptured-error sink.
- `assert_device_error!(expr)` ≈ `ASSERT_DEVICE_ERROR` — asserts the
  enclosed C calls produced exactly one device error.
- Future/poll helpers for async-callback tests.

## Async / callback boundary

WebGPU-Native uses `WGPUFuture` + callback-info (`mode`, `callback`,
`userdata1/2`). yawgpu-core owns a future registry; `wgpuInstanceWaitAny` /
`wgpuInstanceProcessEvents` drive completion. Noop completes futures
synchronously on poll, which keeps ported async validation tests deterministic.

## Key risks / decisions

- **Scope**: full WebGPU + own HAL is large. Mitigation: strict TDD slices by
  Dawn test file; Noop keeps every phase shippable without a GPU.
- **naga**: path dep on `wgpu/naga` (decided). Needed Phase 4+.
- **Wire**: explicitly out of scope; no dawn-wire analog.
- **Header drift**: pin the `webgpu.h` we bindgen against; record its commit
  in `specs/reference/dependencies.md`.
