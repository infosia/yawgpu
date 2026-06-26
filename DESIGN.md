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
                   extern "C" fns        semantics,        Noop|Vk|Metal|  Vulkan /
                   Arc handles           validation,        Gles (T2)      Metal /
                   conv.rs               resource hub                       GLES (T2)
```

## Backend support tiers

| Tier | Backends | Notes |
|---|---|---|
| **Tier 1 — Supported** | Vulkan, Metal | webgpu.h semantics fully mapped; real-GPU conformance verified for the bring-up scope landed so far. |
| **Tier 2 — Experimental (best-effort)** | GLES (Android + Windows) | Opt-in `gles` cargo feature; never in `default`. Paths that do not cleanly map to GLES 3.1 may be rejected at the HAL layer with `HalError` (surfaced as a device error by `yawgpu-core`). Core validation is identical regardless of tier. On Windows the default context backend is ANGLE; an opt-in WGL fallback bypasses ANGLE and uses the host GL driver via `WGL_EXT_create_context_es2_profile` for environments where the locally available ANGLE caps at ES 3.0 — selected through either the `YAWGPU_GLES_BACKEND=wgl` env var or the `YaWGPUGlesContextBackend` chain entry on `WGPUInstanceDescriptor.nextInChain` (chain wins; see `specs/blocks/67-gles-backend.md` "Context backend (Windows)" row). See `CLAUDE.md` "Backend support tiers" and `specs/blocks/67-gles-backend.md`. |

D3D11/D3D12 remain permanently out of scope.

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
  `Noop | Vulkan | Metal | Gles`, backends behind Cargo features.
- **Noop** backend implemented first: synthetic, allocation-tracking, no GPU.
  It is the CI substrate and the TDD substrate for all validation phases.
- **Tier 1 real backends: Vulkan (ash) and Metal (objc2)** — full
  webgpu.h surface, real-GPU end-to-end verified.
- **Tier 2 real backend: GLES** (via `glow` + `khronos-egl` for EGL;
  `windows-sys` for the Windows opt-in WGL context backend), targeting
  Android (native EGL) and Windows (ANGLE default, opt-in WGL fallback
  for ES 3.0-capped ANGLE environments). Best-effort: webgpu.h paths
  that do not cleanly map to GLES 3.1 may be rejected at HAL with
  `HalError`. See `specs/blocks/67-gles-backend.md` for the mapping
  matrix.
- **DirectX (D3D11/D3D12) is permanently out of scope.** The `HalXxx`
  enum stays open to additional variants, but no D3D variant is planned.

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
  Dawn test file; Noop keeps every slice shippable without a GPU.
- **Tint**: the shader frontend (Dawn's WGSL compiler), built from a vendored
  Dawn submodule and driven from Rust via the `yawgpu-tint` C shim. Used for WGSL
  parse/validate, reflection, and per-backend shader emission (MSL / SPIR-V /
  GLSL ES). (Replaced the earlier naga frontend; Tint matches the CTS oracle.)
- **Wire**: explicitly out of scope; no dawn-wire analog.
- **Header drift**: pin the `webgpu.h` we bindgen against; record its commit
  in `specs/reference/dependencies.md`.
