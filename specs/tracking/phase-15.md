# Phase 15 — GLES backend (Tier 2 / experimental)

Status: **P15.0 + P15.1 + P15.1a DONE; P15.2+ PLANNED.** Rules /
plan: `../blocks/67-gles-backend.md`. Roles / loop:
`../reference/workflow.md`.

**Tier:** Tier 2 (best-effort, experimental). The `gles` cargo
feature is the sole experimental signal — no runtime markers
(`AdapterInfo` suffix, `log::warn!`, C `#define`) are added.
`yawgpu.h` vendor extensions (`tiled`, `shader-passthrough`) are
**not** implemented for GLES; the relevant features are not
advertised and the corresponding FFI calls reject GLES devices via
the existing "feature not enabled" / "backend unavailable" paths.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy
--workspace --all-targets -- -D warnings` green on **Noop**.
Per slice **also**: `cargo build -p yawgpu --features gles` +
clippy with the feature on. **Real-GPU end2end**
(`cargo test --features gles -- --ignored`) is run **manually by
the user on Windows ANGLE** and logged here per slice (the dev
machine has ANGLE; see memory `[[windows-native-vulkan-driver]]`
for the analogous Vulkan flow).

**Reused e2e ports:** Phase 7 already ported Dawn end2end
Basic / Compute / Copy targets (`e2e_basic`, `e2e_buffer`,
`e2e_copy`, `e2e_compute_dispatch`). Phase 15 adds **no new
Dawn-derived tests** — it parametrizes the existing tests over
`--features gles`. New GLES-specific direct unit tests (per
CLAUDE.md core principle 1) are added in `yawgpu-hal/src/gles/`.

**Phase ends with the mandatory Phase Review**
(`tracking/phase-15-review.md`, to be created at slice-completion
time).

Methodology: identical to Phase 7 — execution bring-up, not
validation-rule porting. Validation stays in `yawgpu-core`;
backend only executes already-validated work; driver failure →
`HalError` → device error, never panic; **no core-rule relaxation
for Tier 2**.

## P15.0 — Scaffolding + gating harness  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): `gles` cargo
feature wired in `yawgpu-hal` / `yawgpu` / `yawgpu-test` with
optional `glow 0.14` / `khronos-egl 6` (dynamic) / `libloading
0.8` deps (recorded in `reference/dependencies.md`). New
`yawgpu-hal/src/gles/mod.rs` scaffold module mirrors the P7.0
Metal placeholder shape: every fallible entry returns
`HalError::BackendUnavailable { backend: "gles" }`; infallible
creators are allocation-counting no-ops; `use glow as _; use
khronos_egl as _; use libloading as _;` proves the link with
zero EGL/GL calls. `HalBackend::Gles` + `Gles(...)` arms added to
every HAL enum (`HalInstance/HalAdapter/HalDevice/HalQueue/
HalSurface/HalBuffer/HalTexture/HalSampler/HalComputePipeline/
HalRenderPipeline`); per the block 67 Tier 2 policy
`HalTransientAttachment` and `HalSubpassRenderPass` (both
`tiled`-only) do **not** gain a `Gles` variant — the
`HalDevice` tiled-method arms return `BackendUnavailable`
directly. New `HalInstance::create_surface_from_android_native_
window` method added with the four-backend dispatch (Noop ok;
Vulkan / Metal Err with backend-specific messages; Gles
BackendUnavailable). `yawgpu-test` gained `RealBackend::Gles`
+ `gles_backend_available()` returning false in P15.0 +
existing `real_backend_skip_reason` format reused. One
`#[ignore]`'d `yawgpu/tests/e2e_gles_smoke.rs` asserts
unavailability and passes under `--features gles --
--ignored`. Inline `#[cfg(test)] mod tests` covers every new
`pub fn` (24 unit tests in the gles module).

Acceptance (all green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop default; GLES smoke
  `#[ignore]`-skipped as expected)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo build -p yawgpu --features vulkan` ✓ (regression
  clean)
- Metal feature on Windows is pre-existing unbuildable
  (`objc2` is Apple-only); not a regression of this slice.

Original deliverables list (kept for traceability):

Deliverables:

- `yawgpu-hal/Cargo.toml`: `gles = ["dep:glow", "dep:khronos-egl",
  "dep:libloading"]` + workspace deps.
- `yawgpu/Cargo.toml`: `gles = ["yawgpu-hal/gles",
  "yawgpu-test/gles"]`.
- Workspace `naga` features add `glsl-out`.
- `yawgpu-hal/src/lib.rs`: `HalBackend::Gles`; every HAL enum
  (`HalInstance`, `HalAdapter`, `HalDevice`, `HalQueue`, `HalBuffer`,
  `HalTexture`, `HalSampler`, `HalSurface`, `HalComputePipeline`,
  `HalRenderPipeline`) gains a `#[cfg(feature = "gles")] Gles(...)`
  arm. Inline `gles` placeholder module at
  `yawgpu-hal/src/gles/mod.rs` mirroring the P7.0 Metal placeholder
  shape (every fallible entry → `HalError::BackendUnavailable`;
  `enumerate_adapters()` empty; infallible creators are no-ops).
- `HalInstance::create_surface_from_android_native_window`
  introduced (Noop / Vulkan / Metal arms reject; Gles arm = stub).
- `yawgpu-test`: `RealBackend::Gles` + `real_backend_available(Gles)`
  → false in P15.0; `real_backend_skip_reason` updated.
- One `#[ignore]` `yawgpu/tests/e2e_gles_smoke.rs` asserting
  unavailability (proves the harness).
- Documentation: `CLAUDE.md` (Backend support tiers section),
  `DESIGN.md` (Tier section + HAL paragraph), `SPEC.md` (Phase 15
  row + Out of scope update), `specs/blocks/60-real-backends.md`
  (drop GL from out-of-scope), `README.md` (Tier 2 disclaimer +
  ANGLE binary distribution note).

Acceptance:

- Noop `cargo test --workspace` + `clippy --workspace
  --all-targets -D warnings` byte-for-byte unchanged.
- `cargo build -p yawgpu --features gles` clean.
- `cargo clippy -p yawgpu --features gles --all-targets -D warnings`
  clean.
- Smoke test passes under `--features gles -- --ignored`.
- Vulkan + Metal feature builds unchanged.

Coding-agent handoff template (to be issued at slice start):

```
## Task: gles — P15.0 scaffolding + Tier 2 docs

Goal: add a build-only Gles HAL arm + Tier 2 documentation; CI
stays Noop-green; --features gles compiles.

Inputs to read:
- specs/blocks/67-gles-backend.md (this slice)
- specs/tracking/phase-15.md (P15.0 section)
- yawgpu-hal/src/metal/mod.rs (placeholder shape to mirror)
- yawgpu-hal/src/lib.rs (HAL enum dispatch pattern)
- CLAUDE.md / DESIGN.md / SPEC.md (Tier section to add)

Produce:
- yawgpu-hal: feature + Gles arms + stub module
- yawgpu / yawgpu-test: feature forwarding + RealBackend::Gles
- yawgpu/tests/e2e_gles_smoke.rs (#[ignore])
- Documentation edits per the deliverables list

Out of scope: any real EGL / GL code path; surface implementation;
yawgpu.h extension integration; binding to glow API calls.

Acceptance criteria:
- [ ] cargo build -p yawgpu --features gles clean
- [ ] cargo clippy -p yawgpu --features gles --all-targets
  -D warnings clean
- [ ] Noop cargo test --workspace byte-for-byte unchanged
- [ ] e2e_gles_smoke passes under --features gles -- --ignored
- [ ] CLAUDE.md / DESIGN.md / SPEC.md / blocks/60 / README updated
- [ ] no panics in yawgpu-core / yawgpu-hal; CLAUDE.md conventions met

Report back: files changed, any planned deliverables intentionally
deferred (+why).
```

## P15.1 — EGL display + Instance/Adapter/Device/Queue  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): real EGL bring-up
on Windows ANGLE behind the `gles` feature. `yawgpu-hal/src/gles/`
split into per-resource files mirroring `vulkan/`'s layout
(`egl.rs` / `instance.rs` / `adapter.rs` / `device.rs` /
`queue.rs` + stubs for `buffer.rs` / `texture.rs` / `sampler.rs`
/ `surface.rs` / `pipeline.rs`). `egl.rs` loads `libEGL.dll` via
`khronos-egl` dynamic; honors `YAWGPU_ANGLE_PATH` env var by
preloading ANGLE DLLs from a user-specified directory before the
default loader runs (`std::mem::forget` keeps the preloaded
library alive for subsequent `LoadLibrary` calls to resolve the
ANGLE EGL/GLESv2 symbols). `GlesInstance::new` performs
`get_display(EGL_DEFAULT_DISPLAY)` + `initialize` + `bind_api
(OPENGL_ES_API)`; failure on any step ⇒
`HalError::BackendUnavailable`. `enumerate_adapters` returns one
adapter per RGBA8888 + PBUFFER_BIT + OPENGL_ES3_BIT config
(`choose_first_config`), empty on miss. `GlesAdapter::create_
device` builds the real `EGLContext` (MAJOR_VERSION=3, MINOR_
VERSION=1), 1×1 pbuffer surface, make-currents, loads `glow` via
`from_loader_function(eglGetProcAddress)`, parses `GL_VERSION`
via the pure `parse_gles_version` helper (table-tested for "ES
3.1" / "ES 3.2 ANGLE" / "ES 3.0" + reject cases "ES-CM 1.1" /
empty / "OpenGL 4.5"), rejects versions `< 3.1` and tears the
context/surface back down. `GlesDeviceInner` carries the EGL
context + surface + `glow::Context` + `parking_lot::Mutex<()>` +
`AtomicU64` allocation counter; `with_current_context<R>(impl
FnOnce(&glow::Context) -> R)` is the make-current-and-run helper.
`Drop` order: `make_current(None,None,None,None)` →
`destroy_surface` → `destroy_context` → instance Arc Drop runs
`eglTerminate`. `GlesQueue::submit_empty` make-currents +
`glFlush`. Resource creators not in P15.1 scope stay
`unavailable()` (buffer/texture/sampler/pipelines/surface);
infallible creators increment the allocation counter to preserve
the Noop counting contract. `yawgpu-test::gles_backend_
available()` now performs a real instance+adapter+device probe
(mirrors `vulkan_backend_available`); a `yawgpu-hal` dep was
added (optional, gated on the `gles` feature) so the probe can
construct `GlesInstance` directly. New
`yawgpu/tests/e2e_gles_basic.rs` covers adapter name + non-empty
backend, device queue + empty submit, and zero-allocation at
creation; all three pass on ANGLE.

Spec divergence noted: the original block 67 entry for "Error
mapping" mentioned `HalError::BackendOperationFailed`; the
actual `HalError` enum has `BufferOperationFailed`
(plus `Acquire/Present/SwapchainCreationFailed` and the
`backend`-only variants `BackendUnavailable` /
`DeviceCreationFailed` / `QueueSubmissionFailed` /
`ShaderCompilationFailed`). The agent used the existing variants
correctly; the block 67 cell needs a follow-up edit if we want
prose accuracy.

Coverage trade-off (carried into P15.2+): the P15.0 mod.rs
inline tests that constructed `GlesDevice::new_for_scaffold()` /
`GlesQueue` unit values were dropped during the module split
since the production constructors are now `pub(super)` and
require a real EGL chain. This matches the Vulkan/Metal pattern
(no inline tests on driver-required pub fns; coverage comes
from `e2e_*` instead). `parse_gles_version` is the only new pure
function and is fully covered. The `gles_surface_present_is_
covered_by_e2e` test in `surface.rs` is a no-op placeholder; the
real present path lands in P15.6 with the surface implementation
and will get a real test then.

Acceptance (all 8 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop default unchanged)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (**3/3, real ANGLE GPU**)
- `cargo build -p yawgpu --features vulkan` ✓ (regression
  clean)

## P15.1a — FFI selection via standard webgpu.h backendType  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): wired
`wgpuInstanceRequestAdapter` to honor
`WGPURequestAdapterOptions.backendType = WGPUBackendType_OpenGLES`
without touching `YaWGPUInstanceBackendSelect` (the yawgpu.h
vendor extension, off-limits per the user's GLES scope rule).
`WGPUInstanceImpl` gained a `#[cfg(feature = "gles")] gles_core:
Option<Arc<core::Instance>>` field, populated at
`wgpuCreateInstance` by `probe_gles_core()`
(`GlesInstance::new()?` → `HalInstance::Gles(...)` →
`core::Instance::from_hal(...)`; silent `None` on any failure,
no panic). Both `new_noop` and `from_core` route through the new
`with_gles_probe` constructor so the side-instance probe runs
regardless of which primary backend
`YaWGPUInstanceBackendSelect` chose. `wgpuInstanceRequestAdapter`
now reads `options.backendType`: `OpenGLES` →
`select_request_adapter` enumerates from `gles_core`; any other
value (including `Undefined`) → existing primary-instance path
unchanged. When the GLES path is requested but `gles_core` is
`None` (feature absent OR EGL/ANGLE init failed), the callback
fires with `WGPURequestAdapterStatus_Unavailable` and a null
adapter via a new `PendingCallback::RequestAdapterError` variant
(integrated into both `callback_mode()` and the dispatch arm of
`PendingCallback::fire`). `adapter_info_from_core` gained the
`HalBackend::Gles → WGPUBackendType_OpenGLES` arm. New
`yawgpu/tests/e2e_gles_ffi.rs` covers three paths: GLES adapter
returned when requested (assert
`AdapterInfo.backendType == WGPUBackendType_OpenGLES`), Noop
adapter returned for `Undefined` (regression check — the new
branch must not change the default path), and the Unavailable
callback path (self-skipped when GLES *is* available on the
host; the inline `select_request_adapter_opengles_with_no_side_
instance_returns_none` unit test covers the logic directly
either way). The yawgpu.h vendor extension surface is **byte-
for-byte unchanged**.

Spec correction landed alongside: block 67 "Error mapping" cell
was previously claiming `HalError::BackendOperationFailed` (not
a real variant); rewritten to list the actual variants used
(`BufferOperationFailed` + `backend`-only kinds +
message-carrying surface kinds). This closes the corresponding
open follow-up from P15.1.

Acceptance (all 9 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (125 passed = P15.1's 123 + 2 new
  inline tests for `select_request_adapter`; Noop pass count
  delta is just the new tests, no regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3 regression; HAL path unchanged)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3 on real ANGLE; unavailable test
  self-skips because GLES is available)
- `cargo build -p yawgpu --features vulkan` ✓

## P15.2 — Buffer + Queue write/read + B2B copy  *(☐ PLANNED)*

Reuses `e2e_buffer`. Decision required: buffer-mapping fence model
(see `blocks/67` open questions). FFI integration is now in place
(P15.1a), so e2e_buffer can target GLES through the standard
`backendType = WGPUBackendType_OpenGLES` path — no additional
wiring needed beyond the buffer HAL implementation itself.

## P15.3 — Texture/Sampler + B2T/T2B/T2T  *(☐ PLANNED)*

Reuses `e2e_copy` texture subset. Decision required:
storage-texture format gating timing.

## P15.4 — Shader (naga→GLSL ES 3.10) + compute  *(☐ PLANNED)*

Reuses `e2e_compute_dispatch`. Naga `glsl-out` smoke confirmed in
P15.0/P15.1; any uncovered WGSL constructs flow into this slice's
scope.

## P15.5 — Render pipeline + draw  *(☐ PLANNED)*

Reuses `e2e_basic` draw portion. `first_instance` via naga uniform
injection.

## P15.6 — Surface (Android ANativeWindow + Windows ANGLE HWND)  *(☐ PLANNED)*

`examples/triangle` runs under `--features gles` on ANGLE.

## Phase 15 Review  *(☐ PLANNED)*

Mandatory Clean Review Then Fix per `specs/reference/workflow.md`.
Logged in `tracking/phase-15-review.md` at slice-completion time.
Phase 15 cannot be marked COMPLETE with any open CRITICAL/MAJOR
finding. MINORs may defer with explicit rationale.

## Open follow-ups (carried from `blocks/67-gles-backend.md`)

- naga `glsl-out` coverage smoke for Phase 7 e2e shaders.
- Adapter limit mapping reconciliation with core
  `RequiredLimits` validation.
- ANGLE binary distribution wording in `README.md`.
- Buffer mapping fence model definition.
- Storage-texture format gating timing.
- Resource hazard barrier mask defaults.

## Open follow-ups

- ~~**FFI selection wiring for GLES.**~~ **Closed by P15.1a.**
  `wgpuInstanceRequestAdapter` routes
  `backendType == WGPUBackendType_OpenGLES` to a side
  `gles_core` instance; `YaWGPUInstanceBackendSelect` is
  untouched.
- ~~**block 67 "Error mapping" wording.**~~ **Closed by
  P15.1a** (spec correction landed in the same commit).
- **Real test for `GlesSurface::present`.** The current
  placeholder in `surface.rs` is a no-op. Replace with a real
  test when P15.6 brings up surface creation.
- **Generalize `backendType` routing to other backends.**
  P15.1a routes only `OpenGLES`; `Metal` / `Vulkan` / `D3D*`
  values continue to be ignored, and `YaWGPUInstanceBackendSelect`
  remains the selector for those. If the project later wants
  full `backendType` honoring, generalize `select_request_adapter`
  (and decide how it interacts with the yawgpu.h vendor
  extension). Not in Phase 15 scope.
