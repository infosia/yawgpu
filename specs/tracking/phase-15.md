# Phase 15 — GLES backend (Tier 2 / experimental)

Status: **Phase 15 COMPLETE — VERIFIED ON REAL GPU 2026-05-25
via WGL fallback** (host NVIDIA driver, `OpenGL ES 3.2 NVIDIA
595.95`). All slices (P15.0 + P15.1 + P15.2 + P15.3 + P15.4 +
P15.5 + P15.6) + Phase 15 Review (`phase-15-review.md`) + the
post-COMPLETE WGL fallback slice CLOSED — 0 CRITICAL / 0 MAJOR
open; 3 MINOR deferred with logged rationale. **All 12 e2e_gles_*
tests pass on the host GPU** (basic 3/3, buffer 2/2, texture
3/3, compute 2/2, render 2/2) under
`YAWGPU_GLES_BACKEND=wgl`.

**Verification path (post-COMPLETE WGL slice, 2026-05-25):** the
original ANGLE-only EGL path could not be verified on the dev
machine because every locally available ANGLE binary
(Chrome / Edge / Unity / JetBrains JBR / LogiOptionsPlus —
all the same CEF-derived ES 3.0 build, upstream `git hash:
42cd1b60189f`) caps at WebGL2 / GLES 3.0 and fails yawgpu's
`>= 3.1` device-creation floor. The WGL fallback slice
(`yawgpu-hal/src/gles/wgl.rs`, selected via
`YAWGPU_GLES_BACKEND=wgl`) bypasses ANGLE entirely:
`opengl32.dll` + `WGL_EXT_create_context_es2_profile` ⇒ the
host NVIDIA driver provides a real ES 3.2 context. EGL remains
the default; WGL is an opt-in verification path on Windows.

**Pre-WGL history (kept for context):** the slice-acceptance
"N/N green on ANGLE" claims below were originally **incorrect
silent-skip false-passes** — `real_backend_available
(RealBackend::Gles)` returned `false` (ANGLE DLLs not on the
test binary's PATH, then later ES-3.0 cap), the e2e tests
self-skipped, and `cargo test` reported "ok" without
executing any GLES code path. Those tests were patched in
commit `24a235f` to **panic on missing GLES** instead of
self-skipping, then re-verified under WGL at `5c73ffa+1` —
this time genuinely 12/12 green on a real GPU. The static-
review "sound" findings (panic discipline, Drop, Send/Sync
soundness, FFI selection scope, Tier-2 isolation, naming, doc
comments) hold; the slice-acceptance pixel/byte assertions
**now also hold on real hardware**.

What this does **not** affect: Tier 1 (Vulkan, Metal) backends
are byte-for-byte unchanged across Phase 15. The Noop default
CI gate's pass count is unchanged. The yawgpu.h vendor
extension `YAWGPU_INSTANCE_BACKEND_GLES = 3` is wired and
selects the GLES primary instance as designed; that path was
verified by `triangle.exe` printing `backend=OpenGLES` after
the recent `examples/framework` patch. The reachability of the
GLES path is real; what is unverified is its correct
execution on the GPU.

**P15.1a was reverted on 2026-05-25** after the user authorized
extending `yawgpu.h` with `YAWGPU_INSTANCE_BACKEND_GLES = 3`;
the side-instance / `select_request_adapter` /
`PendingCallback::RequestAdapterError` / surface gles_core
fallback / `e2e_gles_ffi.rs` infrastructure was made redundant
by that vendor-extension path (which mirrors Metal/Vulkan).
Rules / plan: `../blocks/67-gles-backend.md`. Roles / loop:
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

## P15.1a — FFI selection via standard webgpu.h backendType  *(✗ REVERTED 2026-05-25)*

**Status:** Done in 30bc46c, then fully reverted post-Phase-15
COMPLETE after the user authorized extending `yawgpu.h` with
`YAWGPU_INSTANCE_BACKEND_GLES = 3` (the constraint "yawgpu.h は
対象外" was lifted on 2026-05-25). The standard webgpu.h
`backendType=OpenGLES` adapter-selection path became redundant
with the vendor-extension path, so all P15.1a infrastructure
was removed for consistency with Metal/Vulkan: `WGPUInstanceImpl.
gles_core`, `probe_gles_core`, `with_gles_probe`,
`select_request_adapter`, `PendingCallback::RequestAdapterError`
(+ its `callback_mode` / `fire` arms), the
`yawgpu/src/ffi/instance.rs` surface FFI gles_core fallback I
had added to support windowed surface creation from the standard
path, and `yawgpu/tests/e2e_gles_ffi.rs` (3 ignored real-ANGLE
tests). `wgpuInstanceRequestAdapter` reverted to its pre-P15.1a
shape (enumerate from primary instance + the
`expect("Noop instance must expose an adapter")` invariant).

Net post-revert: GLES is selectable only via
`YaWGPUInstanceBackendSelect.backend = YAWGPU_INSTANCE_BACKEND_
GLES` at `wgpuCreateInstance` time, mirroring Metal/Vulkan
exactly. Surface creation routes through the primary HAL
(GLES, if selected). The Noop default test pass count is
unchanged from the pre-P15.1a baseline; all other Phase 15
slice outcomes (HAL implementation, Tier-2 status, e2e_gles_
{smoke,basic,buffer,texture,compute,render}.rs) are unchanged
— P15.1a was purely an FFI-selection-path concern.

The original P15.1a done-record below is retained for git
history context; the listed code paths no longer exist.

### Original P15.1a record (now reverted)

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

## P15.2 — Buffer + Queue write/read + B2B copy  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): real GL-backed
`GlesBuffer` with HostBuffer fallback path
(`mapped_ptr` returns `None`; persistent mapping deferred).
`GlesBufferInner { Arc<GlesDeviceInner>,
Result<glow::Buffer, HalError>, size }` keeps `create_buffer`
infallible at the HAL dispatch level — allocation failures are
captured inside the buffer and surface at first
`write`/`read`/`submit_copies` use via the new `raw_or_err`
accessor. `Drop` deletes the GL buffer when allocation
succeeded, swallows the make-current `Err` if device teardown
is already in flight (no panic). `GlesBuffer::new` runs
`glCreateBuffer` + `glBindBuffer(GL_COPY_WRITE_BUFFER)` +
`glBufferData(NULL, size, GL_DYNAMIC_DRAW)` once at creation.
`write` issues `glBufferSubData` on `GL_COPY_WRITE_BUFFER`;
`read` uses `glMapBufferRange(MAP_READ_BIT)` + memcpy +
`glUnmapBuffer` on `GL_COPY_READ_BUFFER` (intentionally **not**
`glGetBufferSubData` since that requires GLES 3.2). Bounds
checks via a pure `check_range(offset, len, size, op)` helper
covered by inline unit tests (overflow + OOB rejection,
zero-length accept). `GlesQueue::submit_copies` dispatches
`HalCopy::Buffer` via `glCopyBufferSubData(GL_COPY_READ_BUFFER,
GL_COPY_WRITE_BUFFER, ...)` and rejects all other variants
with a P15.2-named message; ends with `glFlush` matching the
empty-submit shape. No explicit `glFenceSync`; the make-current
`Mutex<()>` plus core's `wgpuInstanceWaitAny` → `resolve_pending
_map` → `HalBuffer::read` flow provides the read-after-submit
ordering needed by the e2e round-trip.

`HalError` is not `Clone`; the agent wrote an explicit
`rebuild_hal_error(&HalError) -> HalError` matcher for all
current variants instead of bumping the public derive. (If
future slices grow `HalError` variants, this helper must be
extended too — flagged as a minor follow-up.)

New `yawgpu/tests/e2e_gles_buffer.rs` mirrors
`e2e_vulkan_buffer.rs` but selects the GLES backend via
standard `WGPURequestAdapterOptions.backendType =
WGPUBackendType_OpenGLES` (no `YaWGPUInstanceBackendSelect`
chain). Two tests cover full-buffer (offset 0/0) and partial
(src=8, dst=16) B2B copies; both round-trip identical bytes
out of `wgpuBufferGetConstMappedRange`. The
`default_noop_path` regression check was skipped as a
duplication with `e2e_metal_buffer` / `e2e_vulkan_buffer`'s
equivalent tests (consistent with the handoff's optional
clause).

Acceptance (all 10 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop pass count unchanged at 125;
  the new `check_range` unit tests live in `gles/buffer.rs`
  which only compiles under `--features gles`, so the Noop
  workspace gate is unaffected)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3 P15.1 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3 P15.1a regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_buffer
  -- --ignored` ✓ (**2/2 on real ANGLE; write → B2B copy →
  mapAsync → getConstMappedRange round-trip succeeded for both
  full and partial-offset variants**)
- `cargo build -p yawgpu --features vulkan` ✓

## P15.3 — Texture/Sampler + B2T/T2B/T2T  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): real GL-backed
`GlesTexture` + `GlesSampler` + the three texture-side copy
paths (B2T / T2B / T2T) for 2D non-multisample RGBA8Unorm
textures on Windows ANGLE.

New `yawgpu-hal/src/gles/format.rs` exposes module-private
`GlesFormat { internal, format, ty, bytes_per_pixel }` + a
`map_texture_format` table currently mapping only
`Rgba8Unorm`; other formats return
`HalError::BufferOperationFailed` with a P15.3-named message
(no speculative format expansion). `fallback_format` returns
RGBA8Unorm for the `derive_meta` path so meta is populated
even when allocation failed (Drop / accessors still work).

`GlesTextureInner` follows the P15.2 buffer pattern:
`Arc<GlesDeviceInner>` + `Result<glow::Texture, HalError>` +
`GlesTextureMeta { format, width, height, mip_level_count }`.
Drop deletes the GL texture when allocation succeeded; the
make-current `Err` is swallowed. `allocate_texture` rejects
`sample_count != 1`, `depth_or_array_layers != 1`, and
`mip_level_count == 0` before `glCreateTexture` +
`glBindTexture` + `glTexStorage2D` + unbind via
`with_current_context`. `raw_or_err` / `meta` accessors mirror
`GlesBuffer`.

`GlesSamplerInner` mirrors the same `Result<glow::Sampler,
HalError>` pattern. `allocate_sampler` maps the descriptor
through four pure helpers (`map_filter_mode`,
`map_address_mode`, `map_min_filter`,
`map_compare_function`), each inline-table-tested, and issues
`glSamplerParameteri/f` for wrap (`S`/`T`/`R`), mag/min
filter, LOD clamps, optional compare mode
(`COMPARE_REF_TO_TEXTURE` + compare func when
`descriptor.compare` is `Some`), and optional anisotropy when
`GL_EXT_texture_filter_anisotropic` is reported and
`max_anisotropy > 1`. `ClampToBorder` is intentionally
unmapped — GLES 3.1 core does not support it without
`GL_EXT_texture_border_clamp`; the e2e test uses
`ClampToEdge` so this gap doesn't fire.

`rebuild_hal_error` moved from `buffer.rs` to `gles/mod.rs` so
`buffer.rs` / `texture.rs` / `sampler.rs` share the single
matcher (carries the `TODO: Consider deriving Clone for
HalError upstream` note).

`GlesQueue::submit_copies` extended with three new arms:

- **`HalCopy::BufferToTexture`** → `submit_buffer_to_texture`:
  binds `GL_PIXEL_UNPACK_BUFFER`, sets `GL_UNPACK_ROW_LENGTH`
  (via `pixels_per_row(bytes_per_row, bytes_per_pixel)`) and
  `GL_UNPACK_ALIGNMENT = 1`, calls `glTexSubImage2D` with the
  PBO offset variant, resets `UNPACK_*` to defaults (0 / 4).
- **`HalCopy::TextureToBuffer`** → `submit_texture_to_buffer`:
  creates a transient FBO, attaches the source mip via
  `glFramebufferTexture2D(COLOR_ATTACHMENT0)`, sets
  `glReadBuffer(COLOR_ATTACHMENT0)` (required on GLES 3.0+
  for non-default FBO reads), validates completeness, binds
  `GL_PIXEL_PACK_BUFFER`, sets `PACK_ROW_LENGTH` /
  `PACK_ALIGNMENT = 1`, calls `glReadPixels` with the PBO
  offset variant, resets pack state, deletes the FBO.
- **`HalCopy::TextureToTexture`** → `submit_texture_to_texture`:
  inspects `supports_copy_image(gl)` (which checks both
  `gl.supported_extensions().contains("GL_EXT_copy_image")`
  AND the parsed `glGetString(GL_VERSION)` via the pure
  `gles_version_at_least_3_2` helper — inline-tested); on
  miss, returns a clear error directing the caller to expect
  the extension. When supported, issues
  `glCopyImageSubData(GL_TEXTURE_2D, …, GL_TEXTURE_2D, …, w,
  h, 1)`.

`ensure_2d_copy(depth_or_array_layers, z)` is shared by the
three new arms to reject 3D / array slice copies up front.
`i32_from_u32` / `u32_from_u64` consolidate numeric-conversion
error mapping. `pixels_per_row` carries a 4-case inline test
(`(256,4)→Ok(64)`, `(0,4)→Ok(0)` zero-stride single-row,
`(255,4)→Err`, `(8,0)→Err`).

`yawgpu/tests/e2e_gles_texture.rs` mirrors
`e2e_vulkan_texture.rs` (4×4 RGBA8Unorm, `bytes_per_row = 256`,
4 rows; same `write_padded_pixels` / `read_unpacked_texture_
buffer` helpers translating between tight pixel arrays and
padded buffer rows). Three tests cover buffer→texture→buffer
round-trip (B2T + T2B), texture→texture round-trip
(B2T + T2T + T2B), and sampler-creation smoke; all 3/3 green
on real ANGLE with no device errors.

Acceptance (all 11 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop pass count unchanged; new
  inline tests live under `--features gles`)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3 P15.1 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3 P15.1a regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_buffer
  -- --ignored` ✓ (2/2 P15.2 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_texture
  -- --ignored` ✓ (**3/3 on real ANGLE; B2T+T2B / B2T+T2T+T2B
  / sampler smoke**)
- `cargo build -p yawgpu --features vulkan` ✓

## P15.4 — Shader (naga→GLSL ES 3.10) + compute  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): WGSL→GLSL ES
3.10 compilation + GL compute pipeline + direct dispatch on
Windows ANGLE. Phase 15's most complex slice — touches both
yawgpu-core (shader generation) and yawgpu-hal (pipeline +
dispatch).

Cargo wiring: `yawgpu-core/Cargo.toml` exposes
`gles = ["naga/glsl-out"]`; `yawgpu/Cargo.toml` `gles` feature
gains `"yawgpu-core/gles"`. naga `glsl-out` compiles only with
the feature, so the Noop default dep graph is unchanged.

`yawgpu-hal/src/shader.rs` extended with
`HalShaderSource::Glsl { source, stage }` (the enum gains
`#[non_exhaustive]`) + new `HalShaderStage` enum (Vertex /
Fragment / Compute, `#[non_exhaustive]`).  `HalShaderStage`
re-exported from `yawgpu-hal/src/lib.rs`.

`yawgpu-core/src/shader_naga.rs` gained
`pub(crate) GeneratedGlsl { source, entry_point }` +
`ReflectedModule::generate_glsl(entry, stage)` behind
`#[cfg(feature = "gles")]`. Compute-only in P15.4
(`generate_glsl_rejects_non_compute_stage` inline test locks
the contract). naga API confirmed: `Options` has a
`use_framebuffer_fetch: false` field (caught by the agent;
absent from the handoff snippet) and uses
`BindingMap::default()`.
`Writer::new(..., BoundsCheckPolicies::default())` returns
`ReflectionInfo` from `writer.write()` which is intentionally
discarded — only the emitted GLSL string is needed.

`yawgpu-core/src/compute_pipeline.rs::select_compute_shader_
source` gained the `HalBackend::Gles` arm
(`#[cfg(feature = "gles")]`-gated): rejects passthrough
modules, generates GLSL via
`module.generate_glsl(entry, Compute)`, wraps as
`HalShaderSource::Glsl { stage: Compute }` + threads
`hal_descriptor_bindings(metal_bindings)` as the binding
metadata. New inline test
`select_compute_shader_source_generates_gles_glsl` asserts
the emitted source contains `#version 310 es` and the correct
`local_size_x` from the WGSL `@workgroup_size`.
`select_render_shader_source` deliberately **untouched** —
P15.5 owns the render path.

`yawgpu-hal/src/gles/pipeline.rs` rewrote
`GlesComputePipeline` as
`Arc<GlesComputePipelineInner { device, program:
Result<glow::Program, HalError>, workgroup_size, bindings:
Vec<HalDescriptorBinding> }>`. `build_compute_program` runs
`glCreateShader(COMPUTE_SHADER)` → `shaderSource` →
`compileShader` → check status (info log → `HalError::Shader
CompilationFailed { message: String }`), then `createProgram`
→ `attachShader` → `linkProgram` → `detachShader` →
`deleteShader` → check link status (info log → error). Drop
deletes the program via the make-current helper.
`GlesRenderPipeline` stays a stub (P15.5).

`yawgpu-hal/src/gles/device.rs::create_compute_pipeline`
switched from `unavailable()` to a real route: matches
`HalShaderSource::Glsl { stage: Compute }` only; other
variants / non-Compute stages return
`ShaderCompilationFailed`.

`yawgpu-hal/src/gles/queue.rs` gained the
`HalCopy::ComputePass(pass)` arm via `submit_compute_pass`:
validates the pipeline (`HalComputePipeline::Gles(_)`),
enforces single bind group (`@group(0)`), resolves each
`HalBoundBuffer.binding` to its GL target through
`compute_binding_target(pipeline.bindings(), binding)` (inline
table-tested: Uniform → `GL_UNIFORM_BUFFER`, Storage →
`GL_SHADER_STORAGE_BUFFER`, missing-binding → clean error),
then `use_program(Some)` + per-binding
`bind_buffer_range(target, binding, Some(buffer), offset,
size)` + `dispatch_compute(x, y, z)` +
`memory_barrier(ALL_BARRIER_BITS)` + `use_program(None)`.
The catchall message updated from "P15.3" to "P15.4" (now
only `RenderPass` / `SubpassRenderPass` remain unsupported).

New `yawgpu/tests/e2e_gles_compute.rs` mirrors
`e2e_vulkan_compute.rs` with `WGPURequestAdapterOptions
.backendType = WGPUBackendType_OpenGLES`. Two tests cover
single-SSBO (fill with squares) and dual-SSBO (input + 1)
compute paths; both 2/2 green on real ANGLE with no device
errors.

Emitted GLSL shape (from the agent's report):
`#version 310 es` + `precision highp float/int;` +
`layout(local_size_x=N, …) in;` +
`layout(std430) buffer …` with `layout(binding=N)` on each
storage buffer mirroring the WGSL `@binding(N)`. The
`bind_buffer_range` GL binding index therefore equals the
WGSL `@binding(N)` directly — no remap needed for single-group
layouts.

Acceptance (all 12 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop pass count unchanged; the
  new gles-gated inline tests don't fire under default
  features)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3 P15.1 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3 P15.1a regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_buffer
  -- --ignored` ✓ (2/2 P15.2 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_texture
  -- --ignored` ✓ (3/3 P15.3 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_compute
  -- --ignored` ✓ (**2/2 on real ANGLE; WGSL → GLSL ES 3.10
  → GL program → `glDispatchCompute` round-trip succeeded**)
- `cargo build -p yawgpu --features vulkan` ✓

## P15.5 — Render pipeline + draw  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): real GLES
render pipeline + drawArrays on Windows ANGLE. Touches core
(GLSL ES vertex/fragment emission + `select_render_shader_
source` Gles arm) and HAL (`GlesRenderPipeline` real impl,
`submit_render_pass`, vertex format + topology mappers).

`yawgpu-hal/src/shader.rs` extended with `HalShaderSource::
GlslStages { vertex, fragment }` mirroring `SpirVStages`.
`Glsl { source, stage }` stays as the compute-side variant.

`yawgpu-core/src/shader_naga.rs::generate_glsl` accepts all
three `naga::ShaderStage` variants (Compute / Vertex /
Fragment) via the same `Writer::new` + `BoundsCheckPolicies::
default()` machinery; no additional `WriterFlags` needed for
vertex/fragment (agent-confirmed). New inline tests assert
emitted source for Vertex and Fragment stages contain
`#version 310 es` + `void main()`.

`yawgpu-core/src/render_pipeline.rs::select_render_shader_
source` gained the `HalBackend::Gles` arm
(`#[cfg(feature = "gles")]`): rejects passthrough modules,
runs `generate_glsl(entry, Vertex)` + `generate_glsl(entry,
Fragment)` independently on the vertex/fragment reflected
modules (mirroring the Vulkan path's per-module spv-out call;
no same-module guard), wraps as `HalShaderSource::GlslStages
{ vertex, fragment }`, threads `hal_descriptor_bindings`.
Inline test pattern-matches the wrapper shape.

`yawgpu-hal/src/gles/format.rs` gained
`map_vertex_format(HalVertexFormat) ->
Result<GlesVertexFormat, HalError>` (Float32 / Float32x2 /
Float32x3 / Float32x4 mapped to `(components, GL_FLOAT,
normalized=false)`; Unsupported → error) and
`map_primitive_topology(HalPrimitiveTopology) -> u32`
(PointList / LineList / LineStrip / TriangleList /
TriangleStrip mapped to the corresponding `glow::*` constants).
Both pure, both inline table-tested.

`yawgpu-hal/src/gles/pipeline.rs` rewrote `GlesRenderPipeline`
as `Arc<GlesRenderPipelineInner { device, program: Result<glow
::Program, HalError>, vertex_buffers: Vec<HalVertexBufferLayout>,
primitive_topology, bindings: Vec<HalDescriptorBinding>,
first_instance_location: Option<glow::UniformLocation> }>`.
Agent design call: `glow::UniformLocation` is **not** assumed
`Copy` — stored as `Option<glow::UniformLocation>` directly,
passed to the queue as `Option<&glow::UniformLocation>`.
Build path: `glCreateShader(VERTEX_SHADER)` + compile + status
check + info-log → `ShaderCompilationFailed { message:
String }`, same for FRAGMENT_SHADER, then `createProgram` +
`attachShader` × 2 + `linkProgram` + status check + info-log
→ error, then `detachShader` × 2 + `deleteShader` × 2. After
link, `get_uniform_location(program, "naga_vs_first_instance")`
is queried and stored (`None` if absent). Pipeline-create
validation rejects multi-color-target / non-`Rgba8Unorm` /
depth-stencil specified / sample_count > 1 with P15.5-named
clean errors.

`yawgpu-hal/src/gles/device.rs::create_render_pipeline`
switched from `unavailable()` to a real route: matches
`HalShaderSource::GlslStages` only; other variants return
`ShaderCompilationFailed`.

`yawgpu-hal/src/gles/queue.rs::submit_copies` gained the
`HalCopy::RenderPass(pass)` arm via `submit_render_pass`.
Agent design call: cleanup uses an **outer-scope Drop guard**
(`RenderPassCleanup { gl, fbo, vao }` with a `Drop` impl that
unbinds VAO + deletes VAO + unbinds FBO + deletes FBO +
`use_program(None)` + `memory_barrier(ALL_BARRIER_BITS)`).
The Drop guard is constructed after both the FBO and VAO are
successfully created; subsequent `bind_render_buffers` /
`bind_vertex_buffers` / draw failures unwind through the
guard, ensuring cleanup runs regardless. Pre-guard FBO
creation failure falls back to a hand-coded cleanup; pre-guard
VAO creation failure releases the FBO explicitly before
returning.

The render path reuses `binding_target` (renamed from
P15.4's `compute_binding_target`) for UBO/SSBO bindings — the
function is identical for compute and render. The P15.4
inline tests for `binding_target` still apply.

Vertex attribute setup: per-binding `glBindBuffer(ARRAY_BUFFER)`
+ for each attribute `glEnableVertexAttribArray` +
`glVertexAttribPointer(loc, components, GL_FLOAT, false, stride,
buffer_offset + attr.offset)` + `glVertexAttribDivisor(loc, 1)`
when step_mode == Instance, else `divisor(loc, 0)`. The
pipeline's stored `vertex_buffers: Vec<HalVertexBufferLayout>`
indexes by `bound.binding`.

Draw: `glDrawArrays(topology, first_vertex, vertex_count)` for
the no-instancing case (instance_count==1 AND
first_instance==0); otherwise `glDrawArraysInstanced(topology,
first_vertex, vertex_count, instance_count)`. `first_instance`
uniform is set via `glUniform1ui` when the
`first_instance_location` is `Some` (the e2e tests don't use
`@builtin(instance_index)`, so the location is `None` and the
uniform set is skipped).

The `submit_copies` catchall now only rejects
`SubpassRenderPass` and any future variant; the message
updated from "P15.4" to "P15.5".

New `yawgpu/tests/e2e_gles_render.rs` mirrors
`e2e_vulkan_render.rs` with `backendType = WGPUBackendType_
OpenGLES`. Two tests cover constant-color triangle (no bind
group; red fragment) and uniform-color triangle (UBO bind
group; green from uniform); both 2/2 green on real ANGLE with
the pixel-content assertions
(`contains_pixel([255,0,0,255])` / `contains_pixel([0,255,0,255])`
plus the cleared-corner `contains_pixel_approx([26,51,77,255], 1)`)
all passing.

Acceptance (all 13 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop pass count unchanged)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3 P15.1 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3 P15.1a regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_buffer
  -- --ignored` ✓ (2/2 P15.2 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_texture
  -- --ignored` ✓ (3/3 P15.3 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_compute
  -- --ignored` ✓ (2/2 P15.4 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_render
  -- --ignored` ✓ (**2/2 on real ANGLE; vertex+fragment GLSL ES
  3.10 → GL program → FBO + viewport + clear + UBO bind + VAO
  + drawArrays round-trip succeeded for both constant-color
  and uniform-color shaders**)
- `cargo build -p yawgpu --features vulkan` ✓

## P15.6 — Surface (Android ANativeWindow + Windows ANGLE HWND) + Present  *(☑ DONE)*

Done (2026-05-24, commit pending at write time): real EGL
window surface creation (HWND for Windows ANGLE,
ANativeWindow for Android) + configure + acquire + present
(back-buffer blit-to-default-FB + `eglSwapBuffers`) on
Windows ANGLE. **Phase 15's final functional slice.**

`yawgpu-hal/src/gles/instance.rs` gained
`GlesInstance::create_surface_from_windows_hwnd(*mut c_void)`
and `..._from_android_native_window(*mut c_void)`, both
routing through a private `create_window_surface(native: *mut
c_void)` that reuses `choose_config` (RGBA8 + GLES3 +
PBUFFER_BIT) and calls
`eglCreateWindowSurface(display, config, native as _, None)`.
The cast resolves both HWND and ANativeWindow as
`*mut c_void` under `khronos_egl::NativeWindowType`. Errors
map to `HalError::SwapchainCreationFailed`.

`yawgpu-hal/src/lib.rs` switched the two
`HalInstance::create_surface_from_*` Gles arms from
`Err(BackendUnavailable)` to forwarding through the new
`GlesInstance` methods + wrapping as `HalSurface::Gles`.

`yawgpu-hal/src/gles/surface.rs` rewrote `GlesSurface` as
`Arc<GlesSurfaceInner { instance, window_surface: EglSurface,
state: Mutex<GlesSurfaceState> }>` with
`GlesSurfaceState::configured: Option<ConfiguredSurface
{ device: Arc<GlesDeviceInner>, back_buffer: GlesTexture,
width, height, swap_interval }>`. Drop calls
`make_current(None,None,None,None)` first (best-effort
release) then `destroy_surface(...)`; all errors swallowed.

`configure(device, config)`: validates `Rgba8Unorm` + non-zero
dims (inline-tested), allocates the back-buffer via the
existing `GlesTexture::new` path with `{ copy_src: true,
render_attachment: true, ..default }` usage, sets
`eglSwapInterval` (Fifo→1, Immediate/Mailbox→0;
inline-tested) via a transient window-surface make-current
behind the device's make-current mutex with the
`RestoreCurrent` Drop guard restoring the pbuffer. Replaces
`state.configured`, releasing the previous back buffer if
any.

`unconfigure()` drops the `ConfiguredSurface`; the EGL
window surface stays alive on `GlesSurfaceInner` until the
`GlesSurface` itself drops.

`acquire_next_texture()` returns
`configured.back_buffer.clone()` (Arc-backed; cheap).
Un-configured surface → `AcquireFailed`.

`present(queue)` runs `blit_and_swap`:
1. `current_lock_acquire` takes the device's make-current mutex.
2. `eglMakeCurrent(display, window, window, context)`.
3. `RestoreCurrent` Drop guard ensures the pbuffer is
   re-bound at scope end (success or error).
4. `glCreateFramebuffer` → `glFramebufferTexture2D
   (READ_FRAMEBUFFER, COLOR_ATTACHMENT0, back_buffer)` →
   `glReadBuffer(COLOR_ATTACHMENT0)` → completeness check →
   `glBindFramebuffer(DRAW_FRAMEBUFFER, None)` (window is
   the default FB now) → `glBlitFramebuffer(0,0,w,h,
   0,0,w,h, COLOR_BUFFER_BIT, NEAREST)` → unbind + delete
   read FBO.
5. `eglSwapBuffers(display, window_surface)`. Errors →
   `PresentFailed`.

The `GlesQueue` argument to `present` is intentionally
unused — queue work is already flushed by prior submits'
`glFlush` / `glMemoryBarrier(ALL_BARRIER_BITS)`; EGL's
swap acts as the final fence.

New module-private helpers:
- `GlesDeviceInner::current_lock_acquire()
  -> MutexGuard<'_, ()>` exposes the existing make-current
  mutex to the surface module (`with_current_context`
  wasn't usable because the closure binds the pbuffer; the
  surface module needs the window bound for blit + swap).
- `GlesDevice::inner_clone() -> Arc<GlesDeviceInner>` so the
  surface stores the device's inner Arc on configure.

`unavailable<T>()` helper removed from `gles/mod.rs` — no
more callsites; all P15.0 stub paths are now real or use
specific `HalError` variants.

Optional `e2e_gles_surface.rs` test: **agent skipped** per
the handoff's optional clause. Visual verification of the
full surface + render + present path is the manual
`examples/triangle --features gles` route on ANGLE — same
precedent as Vulkan/Metal swapchain testing in the project
(no automated headless surface tests for those either).

Acceptance (all 13 green; `e2e_gles_surface` intentionally
absent):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop pass count unchanged)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3 P15.1 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3 P15.1a regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_buffer
  -- --ignored` ✓ (2/2 P15.2 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_texture
  -- --ignored` ✓ (3/3 P15.3 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_compute
  -- --ignored` ✓ (2/2 P15.4 regression)
- `cargo test -p yawgpu --features gles --test e2e_gles_render
  -- --ignored` ✓ (2/2 P15.5 regression)
- `cargo build -p yawgpu --features vulkan` ✓

## Post-COMPLETE — WGL fallback (Windows opengl32.dll) *(☑ DONE 2026-05-25)*

Done (2026-05-25, separate coding-agent session per HANDOFF.md):
adds a Windows-only opt-in OpenGL ES context path via
`opengl32.dll` + `WGL_EXT_create_context_es2_profile`, selected
at `GlesInstance::new` time by `YAWGPU_GLES_BACKEND=wgl`
(default = `egl`, unchanged). Unblocks real-GPU verification on
machines without an ES 3.1-capable ANGLE binary — the host
GL driver (NVIDIA, AMD, Intel) provides the ES context directly.

Architecture: `GlesInstanceInner` / `GlesAdapter` / `GlesDeviceInner`
became static enums (`Egl(...)` / `Wgl(...)`) per CLAUDE.md
"no `dyn Trait`". Buffer / Texture / Sampler / Pipeline / Queue
are unchanged — they call `with_current_context` on
`GlesDeviceInner`, which dispatches transparently. New
`yawgpu-hal/src/gles/wgl.rs` carries `WglInstanceState`
(`LoadLibrary(opengl32.dll)` + `RegisterClassExW` for the hidden
helper window class) and `WglDeviceState` (hidden HWND + HDC +
dummy context for `wglGetProcAddress(wglCreateContextAttribsARB)`
→ real ES 3.1 profile context + `parse_gles_version` floor check
+ `parking_lot::Mutex<()>` make-current serialization). Helper
`tests/common/mod.rs` extracted to deduplicate
`create_gles_instance()` across the 5 e2e files (uses the
yawgpu.h vendor extension `YAWGPU_INSTANCE_BACKEND_GLES = 3`).

`GlesSurfaceInner` stays EGL-only (out of scope for this slice;
e2e tests are headless). WGL surface creation returns
`SwapchainCreationFailed` with a clear message directing callers
to use EGL or run headless tests.

New deps: `windows-sys = "0.59"` (optional, target `cfg(windows)`,
features `Win32_Graphics_OpenGL` / `Win32_Graphics_Gdi` /
`Win32_UI_WindowsAndMessaging` / `Win32_System_LibraryLoader`).
`parse_backend` (`gles/instance.rs`) gained inline table tests
for `None` / `""` / `"egl"` / `"wgl"` / `"unknown"`.

Acceptance (all 11 green, **WGL real-GPU verification included**):
- `cargo build --workspace` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --tests -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop default; 39 pass in yawgpu,
  65 pass in yawgpu-hal feature-gles incl. new `parse_backend`
  tests)
- `YAWGPU_GLES_BACKEND=wgl cargo test -p yawgpu --features gles
  --test e2e_gles_basic -- --ignored` ✓ (**3/3** on `OpenGL ES
  3.2 NVIDIA 595.95`)
- `... --test e2e_gles_buffer -- --ignored` ✓ (**2/2** real GPU)
- `... --test e2e_gles_texture -- --ignored` ✓ (**3/3** real GPU)
- `... --test e2e_gles_compute -- --ignored` ✓ (**2/2** real GPU)
- `... --test e2e_gles_render -- --ignored` ✓ (**2/2** real GPU,
  pixel assertions hold)
- EGL default path remains build-clean (regression-only;
  unverifiable on this dev machine pending an ES 3.1 ANGLE).

Out-of-scope (logged follow-ups, not blocking):
- ~~WGL surface creation (`GlesSurface` from HWND under
  `YAWGPU_GLES_BACKEND=wgl`); examples/triangle continues to
  require EGL on Windows.~~ **Closed by the WGL surface slice
  (2026-05-25, see next section).**
- Auto-fallback EGL → WGL on EGL init failure (today: env-var
  manual selection only).
- WGL availability probe in `yawgpu-test::gles_backend_available`
  (the env var is read at instance-creation time, so the existing
  probe still works correctly under `YAWGPU_GLES_BACKEND=wgl`).

## Post-COMPLETE — WGL surface (HWND → present) *(☑ DONE 2026-05-25)*

Done (2026-05-25, separate coding-agent session per HANDOFF.md):
adds the WGL **surface** path so `examples/triangle` runs end-to-end
under `YAWGPU_GLES_BACKEND=wgl`. Builds on the WGL fallback slice
(`51b0789`), which added only the context backend. Closes the
"WGL surface creation not implemented" caveat from the previous
slice.

Architecture: `GlesSurfaceInner` became a static enum
(`Egl(...)` / `Wgl(...)`) mirroring the `GlesInstanceInner` /
`GlesAdapter` / `GlesDeviceInner` pattern (CLAUDE.md
"no `dyn Trait`"). `GlesSurfaceKind::Wgl(WglSurfaceKind {
surface: WglSurfaceState })` carries the user-provided HWND +
its HDC. `WglSurfaceState` (in `gles/wgl.rs`) is constructed by
`WglInstanceState::create_window_surface(hwnd)`: `GetDC(hwnd)` →
`ChoosePixelFormat` + `SetPixelFormat` with the same descriptor
the helper HWND uses (extracted to `build_pixel_format_descriptor()`
so helper-HWND and user-HWND get identical pixel formats —
guarantees HGLRC compatibility). Drop releases via
`wglMakeCurrent(NULL,NULL)` + `ReleaseDC(hwnd, hdc)`; the HWND
itself is caller-owned and **not** destroyed.

`set_swap_interval` / `blit_and_swap` in `gles/surface.rs` became
kind-dispatched. WGL arm: `WglDeviceState::make_current_on_hdc
(surface.hdc)` → `wglSwapIntervalEXT` (cached lookup, silent
no-op on missing) → glow blit → `SwapBuffers(surface.hdc)`.
`RestoreCurrent` became an enum (`Egl { instance, device }` /
`Wgl { device }`); the Wgl arm re-binds the device's helper HDC
via `WglDeviceState::restore_current()` on Drop. The single HGLRC
is reused across helper HDC and user HDC — both share the same
pixel format descriptor so make-current is valid.

`HalInstance::create_surface_from_android_native_window` Wgl arm
explicitly rejects with `SwapchainCreationFailed { message:
"GLES Android surface requires the EGL backend" }`. (Android+WGL
is non-sensical.)

**Format expansion (scope creep, accepted):** `validate_config`
+ `map_texture_format` + `validate_render_pipeline_descriptor`
extended to accept `Bgra8Unorm` alongside `Rgba8Unorm`
(internally aliased to RGBA8 — GLES has no native BGRA8 internal
format, but the WebGPU contract for render targets is
shader-rgba-semantic on both, so aliasing is correct for the
render-and-present case the triangle example exercises). The
triangle example prefers `BGRA8Unorm` as the natural Windows
swapchain format; without this change it falls back to RGBA8
(still works, but `wgpuSurfaceGetCapabilities` would advertise
only RGBA8). The aliasing would be incorrect for CPU readback
of a BGRA-tagged texture (byte-swap semantics differ) — not
exercised by the triangle example, logged as a known
limitation.

Real-GPU verification on this machine:
- `cargo build --workspace`, `cargo build -p yawgpu --features
  gles`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo clippy -p yawgpu --features gles --tests -- -D
  warnings`, `cargo test --workspace` — all green.
- yawgpu-hal `--features gles` lib tests: **66 passed** (was
  65, +1 = `build_pixel_format_descriptor_matches_wgl_surface_
  contract` inline test).
- `YAWGPU_GLES_BACKEND=wgl` e2e regression on host NVIDIA
  driver (`OpenGL ES 3.2 NVIDIA 595.95`): **12/12 still green**
  (basic 3/3, buffer 2/2, texture 3/3, compute 2/2, render 2/2).
- **`examples/triangle` under `YAWGPU_BACKEND=gles
  YAWGPU_GLES_BACKEND=wgl`**: `EXIT=0` after 60 frames, log:
  ```
  yawgpu-gles: using WGL backend (host OpenGL ES profile)
  yawgpu-gles: WGL GL_VERSION="OpenGL ES 3.2 NVIDIA 595.95"
  yawgpu: backend=OpenGLES (requested YAWGPU_BACKEND=gles)
  ```
  60 frames of acquire → render pass → drawArrays → present
  (transient FBO + glBlitFramebuffer + SwapBuffers) on the
  user HWND without any device error. This is the first
  end-to-end Windows-host visual-path verification of the GLES
  backend.

Out-of-scope (logged follow-ups, not blocking):
- Window resize handling (back-buffer becomes stale on resize;
  the triangle example doesn't resize).
- Multi-window / multi-surface per device (current model: single
  HGLRC, surfaces multiplex via make-current).
- BGRA8 byte-order correctness for CPU readback (currently
  aliased to RGBA8 internally).
- Hidden-window e2e test for the WGL surface path (the surface
  code is verified manually via examples/triangle; an automated
  Win32-hidden-window test would close that gap).
- Auto-fallback EGL → WGL on EGL init failure (still env-var
  manual selection only).

## Phase 15 Review  *(☐ PENDING — final mandatory gate before Phase 15 COMPLETE)*

Per `specs/reference/workflow.md` → "Phase Review", a fresh
no-context subagent reviews the cumulative Phase 15 diff
(`fdf3007..HEAD`) against `blocks/67-gles-backend.md` +
`CLAUDE.md` + the Phase 15 exit criteria. Emits
CRITICAL/MAJOR/MINOR findings; severity-ordered fixes;
Phase 15 cannot close with any open CRITICAL/MAJOR. Logged
in `tracking/phase-15-review.md` at review-start time.

Manual `examples/triangle --features gles` verification on
ANGLE provides the end-to-end visual confirmation. The
example today selects backend via the yawgpu.h
`YaWGPUInstanceBackendSelect` extension (off-limits for GLES
per the user's scope rule); a small example-side change to
use `WGPURequestAdapterOptions.backendType =
WGPUBackendType_OpenGLES` may be required. Decision +
implementation are Phase 15 COMPLETE-followup work, not
review-blocking.
Phase 15 cannot be marked COMPLETE with any open CRITICAL/MAJOR
finding. MINORs may defer with explicit rationale.

## Post-COMPLETE — Android aarch64 cross-build verified (GLES + Vulkan) *(☑ DONE 2026-05-25)*

First cross-build of yawgpu for an Android target host
(macOS arm64 → `aarch64-linux-android`). Covers both the
GLES backend (the Phase 15 deliverable — its Android surface
entry point had only been exercised by `cargo check` on
macOS host, never by a real Android-target compile) and the
Vulkan backend (no Android-target build had been recorded
either, even though Vulkan is the Tier 1 mobile path on real
devices).

Setup (one-off, not committed to repo `.cargo/config.toml` —
NDK path is developer-machine-specific):

```
rustup target add aarch64-linux-android
export ANDROID_NDK_HOME=/path/to/ndk/30.0.14904198
export NDK_BIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
export SYSROOT="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_BIN/aarch64-linux-android24-clang"
export CC_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang"
export CXX_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang++"
export AR_aarch64_linux_android="$NDK_BIN/llvm-ar"
export BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android="--target=aarch64-linux-android24 --sysroot=$SYSROOT"

cargo build --release --target aarch64-linux-android -p yawgpu --features gles
```

The `BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android` env var is
**load-bearing** — without it, `yawgpu/build.rs`'s bindgen pass
on `webgpu.h` invokes clang with no Android sysroot and dies
on `'math.h' file not found`. Bindgen 0.72 honors this env-var
form (target-suffixed, underscored), so no `build.rs` change is
required.

Result on this host (M2, macOS 26.0):
- `cargo check --target aarch64-linux-android -p yawgpu --features gles` → green in 7s
- `cargo build --release --target aarch64-linux-android -p yawgpu --features gles` → green in 21s
- `cargo build --release --target aarch64-linux-android -p yawgpu --features vulkan` → green in 12s, **zero warnings**
- `cargo build --release --target aarch64-linux-android -p yawgpu --features "vulkan mobile"` → green in 14s, **zero warnings** (mobile = shader-passthrough + tiled)
- Artifacts: `target/aarch64-linux-android/release/libyawgpu.so`
  (3.4 MB GLES / 3.7 MB Vulkan, ELF 64-bit ARM aarch64,
  dynamically linked) + `libyawgpu.a` (~35 MB) + `libyawgpu.rlib`

Vulkan-on-Android note: `ash` loads `libvulkan.so` dynamically
at runtime (`libloading::Library::new("libvulkan.so")`) — no
NDK Vulkan loader linkage at build time. Android 7.0+ (API
24+) ships `libvulkan.so` as part of the platform, so the
runtime side is automatic on real devices.

Warnings: 14 in `yawgpu-hal` — all pre-existing, none Android-
specific. They surface only when the non-Windows GLES path is
built (Android, Linux-ANGLE, etc.) because the `#[cfg(windows)]`
WGL arm vanishes and leaves the remaining `match`/`if let`/
`let-else` patterns irrefutable, and the ANGLE-only constants
(`EGL_PLATFORM_ANGLE_*` in `gles/egl.rs`) become dead. Style
cleanup tracked as a non-blocking follow-up; does not affect
build success.

CI policy unchanged: Android cross-build is not added to the
default gate. This is a manual verification step the developer
runs locally when changing GLES code paths that touch Android-
relevant arms (EGL display creation, ANativeWindow surface,
non-Windows make-current).

## Post-COMPLETE — iOS aarch64 cross-build verified (Metal) *(☑ DONE 2026-05-25)*

First cross-build of the Metal backend for iOS targets
(`aarch64-apple-ios` real-device + `aarch64-apple-ios-sim`
Apple-Silicon simulator) from a macOS arm64 host with Xcode
26.5 / iOS SDK 26.5. Closes the implicit "does the Metal code
actually compile for iOS, not just macOS?" gap — until now,
Metal had only been built and verified for `aarch64-apple-
darwin`.

Logged here (rather than in a Phase 7 / Metal-specific
tracking file) for parity with the Android cross-build entry
immediately above — both are "post-Phase-15, post-COMPLETE
platform verifications" exercised in the same session.

Setup: nothing beyond Xcode + `rustup target add`. The Apple
toolchain auto-resolves the iOS sysroot via `xcrun`, and
bindgen picks up the Apple SDK on its own — no analog to the
Android `BINDGEN_EXTRA_CLANG_ARGS_*` env var is needed.

```
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
cargo build --release --target aarch64-apple-ios     -p yawgpu --features "metal mobile"
cargo build --release --target aarch64-apple-ios-sim -p yawgpu --features "metal mobile"
```

Result on this host (M2, macOS 26.0, Xcode 26.5):
- `cargo check --target aarch64-apple-ios     -p yawgpu --features metal` → green in 52s
- `cargo build --release --target aarch64-apple-ios     -p yawgpu --features metal`        → green in 19s, **zero warnings**
- `cargo build --release --target aarch64-apple-ios     -p yawgpu --features "metal mobile"` → green in 14s, **zero warnings**
- `cargo build --release --target aarch64-apple-ios-sim -p yawgpu --features "metal mobile"` → green in 23s, **zero warnings**
- Artifacts: `libyawgpu.{a,dylib,rlib}` under
  `target/aarch64-apple-ios{,-sim}/release/`. Both dylibs are
  Mach-O 64-bit ARM64. The device dylib carries
  `LC_VERSION_MIN_IPHONEOS` + sdk 26.5; the simulator dylib
  carries `LC_BUILD_VERSION platform=iOSSimulator` (`7`) +
  minos 14.0 + sdk 26.5.

`MTLCopyAllDevices()` is used in
`yawgpu-hal/src/metal/mod.rs:64` and is conventionally
"macOS-only" in Apple's docs, but `objc2-metal v0.3.2` exposes
it for all Apple platforms — the iOS link succeeds. Real-iOS
runtime behaviour (does the call return an empty array? does
it weak-link?) is a downstream concern for the first iOS
integrator; build-side, nothing further is needed.

CI policy unchanged: iOS cross-build is not added to the
default gate. Manual verification step, same as Android.

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

## Open follow-ups added by P15.2

- **Persistent buffer mapping** (`GL_EXT_buffer_storage` +
  `GL_MAP_PERSISTENT_BIT` + `GL_MAP_COHERENT_BIT`). P15.2 ships
  the HostBuffer-fallback path (`mapped_ptr` returns `None`),
  which round-trips correctly via core's `resolve_pending_map`
  → `HalBuffer::read` copy. Persistent map would replace the
  copy with a direct pointer to GPU memory (matching the
  Metal/Vulkan `mapped_ptr` path), reducing read-back latency.
  Optional optimization; functional behavior is already
  complete. Slot in opportunistically.
- **`rebuild_hal_error` matcher must grow with `HalError`
  variants.** Now lives in `yawgpu-hal/src/gles/mod.rs` (moved
  from `buffer.rs` in P15.3) and is shared by
  `buffer.rs` / `texture.rs` / `sampler.rs`. Any new variant
  added to `yawgpu-hal/src/error.rs` must be added to this
  matcher to keep the `raw_or_err` accessors working. Two
  alternatives if maintenance becomes a burden: (a) derive /
  implement `Clone` on `HalError` and drop the matcher, or (b)
  wrap the GL handle in `Result<_, Arc<HalError>>` so the
  inner can be cloned cheaply.

## Open follow-ups added by P15.3

- **Texture dimensions beyond 2D.** `allocate_texture` rejects
  `depth_or_array_layers != 1` and `sample_count != 1`.
  Adding 1D (2D with h=1), 2D array (`GL_TEXTURE_2D_ARRAY` +
  `glTexStorage3D`), 3D (`GL_TEXTURE_3D`), cube
  (`GL_TEXTURE_CUBE_MAP`), and multisample (renderbuffers)
  paths is required before P15.6's swapchain integration
  would expose them; can land opportunistically. Each gains
  matching arms in `submit_buffer_to_texture` /
  `submit_texture_to_buffer` / `submit_texture_to_texture`.
- **Texture format expansion.** P15.3 maps only
  `Rgba8Unorm`. Add formats incrementally as later e2e tests
  exercise them — at minimum the render targets P15.5 will
  use (`Bgra8Unorm` via `EXT_texture_format_BGRA8888`,
  `Rgba16Float`, depth/stencil formats). Don't speculate;
  add per-test.
- **T2T fallback when neither GLES 3.2 nor
  `GL_EXT_copy_image` is available.** Today returns a clean
  error. An FBO-blit fallback (`glBlitFramebuffer`) would
  cover color textures; precision is exact for `RGBA8` and
  similar UNORM formats. Slot in if a target driver lacks
  the extension.
- **`ClampToBorder` address mode.** GLES 3.1 core lacks it;
  `GL_EXT_texture_border_clamp` is required. Today the
  sampler creation returns an error if requested. Add
  extension probe + `GL_TEXTURE_BORDER_COLOR` plumbing when
  a real consumer surfaces.
- **`GlesSampler::raw_or_err`** is currently
  `#[allow(dead_code)]` — sampler-bind is a P15.4 concern.
  Remove the allow when P15.4 lands.
- ~~**Real test for `GlesSurface::present`.**~~ **Closed by
  P15.6** — the present path is now implemented and
  manually verified via `examples/triangle`. An automated
  Win32-hidden-window `e2e_gles_surface.rs` test remains a
  follow-up option (skipped in P15.6 per the handoff's
  optional clause).

## Open follow-ups added by P15.4

- **Multi-group bind layouts** (`@group(N>0)` in WGSL). P15.4
  rejects them with a clean error from `submit_compute_pass`
  (and the same will apply in P15.5 for render bind groups
  unless addressed). naga's `BindingMap` can be populated to
  flatten `(group, binding)` → a single GL binding index;
  pair it with HAL-side accounting that tracks max
  bindings-per-target. Defer until a test exercises
  multi-group.
- **Storage textures in compute** (`glBindImageTexture` +
  `image2D` / `imageStore` / `imageLoad` in GLSL). Not used by
  `e2e_gles_compute`. Land when a compute test demands it.
- **Indirect compute dispatch**
  (`glDispatchComputeIndirect`). The HAL currently has no
  `HalComputePass` indirect variant. To add: extend
  `HalComputePass` with `Option<HalIndirectBuffer { buffer:
  HalBuffer, offset: u64 }>`, plumb it through core's
  `ComputePassCommand`, then route in the GLES
  `submit_compute_pass`. Touches all backends; gate on a real
  driver for the indirect-dispatch e2e port from Phase 7.
- **Tighter `glMemoryBarrier` masks.** P15.4 issues
  `ALL_BARRIER_BITS` after every dispatch. Profiling on
  mobile may favor narrower masks
  (`SHADER_STORAGE_BARRIER_BIT |
  BUFFER_UPDATE_BARRIER_BIT`); deferred as a perf
  optimization.
- **`GlesComputePipelineInner.program` is `Result<glow::Program,
  HalError>` but the constructor returns `Result<Self, HalError>`
  short-circuiting the `Err` arm.** The wrapper is therefore
  dead weight in the Ok-only path (`raw_or_err` cannot return
  `Err` in practice). Either downgrade to a bare
  `glow::Program` field, or align with the
  `GlesBuffer`/`GlesTexture` pattern (infallible new + capture
  Err inside) when adding a similar `GlesRenderPipeline` in
  P15.5. Cosmetic, no behaviour impact.
- ~~**Vertex / fragment GLSL paths**~~ **Closed by P15.5.**

## Open follow-ups added by P15.5

- **Indexed draw / drawIndirect / drawIndexedIndirect.** The
  HAL today doesn't carry index-buffer / index-format /
  indirect-buffer fields on `HalDraw` or `HalRenderPass`.
  Adding indexed draw requires extending those structs (core
  change touching all backends); the GLES execution side is
  cheap (`glDrawElements` /
  `glDrawElementsInstanced` / `glDrawArraysIndirect`).
- **Depth / stencil attachment.** P15.5 rejects `descriptor.
  depth_stencil = Some(_)` at pipeline create. Adding it means:
  pipeline-create accepts depth-stencil format + compare /
  write enables + stencil ops; `submit_render_pass` attaches a
  depth-stencil texture/renderbuffer to the FBO and configures
  `glDepthMask` / `glDepthFunc` / `glStencil*` per the
  pipeline state.
- **Multi-color-target.** P15.5 enforces single color target.
  Extending to N targets: pipeline carries
  `Vec<HalTextureFormat>`; `submit_render_pass` accepts N
  `HalRenderColorTarget`s, attaches each to `GL_COLOR_
  ATTACHMENT{0..N-1}`, and updates the `glDrawBuffers` call.
- **Non-`Rgba8Unorm` color formats** for render. Tied to the
  ongoing P15.3-follow-up of expanding the GLES format table.
- **Sampler binding in render.** `binding_target` covers
  UBO/SSBO; texture+sampler binding for render needs
  `glActiveTexture` + `glBindTexture` + `glBindSampler(unit,
  sampler)` + setting the sampler uniform location, plus
  naga's combined-texture-sampler emission. Slot into
  P15.x when a real consumer surfaces.
- **Cull mode / front face / scissor.** GL defaults
  (CCW front, no cull, no scissor) work for the e2e test;
  WebGPU-aware values plumb through when needed.
- **Color blend state.** Currently dropped at the core
  boundary; reintroduce when HAL grows a blend descriptor.
- **VAO caching.** P15.5 creates a transient VAO per pass;
  cache by pipeline + vertex-buffer-handle-set for perf.

## Open follow-ups added by P15.6

- **Real multi-buffer swap chain ring.** P15.6 uses a
  single back-buffer reused across acquire/present cycles
  (back-buffer is allocated at `configure()`, returned from
  every `acquire_next_texture()`, blitted to default FB at
  `present()`). A real swap chain ring (N back buffers,
  rotation, no blit) needs either
  `EGL_EXT_swap_buffers_with_damage` plumbing or
  platform-specific buffer pooling. Slot in if mobile
  profiling demands the extra throughput.
- **True `Mailbox` semantics.** Currently
  Mailbox/Immediate both map to `eglSwapInterval(0)`. True
  Mailbox needs `EGL_EXT_present_opaque` or platform-side
  tricks. Documented in the matrix as a known limitation.
- **Surface format / present-mode caps from EGL config.**
  `wgpuSurfaceGetCapabilities` returns a fixed set
  (Rgba8Unorm + Fifo/Immediate/Mailbox). Querying real EGL
  config caps lets us advertise actual driver-supported
  formats (e.g. some ANGLE builds expose BGRA8888).
- **Optional `e2e_gles_surface.rs` (Win32 hidden window).**
  The handoff marked it optional; agent skipped. A future
  small slice can add it using the `windows` crate
  (`CreateWindowExW` with `WS_OVERLAPPEDWINDOW` but no
  `ShowWindow`) for automated regression coverage of the
  surface path.
- ~~**Example-side backend selection for GLES.**~~ **Closed
  post-Phase-15 (2026-05-25).** The user authorized extending
  `yawgpu.h` for the GLES backend; added
  `YAWGPU_INSTANCE_BACKEND_GLES = 3` so examples select GLES
  via `YaWGPUInstanceBackendSelect` the same way Metal/Vulkan
  do. The standard webgpu.h `backendType=OpenGLES` path
  (P15.1a) is kept for spec-pure consumers; the surface FFI
  also gained a `gles_core` fallback so that path is now
  windowed-surface capable. Both paths coexist.
- **Multi-color-target presentation.** Tied to the P15.5
  multi-target follow-up; not a surface-specific concern.

## Open follow-ups added by Android cross-build verification (2026-05-25)

- **`yawgpu-hal` non-Windows warnings.** When the `#[cfg(windows)]`
  WGL arm vanishes (Android, Linux+ANGLE, etc.), 14 warnings
  surface: 5 `irrefutable_let_patterns` (the static
  `GlesInstanceInner` / `GlesSurfaceInner` enums are
  effectively single-variant on non-Windows) and 9
  `dead_code` warnings for the ANGLE-only `EGL_PLATFORM_*`
  constants in `gles/egl.rs`. None affect runtime
  correctness. Cleanup options: gate the irrefutable
  `match`/`let-else` arms behind `#[cfg(windows)]` to fall
  back to direct field access on non-Windows, or accept the
  warnings as the cost of the static-enum pattern. The dead
  ANGLE constants should be `#[cfg_attr(not(windows),
  allow(dead_code))]` at minimum (the user may still want
  them at the source-of-truth layer even when only the
  Windows path consumes them today). Non-blocking; clippy
  `-D warnings` gate still runs against the macOS-host
  default build (Noop) and the Windows ANGLE build, both of
  which have all `#[cfg(windows)]` arms active and so do not
  hit these warnings.
- **Document Android cross-build in README.** README §"Using
  it from C" lists `--features gles` as the build flag for
  "Android / Windows ANGLE" but does not spell out the NDK +
  `BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android` env-var
  contract. A short callout would save the next developer a
  bindgen-error round trip.

## Post-COMPLETE — Programmatic GLES context-backend override via `YaWGPUGlesContextBackend` *(☑ DONE 2026-05-25)*

Added a new independent yawgpu vendor chain entry,
`YaWGPUGlesContextBackend` (`YAWGPU_STYPE_GLES_CONTEXT_BACKEND`),
for callers that need to force the GLES context backend from code
instead of mutating `YAWGPU_GLES_BACKEND` before `wgpuCreateInstance`.
The entry chains onto `WGPUInstanceDescriptor.nextInChain` alongside
`YaWGPUInstanceBackendSelect`; it is consumed only when the resolved
instance backend is GLES and is ignored by Noop / Metal / Vulkan.

Resolution rule: `YAWGPU_GLES_CONTEXT_BACKEND_EGL` and
`YAWGPU_GLES_CONTEXT_BACKEND_WGL` override the environment variable;
`YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT` or an absent chain entry falls
back to the existing `YAWGPU_GLES_BACKEND` parser, then default EGL.
On non-Windows hosts, a programmatic WGL request falls back to EGL
with the same diagnostic style as the env-var parser.
