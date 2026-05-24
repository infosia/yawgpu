# Phase 15 — GLES backend (Tier 2 / experimental)

Status: **P15.0 + P15.1 + P15.1a + P15.2 + P15.3 + P15.4 DONE;
P15.5+ PLANNED.** Rules / plan: `../blocks/67-gles-backend.md`.
Roles / loop: `../reference/workflow.md`.

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
- **Real test for `GlesSurface::present`.** Still a no-op
  placeholder; revisit in P15.6.

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
- **Vertex / fragment GLSL paths in
  `ReflectedModule::generate_glsl`.** Currently returns the
  "only supports compute" error for non-Compute stages. P15.5
  needs vertex+fragment emission (and `select_render_shader_
  source` Gles arm).
