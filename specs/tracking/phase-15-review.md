# Phase 15 Review — Clean Review Then Fix

Process: `../reference/workflow.md` → "Phase Review". Fresh
no-context reviewer over the cumulative Phase-15 production diff
(`fdf3007..b233d64`, P15.0 + P15.1 + P15.1a + P15.2 + P15.3 + P15.4 +
P15.5 + P15.6) + `blocks/67-gles-backend.md` + `phase-15.md` slice
notes + `CLAUDE.md` + `reference/naming-conventions.md`.

Result: **0 CRITICAL, 1 MAJOR, 5 MINOR**.

Reviewer verified sound (no finding): no `unwrap`/`expect`/`panic!`/
unchecked-index/`as`-truncation reachable from the C ABI on hostile
input in `yawgpu-core`/`yawgpu-hal`'s GLES path (all `khronos_egl` /
`glow` fallibles → `HalError`; offset/length casts via
`try_from`/`i32_from_u32`/`u32_from_u64`); GL resource lifetimes
correct (Arc-held instance > device > resources; per-operation FBO/
VAO scoped within make-current; `RenderPassCleanup` /
`RestoreCurrent` Drop guards on render-pass and present); every
`unsafe impl Send/Sync` for the `Gles*` types matched by a
make-current-mutex-protected SAFETY comment, no shared-mutation
escape; **Tier 2 / core-validation isolation holds** — the
`HalBackend::Gles` arms in `yawgpu-core/src/{compute_pipeline,
render_pipeline,shader_naga}.rs` only **add** code paths, never
weaken existing rules; **FFI selection scope (P15.1a) clean** —
`wgpuInstanceRequestAdapter` only branches on
`WGPUBackendType_OpenGLES`, all other `backendType` values
(including `Undefined`) route through the original primary path
unchanged; `YaWGPUInstanceBackendSelect` byte-for-byte unchanged;
new `PendingCallback::RequestAdapterError` variant symmetric with
the existing pending-callback machinery (mode fan-out + `fire`
dispatch + owned-string-view leak-free); spec / matrix conformance
clean for ✗ Deferred items (3D / array / cube / multisample
textures, multi-group bind layouts, non-RGBA8Unorm color targets,
depth/stencil at pipeline create, `glCopyImageSubData` driver-miss
— all return clean errors, no panics, no silent no-ops); Noop /
Vulkan / Metal backends byte-for-byte unchanged (`git diff
fdf3007..b233d64 -- yawgpu-hal/src/{noop,vulkan,metal}` empty); the
HAL dispatch in `yawgpu-hal/src/lib.rs` only adds `#[cfg(feature =
"gles")] Gles(...)` variants and arms, no existing arm semantics
changed; naming conventions (`GlesXxx` inner types, `map_*`
converters, `submit_*` queue handlers) match the Vulkan / Metal
precedent.

## Triage + disposition

| ID | Sev | File:line | Finding | Disposition |
|---|---|---|---|---|
| q1 | MAJOR | `yawgpu-hal/src/gles/queue.rs:197-203` | `submit_render_pass` rejects `pass.pipeline == None` with `BufferOperationFailed { message: "render pass requires a GLES pipeline" }`. `yawgpu-core::queue::hal_render_pass_execution` emits `HalRenderPass { pipeline: None, draw: None, ... }` for clear-only passes; Vulkan handles via temporary render-pass-begin/end, Metal via the encoder's load_op clear. GLES rejects, so a WebGPU validated op with a trivial GLES 3.1 mapping (`glClear`) fails on the Gles backend with a confusing message. | **FIX (q1 handoff slice)** — Phase 15 cannot close MAJOR-open. In `submit_render_pass`, branch on `pass.pipeline.is_none()`: take the FBO + viewport + load_op path (already isolated in `create_render_fbo`) but skip the VAO + draw + program-use; `RenderPassCleanup` continues to handle FBO/VAO cleanup. |
| q2 | MINOR | `yawgpu/src/ffi/mod.rs:947 + yawgpu/src/ffi/instance.rs:97-142` | `has_supported_surface_source` accepts `WGPUSType_SurfaceSourceAndroidNativeWindow` but no `find_android_native_window_source` exists; `wgpuInstanceCreateSurface` never calls `core.create_surface_from_android_native_window`. Android-from-C callers get a non-error `WGPUSurface` whose `hal: None` then errors at `configure()`. HAL side is fully wired (P15.6); only the FFI bridge is missing. | **FIX (q1 handoff slice)** — Android is an explicit Phase 15 target; the matrix row `Surface (Android)` claims "code path implemented", which is true at HAL level but misleading at FFI level. Cheap fix: clone `find_windows_hwnd_source` shape into a `find_android_native_window_source` helper that resolves the chained `WGPUSurfaceSourceAndroidNativeWindow.window` and route it through `core.create_surface_from_android_native_window`. |
| q3 | MINOR | `yawgpu-hal/src/gles/queue.rs:680-715` | `i32_from_u32`, `u32_from_u64`, `ensure_2d_copy` are pure helpers without inline `#[cfg(test)] mod tests` coverage. Existing peers (`pixels_per_row`, `check_range`, `parse_gles_version`, `gles_version_at_least_3_2`, `binding_target`, sampler mappers, format mappers, vertex-format mapper, primitive-topology mapper, `validate_config`, `swap_interval_for_present_mode`) all have inline table tests; symmetry says these three should too. | **FIX (q1 handoff slice)** — three small table tests, no behavior change. |
| q4 | MINOR | `yawgpu-hal/src/gles/pipeline.rs:13-18 + 84-91` | `GlesComputePipelineInner.program` / `GlesRenderPipelineInner.program` are typed `Result<glow::Program, HalError>` but the constructors short-circuit on `Err` (`?` propagation at construction), so the `Err` arm is unreachable in production. Dead wrapping that keeps `rebuild_hal_error` on every variant's maintenance path. | **DEFER (logged)** — already entered as a P15.4 follow-up in `phase-15.md`; cosmetic, no behavior bug. Re-evaluate alongside `HalError: Clone` (see q5). |
| q5 | MINOR | `yawgpu-hal/src/gles/mod.rs:30-62` | `rebuild_hal_error` hand-rolled per-variant matcher because `HalError` is not `Clone`. Every new `HalError` variant added upstream must be added here too, otherwise `raw_or_err` accessors emit wrong-variant errors. Pre-existing follow-up. | **DEFER (logged)** — already entered as a P15.2/P15.3 follow-up. The "fix" is upstream (`derive(Clone)` on `HalError`, or wrap GL handles in `Result<_, Arc<HalError>>`). Out of scope for the Phase 15 close. |
| q6 | MINOR | `yawgpu-hal/src/gles/queue.rs:506-585` + `yawgpu-hal/src/gles/surface.rs:300-339` | `submit_texture_to_buffer` and `blit_back_buffer_to_window` build a transient FBO with hand-rolled cleanup at each error site, rather than the `RenderPassCleanup`-style Drop guard P15.5 uses. Today no `?`-bearing operations sit between FBO creation and unconditional cleanup, so a leak is impossible — but the asymmetry invites a future edit (e.g. adding a fallible bounds check between `bind_framebuffer` and `delete_framebuffer`) to silently leak. | **DEFER (logged)** — defense-in-depth, no current bug. Add as a new follow-up entry in `phase-15.md` under "Open follow-ups added by P15.6". Pattern: factor a `TransientFbo { gl, fbo }` Drop-guard alongside `RenderPassCleanup` and reuse from both call sites. |

No false positives. **1 MAJOR + 3 MINOR fixes** routed through the
q1 fix handoff (handed to the coding agent). **2 MINORs** deferred
with the rationale logged above (and as follow-up entries in
`phase-15.md`). Per workflow.md, MINOR may defer with explicit
written rationale; phase cannot close with any open CRITICAL/MAJOR.

## Resolution log

**CLOSED** (2026-05-25). Q1 MAJOR + Q2/Q3 MINOR fixed in the
fix-slice; Q4/Q5/Q6 MINOR deferred with rationale (logged above
and as follow-up entries in `phase-15.md`).

Fix-slice changes:
- **Q1** (`yawgpu-hal/src/gles/queue.rs`): `submit_render_pass`
  refactored — FBO is created first (load_op clear runs
  unconditionally), then a three-arm match on `pass.pipeline`:
  `None` → `RenderPassCleanup { vao: None }` + `return Ok(())`
  (clear-only); `Some(Gles)` → existing draw path; `Some(_)` →
  Drop-guarded `BufferOperationFailed`. `RenderPassCleanup.vao`
  promoted to `Option<glow::VertexArray>`; Drop no-ops the
  VAO unbind/delete when `None`. The error path during VAO
  creation also routes through the Drop guard for FBO release
  (defense-in-depth bonus). `e2e_gles_render` regression-clean.
- **Q2** (`yawgpu-core/src/instance.rs` +
  `yawgpu/src/ffi/{instance,mod}.rs`):
  `core::Instance::create_surface_from_android_native_window`
  added mirroring the existing Windows HWND method (with a Noop
  inline unit test).
  `find_android_native_window_source` helper added to
  `ffi/mod.rs` mirroring `find_windows_hwnd_source`.
  `wgpuInstanceCreateSurface` gained an Android branch after
  the existing HWND check (gated by a `found_hwnd_source`
  flag so metal/hwnd/android priority order is explicit). The
  metal/hwnd/no-source paths are byte-for-byte unchanged.
- **Q3** (`yawgpu-hal/src/gles/queue.rs::tests`): three new
  inline tests — `i32_from_u32_accepts_in_range_and_rejects
  _overflow`, `u32_from_u64_accepts_in_range_and_rejects_overflow`,
  `ensure_2d_copy_accepts_layer_one_z_zero_only`.

Gate at close (all 13 green):
- `cargo build -p yawgpu` (Noop default) ✓
- `cargo build -p yawgpu --features gles` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo clippy -p yawgpu --features gles --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (Noop pass count unchanged from
  pre-fix; core gained 1 new inline test, gles queue gained 3)
- `cargo test -p yawgpu --features gles --test e2e_gles_smoke
  -- --ignored` ✓ (1/1)
- `cargo test -p yawgpu --features gles --test e2e_gles_basic
  -- --ignored` ✓ (3/3)
- `cargo test -p yawgpu --features gles --test e2e_gles_ffi
  -- --ignored` ✓ (3/3)
- `cargo test -p yawgpu --features gles --test e2e_gles_buffer
  -- --ignored` ✓ (2/2)
- `cargo test -p yawgpu --features gles --test e2e_gles_texture
  -- --ignored` ✓ (3/3)
- `cargo test -p yawgpu --features gles --test e2e_gles_compute
  -- --ignored` ✓ (2/2)
- `cargo test -p yawgpu --features gles --test e2e_gles_render
  -- --ignored` ✓ (2/2; Q1 regression clean)
- `cargo build -p yawgpu --features vulkan` ✓

**Phase 15 COMPLETE.** Tier 2 / experimental GLES backend
covers the webgpu.h core surface (Instance / Adapter / Device /
Queue / Buffer + B2B / Texture + Sampler + B2T/T2B/T2T /
Compute + Dispatch / Render + Draw / Surface + Present) on
Windows ANGLE with HAL-side Android ANativeWindow support
(FFI bridge now wired). Core validation stays Tier-independent.
Noop / Vulkan / Metal backends byte-for-byte unchanged.
