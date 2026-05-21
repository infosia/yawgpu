# Phase 12 — Phase Review (mandatory)

Status: **COMPLETE** (2026-05-21 — 0 CRITICAL / 0 MAJOR; MINOR #1 + #2
fixed in `37b6f89`, #3 + #4 dropped with rationale). Per
`../reference/workflow.md` ("Phase Review"): a fresh no-context
subagent reviewed the cumulative Phase 12 diff (`2417855^..HEAD` —
P12.0 spec, P12.1 cross-platform `VulkanInstance::new`, P12.2 HWND
surface in HAL, P12.3 core+FFI wiring, P12.4 Win32 framework +
surface helper, P12.5 CMake/docs). CRITICAL/MAJOR must be fixed
before Phase 12 is COMPLETE; MINOR may be deferred with written
rationale.

## Review headline

**0 CRITICAL / 0 MAJOR / 4 MINOR.** All Phase-12 focus areas
verified by the reviewer (and re-checked against the gate):

Positive findings (verified, no action):
- **No-panic across the C ABI**: `find_windows_hwnd_source` mirrors
  `find_metal_layer_source`; null hwnd / null hinstance / both route
  through `create_surface_from_windows_hwnd` → `real_hal_surface` →
  error surface (real instance) or Noop surface (Noop), never panic.
  Covered by `surface_validation.rs::..._does_not_panic_on_noop` and
  the `ffi/mod.rs` HWND unit test. The `WGPUSurfaceSourceWindowsHWND`
  decode reads `(hinstance, hwnd)` in header field order — ABI correct.
- **`match` totality**: `HalInstance::create_surface_from_windows_hwnd`
  mirrors the metal twin incl. the `#[cfg(not(any(metal, vulkan)))]
  let _ = (hinstance, hwnd);` guard; total under every feature combo;
  Metal arm returns `SwapchainCreationFailed`, not `unreachable!`/panic.
- **macOS non-regression**: R85-1's capability-driven list enables the
  same extensions on MoltenVK (all three present → all enabled,
  portability flag still set); only the ordering changed, which is
  insignificant to Vulkan.
- **Block 90**: every new `pub fn` (`instance_extension_config`,
  HAL/core HWND fns) ships an inline test in the same commit; Noop arms
  are GPU-free; the real-Vulkan ones are correctly `#[ignore]`d.
- **Refcount**: `clone_handle(instance, ...)` is taken once and moved
  into `WGPUSurfaceImpl._instance` in both the error and success
  branches — no extra Arc clone, no early-return leak. Same as metal.
- **No GLFW on Windows**: `examples/CMakeLists.txt` `WIN32` branch sets
  `YAWGPU_GLFW_FOUND OFF` and never calls `find_package(glfw3)`;
  `framework/CMakeLists.txt` links only `user32 gdi32` on `WIN32`.

Independent verification by Claude (host: Windows 11 + VS 2022 +
Vulkan SDK 1.3.296.0): Noop `cargo test --workspace` + `cargo clippy
--workspace --all-targets -- -D warnings` clean; `cargo clippy -p
yawgpu-hal --features vulkan --all-targets -- -D warnings` clean;
`cmake -S examples -B examples/build -DYAWGPU_FEATURE=vulkan` + build →
all 8 examples (incl. windowed) compile `/W4`-clean; `enumerate_adapters`
with `YAWGPU_BACKEND=vulkan` reports a real GPU (backendType 6); the
three windowed examples run on-screen (user-confirmed).

## MINOR findings + triage

| # | File:line | Finding | Decision |
|---|---|---|---|
| 1 | `examples/framework/framework_windows.c:78-129` | Window-class atom leaks on a creation-failure path (`AdjustWindowRectEx`/`CreateWindowExW` fail after the class was registered; count stays 0 so it is never unregistered). Error-path only, no UB. | **FIX** → handoff P12-Review-Fix (Fix 1) |
| 2 | `examples/framework/framework_windows.c:131-147` | `free(window)` leaves the destroyed hwnd's `GWLP_USERDATA` dangling. Safe today (WM_DESTROY is dispatched synchronously by `DestroyWindow` before `free`), but a latent footgun. | **FIX** → handoff P12-Review-Fix (Fix 2) |
| 3 | `yawgpu-hal/src/vulkan/mod.rs` `instance_extension_config` | Enabled-extension list order changed vs. the old hard-coded array. | **DROP** — Vulkan extension order is insignificant and the portability flag is still set iff present; R85-1's "identical on MoltenVK" holds behaviorally. Not a defect. |
| 4 | `yawgpu-hal/src/vulkan/mod.rs` `has_instance_extension` | One-line helper could be inlined into its sole caller. | **DROP** — the named helper documents intent and is harmless; inlining is not an improvement. |

Both FIX items are C example code (not Block-90 Rust), so they carry
no inline Rust unit test; verification is a clean Windows build. They
are fixed together in one commit via handoff **P12-Review-Fix**.

## Gate

Phase 12 has **0 CRITICAL / 0 MAJOR**, and the two accepted MINOR
fixes have landed (user elected to fix rather than defer). Dropped
MINORs (#3, #4) recorded above with rationale. **Phase 12 COMPLETE.**

## Fix log

- P12-Review-Fix (`framework_windows.c` MINOR #1 + #2): commit
  `37b6f89`. Factored `yawgpu_unregister_window_class_if_unused()`
  called on both `yawgpu_window_create` failure paths + in destroy
  (no class-atom leak); cleared `GWLP_USERDATA` before `DestroyWindow`
  (no dangling USERDATA). Windows examples rebuilt `/W4`-clean
  (framework + the three windowed exes); Rust untouched, gate still
  green.
