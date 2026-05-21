# Block 85 — Windows windowed examples (HWND surface + Win32 windowing)

Phase 9 (`80-examples.md`) brought a real window→surface→swapchain
path to **macOS** only: GLFW + `CAMetalLayer` in `framework_macos.m`,
wired through `WGPUSurfaceSourceMetalLayer`. The CMake gate is
`glfw3_FOUND AND APPLE`, and the three windowed examples
(`surface_smoke`, `triangle`, `hello_triangle`) hard-code the
metal-layer surface source in their `main.c`.

Block 85 extends that path to **Windows**: every example —
including the three windowed ones — builds and runs on Windows,
rendering for real via the **Vulkan** backend, with **no GLFW
dependency** (native Win32 windowing). Metal stays macOS-only;
Windows has exactly one real backend (Vulkan).

This is **Phase 12**. Plan/tracking: `../tracking/phase-12.md`.
Roles/loop: `../reference/workflow.md`. Windows toolchain rules
(libclang, enum-bridge pattern) live in `95-windows-build.md` and
are unchanged here — Phase 11 already made the workspace compile on
Windows; Block 85 is about making the *windowed examples* run there.

## Scope decisions (authoritative)

- **Backend: Vulkan only on Windows.** Metal is macOS-only and is
  not touched. The Noop default still works on Windows (windowed
  examples open a window and acquire Noop surface textures, but
  Noop has no real swapchain — same P8.6 behavior as before).
- **Windowing: native Win32, no GLFW.** A new
  `examples/framework/framework_windows.c` implements the opaque
  `YawgpuWindow` with raw Win32 (`RegisterClassExW` /
  `CreateWindowExW` / a `WM_*` message pump). It links only
  `user32`/`gdi32` (already available with the MSVC toolchain).
  macOS keeps GLFW + `framework_macos.m` unchanged. Do **not** add
  a GLFW dependency on Windows; do **not** vendor any windowing
  library.
- **Surface selection is framework-internal.** A new framework
  helper `yawgpu_window_create_surface(instance, window, label)`
  builds the correct surface-source chain per platform
  (`WGPUSurfaceSourceMetalLayer` on macOS,
  `WGPUSurfaceSourceWindowsHWND` on Windows) and calls
  `wgpuInstanceCreateSurface`. The three windowed examples'
  `main.c` call this helper and contain **no** platform `#ifdef`
  and **no** surface-source struct of their own.
- **HWND→`VkSurfaceKHR` is a real HAL path.** Today
  `WGPUSurfaceSourceWindowsHWND` is recognized by FFI validation
  (`is_supported_surface_source`) but produces a **Noop** surface
  (`hal: None`) because no HWND creation path exists. Block 85
  adds the real path through HAL → core → FFI, mirroring the
  existing metal-layer path exactly.
- **Cross-platform Vulkan instance.** `VulkanInstance::new` must
  stop hard-coding macOS-only instance extensions
  (`KHR_PORTABILITY_ENUMERATION`, `EXT_METAL_SURFACE`) so that it
  succeeds on Windows (where it must enable `KHR_WIN32_SURFACE`
  instead). The selection is **capability-driven** (enable an
  extension only if the loader reports it present), not
  `cfg!(target_os)`-driven, so a single code path serves all hosts
  and a missing optional extension never aborts instance creation.

## API contract (the new/changed public surface)

All new `pub fn`s ship with an inline `#[cfg(test)] mod tests`
unit test in the **same commit** (Block 90, principle 1). The
Noop arm of each must be exercisable with no GPU (principle 2).

### R85-1 — `VulkanInstance::new` capability-driven extensions

`yawgpu-hal/src/vulkan/mod.rs`. `new()` queries
`entry.enumerate_instance_extension_properties(None)` and builds
the enabled-extension list from what is actually present:

- Always require `VK_KHR_surface` (`vk::KHR_SURFACE_NAME`). If
  absent, return `HalError::BackendUnavailable { backend }`.
- Add `VK_EXT_metal_surface` (`vk::EXT_METAL_SURFACE_NAME`) iff
  present (macOS / MoltenVK).
- Add `VK_KHR_win32_surface` (`vk::KHR_WIN32_SURFACE_NAME`) iff
  present (Windows).
- Add `VK_KHR_portability_enumeration` iff present, and set the
  `ENUMERATE_PORTABILITY_KHR` create flag **only** when it was
  added (MoltenVK needs it; a conformant desktop ICD does not have
  it and must not receive the flag).

The behavior on macOS must be identical to today (same extensions
get enabled because they are present). The signature is unchanged
(`pub fn new() -> Result<Self, HalError>`); its existing unit test
is updated to assert success on the host where a Vulkan loader is
available, and the no-loader path still maps to
`BackendUnavailable`.

### R85-2 — HWND surface creation in the Vulkan HAL

`yawgpu-hal/src/vulkan/mod.rs`. New:

```rust
/// # Safety
/// `hwnd` must be a valid Win32 window handle and `hinstance` its
/// owning module handle (or null, per VK_KHR_win32_surface rules).
pub unsafe fn create_surface_from_windows_hwnd(
    &self,
    hinstance: *mut c_void,
    hwnd: *mut c_void,
) -> Result<VulkanSurface, HalError>
```

- Rejects a null `hwnd` with
  `HalError::SwapchainCreationFailed { backend, message: "surface hwnd is null" }`
  (mirrors the null-layer rejection in
  `create_surface_from_metal_layer`).
- Uses `ash::khr::win32_surface::Instance` +
  `vk::Win32SurfaceCreateInfoKHR::default().hinstance(..).hwnd(..)`
  and `create_win32_surface`, wrapping the result in the same
  `VulkanSurface { instance, surface, swapchain: None, config:
  None, current_image_index: None }` shape as the metal path.
- **Compiles on every host.** The `ash::khr::win32_surface` loader
  type exists in `ash` regardless of target OS, so this function is
  *not* `cfg`-gated. It only succeeds at runtime when the instance
  enabled `VK_KHR_win32_surface` (R85-1) on a Windows host.

### R85-3 — `HalInstance::create_surface_from_windows_hwnd`

`yawgpu-hal/src/lib.rs`. Mirrors
`create_surface_from_metal_layer` (lib.rs:82):

```rust
/// # Safety
/// `hwnd` must be a valid Win32 window handle; `hinstance` its
/// owning module handle or null.
pub unsafe fn create_surface_from_windows_hwnd(
    &self,
    hinstance: *mut c_void,
    hwnd: *mut c_void,
) -> Result<HalSurface, HalError>
```

- `Noop` arm → `Ok(HalSurface::Noop)` (ignores the pointers; the
  no-GPU unit test target).
- `Vulkan` arm → delegates to R85-2, maps to `HalSurface::Vulkan`.
- `Metal` arm → returns
  `HalError::SwapchainCreationFailed { backend: "metal", message:
  "HWND surface is not supported on Metal" }` (an HWND can never be
  a `CAMetalLayer`; surfacing this as an error rather than a panic
  honors the no-panic rule). The arm exists so the `match` is total
  under `--features metal`.

### R85-4 — `core::Instance::create_surface_from_windows_hwnd`

`yawgpu-core/src/instance.rs`. Mirrors the metal-layer wrapper
(instance.rs:67), forwarding to the HAL and returning the core
`Surface` type. Same safety contract. Unit test: a Noop instance
returns a Noop-backed surface (no GPU).

### R85-5 — FFI: wire HWND into `wgpuInstanceCreateSurface`

`yawgpu/src/ffi/`. Two changes, mirroring `find_metal_layer_source`
(`ffi/mod.rs:832`) and its use in `wgpuInstanceCreateSurface`
(`ffi/instance.rs:99-107`):

- Add `find_windows_hwnd_source(chain) -> Option<(*mut c_void,
  *mut c_void)>` returning `(hinstance, hwnd)` from a
  `WGPUSurfaceSourceWindowsHWND` link
  (`native::WGPUSType_SurfaceSourceWindowsHWND`), or `None`.
- In `wgpuInstanceCreateSurface`, after the metal-layer attempt,
  try the HWND source: if a metal layer was not found, look for an
  HWND source and call
  `instance.core.create_surface_from_windows_hwnd(hinstance, hwnd)`.
  The `real_surface_creation_failed` / `is_error` accounting must
  treat a found-but-failed HWND source the same way it treats a
  found-but-failed metal layer (error surface only when a real HAL
  instance was asked to wrap a real handle and failed).
- `is_supported_surface_source` already lists
  `WGPUSurfaceSourceWindowsHWND`; no validation change is needed.

The Noop path is unchanged: with no metal layer and no HWND (or on
a Noop instance), the surface is created with `hal: None`, exactly
as today.

### R85-6 — Framework windowing + surface helper (examples)

`examples/framework/`. Not Rust `pub fn`s (C, not covered by Block
90), but the behavior contract:

- **`framework.h`**: add
  `WGPUSurface yawgpu_window_create_surface(WGPUInstance instance,
  YawgpuWindow *window, const char *label);`. Keep the existing
  `yawgpu_window_*` declarations. The `yawgpu_window_metal_layer`
  accessor may stay (used internally by the macOS helper) or be
  made internal — implementer's choice, but the windowed examples
  must no longer call it directly.
- **`framework_windows.c`** (new): implements `YawgpuWindow` for
  Win32 — `struct YawgpuWindow { HWND hwnd; HINSTANCE hinstance;
  /* close flag updated by WM_CLOSE/WM_DESTROY */ }`. Implements
  `yawgpu_window_create` (registers a class once, creates a
  top-level window sized to the requested client area via
  `AdjustWindowRectEx`), `yawgpu_window_destroy`,
  `yawgpu_window_should_close`, `yawgpu_window_poll_events`
  (`PeekMessageW`/`Translate`/`Dispatch` loop, non-blocking),
  `yawgpu_window_framebuffer_size` (`GetClientRect`). UTF-16 Win32
  `*W` APIs (the title is ASCII-only in examples; a minimal
  widen is fine).
- **Surface helper**: `framework.c` (or per-platform file) provides
  `yawgpu_window_create_surface`. On macOS it builds the existing
  `WGPUSurfaceSourceMetalLayer` chain from the window's
  `CAMetalLayer`; on Windows it builds a
  `WGPUSurfaceSourceWindowsHWND` chain with the window's
  `hwnd`/`hinstance`. Returns the `WGPUSurface` or `NULL` on
  failure (logging to stderr, like the existing framework helpers).
- The three windowed `main.c` (`triangle`, `hello_triangle`,
  `surface_smoke`) replace their inline
  `WGPUSurfaceSourceMetalLayer` block with a single
  `yawgpu_window_create_surface(...)` call. No other behavioral
  change (same configure / acquire / draw / present loop, same ~60
  frame cap).

### R85-7 — CMake builds windowed examples on Windows

`examples/CMakeLists.txt` + `examples/framework/CMakeLists.txt`:

- The windowed-examples gate becomes: **macOS** → as today
  (`glfw3_FOUND AND APPLE`); **Windows** → always on (no GLFW
  probe), windowed examples added unconditionally. Express this as
  a single `YAWGPU_WINDOWED_EXAMPLES` boolean set per platform so
  the `add_subdirectory(surface_smoke|triangle|hello_triangle)`
  block stays one `if(YAWGPU_WINDOWED_EXAMPLES)`.
- `framework` library sources: `framework_macos.m` on Apple (as
  today), `framework_windows.c` on `WIN32`.
- Windows windowed targets link `user32` and `gdi32` (and the
  existing `yawgpu_c_api`). No GLFW link on Windows.
- The Cargo feature for the example build on Windows is `vulkan`
  (`-DYAWGPU_FEATURE=vulkan`), reusing the existing
  `target-vulkan` directory + dll-path logic (`WIN32` branch at
  CMakeLists.txt:57 already names `yawgpu.dll`). The built
  `yawgpu.dll` must be discoverable at runtime — document the
  copy/`PATH` step in the README (R85-8); CMake may optionally add
  a post-build `copy_if_different` of the dll next to each example
  exe (implementer's choice, but if added it must not break the
  macOS/Linux builds).

### R85-8 — Docs

`examples/README.md`: add a Windows section — prerequisites
(Vulkan SDK / a Vulkan-capable GPU driver providing an ICD; MSVC
toolchain; LLVM/`LIBCLANG_PATH` per Block 95), the
`-DYAWGPU_FEATURE=vulkan` configure line, `YAWGPU_BACKEND=vulkan`
at runtime, and the `yawgpu.dll`-on-`PATH` note. State that windowed
examples on Windows use native Win32 (no GLFW needed) while
macOS/Linux windowed examples still require GLFW.

## Verification

- **Noop gate unchanged & green** on all hosts: `cargo test
  --workspace` + `cargo clippy --workspace --all-targets -- -D
  warnings`. New unit tests (R85-1…R85-5) pass on Noop with no GPU.
- **Per-feature build clean**: `cargo build -p yawgpu --features
  vulkan` on Windows and macOS; `--features metal` on macOS.
- **macOS unchanged**: the three windowed examples still build via
  GLFW + metal-layer and still render (real-GPU run by Claude on the
  host, logged).
- **Windows build**: `cmake -S examples -B examples/build
  -DYAWGPU_FEATURE=vulkan && cmake --build examples/build` produces
  every example exe, including the three windowed ones.
- **Windows run** (by the user on the host, window presents):
  `YAWGPU_BACKEND=vulkan` → each windowed example opens a window and
  renders ~60 frames then exits 0; `YAWGPU_BACKEND=noop` → window
  opens, acquires Noop textures, exits cleanly. Logged in
  `tracking/phase-12.md`.

## Out of scope

- GL / D3D backends; Metal on non-Apple; Vulkan on macOS beyond the
  existing MoltenVK path.
- Xlib / Wayland / XCB / Android surface sources (the enum values
  stay validation-recognized but inert, as today). Linux windowed
  examples remain GLFW-gated and are not part of this phase.
- CI: a Windows *example-build* job is a nice-to-have follow-up, not
  required here (the Phase-11 Windows CI matrix already guards the
  Rust workspace build).
- Changing the Noop surface contract (no real swapchain on Noop).
