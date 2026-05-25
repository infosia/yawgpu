# Phase 9 — Phase Review (mandatory)

Status: in progress. Per `../reference/workflow.md` ("Phase
Review"): a fresh no-context subagent reviewed the cumulative
Phase 9 diff (`5c8dfa4..HEAD` — from after the Phase 8 review
commit through the P9.4 hello_triangle commit, covering P9.0 –
P9.4) and emitted findings; CRITICAL/MAJOR must be fixed before
Phase 9 is COMPLETE.

## Review headline

**Phase 9 review: 0 critical, 4 major, 7 minor.**

Positive findings (no action): Noop SF3 boundary
(`surface_validation` 4/4) preserved byte-for-byte; all Vulkan
resource-creation paths have correct cleanup-on-error chains;
`VulkanTextureInner`'s `owns_image: bool` + `memory: Option<...>`
correctly distinguishes owned vs. swapchain images;
`objc2`-family migration logged in `../reference/
dependencies.md`; the `CARGO_TARGET_DIR=target-${YAWGPU_FEATURE}`
cmake change has an in-source comment explaining the dylib-
collision it prevents; `.gitignore` covers `/target-*` +
`/examples/build*`; surface texture Arc lifetimes (releasing a
surface texture drops the cached `Arc<VulkanTextureInner>` back
to swapchain's `images` Vec without destroying the underlying
image) are correct.

## MAJOR findings — must fix

### K1 — Pointer-consuming `pub fn`s must be `unsafe`
Public safe Rust functions take a `*mut c_void` and pass it to
`Retained::retain` / `vkCreateMetalSurfaceEXT` / Objective-C
runtime, all of which require a valid `CAMetalLayer` instance
pointer. A safe Rust caller can trigger UB by passing arbitrary
addresses. The FFI entry is already `unsafe extern "C"`, but the
Rust API leaks the unsafety into safe code.
- `yawgpu-hal/src/metal/mod.rs` (`MetalSurface::from_layer`)
- `yawgpu-hal/src/vulkan/mod.rs`
  (`VulkanInstance::create_surface_from_metal_layer`)
- `yawgpu-hal/src/lib.rs`
  (`HalInstance::create_surface_from_metal_layer`)
- `yawgpu-core/src/lib.rs`
  (`Instance::create_surface_from_metal_layer`)

**Fix:** mark all four `pub unsafe fn ...` with a `# Safety`
section requiring the pointer be a valid (non-dangling)
`CAMetalLayer` instance. Update the single caller
(`wgpuInstanceCreateSurface`) to invoke them inside the existing
`unsafe` block.

### K2 — `wgpuInstanceCreateSurface` swallows real-backend surface-creation errors
The current FFI computes:
```rust
let hal = layer
    .and_then(|layer| instance.core.create_surface_from_metal_layer(layer).ok())
    .and_then(real_hal_surface);
```
`.ok()` silently swallows `HalError::SwapchainCreationFailed`
(null layer, `vkCreateMetalSurfaceEXT` failure, etc.). The
`is_error` flag only checks the chained-struct sType, so on a
real backend a malformed CAMetalLayer pointer yields a "valid"
surface that always returns `Lost` from `GetCurrentTexture` —
no diagnostic.
- `yawgpu/src/lib.rs` `wgpuInstanceCreateSurface`

**Fix:** when the chain provides a layer AND the instance is a
real backend (Metal or Vulkan) AND the HAL surface could not be
created, set `is_error = true` so the surface fails fast and
subsequent operations report Error.

### K3 — Swapchain image view can outlive the swapchain (UAF on disordered Release)
`VulkanTexture` clones — held by `WGPUTextureImpl` via
`core::Texture::from_hal` — keep `Arc<VulkanTextureInner>` alive
even after the surface's swapchain is dropped. If the user
releases the surface before releasing a still-held surface
texture, `VulkanSwapchain::drop` calls `destroy_swapchain` while
a `VulkanTextureInner` with a live `vk::ImageView` still
references one of that swapchain's now-destroyed images. The
user's subsequent `wgpuTextureRelease` then calls
`destroy_image_view` on a view of a destroyed image — Vulkan
validation error / UB. The bundled examples release the surface
texture each frame before the surface, so they don't trigger
it; the C ABI permits the inverse order, however.
- `yawgpu-hal/src/vulkan/mod.rs` `VulkanSurface` /
  `VulkanSwapchain` / `VulkanTextureInner`

**Fix:** make `VulkanSwapchain` ref-counted via
`Arc<VulkanSwapchainInner>` (destroy happens in
`Drop for VulkanSwapchainInner`), and have each swapchain-
image `VulkanTextureInner` carry a strong `Arc<...Inner>` into
the swapchain so the swapchain handle survives until the last
image view drops. `VulkanSurface::unconfigure` /
`VulkanSurface::drop` then merely drop their reference; actual
`vkDestroySwapchainKHR` waits for the last texture.

### K4 — `HalSurfaceConfiguration` / `HalPresentMode` missing `#[non_exhaustive]`
CLAUDE.md "code conventions" requires `#[non_exhaustive]` on
extensible public enums/structs. Both new types are extensible
(more present modes, more config fields — alpha_mode,
view_formats, etc.) and per CLAUDE.md must be tagged. Tagged
MAJOR because exhaustive matches on these become a breaking
change to remove later — exactly what `#[non_exhaustive]`
prevents.
- `yawgpu-hal/src/lib.rs`

**Fix:** add `#[non_exhaustive]` to both `HalSurfaceConfiguration`
struct and `HalPresentMode` enum.

## MINOR findings — deferred with rationale

### m1 — `framework_macos.m` is single-window only (`glfwTerminate` per destroy)
- `examples/framework/framework_macos.m` —
  `yawgpu_window_destroy` calls `glfwTerminate()`; safe for the
  current three single-window examples; multi-window would need
  refcounting or split init/shutdown.
  **Closed 2026-05-25** — added a `yawgpu_window_count`
  refcount mirroring the `framework_windows.c` pattern;
  `glfwInit()` runs once before the first window and
  `glfwTerminate()` only after the last window is destroyed.
  Single-window examples are byte-for-byte unchanged.

### m2 — `shader.wgsl` is CWD-relative, not binary-dir relative
- `examples/framework/framework.c` `yawgpu_load_wgsl_shader`
  uses `fopen(path, "rb")`. CMake stages `shader.wgsl` next to
  the binary but `fopen` resolves relative to CWD. Running
  from the repo root yields a "failed to load shader.wgsl"
  error.
  **Closed 2026-05-25** — added `yawgpu_set_argv0(argv0)` to the
  framework; when called, `yawgpu_load_wgsl_shader` resolves
  relative paths against the binary's directory derived from
  `argv[0]`. `triangle` / `hello_triangle` now forward
  `argv[0]` from `main`; absolute paths and the legacy
  cwd-relative behaviour (when `yawgpu_set_argv0` is not
  called) remain supported byte-for-byte.

### m3 — `MetalSurface::present` uses `drawable.present()`
- `yawgpu-hal/src/metal/mod.rs` — Apple's canonical pattern is
  `[command_buffer presentDrawable:drawable]`; the chosen
  `[drawable present]` relies on Metal's automatic tracking
  (the drawable knows which CBs wrote its texture and waits for
  them) + the fact that `submit_copies` already does
  `waitUntilCompleted()` before present. Correct but unusual;
  **deferred** — a doc-comment explaining the choice is
  sufficient and the canonical pattern can land with the
  semaphore-driven Vulkan refactor (m4).

### m4 — Vulkan present path has four CPU stalls per frame
- `yawgpu-hal/src/vulkan/mod.rs` `VulkanSurface::present` /
  `transition_swapchain_image_to_present` — fence wait,
  submit-copies queue_wait_idle, transition submit +
  queue_wait_idle, vkQueuePresentKHR + queue_wait_idle. Smoke-
  test-acceptable; a semaphore-driven path is a sizable
  refactor. **Deferred** — tracked implicitly by the
  ComputeBoids deferred slice which would need it for real
  perf.

### m5 — `unsafe impl Send/Sync for MetalSurface` is loose for `CAMetalLayer` thread-confinement
- `yawgpu-hal/src/metal/mod.rs` — `CAMetalLayer` mutating
  accessors are main-thread-only; current examples all run on
  main thread (GLFW requirement).
  **Closed 2026-05-25** — added a SAFETY block above
  `unsafe impl Send / Sync for MetalSurface` in
  `yawgpu-hal/src/metal/surface.rs` documenting the main-thread
  invariant for the mutating accessors driven by
  `MetalSurface::configure`. The loose `Send` / `Sync` matches
  the rest of the Metal HAL; tightening (e.g. wrapping mutating
  ops in a main-thread runner) is left to the future HAL
  Send/Sync audit, which can pick this comment up as the
  starting reference.

### m6 — `surface_smoke/main.c` is less defensive than `triangle`/`hello_triangle`
- `examples/surface_smoke/main.c` — no `if (!commands)` check
  after `wgpuCommandEncoderFinish`.
  **Closed 2026-05-25** — verified the
  `surface_smoke/main.c:197-203` `if (!commands)` branch is
  already present and matches the `triangle` / `hello_triangle`
  pattern (released elsewhere; spec entry was stale). No code
  change.

### m7 — Example failure paths leak handles before exit
- `examples/compute/main.c`, `examples/device_info/main.c`
  return EXIT_FAILURE on map/finish failure without going
  through the cleanup block.
  **Closed 2026-05-25** — `compute/main.c` rewritten to use
  `goto cleanup;` with an `exit_status` and `mapped` flag so
  both failure sites (`map_state` not Success, `GetConstMappedRange`
  returning null) tear down every allocated handle in the
  same order as the success path. `device_info/main.c` was
  already correct — its single failure path at lines 121-127
  already calls `yawgpu_context_release` before
  `EXIT_FAILURE` (spec entry was stale).

## Fix log (2026-05-20)

All four MAJOR findings fixed in a single follow-up commit:

- **K1 fixed**: `pub fn` → `pub unsafe fn` + `# Safety` doc on
  `HalInstance::create_surface_from_metal_layer`,
  `MetalSurface::from_layer`,
  `VulkanInstance::create_surface_from_metal_layer`,
  `Instance::create_surface_from_metal_layer`. The single call
  site in `wgpuInstanceCreateSurface` wraps the new unsafe call
  in an explicit `unsafe { ... }` block. No clippy warnings.
- **K2 fixed**: added `is_real_hal_instance(&HalInstance)`
  helper + `Instance::hal()` accessor; `wgpuInstanceCreateSurface`
  now computes `is_error = surface_source_is_unsupported ||
  (layer.is_some() && is_real_hal_instance(...) && hal.is_none()
  )`. Noop path is unaffected — `is_real_hal_instance(Noop) ==
  false`, so descriptor-only Noop surfaces still produce
  `is_error == false`. `surface_validation` 4/4 remains green.
- **K3 fixed (UAF on view-of-destroyed-image path)**:
  `VulkanSwapchain` → `Arc<VulkanSwapchainInner>` with
  `Drop for VulkanSwapchainInner` doing
  `images.clear(); destroy_swapchain`. `VulkanSurface.swapchain`
  is now `Option<Arc<VulkanSwapchainInner>>`. Each
  `VulkanTexture` gains a `swapchain:
  Option<Arc<VulkanSwapchainInner>>` field — the cached images
  inside the swapchain inner have `swapchain: None` (no cycle),
  but each returned `VulkanTexture` clone from
  `acquire_next_texture` has `swapchain: Some(Arc::clone(
  swapchain))` so the user's outstanding texture clone keeps
  the swapchain alive until released. `VulkanTexture` field
  drop order (`inner` then `swapchain`) destroys the image view
  before the swapchain handle on the disordered-release path.
- **K4 fixed**: `#[non_exhaustive]` on both
  `HalSurfaceConfiguration` (struct) and `HalPresentMode`
  (enum). A small `HalSurfaceConfiguration::new(...)`
  constructor was added so external construction is still
  possible after the attribute; FFI updated to use it.

## Residual follow-up (deferred — does not block Phase 9 COMPLETE)

- **K3-r** — `vkDestroySurfaceKHR` is called from
  `Drop for VulkanSurface` immediately after dropping
  `self.swapchain = None`. On the disordered-release path
  (user releases the surface while still holding a surface
  texture), the swapchain Arc refcount drops to N (the user's
  clones) and the swapchain VkSwapchainKHR remains live; the
  subsequent `destroy_surface` then violates Vulkan spec VUID
  `VkDestroySurfaceKHR-surface-01266` ("All `VkSwapchainKHR`s
  created from `surface` must have been destroyed prior to
  destroying `surface`"). This is a validation issue, not a
  memory-safety bug on most loaders, and is **distinct from
  the original MAJOR K3 finding** (which was about
  `destroy_image_view` of a view referencing a destroyed
  image — that path is now safe). The complete fix is an
  additional Arc-share of `vk::SurfaceKHR` into
  `VulkanSwapchainInner` (or moving the surface handle's
  ownership down into the swapchain inner so destruction
  cascades), tracked as a Phase-9-follow-up alongside the
  ComputeBoids deferred slice (which would benefit from the
  same semaphore-driven path m4). The bundled examples
  release each surface texture before
  `wgpuSurfaceUnconfigure` / `wgpuSurfaceRelease` and do not
  trigger this path; real-GPU-verified runs of all three windowed
  examples (`surface_smoke`, `triangle`, `hello_triangle`)
  on both Metal and Vulkan exit 0 after these fixes.

## Verification (real-GPU, 2026-05-20)

- Noop `cargo test --workspace` **58/58 binaries** + clippy
  clean. `surface_validation` 4/4 byte-for-byte unchanged.
- `cargo clippy -p yawgpu --features metal/vulkan --all-
  targets -- -D warnings` clean.
- Phase-7 e2e regression: Metal `e2e_metal_*` **17/17** +
  Vulkan `e2e_vulkan_*` **15/15** (basic 3/buffer 3/texture
  4/compute 3/render 3+2/smoke 1; no regression from the
  K3 swapchain Arc refactor).
- real-GPU windowed examples: `surface_smoke`, `triangle`,
  `hello_triangle` all open the window, render, and exit 0
  on both `YAWGPU_BACKEND=metal` and `YAWGPU_BACKEND=vulkan`
  (Vulkan with `$VULKAN_SDK` sourced).

## Status

**Phase 9 COMPLETE** — 0 critical / 0 major / 7 minor (all
deferred with rationale above) + 1 minor residual K3-r
follow-up. Block 80 inventory satisfied: enumerate_adapters,
compute, device_info (headless P9.0), capture (offscreen
T2B+PNG P9.1), surface_smoke + triangle + hello_triangle
(windowed P9.2–P9.4). Deferred Dawn samples logged:
ComputeBoids (post-Phase-9 follow-up), Animometer +
ManualSurfaceTest (already deferred in block-80).
