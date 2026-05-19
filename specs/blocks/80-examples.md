# Block 80 — Examples + real surface/presentation (Phase 9)

Phase 9 ports the **C samples** of Dawn (`dawn/src/dawn/
samples/`) and wgpu-native (`wgpu-native/examples/`) into
yawgpu, exercising the webgpu.h C ABI yawgpu exposes. It also **lifts
the SF3 "real presentation N/A on Noop" boundary** by adding a real
window + surface/swapchain path on the Phase-7 Metal/Vulkan backends
(user-approved scope: windowed samples included; C-program form).

## Scope decisions (authoritative)

- **Form: C programs** under `examples/`, linked against yawgpu's
  `staticlib` (`libyawgpu.a`) + the vendored
  `yawgpu/ffi/webgpu-headers/webgpu.h`. wgpu-native's `examples/`
  C sources are the port template (small diffs: yawgpu uses the same
  Dawn `webgpu.h`; the only yawgpu-specific bit is the
  `WGPUYawgpuInstanceBackendSelect` chained struct to pick
  Metal/Vulkan, Noop default).
- **No hard cmake/glfw dependency.** `cmake`/`glfw` are not installed
  on the dev machine. Build via a plain driver (a `Makefile` or
  `examples/build.sh` using `cc`), not CMake. **Headless** examples
  need only `cc` + `libyawgpu.a`. **Windowed** examples need GLFW —
  gate them: build/run only when `glfw3` is found (`pkg-config` /
  `brew --prefix glfw`); skip with a clear message otherwise. Do not
  vendor GLFW.
- **Gating / verification** (mirrors Phase 7): the Noop CI gate
  (`cargo test --workspace` + clippy) is unchanged and must stay
  green — examples are **not** part of `cargo test`. Headless
  examples must **build+link** (proof) and run on Noop where
  meaningful (enumerate/info; compute validates). Real-GPU runs
  (Metal, Vulkan/MoltenVK on the Apple Silicon) are executed **by Claude
  directly** (per `[[claude-runs-real-gpu-tests]]`) and logged in
  `tracking/phase-9.md` per slice. Windowed samples are run by
  Claude on the host (windows can open in this environment) or, if a
  window cannot be presented headlessly, marked "build-verified +
  manual" and logged.
- **Real surface/presentation** (lifts SF3): implement a real
  window→surface→swapchain path on Metal (CAMetalLayer + drawable)
  and Vulkan (MoltenVK `VK_EXT_metal_surface` + `VkSwapchainKHR`),
  wired through `wgpuInstanceCreateSurface`
  (`WGPUSurfaceSourceMetalLayer` from the GLFW NSWindow's layer),
  `wgpuSurfaceConfigure`, `wgpuSurfaceGetCurrentTexture` (real
  backbuffer image as a yawgpu `Texture`), `wgpuSurfacePresent`.
  Noop surface stays the P8.6 descriptor/arg-validation behavior
  (no real swapchain) — only the real backends gain presentation.
  Update the block-70 SF3 ✗ N/A note to "real on Metal/Vulkan with a
  window (P9.2); still N/A on Noop".
- Out of scope (unchanged): GL/D3D, Dawn `wire/`, Dawn samples that
  *require* the Dawn C++ webgpu_cpp wrapper or Dawn-internal
  `SampleUtils` (port the C-expressible subset; rewrite minimal C
  using the C ABI, or record which are C++-wrapper-bound → skip).

## Sample inventory → portability

wgpu-native (`Rust/wgpu-native/examples/`, C + webgpu.h — closest):
- `enumerate_adapters` — headless, trivial. ✅
- `compute` — headless storage-buffer dispatch + readback
  (shader.wgsl). ✅ (Noop validates; real Metal/Vulkan executes)
- `capture` — headless offscreen render → texture → buffer readback
  → PPM/PNG file. ✅ (real backends; mirrors the e2e render+T2B)
- `triangle` — **windowed** (GLFW surface + present). ✅ via P9.2.
- `texture_arrays` / `immediates` / `metal_interop` — feature-/
  platform-specific; port only if the feature exists in yawgpu
  (record N/A otherwise).

Dawn (`C/dawn/src/dawn/samples/`, C++ + SampleUtils/GLFW):
- `DawnInfo` — adapter/device/limits/features dump. ✅ (rewrite as C
  `device_info`)
- `HelloTriangle` — windowed. ✅ via P9.2 (C rewrite).
- `ComputeBoids` — compute + windowed render. ◐ (compute part ✅;
  windowed via P9.2 if feasible).
- `Animometer` / `ManualSurfaceTest` — windowed stress/manual; port
  if cheap after P9.2, else record deferred.

## Slices

- **P9.0** C example build scaffold (no cmake/glfw) + **headless**:
  `enumerate_adapters`, `compute`, `device_info` (DawnInfo). Proves
  the `cc` + `libyawgpu.a` + `webgpu.h` link and the backend-select
  struct from C. De-risk.
- **P9.1** `capture` — offscreen render→readback→image file (real
  Metal/Vulkan; no window).
- **P9.2** Real window→surface→swapchain (GLFW-gated): Metal
  CAMetalLayer + Vulkan VkSwapchainKHR; wire CreateSurface/Configure/
  GetCurrentTexture/Present; lifts SF3 for real backends.
- **P9.3** `triangle` (wgpu-native windowed) on the real surface.
- **P9.4** Dawn `HelloTriangle` (C rewrite) + `ComputeBoids`
  (compute±windowed) as feasible; record any C++-wrapper-bound or
  windowed-infeasible as deferred/N-A.
- **Phase 9 Review** (mandatory) → COMPLETE.

## Exit criteria

- Headless examples build+link and run (Noop where meaningful; real
  Metal/Vulkan real-GPU-run, logged); windowed examples build (GLFW-gated)
  and run on the real backends (real-GPU, logged) — SF3 real-presentation
  path implemented for Metal/Vulkan.
- Noop `cargo test --workspace` + clippy gate **unchanged & green**
  (examples excluded from `cargo test`); per-slice `--features
  metal`/`vulkan` build clean.
- One commit per slice (`phase-9: <slice> — <short>`); divergences/
  N-A recorded; mandatory Phase 9 Review logged in
  `tracking/phase-9-review.md`.
