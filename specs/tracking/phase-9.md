# Phase 9 — Examples + real surface/presentation

Status: **in progress** (P9.0 active). Rules/plan:
`../blocks/80-examples.md`. Roles/loop: `../reference/workflow.md`.

Post-core phase (user-requested): port Dawn + wgpu-native **C
samples** into `examples/`, and lift the SF3 "real presentation N/A"
boundary by adding a real window→surface→swapchain path on the
Phase-7 Metal/Vulkan backends. Form = C programs linked against
`libyawgpu.a` + vendored `webgpu.h`; **no hard cmake/glfw dep**
(plain `cc` driver; glfw gated to windowed slices). Permanent gate
unchanged: `cargo test --workspace` + `cargo clippy --workspace
--all-targets -- -D warnings` green on Noop (examples are NOT in
`cargo test`). Real-GPU example runs done by Claude on the Apple Silicon
(per `claude-runs-real-gpu-tests` memory), logged per slice.
**Phase ends with the mandatory Phase Review**
(`tracking/phase-9-review.md`).

## P9.0 — C example build scaffold + headless  *(active)*
`examples/` CMake tree (mirrors wgpu-native layout) + shared
framework; port
`enumerate_adapters`, `compute` (shader.wgsl + storage readback),
`device_info` (Dawn `DawnInfo` as C). Headless. Proves the C ↔
`libyawgpu.a` link + the `WGPUYawgpuInstanceBackendSelect` struct
from C. Noop: enumerate/info run, compute validates; real Metal/
Vulkan: Claude real-GPU-run.

## P9.1 — `capture` (offscreen render → image file)  *(after P9.0)*
wgpu-native `capture` port: offscreen render pipeline → texture →
T2B → PPM/PNG. Real Metal/Vulkan (reuses P7.5/P7.6e + P7.3/P7.6c).
No window.

## P9.2 — Real window→surface→swapchain (GLFW-gated)  *(after P9.1)*
Metal CAMetalLayer+drawable, Vulkan MoltenVK metal-surface+
VkSwapchainKHR; wire wgpuInstanceCreateSurface(SurfaceSourceMetalLayer
from GLFW NSWindow)/Configure/GetCurrentTexture(real backbuffer)/
Present. Noop surface unchanged (P8.6). Updates block-70 SF3 note
(real on Metal/Vulkan w/ window; still N/A on Noop).

## P9.3 — `triangle` (windowed)  *(after P9.2)*
wgpu-native `triangle` on the real surface (GLFW-gated).

## P9.4 — Dawn samples (HelloTriangle / ComputeBoids …)  *(after P9.3)*
C rewrites of the C-ABI-expressible Dawn samples; record C++-wrapper-
bound / windowed-infeasible ones as deferred/N-A. Then Phase 9
Review.

## Exit criteria

- Headless examples build+link + run (Noop where meaningful; real
  GPU real-GPU-run logged); windowed examples build (GLFW-gated) + real-
  backend run (M2 logged); SF3 real path implemented for Metal/
  Vulkan.
- Noop gate unchanged & green; per-slice `--features metal/vulkan`
  build clean. One commit per slice. Mandatory Phase 9 Review.
