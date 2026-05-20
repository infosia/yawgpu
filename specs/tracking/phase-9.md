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

## P9.0 — C example build scaffold + headless  *(☑ DONE — partial; one tracked known issue)*

Done: `examples/` CMake tree (mirrors wgpu-native layout) + shared
`framework/` (request-adapter/device, error printer, vendor
`WGPUYawgpuInstanceBackendSelect` declaration + env-driven
selection); ported `enumerate_adapters` (wgpu-native, adapter info
print), `compute` (storage-buffer Collatz + shader.wgsl + map-read),
`device_info` (Dawn `DawnInfo` rewritten in C: adapter+limits+
features). `.gitignore` excludes `examples/build/`; `examples/
README.md` documents `brew install cmake glfw` + the
`YAWGPU_BACKEND` env. Header pulled from
`yawgpu/ffi/webgpu-headers`, libyawgpu built via a CMake
`cargo build` custom target + rpath. No `cargo test` workspace
member added; Noop gate untouched.

**Side-effect cleanup (also in P9.0):** the Metal HAL was migrated
from the deprecated `metal 0.33.0` crate to `objc2 0.6` +
`objc2-foundation 0.3` + `objc2-metal 0.3` (matching `mgpu`).
This resolves the Phase-7 Review MINOR about the deprecated `objc`
ecosystem. Phase-7 Metal HAL public API unchanged; the full
`e2e_metal_*` suite on the host passes after migration
(basic/buffer/texture/compute/render/smoke). Also added a write-
through on `Buffer::unmap` for Write/mappedAtCreation mappings (host
→ `HalBuffer::write`) — symmetric with P7.2's map-read direction.

**Verification (real-GPU, 2026-05-20):**
- Noop `cargo test --workspace` 58 binaries green + clippy clean.
- `cargo build/clippy -p yawgpu --features metal` + `--features
  vulkan` clean.
- Phase-7 e2e regression: **Metal 6/6, Vulkan 5/5 binaries** all
  green.
- Examples runs:
  - Noop default: `enumerate_adapters` ✅, `device_info` ✅,
    `compute` ✅ (validates, prints zeros + Noop note).
  - Metal: `enumerate_adapters` ✅ (`Apple Silicon`, backendType 5),
    `device_info` ✅, `compute` ❌ → see tracked issue below.
  - Vulkan (MoltenVK; `$VULKAN_SDK` env required at runtime):
    `enumerate_adapters` ✅ (`Apple Silicon`, backendType 6),
    `device_info` ✅, `compute` ✅ → real Collatz `[0,1,7,2]`.

### P9.0 follow-up — RESOLVED (2026-05-20)
Diagnosed and fixed by adopting `mgpu`'s direct-mapping model:
- HAL: added `HalBuffer::mapped_ptr() -> Option<NonNull<u8>>`
  (Metal `MTLBuffer.contents()` for Shared storage, Vulkan
  persistent `vkMapMemory` HOST_VISIBLE|COHERENT, Noop `None`).
- core `Buffer::mapped_range` prefers `hal.mapped_ptr` over the
  intermediate `HostBuffer`; writes via `GetMappedRange` land
  **directly** in the real backend buffer.
- `Buffer::unmap` write-through and `resolve_pending_map`
  Read-copy are **skipped** when `mapped_ptr.is_some()` (real
  backends now read/write the GPU-coherent memory directly; Noop
  unchanged via the `HostBuffer` fallback).
- The earlier `examples/framework` `queueWriteBuffer` workaround
  was reverted: `yawgpu_create_buffer_init` is back to the
  canonical `mappedAtCreation` + memcpy + Unmap path that mgpu
  uses.

**Final P9.0 verification (real-GPU, 2026-05-20):**
- Noop `cargo test --workspace` **58 binaries green** + clippy
  clean; `buffer_map`/`buffer_mapped_range` 9/8 unregressed.
- `--features metal/vulkan` build/clippy clean.
- Phase-7 e2e: **Metal 6/6 binaries + Vulkan 5/5 binaries** all
  green (no regression).
- Examples on Metal **and** Vulkan: `enumerate_adapters` ✅
  (`Apple Silicon`, backendType 5/6), `device_info` ✅, **`compute`
  → real Collatz `[0, 1, 7, 2]`** on both real backends, and Noop
  validates.

### New known issue (separate; tracked Phase-9 follow-up)
yawgpu's naga MSL backend does not emit the "sizes buffer" slot
required for **runtime-sized storage arrays** (`var<storage>
values: array<u32>;`). Such shaders fail Metal compute-pipeline
creation with `mapping for sizes buffer is missing`. Fixed-size
arrays (`array<u32, N>`) compile cleanly on Metal/Vulkan/Noop.
`examples/compute/shader.wgsl` consequently uses `array<u32, 4>`
matching the input length, matching `mgpu/hello_compute`'s
`array<u32, 256>`. Supporting runtime-sized storage arrays on
Metal requires extending the binding map with a sizes-buffer
argument and wiring it from compute-pipeline reflection through
dispatch. Does NOT block P9.1+ (capture/triangle/etc. use
fixed-size buffers).

## P9.1 — `capture` (offscreen render → image file)  *(☑ DONE — real-GPU-verified)*

Done: `examples/capture/` (CMakeLists.txt + main.c + vendored
`stb_image_write.h` MIT/PublicDomain). 100×200 RGBA8Unorm
`RenderAttachment|CopySrc` texture, clear-only render pass (clear
to red `(1,0,0,1)`, no pipeline/no draw — matches wgpu-native's
baseline), `CopyTextureToBuffer` with `padded_bytes_per_row`
256-aligned, `MapAsync(Read)` + `GetConstMappedRange` + `stbi_write_png`
with stride arg = padded_bytes_per_row → `red.png`. `examples/
CMakeLists.txt` adds `add_subdirectory(capture)`; README updated.

**Phase-7 execution gap surfaced and fixed in the same slice:**
yawgpu's `hal_render_pass_execution` required a pipeline (`pass
.pipeline.hal()?`); WebGPU-spec-valid **clear-only** render passes
(no SetPipeline / no Draw, used by wgpu-native's capture) were
silently skipped → undefined/uninitialized texture → wrong PNG
output. Fix: made `RenderPassCommand.pipeline`/`draw` and
`HalRenderPass.pipeline`/`draw` `Option`s; core
`hal_render_pass_execution` emits `pipeline: None` / `draw: None`
when absent (no error). Metal & Vulkan `encode_render_pass`
conditionally bind pipeline + set buffers + draw only when
`pipeline.is_some()`; the begin / load+clear / store / end path
runs unconditionally, so clear-only passes now execute the clear.
The `Some(pipeline)` branch is byte-for-byte unchanged (Phase-7
`e2e_metal_render` 3/3 + `e2e_vulkan_render` 2/2 remain green on
the host). P6 validation unchanged (C37 only trips on draw).

**Verification (real-GPU, 2026-05-20):**
- Noop `cargo test --workspace` **58 binaries green** + clippy
  clean.
- `--features metal/vulkan` build+clippy clean.
- Phase-7 e2e regression: Metal `e2e_metal_render` 3/3 + Vulkan
  `e2e_vulkan_render` 2/2 green (no regression).
- real-GPU `capture`: writes 100×200 PNG. PNG pixel decode (corner +
  center): **`(255, 0, 0, 255)` solid red on both Metal AND
  Vulkan** ✅. Noop writes a 100×200 PNG (uninitialized memory
  contents — expected; Noop does not actually render).

## P9.2 — Real window→surface→swapchain (GLFW-gated)  *(☑ DONE — real-GPU-verified)*

Done: real `HalSurface` enum (Noop/Metal/Vulkan) + configure /
acquire_next_texture / present API in `yawgpu-hal`; FFI wiring in
`wgpuInstanceCreateSurface` decodes `WGPUSurfaceSourceMetalLayer`
chained source and creates the real HAL surface,
`wgpuSurfaceConfigure` / `Unconfigure` drive HAL state,
`wgpuSurfaceGetCurrentTexture` acquires a drawable / swapchain
image as a `core::Texture::from_hal` handle, `wgpuSurfacePresent`
calls HAL present. Noop surface stays P8.6 descriptor-only (SF3
N/A boundary preserved — `surface_validation` 4/4 unchanged).

- **Metal arm:** `MetalSurface { layer, current_drawable, config }`
  retains a `CAMetalLayer` (from `objc2-quartz-core`), sets
  pixelFormat/drawableSize/framebufferOnly=false on configure,
  `layer.nextDrawable()` → `MTLTexture` wrapped as `MetalTexture`
  for the swapchain image, `drawable.present()` on present (Metal
  tracks command-buffer usage of the drawable and presents after
  those buffers finish — no explicit `presentDrawable:`/commit
  required for the simple smoke case).
- **Vulkan arm:** `VulkanSurface { surface, swapchain, … }` —
  `vkCreateMetalSurfaceEXT` (MoltenVK on Apple Silicon),
  `VkSwapchainKHR` over `KHR_swapchain` (instance gains
  `KHR_SURFACE` + `EXT_METAL_SURFACE` extensions; device gains
  `KHR_SWAPCHAIN` when present). Swapchain `VkImage` wrapped as
  `VulkanTexture` with `owns_image=false, memory=None` (image
  lifetime owned by swapchain). Acquire uses a transient fence;
  present transitions `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR`
  via a one-shot command buffer + `vkQueuePresentKHR`. P7.6
  render-pass execution unchanged for the
  `UNDEFINED→COLOR_ATTACHMENT_OPTIMAL` direction.
- **GLFW + Cocoa shim:** new `examples/framework/framework_macos.m`
  (Objective-C, Apple-only) — opens a GLFW window via
  `GLFW_CLIENT_API=GLFW_NO_API`, extracts NSWindow via
  `glfwGetCocoaWindow`, sets `wantsLayer=YES` and attaches a
  `[CAMetalLayer layer]` (framebufferOnly=NO) to the contentView,
  returns the layer pointer for the
  `WGPUSurfaceSourceMetalLayer` chained source. CMake gates the
  `.m` and the `surface_smoke` subdir on `find_package(glfw3)` +
  APPLE; non-Apple builds (or systems without glfw3) skip the
  windowed subdir cleanly without affecting headless examples.
- **`examples/surface_smoke/`** opens an 800×600 window, picks a
  supported surface format (BGRA8/RGBA8 Unorm) via
  `wgpuSurfaceGetCapabilities`, configures with present_mode Fifo,
  runs up to 60 frames each acquiring the surface texture,
  beginning a clear-only render pass (reuses P9.1's Optional
  pipeline/draw path — clears to slate `(0.1, 0.2, 0.3, 1)`),
  submitting, presenting, polling events. Exits 0 cleanly on
  loop end or window close.
- **CMake `CARGO_TARGET_DIR` per feature:** discovered during M2
  verification — the cmake `cargo build` invocation now sets
  `CARGO_TARGET_DIR=target/target-${YAWGPU_FEATURE}` (so
  `target-metal/debug` and `target-vulkan/debug` hold distinct
  dylibs). Without this, the metal and vulkan dylibs would
  collide at `target/debug/libyawgpu.dylib`, the
  `#[cfg(not(feature = "metal"))]` /
  `#[cfg(not(feature = "vulkan"))]` arms of `wgpuCreateInstance`
  would silently fall back to Noop for whichever backend was
  not the most-recent cargo build, and the surface would dispatch
  through the Noop arm returning `GetCurrentTextureStatus_Lost`.

**Verification (real-GPU, 2026-05-20):**
- Noop `cargo test --workspace` **58/58 binaries** + clippy clean
  (`surface_validation` 4/4 unchanged — SF3 Noop boundary
  preserved).
- `cargo build/clippy -p yawgpu --features metal/vulkan` clean.
- Phase-7 e2e regression: Metal basic 3 / buffer 3 / texture 4 /
  compute 3 / render 3 / smoke 1 = **17/17**; Vulkan basic 3 /
  buffer 3 / texture 4 / compute 3 / render 2 = **15/15**. No
  regression from the swapchain-image VulkanTexture refactor
  (`owns_image` + optional `memory`) nor from the surface FFI
  rewire.
- real-GPU `surface_smoke` (Claude's Bash, foreground): both
  `YAWGPU_BACKEND=metal` and (with `$VULKAN_SDK` env sourced)
  `YAWGPU_BACKEND=vulkan` open the window, render 60 cleared
  frames, exit 0. Logged that Claude can run windowed examples
  in-session (memory: `claude-runs-windowed-examples`).

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
