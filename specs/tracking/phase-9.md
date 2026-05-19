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

### P9.0 known issue (tracked Phase-9 follow-up)
`examples/compute` reads `[0,0,0,0]` on the **Metal** backend
whether the storage buffer is seeded via `mappedAtCreation` or
`queueWriteBuffer`, while Vulkan/MoltenVK produces the expected
Collatz `[0,1,7,2]` on the same core code path. The Rust
`e2e_metal_compute` (using `queueWriteBuffer`) passes 3/3 on the
same M2 — so the compute pipeline, blit copy, and map-read paths
are individually sound on Metal. The example framework currently
seeds `CopyDst` buffers via `queueWriteBuffer` (a partial portable
workaround); the symptom remains Metal-specific. Likely cause: a
sequencing / autoreleasepool / FFI-marshalling difference between
the C `cdylib` Metal compute path and the Rust e2e path. Needs
deeper investigation as a Phase-9 follow-up; does NOT block P9.1+.

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
