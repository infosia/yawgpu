# Phase 12 — Windows windowed examples (HWND surface + Win32 windowing)

Status: **PLANNED** (2026-05-21 — P12.0 spec authored). Rules/plan:
`../blocks/85-windows-surface.md`. Roles/loop:
`../reference/workflow.md`. Windows toolchain rules:
`../blocks/95-windows-build.md` (unchanged).

Goal: make **every** example build and run on Windows, including the
three windowed ones (`surface_smoke`, `triangle`, `hello_triangle`),
rendering for real via the **Vulkan** backend with **no GLFW
dependency** (native Win32 windowing). Metal stays macOS-only. The
Noop gate (`cargo test --workspace` + `cargo clippy --workspace
--all-targets -- -D warnings`) stays green on every host; examples
are not `cargo test` members. Real windowed runs are done on the host
(window presents) and logged here.

Decisions (user-approved 2026-05-21):
- Windows windowing = **native Win32** (`framework_windows.c`), no
  GLFW dependency on Windows.
- Surface-source selection = **framework helper**
  (`yawgpu_window_create_surface`), so windowed `main.c` stay
  platform-agnostic (no `#ifdef`).

## Why these slices, in this order

The window cannot render until the whole HWND→`VkSurfaceKHR` path
exists, and that path is dead unless `VulkanInstance::new` first
stops being macOS-only. So the Rust/HAL plumbing (P12.1→P12.3) lands
before the C examples (P12.4→P12.5). Each Rust slice is independently
testable on Noop; the Vulkan arms are build-verified everywhere and
run-verified on Windows in P12.5.

## Slices

- **P12.0** Spec authoring (Claude). `blocks/85-windows-surface.md` +
  this tracking file. *(☑ DONE 2026-05-21.)*
- **P12.1** Cross-platform `VulkanInstance::new` — capability-driven
  instance-extension selection (R85-1). De-risks the whole phase:
  proves a Vulkan instance comes up on Windows. *(☑ DONE 2026-05-21,
  commit `02e8e46`; one clippy revision P12.1-R1 applied by Claude.)*
- **P12.2** HWND surface through HAL — `VulkanInstance` +
  `HalInstance` `create_surface_from_windows_hwnd` (R85-2, R85-3).
  *(☑ DONE 2026-05-21, commit `6b4f591`.)*
- **P12.3** core + FFI wiring —
  `core::Instance::create_surface_from_windows_hwnd` +
  `find_windows_hwnd_source` + `wgpuInstanceCreateSurface` (R85-4,
  R85-5). *(☑ DONE 2026-05-21, commit `10163ce`.)*
- **P12.4** Framework Win32 windowing + surface helper, examples
  refactor (R85-6). *(☑ DONE 2026-05-21, commit `74d0261`.)*
- **P12.5** CMake builds windowed examples on Windows; docs; host
  run-verification (R85-7, R85-8). *(☑ DONE 2026-05-21, commit
  `9e93b86`; build + headless real-GPU verified by Claude. Windowed
  on-screen run pending on the host — see run log.)*
- **Phase 12 Review** (mandatory) → COMPLETE
  (`tracking/phase-12-review.md`).

## Exit criteria

- Noop `cargo test --workspace` + clippy gate **unchanged & green**
  on every host; new R85-1…R85-5 unit tests pass on Noop with no GPU.
- `cargo build -p yawgpu --features vulkan` clean on Windows & macOS;
  `--features metal` clean on macOS.
- macOS windowed examples still build (GLFW + metal-layer) and render
  (real-GPU, logged) — no regression.
- Windows: `cmake -S examples -B examples/build
  -DYAWGPU_FEATURE=vulkan && cmake --build examples/build` builds all
  examples incl. windowed; each windowed exe with
  `YAWGPU_BACKEND=vulkan` opens a window and renders ~60 frames then
  exits 0 (logged here).
- One commit per slice (`phase-12: <slice> — <short>`); mandatory
  Phase 12 Review logged in `tracking/phase-12-review.md`. Phase
  cannot be COMPLETE with any open CRITICAL/MAJOR.

## Phase 12 Review — focus areas (for the fresh reviewer)

- **No-panic across the C ABI**: `wgpuInstanceCreateSurface` with a
  malformed/partial `WGPUSurfaceSourceWindowsHWND` chain (null hwnd,
  null hinstance, both) must route to an error surface, never panic.
- **`match` totality / `cfg` arms**: every new `match self` over
  `HalInstance` covers Noop/Vulkan/Metal under each feature
  combination; the Metal arm of the HWND path returns an error, not
  `unreachable!`/panic.
- **macOS regression**: R85-1's capability-driven extension list
  enables exactly the same extensions on MoltenVK as before
  (portability flag still set there); the metal-layer path is byte-
  for-byte equivalent in behavior.
- **Block 90**: every new/changed `pub fn` (R85-1…R85-4) has an
  inline unit test in the same commit; Noop-reachable.
- **Refcount/leak**: the new surface path clones/drops the instance
  handle exactly like the metal-layer path (no extra `Arc` leak in
  the error-surface branch).
- **No GLFW on Windows**: confirm the Windows CMake path links only
  `user32`/`gdi32` and never probes/links GLFW.

## Run log

### P12.1 (2026-05-21, host: Windows 11 + Vulkan SDK 1.3.296.0)
- `cargo clippy -p yawgpu-hal --features vulkan --all-targets -- -D
  warnings` clean (after P12.1-R1 fixed a `clippy::manual_contains`
  in `has_instance_extension`).
- `cargo test -p yawgpu-hal --features vulkan`: 39 passed, 22 ignored
  (the 3 new `instance_extension_config` tests pass).
- Noop default gate: `cargo clippy --workspace --all-targets -- -D
  warnings` clean; `cargo test --workspace` clean.
- Real-instance check (Windows ICD): the `#[ignore]`d
  `vulkan_instance_new_constructs` and
  `vulkan_instance_enumerate_adapters_returns_devices` both pass when
  run with `--ignored` — Vulkan instance constructs and enumerates a
  physical device on Windows. Phase de-risk goal met.
- Process note: coding agent committed P12.0 spec (`2417855`) — left
  as-is (content correct). Incidental `.gitignore` (`/AGENTS.md`)
  change committed separately (`1cf35e8`).

### P12.2 (2026-05-21, host: Windows 11 + Vulkan SDK 1.3.296.0)
- `HalInstance` + `VulkanInstance` `create_surface_from_windows_hwnd`
  added (R85-2/R85-3), mirroring the metal-layer twin; diff is
  `lib.rs` +41 / `vulkan/mod.rs` +52 only (no core/FFI, no metal-path
  change).
- Noop default gate: `cargo test --workspace` + `cargo clippy
  --workspace --all-targets -- -D warnings` clean.
- `cargo clippy -p yawgpu-hal --features vulkan --all-targets -- -D
  warnings` clean; `cargo test -p yawgpu-hal --features vulkan`:
  40 passed, 23 ignored (new Noop `..._noop_ignores_pointers` passes).
- Real-instance check (Windows ICD): the `#[ignore]`d
  `vulkan_instance_create_surface_from_windows_hwnd_rejects_null_hwnd`
  passes with `--ignored` — confirms the `hinstance/hwnd as _` casts
  compile/link against ash 0.38 win32 types and null-hwnd is rejected
  on a real instance.

### P12.3 / P12.4 / P12.5 (2026-05-21, host: Windows 11 + VS 2022 +
Vulkan SDK 1.3.296.0; coding agent delivered all three in one pass,
split into commits `10163ce` / `74d0261` / `9e93b86`)
- P12.3 FFI weaving traced across all cases (metal present / Noop /
  real success+fail, HWND present / Noop / real success+fail, no
  source, both present → metal wins); null-hwnd routes to error
  surface (real) / Noop surface (Noop), no panic across the C ABI.
- Noop default gate: `cargo test --workspace` clean (5 new HWND tests
  pass) + `cargo clippy --workspace --all-targets -- -D warnings`
  clean. `cargo build -p yawgpu --features vulkan` clean.
- Examples build (the real P12.4/P12.5 test): `cmake -S examples -B
  examples/build -DYAWGPU_FEATURE=vulkan` + `cmake --build` →
  **all 8 examples build, incl. windowed `triangle` /
  `hello_triangle` / `surface_smoke`**, `/W4` warning-free. `yawgpu.dll`
  auto-copied next to each exe.
- Real-GPU headless run: `enumerate_adapters` with
  `YAWGPU_BACKEND=vulkan` reports `NVIDIA GeForce RTX 5060 Ti`,
  backendType 6 — the C FFI → Vulkan path works end-to-end on Windows
  hardware.
- **Pending (host, on-screen):** running the three windowed exes with
  `YAWGPU_BACKEND=vulkan` to confirm a window opens and renders ~60
  frames then exits 0. Build + headless Vulkan verified; on-screen
  present is the user's manual step.
- MINOR (logged, accepted): the three windowed `main.c` gained a
  `WGPUSurfaceGetCurrentTextureStatus_Lost` frame-skip — a small
  robustness addition beyond R85-6's "no render-loop change", kept.
