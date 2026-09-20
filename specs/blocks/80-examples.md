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
- **Build: CMake + GLFW** (user installs `brew install cmake glfw`).
  Mirror wgpu-native's `examples/` CMake layout (a top-level
  `examples/CMakeLists.txt` + per-example dirs + a shared
  `framework`); the wgpu-native `CMakeLists.txt`/`main.c` port nearly
  1:1 — point the import at yawgpu's built lib
  (`libyawgpu.{a,dylib}` from `cargo build -p yawgpu [--features
  metal|vulkan]`) + the vendored `yawgpu/ffi/webgpu-headers/
  webgpu.h`. CMake `find_package(glfw3)` for windowed examples;
  headless ones don't link GLFW. Do not vendor GLFW. The Rust
  workspace build is unaffected (examples are a separate CMake tree,
  not cargo workspace members).
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

- **P9.0** C example CMake scaffold (wgpu-native layout) +
  **headless**: `enumerate_adapters`, `compute`, `device_info`
  (DawnInfo). Proves the CMake → `libyawgpu` + `webgpu.h` link and
  the backend-select struct from C. De-risk.
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

---

## Post-COMPLETE slice — P9.5: C framework future-wait regression (filed 2026-09-20)

Filed after a clean-install bring-up on Linux (native Vulkan, NVIDIA
RTX 5060 Ti). **Separate from Block 98** (the `yawgpu-tint` stub-cache
build bug found in the same session) — this one is a run-time regression
in the C example framework, not a build issue.

### Problem

`examples/framework/framework.c:180-187`:

```c
void yawgpu_wait_for_future(WGPUInstance instance, WGPUFuture future) {
    wgpuInstanceProcessEvents(instance);
    WGPUFutureWaitInfo wait_info = { .future = future, .completed = 0 };
    (void)wgpuInstanceWaitAny(instance, 1, &wait_info, 0);
}
```

This is a **single non-blocking pass**: `wgpuInstanceWaitAny` with
`timeoutNS == 0` is a poll, and its result is discarded, so the helper
returns whether or not the future completed. It was correct while queue
submission was synchronous — on Noop a map future completes at
registration, so one pass always suffices.

**Block 96 B2 (`a2332c4`, 2026-08-08, "perf(hal): make Metal submission
asynchronous") made submission asynchronous and updated the *Rust*
harness accordingly** — `yawgpu-test/src/lib.rs:377-402` now loops,
with a comment stating the rule exactly:

> A single non-blocking pass is not enough once queue submission is
> asynchronous: `wgpuInstanceWaitAny` with a zero timeout is a poll, and
> a future gated on GPU completion legitimately reports TimedOut until
> the submission finishes.

`examples/framework/framework.c` was last touched `f75ce4d`
(2026-06-29), i.e. **before** that change, and never followed. The C
framework is therefore stuck on the pre-async contract.

### Observed

With `-DYAWGPU_FEATURE=vulkan` on native Vulkan:

- `enumerate_adapters` ✅ — reports the real adapter
  (`NVIDIA GeForce RTX 5060 Ti`); adapter/device futures complete
  synchronously, so the one-shot poll happens to work.
- `compute` ❌ — `readback map did not complete successfully`
  (`examples/compute/main.c:192-193`: `map_state.called == false`).
- `capture` ❌ — same failure; no `red.png` written.
- All three ✅ on Noop (immediate completion masks the bug).

The corresponding Rust e2e coverage stays green (`e2e_vulkan_*`:
78 passed / 0 failed across 20 suites) precisely because it uses the
looping harness — which is why CI never caught this.

### Rules

- **R1 — Wait, don't poll.** `yawgpu_wait_for_future` drives the
  instance event loop until the future reports completed, rather than
  making a single pass. It calls `wgpuInstanceProcessEvents` then
  `wgpuInstanceWaitAny`, and **inspects** `wait_info.completed` /
  the returned `WGPUWaitStatus` instead of discarding the result.

- **R2 — Bounded.** The loop is bounded by a wall-clock deadline so a
  future that never completes returns instead of hanging an example.
  Mirror the Rust harness's shape; the deadline value may differ (an
  example waiting on real GPU work may warrant longer than the
  harness's 5s).

- **R3 — Error is terminal.** `WGPUWaitStatus_Error` ends the wait
  immediately; it is not retried until the deadline.

- **R4 — Caller-visible outcome.** The helper reports whether the
  future completed, so callers can distinguish "completed" from
  "timed out" instead of inferring it from their own callback flag.
  Changing the signature from `void` is permitted; every caller in
  `examples/` is updated accordingly. `examples/` is not a Cargo
  workspace member and ships no C ABI, so this is not a public-API
  break.

- **R5 — No Noop regression.** Every example that passes on Noop today
  still passes, with no added latency on the immediate-completion path
  (a future already complete on the first pass returns without
  sleeping).

- **R6 — Single wait implementation.** The fixed helper is the only
  future-wait in `examples/`; `yawgpu_request_adapter` /
  `yawgpu_request_device` (`framework.c:189-220`) keep delegating to
  it rather than growing their own loops.

### Verification

- Noop, unchanged: `enumerate_adapters`, `device_info`, `compute`,
  `capture` all run clean (build tree without `YAWGPU_FEATURE`).
- Real Vulkan (Linux / NVIDIA, this host), `-DYAWGPU_FEATURE=vulkan`:
  `compute` prints the real Collatz result rather than the readback
  error, and `capture` writes `red.png`. Run by Claude per
  `[[claude-runs-real-gpu-tests]]` and logged.
- Rust gates unchanged and green: `cargo test --workspace` +
  `cargo clippy --workspace --all-targets -- -D warnings`.
- Real Metal re-verification is **not** required on this host (no Apple
  hardware); the change is platform-independent C. Record as
  build-verified-elsewhere / deferred to the next macOS session.

### Follow-up (not this slice)

Examples are excluded from `cargo test`, so no automated gate would have
caught this. Consider a real-GPU smoke gate that runs the headless C
examples under `--features vulkan` alongside the `e2e_vulkan_*` suites.
Tracked here as an open question, not scope.
