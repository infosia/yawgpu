# GLES — Linux EGL device selection (Block 67 post-COMPLETE addition)

Ledger for the "Linux EGL device selection" section of
`specs/blocks/67-gles-backend.md`. Opened 2026-09-21 with the L2 cascade;
L4 completes it once L3 puts the renderer strings in the adapter name.

## Host

| | |
|---|---|
| GPU 0 | NVIDIA GeForce RTX 5060 Ti, proprietary driver 595.91.07 |
| GPU 1 | AMD Raphael iGPU (radeonsi / Mesa 26.0.8) |
| OS | Ubuntu, `x86_64-unknown-linux-gnu`, Wayland session |
| EGL devices | 4, per `eglQueryDevicesEXT` (see the block's problem table) |

## Before (HEAD `ec62d43`)

`eglGetDisplay(EGL_DEFAULT_DISPLAY)` → libglvnd → Mesa → **llvmpipe**
(software). The whole `e2e_gles_*` suite passed there, so nothing in the
test log distinguished it from a real-GPU run.

## After L2 + L3 (`auto`, no env vars)

```
yawgpu-gles: selected EGL device 0 (hardware) via EGL_PLATFORM_DEVICE_EXT: GL_RENDERER="NVIDIA GeForce RTX 5060 Ti/PCIe/SSE2"
adapter name: yawgpu GLES Adapter (EGL) — NVIDIA GeForce RTX 5060 Ti/PCIe/SSE2 / OpenGL ES 3.2 NVIDIA 595.91.07
```

`e2e_gles_*` **15/15 green** on it, and `yawgpu-hal --features gles --lib`
**215/215**.

Under `YAWGPU_GLES_EGL_DEVICE=3` the same two lines read:

```
yawgpu-gles: selected EGL device 3 (software) via EGL_PLATFORM_DEVICE_EXT: GL_RENDERER="llvmpipe (LLVM 21.1.8, 256 bits)"
adapter name: yawgpu GLES Adapter (EGL) — llvmpipe (LLVM 21.1.8, 256 bits) / OpenGL ES 3.2 Mesa 26.0.8-1ubuntu0.3
```

That difference is the whole point of L3: before it, both runs produced the
identical constant `"yawgpu GLES Adapter (EGL)"` and an otherwise identical
green log.

## Selection matrix (measured 2026-09-21, `e2e_gles_basic --ignored --nocapture`)

| `YAWGPU_GLES_EGL_DEVICE` | diagnostic | selected | spec rule |
|---|---|---|---|
| unset / `auto` | `selected EGL device 0 (hardware)` | NVIDIA | D1 step 3 — first validated hardware device |
| `default` | `…=default; using the default display` | Mesa llvmpipe | D3 — escape hatch reproduces pre-L2 behaviour |
| `3` | `selected EGL device 3 (software)` | Mesa llvmpipe | D3 — explicit index reaches the software device |
| `software` | `selected EGL device 3 (software)` | Mesa llvmpipe | D3 |
| `2` | `selected EGL device 2 (hardware)` | AMD radeonsi | D3 |
| `1` | `eglInitialize failed for EGL device 1: NotInitialized` then `no EGL device satisfied …Index(1); using the default display` | default display | **D2 + D3** — the validated-candidate rule and the runtime fallback, both exercised |
| `99` | `…Index(99) cannot be satisfied by the 4 enumerated EGL device(s); using the default display` | default display | D3 — unsatisfiable ask falls to step 4, **not** to `Auto` |
| `nvidia` | `unknown YAWGPU_GLES_EGL_DEVICE="nvidia"; falling back to auto` then `selected EGL device 0` | NVIDIA | D3 — unparseable degrades to `Auto` at parse time |

The `1` row is the one worth keeping: device 1 is Mesa's view of the NVIDIA
DRM node and fails `eglInitialize` although it enumerates. It is why D2
validates every candidate instead of trusting enumeration, and why the two
fallbacks in D3 are kept distinct — a pinned-but-unusable device must not
silently become a different device.

## Finding — unorm8 tie rounding is vendor-dependent

Two `gles/queue.rs` unit tests
(`submit_compute_pass_writes_rgba8_storage_texture`,
`submit_render_pass_binds_whole_size_uniform_buffer_at_offset`) write
`vec4(0.25, 0.5, 0.75, 1.0)` and asserted exactly `[64, 128, 191, 255]`.
They failed the moment L2 moved them onto real hardware: NVIDIA reads back
`[64, **127**, 191, 255]`.

Not a yawgpu defect. `0.25 × 255 = 63.75` and `0.75 × 255 = 191.25` round
unambiguously, but `0.5 × 255 = 127.5` is an exact tie, and the float →
normalized-fixed-point conversion rounds to nearest with the **tie direction
implementation-defined**. Mesa rounds up, NVIDIA rounds down; both conform.

Fixed in the tests only, by a helper that accepts either tie direction while
still pinning the three unambiguous channels exactly. No production code
changed. **The assertions had been silently pinned to llvmpipe** — the exact
class of blindness this block exists to remove, found on its first real-GPU
run, which is the strongest available evidence that the fix was worth making.

No other test in the 213 changed behaviour between llvmpipe, radeonsi and
NVIDIA.

## Not covered

- **Non-Linux compilation.** Acceptance criterion 4 (Android / Windows
  execute no new code) holds by construction — the cascade and the
  `EglDeviceChoice` parser are `#[cfg(target_os = "linux")]`, and
  `target_os = "android"` is a distinct value — but it was **not compiled**
  for those targets: only `x86_64-unknown-linux-gnu` is installed on this
  host. Stated rather than silently omitted, per
  `tracking/toolchain-clippy-1-98.md` R4.
- Per-mode `GL_RENDERER` / `GL_VERSION` for the six modes other than `auto`
  and `3`: L3 makes them readable from any run, but only those two were
  captured in full. L4 fills the rest if it is worth the runs.
- Windowed presentation: out of scope (D5); a device display is headless.
