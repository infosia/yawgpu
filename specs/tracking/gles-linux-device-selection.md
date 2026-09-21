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

## Before (HEAD `1cd47e8`)

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
| `1` / `2` | one of them `selected EGL device N (hardware)` → AMD radeonsi; the other `eglInitialize failed for EGL device N: NotInitialized` then `no EGL device satisfied …Index(N); using the default display` | AMD radeonsi / default display | **D2 + D3** — the validated-candidate rule and the runtime fallback, both exercised. **Which index is which is not stable** — see below |
| `99` | `…Index(99) cannot be satisfied by the 4 enumerated EGL device(s); using the default display` | default display | D3 — unsatisfiable ask falls to step 4, **not** to `Auto` |
| `nvidia` | `unknown YAWGPU_GLES_EGL_DEVICE="nvidia"; falling back to auto` then `selected EGL device 0` | NVIDIA | D3 — unparseable degrades to `Auto` at parse time |

The `1`/`2` row is the one worth keeping: one of those two devices
enumerates and still fails `eglInitialize`. It is why D2 validates every
candidate instead of trusting enumeration, and why the two fallbacks in D3
are kept distinct — a pinned-but-unusable device must not silently become a
different device.

**Indices are not stable.** First measurement (morning): `1` = fails
`eglInitialize`, `2` = AMD radeonsi. Re-measured the same day during the
Phase Review, and again three times after it: `1` = AMD radeonsi, `2` =
fails. `0` (NVIDIA) and `3` (llvmpipe) did not move in any run, and `auto`
selected NVIDIA every time. `eglQueryDevicesEXT` ordering is
implementation-defined, so `YAWGPU_GLES_EGL_DEVICE=<n>` pins an enumeration
slot rather than a device; `auto` / `software` / `default` are the
reproducible selectors. Possible future work, out of scope for this block:
a renderer-substring selector (`YAWGPU_GLES_EGL_DEVICE=nvidia` matching
`GL_RENDERER`) would give a stable way to name a specific GPU.

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

## Phase Review (2026-09-21)

Clean Review per `specs/reference/workflow.md`: a fresh subagent with no
session context, given the cumulative diff `5e79401..HEAD`, this block's
spec, `CLAUDE.md` and the naming conventions. It ran the gates itself and
reproduced acceptance criteria 1, 2, 3 and 5.

**0 CRITICAL, 2 MAJOR, 9 MINOR.** The reviewer separately confirmed as
correct: all three transmuted EGL extension signatures against the registry
specs, the constants, the null-`devices` count form, `EglDisplay::from_ptr`,
resource cleanup on every probe error path, D5's headless constraint, and
the `chunks_exact` → `as_chunks` rewrites.

| id | severity | finding | disposition |
|---|---|---|---|
| M1 | MAJOR | `GlesAdapter::name()` had no direct inline unit test (Block 90 requires one; integration coverage does not satisfy it), and `gles_adapter_name_is_present` asserted only `contains("OpenGL ES")` — which the old constant and the degraded `unknown` form both satisfy. `GL_RENDERER` was asserted nowhere. | **Fixed** — inline live-adapter test asserting the name carries the driver's `GL_RENDERER`, plus a strengthened e2e assertion |
| M2 | MAJOR | The section still opened with "SPEC ONLY — not implemented" and the matrix row still said NOT implemented, contradicting the slice records 220 lines below. A coding agent reading top-down would conclude the cascade does not exist. | **Fixed** (spec) — banner, matrix cell and acceptance roll-up updated |
| m1 | MINOR | EGL enumeration indices are not stable across sessions; the ledger's `1`/`2` rows no longer reproduced. | **Fixed** (spec + this doc) — D3 now states an index pins a slot, not a device |
| m2 | MINOR | `device_is_software` classified a NULL extension-string query as *hardware*, so a software device could enter the `Auto` list and be selected while the log claimed hardware. | **Fixed** — classification is three-valued; unknown devices stay eligible but are ordered after known hardware, with a diagnostic |
| m3 | MINOR | The winning candidate is capability-probed twice: once to validate it, once in `GlesAdapter::new_egl`. | **Deferred** — see rationale below |
| m4 | MINOR | The L3 note claimed the panic exposure was "unchanged"; L3 in fact adds a `GL_RENDERER` read on the EGL arm and both reads on the WGL arm. | **Fixed** (spec wording + one doc comment) |
| m5 | MINOR | `GlesDriverInfo` derived `Clone`/`Debug`/`Default`, none used. | **Fixed** |
| m6 | MINOR | Seven `resolve_device_candidates` tests did not follow Block 90's `<fn_name>_<scenario>` naming. | **Fixed** |
| m7 | MINOR | `load_client_proc<T>`'s `transmute_copy` has no size check; a future non-fn-pointer `T` would read out of bounds with no compile error. | **Fixed** — `debug_assert_eq!` on the sizes |
| m8 | MINOR | `"+3"` — the one parse case D3 singles out as a deliberate decision — had no test. | **Fixed** |
| m9 | MINOR | Lost blank line between two fns; the "no `Drop`, never `eglTerminate`" comment no longer described the code (the cascade can leave several displays initialized). | **Fixed** — comment extended, behaviour unchanged |

No finding was dropped as a false positive.

### m3 deferral rationale (required by the workflow for a deferred MINOR)

`try_device_display` runs the full `query_egl_adapter_caps` to validate a
candidate per D2, discards the `GlesAdapterCaps` and keeps only the driver
strings; `GlesAdapter::new_egl` then re-runs the identical probe on the same
display and config. Linux therefore pays two extra context
create/make-current/destroy cycles, plus a duplicate `choose_config`, per
`GlesInstance::new()`.

Deferred because the fix is a design change, not a cleanup: the probe result
would have to be carried across the instance/adapter seam — cached in
`EglInstanceState` or returned from display selection — and that seam is
shared with the Android and Windows paths, which do not run a cascade at
all. Instance creation is not a hot path (once per `wgpuCreateInstance`, and
the probe is a 1×1 pbuffer), so the cost is bounded and one-off. Revisit if
instance-creation latency ever matters, or when the adapter layer is next
touched for another reason.
