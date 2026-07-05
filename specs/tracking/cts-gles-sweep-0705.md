# CTS GLES sweep 2026-07-05 (Linux Mesa, llvmpipe + crocus) — findings

Status: **root-caused; fixes pending handoff**. First CTS run against the
Tier 2 GLES backend, on the Linux host (Haswell). Backend forced via
`CTS_YAWGPU_BACKEND=gles` (already supported by webgpu-native-cts's
yawgpu shim); library `target-gles/release` built with
`--features vulkan,gles`. Driver profiles on this host: **crocus**
(Haswell iGPU, ES 3.1 floor) and **llvmpipe**
(`LIBGL_ALWAYS_SOFTWARE=1`, ES 3.2, no GPU-wedge risk).

## Timeline / incidents

- 18:33 `--workers 4` crocus full sweep (`webgpu:*`): whole-machine
  hard stop at ~18:36 (journal cuts off, no i915 errors, no clean
  shutdown). Initially suspected thermal (intel_powerclamp lines) —
  **ruled out**: powerclamp cycling is chronic on this host (228
  events in a benign same-day boot that shut down cleanly) and the
  user observed no fan spin-up. Workers had already crash-resumed
  (`--shard-from ~347-350`) before the freeze.
- llvmpipe re-runs reproduced the worker crash deterministically:
  abort after exactly 107 `create_egl_device` calls, independent of
  `ulimit -v` (8 GiB) and `ulimit -n` (4096). Byte-identical logs
  across runs.
- `EGL_LOG_LEVEL=debug` pinpointed the failing call:
  `EGL user error 0x3001 (EGL_NOT_INITIALIZED) in eglChooseConfig`.
- Control experiments (raw EGL loop, llvmpipe): 200 clean
  create/destroy context cycles OK; 200 *leaked* contexts also OK —
  no Mesa-side context-count limit. The failure is yawgpu-side.

## Finding G-1 — `wgpuInstanceRequestAdapter` panics (abort) on empty enumeration

`yawgpu/src/ffi/instance.rs:337`:
`.expect("Noop instance must expose an adapter")` fires when
`enumerate_adapters_with_feature_level` returns an empty list. The
message betrays the Noop-era assumption; for real backends an empty
list is a legitimate spec-level outcome ("no adapter available") and
must be delivered as a `RequestAdapter` callback with a non-success
status and null adapter, not a panic. Panicking across the
`extern "C"` boundary in the cdylib aborts the process
(`fatal runtime error: failed to initiate panic`). This is the direct
crash: every CTS worker crash-resume traces to this expect.
CLAUDE.md core principle 3's FFI `expect` exception covers invalid
handles/null, **not** this condition.

## Finding G-2 — `EglInstanceState::Drop` calls `eglTerminate` on a process-global display (root cause)

`yawgpu-hal/src/gles/instance.rs:35`. EGL display handles are
process-global: `eglGetDisplay(EGL_DEFAULT_DISPLAY)` returns the same
display to every caller in the process. When the CTS harness overlaps
instance lifetimes (new instance initialized before the old one is
released), the old instance's Drop terminates the display **under the
live instance**, and every subsequent EGL call on it fails
(`EGL_NOT_INITIALIZED`) → `enumerate_adapters` returns empty → G-1
aborts the process. Live GL contexts on a terminated display are also
undefined behaviour at the driver level — the leading suspect for the
18:36 whole-machine freeze on crocus/i915 (4 workers × churning
contexts on real hardware), though that attribution is plausible, not
proven. Precedent: wgpu-hal deliberately never calls `eglTerminate`
for exactly this reason.

## Finding G-3 (minor) — `enumerate_adapters` silently maps EGL errors to an empty list

`yawgpu-hal/src/gles/instance.rs:75-77`: `choose_config` failure →
`unwrap_or_default()` with no diagnostic, unlike every other GLES
bring-up failure path (which `eprintln!`s). Made G-2 needlessly hard
to localize: the process died with zero GLES diagnostics.

## Fix plan (handoff T-G1..T-G3)

- **T-G1**: empty enumeration in `wgpuInstanceRequestAdapter` →
  register the callback with an unavailable/error status and null
  adapter per webgpu.h semantics; never `expect`. Inline unit test.
- **T-G2**: stop terminating the display in `EglInstanceState::Drop`
  (leak-by-design, wgpu-hal precedent), with a comment documenting the
  process-global display semantics. Inline unit test:
  two overlapping `GlesInstance` lifetimes, drop the first, the
  second must still enumerate (feature-gated e2e-style unit test that
  self-skips when EGL is unavailable).
- **T-G3**: log the EGL error on the `choose_config` failure path in
  `enumerate_adapters`.

## Verification once fixed

1. Repro (must run clean end-to-end, no abort):
   `CTS_YAWGPU_BACKEND=gles LIBGL_ALWAYS_SOFTWARE=1 ./build-yawgpu-release/cts 'webgpu:api,operation,buffers,*' 'webgpu:api,operation,command_buffer,*'`
2. llvmpipe chunked full sweep (transcripts/gles-sweep-llvmpipe.sh).
3. crocus real-GPU sweep, supervised, quarantining the two zero-dim
   indirect-dispatch files (permanent quarantine per the 0704 sweep
   doc — same Haswell hardware wedge applies to
   `glDispatchComputeIndirect`).

## Operational notes (this host)

- CTS sweep worker count: the machine sustains `--workers 2`
  (~92 °C package, powerclamp-managed); `--workers 4` is untested
  post-fix and was in flight during the freeze — prefer ≤2 until the
  fixed library survives a full llvmpipe sweep.
- The docs/06-build-and-run.md example query
  `webgpu:api,validation,createBuffer:*` no longer matches the
  catalog (`webgpu:api,validation,buffer,create:*` is current) —
  runner silently reports pass=0 for unmatched queries (cts-repo docs
  fix candidate).

## Finding G-4 — self-deadlock in GLES `ResolveQuerySet` submission (2026-07-05 evening)

Repro (hangs forever, single case):
`CTS_YAWGPU_BACKEND=gles cts 'webgpu:api,validation,encoding,queries,resolveQuerySet:queryset_and_destination_buffer_state:querySetState="valid";destinationState="valid"'`
— the only case of the file that executes a real resolve; the other 8
(error-path) cases pass. Bisected from the full api,validation chunk
stalling at any worker count.

Cause: `submit_copies` (yawgpu-hal/src/gles/queue.rs, `ResolveQuerySet`
arm) calls `resolve.destination.write(...)` **inside**
`with_current_context`, and `GlesBuffer::write`
(yawgpu-hal/src/gles/buffer.rs:77) re-acquires the same device
`current_lock` — parking_lot mutexes are non-reentrant, so the queue
thread self-deadlocks. Every submission containing a resolveQuerySet
hangs the device permanently; this is also what stalled the 21:08
`--workers 4` crocus sweep (workers block one by one as their shard
reaches this file — worker count is irrelevant).

Fix (T-G4): perform the destination write with the already-current `gl`
inside the held lock (factor a lock-free `write_with_gl(gl, offset,
data)` helper out of `GlesBuffer::write` and use it from both call
sites), or hoist the resolve write outside the context closure. Inline
unit test: a submit containing a ResolveQuerySet copy against a real
GLES device must complete (self-skip when EGL unavailable) — a plain
Noop-level test cannot catch the re-entrancy.

## api,validation fail clusters observed during bisect (crocus, workers=2, to triage after G-4)

- files compute_pipeline..encoding,queries,general (bisect A1): fail=12,542 (pass=55,021 skip=104,379)
- image_copy,{buffer_related,buffer_texture_copies,layout_related}: fail=375+38+2,612
- error_scope: fail=12; buffer,*+capability_checks,*: fail=75

## Latent hazard noted during the T-G4 re-entrancy audit (not fixed; no current path triggers it)

The `Drop` impls of `GlesBufferInner` (buffer.rs:20), the
`GlesRenderPipeline`/`GlesComputePipeline` inners (pipeline.rs:36, 113),
`GlesSampler` (sampler.rs:21) and `GlesTexture` (texture.rs:30) each
acquire `with_current_context`. If the last `Arc` of any of these were
dropped from inside a context closure, the same self-deadlock as G-4
would occur. No production path currently does this (`submit_copies`
only borrows from `&[HalCopy]`), but any future change that moves owned
HAL resources into the submit path must keep this in mind.

## Finding G-5 — draw path rejects vertex buffers bound at slots the pipeline does not use

`bind_vertex_buffers` (yawgpu-hal/src/gles/queue.rs:1121-1130) iterates
`pass.vertex_buffers` and returns `HalError` ("vertex buffer binding is
missing from pipeline layout") when a bound slot has no layout entry in
the pipeline. WebGPU semantics: vertex buffers bound beyond the
pipeline's declared layouts are simply ignored at draw time (the CTS
draw suite binds buffers at extra slots constantly). Largest single
api,validation fail cluster: 11,988. Fix (T-G5): `continue` instead of
erroring for slots with no pipeline layout entry.

## Finding G-6 — Mesa rejects fragment-less render program links (ANGLE accepted them)

638+18 fails: "GLES render program link failed: error: program lacks a
fragment shader" (vertex-only pipelines, e.g. layout_shader_compat).
The mapping matrix noted vertex-only pipelines work "where GLES program
linking accepts a fragment-less program" — ANGLE accepts, Mesa/crocus
does not (GLSL ES requires both stages). Fix (T-G6): when a render
pipeline has no fragment stage, attach a minimal fragment shader
(`#version 310 es\nvoid main() {}`) at link time so vertex-only
pipelines link on every driver.

## Remaining api,validation clusters (larger slices, separate handoffs)

- Color-target formats beyond RGBA8/BGRA8 (5,083) — widen the
  renderable-format table per GLES 3.1 (+EXT_color_buffer_float);
  needs glClearBuffer{f,i,ui}v for integer formats.
- 1D/3D/array texture copies (3,594 + 311 T2B framebuffer-incomplete +
  257 format-unsupported) — P15.3 deferral.
- MRT >1 color target (1,270 + 44 + 34 sparse/non-zero slot) — F-040
  slice 1 deferral; GLES 3.1 glDrawBuffers.
- Unorm8x4Bgra vertex format (364) — no ES equivalent; candidate for
  permanent Tier-2 catalogue (Dawn emulates via shader swizzle).
- rgba8unorm read-write storage (180) — core-level message; verify
  against spec tiering before touching (may be correct behaviour).
