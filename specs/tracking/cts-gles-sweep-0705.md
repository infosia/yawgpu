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
