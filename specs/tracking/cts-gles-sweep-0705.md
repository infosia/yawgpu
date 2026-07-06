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

## Overnight fix session 2026-07-05/06 — api,validation ledger (crocus, workers=2)

| Slice | Commit | fail after |
|---|---|---|
| start of session | 82de24f (G-4) | 24,553 |
| G-5 ignore undeclared VB slots + G-6 stub fragment shader | a975951 | 13,052 |
| T-G7 GLES 3.1 color-renderable set + integer clears | 12aefe9 | — |
| T-G8 MRT via glDrawBuffers (uniform per-target state) | d43b5bf | 7,531 |
| T-G11 base-vertex draws (OES/EXT_draw_elements_base_vertex) | fad5a01 | — |
| T-G9 2D-array/3D copies (+ Mesa 3D-layer PBO readback bug workaround) | 2139a0f | 3,759 |
| T-G12 extension-gated float color targets | 5d0e244 | **2,616** |

pass 185,492 → 207,429; crash 0 throughout; draw file 12,006 → 56.

## Remaining clusters (2,616) — classification

Permanent Tier-2 catalogue candidates (need sign-off, then CTS
expectations entries):
- 852 residual non-renderable color targets (snorm / bgra8unorm-srgb /
  formats GLES cannot render to even with extensions — enumerate before
  sign-off)
- 364 Unorm8x4Bgra vertex format (no ES equivalent; Dawn emulates via
  shader swizzle — implementable but costly)

Policy decision needed (adapter limits truthfulness vs binding remap):
- 330+330 shader compile fails: `layout(binding=999)` exceeds GL UBO
  limits (adapter advertises WebGPU-default maxBindingsPerBindGroup),
  `samplerCubeArray` reserved on ES 3.1 (cube-array needs ES 3.2/EXT).
  Options: report真 GLES limits on the adapter (sub-WebGPU-minimum) or
  implement the block-67 linear binding remap for group/binding.
- 180 "unsupported read-write storage texture format" (core-level
  message — verify against spec tiering first).
- 32 "supports only bind group 0" (render 16 + compute 16) — the
  full bind-group remap design (same thread as above).

Small implementables:
- 54 indexed-indirect index-buffer offset restriction
- 17 framebuffer-incomplete T2B residuals
- 373 "texture format not supported (P15.3)" — mostly depth/stencil
  copy formats; depth readback needs blit tricks on ES (part catalog,
  part implementable)

Next feature slice: F-040 slice 2 MSAA/resolve (41 direct fails here,
larger effect expected in api,operation).

## api,operation sweep (2026-07-06 morning, crocus, workers=2, per-dir)

Totals: ~76,000 fail / crash 0. command_buffer needed ~8.5 min alone
(slow, not stalled). Clusters:

- **command_buffer 72,031** — three families: (1) T2B "texture padding
  mismatch ... got 0" (~12k+): readback zeroes destination-buffer
  padding bytes the CTS expects preserved; (2) "framebuffer incomplete
  for texture-to-buffer copy" 3,848: T2B of non-color-renderable
  formats cannot use the FBO/glReadPixels path; (3) large pixel-mismatch
  families in image_copy origins/extents/array cases (real copy bugs).
- **render_pipeline 3,029** — nearly all "queue submit cannot use an
  error command buffer": encode-time HAL rejection poisons the CB; the
  dominant underlying rejection is the catalogued-but-unimplemented
  **texture/sampler bindings in render/compute passes** (also 60 direct
  hits in texture_view). Implementing GLES texture/sampler binding is
  the single biggest unlock for api,operation and shader,execution.
- rendering 427 (334 P15.3 formats, 72 indexed-indirect offset),
  memory_sync 234 (166+64 "binding size exceeds GLES limit"),
  texture_view 210, render_pass 63, others < 30 each.

## Agreed next campaign (user-approved 2026-07-06 morning; in this order)

1. **T-G13 — sampled texture + sampler bindings** (render + compute,
   group 0, storage textures stay rejected → T-G14). Key unlock for
   api,operation and the untouched shader,execution area. Investigate
   Tint's combined-sampler GLSL emission first (yawgpu-tint BindingRemap
   plumbing exists); assign texture units at link time (tint_immediates
   uniform pattern), bind via glActiveTexture/glBindTexture/glBindSampler
   at draw/dispatch; mip subrange via TEXTURE_BASE_LEVEL/MAX_LEVEL;
   array-layer-subrange views and cube-array return HalError. A stopped
   agent had just started this — restart fresh.
2. T2B padding-zeroing bug (~12k in command_buffer) + copy-correctness
   families (image_copy origins/extents).
3. Decision 2a implementation: adapter reports true GLES limits
   (maxBindingsPerBindGroup et al from GL queries) and stops advertising
   norm16/texture-formats-tier features it cannot render.
4. T2B of non-color-renderable formats (3,848).
5. Tier-2 permanent catalogue (user-approved): snorm/16-bit-norm/
   bgra8unorm-srgb render targets, Unorm8x4Bgra vertex format →
   block-67 matrix entries + webgpu-native-cts expectations file.

Verification loop per slice: unit gates → target-gles release rebuild →
targeted CTS file(s) → full-area re-measure → commit. CTS runs:
workers=2, one GPU process at a time, never stack GPU loads
(see memory: no-heavy-load-on-this-host).

## 2026-07-06 Codex-agent session: MSAA + the sample_mask onion (T-G15..T-G18)

Landed in one interleaved change set (commit below):
- T-G15 MSAA: glTexStorage2DMultisample textures, MSAA passes,
  glBlitFramebuffer resolve, glSampleMaski sample mask,
  alpha-to-coverage; real-EGL resolve/mask tests.
- T-G16 diagnosability: CommandBuffer/BindGroup/TextureView retain
  their creation/finish error message; submit and setBindGroup errors
  append it ("...: <original>"). This peeled a 4-layer error onion that
  was previously a single opaque "error command buffer" (CTS harness
  keeps only the LAST uncaptured error — harness.cpp:349).
- T-G17: Tint texture_builtins_from_uniform metadata UBO exposed
  through the shim and bound by GLES pipelines (textureLoad with
  explicit binding remaps previously failed pipeline creation with
  Codegen("texture missing from texture_builtins_from_uniform list")).
- T-G18: depth/stencil format mappings (depth16..depth32f-stencil8,
  stencil8 gated on OES_texture_stencil8), DEPTH_STENCIL_TEXTURE_MODE
  aspect binding, and a per-device internal NEAREST placeholder
  sampler — placeholder (samplerless textureLoad) bindings previously
  left default LINEAR filtering on integer/stencil textures, making
  them incomplete (reads returned 0; llvmpipe confirmed it was not a
  driver quirk).

CTS render_pipeline dir: pass 36 -> 366, fail 3,029 -> 2,699 (rest is
MSAA per-sample behavioural correctness in the huge sample_mask file).
Next: rerun full api,validation + api,operation ledgers, then the
remaining campaign order stands.

## Ledger update after T-G15..T-G18 (2026-07-06)

- api,validation: fail 2,616 -> **2,280** (pass 207,765; crash 0).
- api,operation command_buffer: 72,031 -> 72,844 (slightly up: depth
  formats now create, exposing more copy cases to the known T2B
  bugs — the untouched campaign slice #2).
- render_pipeline: 3,029 -> 2,699 (rest: MSAA per-sample behaviour in
  sample_mask).

Next session: campaign slice #2 (T2B padding-zeroing + copy
correctness, ~12k+ single-bug candidates in command_buffer), then
limits truthfulness (#3), non-renderable readback (#4), catalogue (#5).
Codex-agent ops note: Codex sandbox has no EGL — hardware verification
always runs on the orchestrator side; the hypothesis->instrument->
hardware-output loop works well.

## Slice 2 progress (2026-07-06, Codex agent)

- 2a (b20ff6c): T2B padding preservation — command_buffer 72,844 -> 60,354.
- 2b (80eef90): ClearTexture lazy-init was a GLES no-op + sampled-bind
  BASE/MAX_LEVEL poisoning of later copies — 60,354 -> **37,669**.
- Remaining command_buffer clusters: non-renderable-format T2B
  (framebuffer-incomplete, 3,848 — slice #4), depth readback rejection
  (1,196 — slice #4/catalogue), and a ~30k pixel-mismatch tail to
  re-cluster after slice #4 (many were secondary to the fixed bugs).
- Campaign slices #3 (limits truthfulness) and #5 (catalogue) unchanged.

## Slice 4 (2026-07-06): compute-shader T2B fallback landed

command_buffer 37,669 -> **22,739** (snorm/norm16 + depth-aspect
readbacks now work; stencil readback stays catalogued). Session
command_buffer total: 72,844 -> 22,739. Remaining: re-cluster the
~22k tail (T2T format-conversion families suspected), then slices
#3 (limits truthfulness) and #5 (catalogue).

## Slice 4b exploration (2026-07-06 late) — REVERTED, findings kept

Attempted: rgb9e5ufloat compute-fallback + PBO->client-row readback
rewrite + OES_copy_image detection. Unit-level copy matrix (formats x
offsets x origins x layers) went fully green, but CTS command_buffer
regressed (22,739 -> 23,850 -> 24,116), so the working tree was
reverted to 661991a per the beat-the-checkpoint rule. Salvageable
pieces saved in webgpu-native-cts/transcripts/
slice4b-exploration-reverted.patch: (1) the comprehensive matrix test,
(2) canonical Rust RGB9E5 reference encoder (layout R 0..8, G 9..17,
B 18..26, exp 27..31), (3) GL_OES_copy_image detection. Lesson: the
client-row readback rewrite likely regressed cases the matrix does not
cover (3D, mips, T2T interactions) — next attempt should change ONE
path at a time and re-run CTS per change. Checkpoint stands at
command_buffer fail=22,739 (commit 5736aea/661991a).

## Slice 4b pieces (2026-07-06 night, one-at-a-time discipline)

- Piece 1 (eb9f172): rgb9e5 compute fallback + copy matrix test — 22,739 -> 22,503.
- Piece 2 (a602bc7): GL_OES_copy_image detection — -> 22,434.
- Piece 3 (7f08953): non-attachable lazy-init clear via zero upload — -> **19,331** (fb-incomplete family eliminated).
- Piece 4 REVERTED (patch: webgpu-native-cts/transcripts/slice4b-piece4-3d-b2t-reverted.patch): 3D B2T row-by-row upload regressed CTS to 20,968 despite green matrix incl. the new 4x4x3 partial-B2T case. The "byte 48 expected 0 got 1" family (~3.5k, r8snorm/3d origins_and_extents) remains UNSOLVED — the failing shape differs from all matrix cases; next attempt should extract the exact CTS subcase parameters (--list-cases on that test) and replicate one verbatim before changing production code.
- Remaining command_buffer: 19,331 = unsolved padding family (~3.5k) + stencil readback catalogue (908) + long tail to re-cluster.

## Slice 4b piece 5 WIP (paused mid-investigation; patch saved)

The r8snorm/3d rows_per_image>height family is REPRODUCED by an
exhaustive CTS-replica unit test (in
webgpu-native-cts/transcripts/slice4b-piece5-slicestride-wip.patch,
together with a per-slice B2T upload attempt that did NOT fix it —
CTS 19,331 -> 19,373, replica still failing identically). Open
question for the resumption: the replica reports "byte 32 expected 63"
inside ROW PADDING for a width-3 texture at read_bpr=256 — byte 32
cannot be texel data at that stride, so the replica's own
expected-buffer stride math is suspect; verify the checker semantics
in webgpu-native-cts harness.cpp:601-646 before trusting the repro,
THEN re-attempt the production fix. Committed checkpoint remains
command_buffer fail=19,331 (repo green, 156 gles tests).

## Slice 3a (2026-07-06): truthful GL-queried adapter limits — DONE (61dd95b)

api,validation: pass 207,765 -> 209,624, skip 142,270 -> 140,442,
fail 2,280 -> 2,249, crash 0. The binding-overflow shader-compile
class (`layout(binding=999)` exceeds UBO points) collapsed ~624 -> 12
because maxBindingsPerBindGroup is now min-of-GL-binding-points, not
1000. Net fail barely moved only because ~1,828 previously-skipped
cases now execute (and mostly pass).

Residual api,validation shader-compile fails re-classified:
- ~50 cube-array (`samplerCubeArray` reserved on ES 3.1) — cube-array
  is CORE WebGPU, so this is a Tier-2 hardware gap → catalogue.
- ~168 storage-texture layout identifiers (r16/rg16/rgba16 [s]norm,
  r32ui as read-write image) + 180 "unsupported read-write storage
  texture format" — these come from advertising TextureFormatsTier2
  (read-write storage). GLES 3.1 has glBindImageTexture but yawgpu has
  NOT wired GLES storage textures (T-G14 deferred), and ES image
  load/store supports only a limited format subset anyway.

### DECISION NEEDED — storage textures on GLES
CTS expects these to work because the adapter advertises the storage
capability. Options: (A) implement GLES storage textures (T-G14:
glBindImageTexture; sizable, unlocks ~350+ fails properly, ES-format
subset must still be catalogued); (B) catalogue as Tier-2 unsupported
in the expectations file (fast, honest, leaves the capability
un-executable). Storage textures are core WebGPU, not an optional
feature, so "stop advertising" is not clean here.

Clearly-catalogue (approved slice 5, no decision): 852 non-renderable
color targets, 364 Unorm8x4Bgra vertex format, cube-array, stencil
readback — block-67 matrix + expectations file.

## Slice 3b + 5 complete (2026-07-06) — api,validation fail 2,280 -> 595

- 3a truthful limits (61dd95b): binding-overflow ~624 -> 12.
- 3b feature truthfulness (6b105c4): drop TextureFormatsTier1/Tier2/
  Bgra8UnormStorage/TimestampQuery on GLES; Rg11b10/Float32Filterable
  extension-gated. norm16/snorm/tier render-target + storage-format
  cases now SKIP. fail 2,249 -> 959, skip +16.5k. Vulkan/Metal/Noop
  advertisement unchanged (workspace suite green).
- Unorm8x4Bgra vertex format (169006c): implemented HAL-side (GL_BGRA
  attribute size when EXT/ARB_vertex_array_bgra present; crocus lacks
  it so accept-only with R/B execution divergence). vertex_state 0 fail.
- Catalogue: block-67 "CTS-confirmed Tier-2 catalogue" section records
  cube-array, stencil readback, >1 bind group, and the Unorm8x4Bgra
  execution divergence. The expectations-file route was ABANDONED — CTS
  failures here are subcase-specific and the file is only case-granular,
  producing heavy xpass noise.

api,validation residual 595: storage textures ~204 (T-G14, NEXT),
maxBindingsPerBindGroup off-by-one ~100 (follow-up), indexed-indirect
54, depth-stencil copy bytes_per_row 57, single-bind-group 32,
cube-array 50, stencil 7. crash 0.

## T-G14 storage textures done (d0ef874)

Tint emits storage textures as image uniforms with layout(binding=N) =>
glBindImageTexture(unit=N). GLES image-loadstore required format set
mapped; others HalError. Real-EGL compute write/read_write tests pass.
api,operation,storage_texture 6 -> 0 fail. api,validation UNCHANGED at
595 — its storage residual is NOT a storage-execution gap.

### FINDING (Tier-independent, needs a decision): core error-routing
The 204 api,validation "StorageBinding texture format must support"
(102) + "cannot create a view from an error texture" (102) are in
state,device_lost,destroy:{createTexture,createView} iterating
format="r8unorm";usageType="storage" — an invalid combo (r8unorm is
not storage-capable in WebGPU, any tier). CTS marks our rejection
"unexpected validation error", i.e. Dawn produces an error-texture the
scope catches while yawgpu routes it to the uncaptured device-error
sink. This is a CORE error-routing divergence, Tier-independent (should
reproduce on Vulkan) — NOT GLES-specific. Candidate to verify against
the Vulkan sweep and fix in core (would help Vulkan conformance too).

## api,validation residual 595 (post T-G14) — disposition
- 204 core error-routing (above) — cross-backend, verify + core fix.
- 122 cube-array — catalogued (block-67).
- 106 single-bind-group (>group 0) — deferred GLES impl.
- 57 depth32float-stencil8 copy bytes_per_row — investigate (over-strict?).
- 54 indexed-indirect draw restriction — investigate/impl.
- ~50 residual incl. maxBindings off-by-one follow-up.
crash 0 throughout. Campaign api,validation arc: 24,553 -> 595.

## #1 resolved: NOT a core bug — CTS-port storage-skip gap (2026-07-06)

The 204 "StorageBinding format must support" / "view from error texture"
were NOT Tier-independent (my earlier note was wrong): the SAME case
PASSES on yawgpu-Vulkan and FAILS on yawgpu-GLES. Root cause: slice 3b
correctly stopped advertising TextureFormatsTier1 on GLES, so r8unorm/
rg8/r16/rg16/rgb10a2/rg11b10 + storage is (correctly) invalid there,
while Vulkan (tier1 present) allows it. The webgpu-native-cts
device_lost destroy tests filtered these combos in via
isPossiblyStorageReadable but only skipped on the format's BASE feature,
never on storage-usability — so they ran an unsupported combo. Fixed in
the cts repo (device_lost/destroy.spec.cpp: isFormatUsableAsStorageOnDevice
gate, commit ab67f80): GLES skips (204 -> skip), Vulkan unaffected
(pass 726, skip 0). yawgpu behavior was correct on both backends — no
core change. api,validation fail 595 -> 391.

api,validation residual 391: cube-array 122 (catalogued), single-bind-
group 106 (deferred impl), depth-stencil copy bytes_per_row 57,
indexed-indirect 54, maxBindings off-by-one ~40, stencil readback 7.

## shader,execution FIRST sweep (2026-07-06, crocus, workers=2)

shader,validation: 369,753 pass / 0 fail / 0 crash — fully clean (Tint).

shader,execution: pass 167,098 / skip 516,424 / fail 151,140 / crash 2.
Fail clusters:
- **~81k single-bind-group**: "GLES render/compute pass supports only
  bind group 0" (53,457) + its poisoned-CB secondaries ("error command
  buffer: render pass" 18,668 + "compute" 7,454 + compute-group-0
  2,603). shader,execution builtins bind @group(1..3). DOMINANT lever —
  implementing multi-bind-group (linear binding remap) unlocks ~81k.
- **~30k texture sampling correctness**: textureSampleLevel / textureGather
  value mismatches (mip level / filtering / gather component). Real
  bugs in the T-G13 sampling path — mip-level sampling and gather.
- **crash 2 (SEGFAULT)**: textureDimensions:sampled_and_multisampled on
  depth24plus-stencil8 / depth32float-stencil8 — signal 11 in the GLES
  backend. MUST-FIX (no library segfault).
- smaller: access 40, shader_io 87, memory_model 52, statement 5,
  flow_control 2.

Next: (1) fix the segfault (must), (2) multi-bind-group implementation
(biggest lever, ~81k), (3) texture sampling correctness (~30k).

## shader,execution crash disposition (2026-07-06)
The 2 crashes (textureDimensions stencil-only aspect on packed
depth/stencil) are a suspected Mesa/crocus driver bug in textureSize()
on a stencil-mode sampler — texelFetch works, textureSize crashes; a
bind-time guard was tried and REVERTED (it broke the passing T-G18
texelFetch readback). Catalogued in block-67. Next: multi-bind-group
(the ~81k single-bind-group lever).

## Multi-bind-group DONE (85b4880) — shader,execution bind-group-0 eliminated

Flat per-class binding remap (core computes, shared by Tint emission +
HAL glBindBufferRange/glBindImageTexture); textures/samplers needed
only guard removal (resolve by uniform name). Batch-5 bind-group-0
fails 26k -> 0; new cross-group compute+render tests pass; workspace
green. Those cases now BUILD and advance to the next limitation rather
than failing at bind time.

### shader,execution next layers (post multi-bind-group)
- **~26k+ "GLES render pass supports only 2D depth-stencil attachments"**
  — now the dominant cluster (was masked behind bind-group-0). Surfaces
  in texture-builtin tests (textureGather/textureSampleLevel on depth/
  stencil formats) via the CTS result-collection render pass. Investigate:
  over-strict HAL check vs a real non-2D-DS-attachment the harness uses.
- **~30k texture sampling correctness** — textureSampleLevel /
  textureGather value mismatches (mip level, filtering, gather
  component ordering). Real T-G13 sampling-path bugs.
- 2 driver-crash cases catalogued (block-67).
A full shader,execution re-sweep is due after the 2D-DS-attachment layer
to get the true post-multi-group number.

## Layered depth-stencil attachments DONE (e185afc)
'2D depth-stencil attachments' cluster 26.5k -> 0 (batch-5 fail 72,278
-> 48,428, pass +24k). Remaining shader,execution = texture sampling
CORRECTNESS, characterized (batch-5 sample):
- textureGather 13,770 + textureGatherCompare 9,720 = ~23.5k — fails on
  BOTH nearest AND linear, so it's the gather operation itself
  (texel/component ordering or offset), not filtering. BIGGEST.
- textureSample 5,349
- textureDimensions 1,602 — size-query value mismatches (suspect the
  T-G13 BASE_LEVEL/MAX_LEVEL bind state affecting textureSize, or the
  metadata UBO values)
- textureLoad 1,456, textureNumLevels 123, textureNumSamples 12
These are MULTIPLE independent sampling bugs, each a hardware-loop
debug. Priority order: textureGather family (23.5k), then textureSample,
then textureDimensions. A full shader,execution re-sweep is due to get
the true post-fix total (batch-5 alone: 74,228 -> 48,428).

## Clean consistent 4-area sweep (2026-07-06, yawgpu a94ab06) — for README table

Per-subcase, crocus, workers=2, raw (no expectations), 2 files quarantined
(F-126 zero-dim indirect wedge):

| area | pass | skip | fail | crash |
|---|---|---|---|---|
| api/validation (124) | 194,805 | 157,163 | 347 | 0 |
| api/operation (67) | 132,831 | 76,698 | 19,932 | 0 |
| shader/execution (239) | 217,273 | 516,424 | 100,965 | 2 |
| shader/validation (207) | 369,753 | 297,389 | 0 | 0 |
| TOTAL | 914,662 | 1,047,674 | 121,244 | 2 |

Published in webgpu-native-cts/README.md (commit 466ec89). shader/execution
100,965 (down from 151,140 first-sweep, via multi-bind-group + layered DS
attachments); remaining is texture-sampling correctness (textureGather
family biggest). api/operation 19,932 (command_buffer copy correctness).

## Tier-1 sampling: CUBE investigation — HARD, reverted to decision point (2026-07-06)

shader,execution sampling mismatches split: dim="cube" ~13.7k (single
root cause) + non-cube ~85k (gather/compare/sample, heterogeneous).
Started with cube. Root cause: GLES creates a 2d/6-layer cube-viewable
texture as TEXTURE_2D_ARRAY; a Cube view binds samplerCube which needs
TEXTURE_CUBE_MAP => incomplete => samples 0.

Attempted fix (Codex, 25 files, saved as
webgpu-native-cts/transcripts/cube-wip-reverted.patch): plumb
textureBindingViewDimension from the C descriptor chain through core to
HalTextureDescriptor; GLES creates TEXTURE_CUBE_MAP + per-face
glTexSubImage2D upload + cube bind. REVERTED because it does not fix CTS
and left a failing test:
1. **CTS does not set textureBindingViewDimension** (grep: only in
   reflection.spec.cpp, NOT the texture-builtin harness) — the CTS
   oracle is Dawn on Metal/Vulkan, which cube-views any 6-layer 2d
   texture without the hint. So CTS-created cube textures arrive with no
   signal => created as 2D_ARRAY => cube view returns HalError
   "cube texture views require a cube-compatible texture".
2. **Even WITH the hint** (the hand-written test set it): creation =
   CUBE_MAP, all 6 faces upload GL-clean (verified with probes:
   face_target 0x8515-0x851a, err=0), bind CUBE_MAP err=0 — but sampling
   still returns ~0. A residual cube-completeness/sampling issue remains
   unsolved.

### DECISION NEEDED — cube strategy on GLES
GL coples storage+view and ES 3.1 has no glTextureView, so a 2d-array
texture cannot be aliased as cube. Options:
- (A) CTS-port fix: set textureBindingViewDimension=cube for cube tests
  (like the device_lost storage-skip fix — legitimate, matches what a
  GL-targeting app must do) + debug the residual completeness issue.
  Full cube support (~13.7k) but two problems to finish.
- (B) Heuristic: create every cube-compatible 6-layer 2d texture as
  CUBE_MAP. Risks 2d-array views of the same texture (WebGPU default
  binding view for 6-layer 2d is 2d-array, not cube).
- (C) Catalogue cube as a permanent-ish Tier-2 GLES gap and pivot to the
  larger non-cube sampling clusters (~85k: gather/compare/sample), which
  are independent of the GL cube limitation and may be more tractable.

## Non-cube sampling (option C) characterization — deep iterative tail (2026-07-06)

Pivoted to non-cube per user (C). Characterized the ~58k non-cube
sampling fails (crocus, from the clean sweep):
- 3d textures: PASS (16,434, 0 fail) — 3d sampling is correct.
- plain 2d: mostly pass.
- 2d-ARRAY textures (array_2d/arrayed_2d/sampled_array_2d): MIXED
  pass/fail. Not "returns 0 wholesale" — e.g. textureSampleLevel on a
  rgba8unorm 2d-array: "call 0 component 1: expected 0.945, got 0" —
  a specific COMPONENT/layer/coord is wrong while others pass. Subtle.
- DEPTH textures (depth_2d/depth_array_2d + textureSampleCompare/
  CompareLevel/GatherCompare ~24k): the GLES sampler ALREADY sets
  TEXTURE_COMPARE_MODE=COMPARE_REF_TO_TEXTURE + FUNC correctly
  (gles/sampler.rs:108), so the failure is depth-data-upload or
  depth-texture binding for shadow sampling, not a missing compare mode.
- array_3d_coords / arrayed_3d_coords (all-fail): cube-array (unsupported
  ES 3.1, catalogued).

Finding: the remaining sampling correctness (cube AND non-cube) is a
DEEP, SUBTLE, MULTI-BUG iterative tail — component/layer-specific 2d-array
errors, depth-shadow data/binding, cube. Each is a separate
hardware-loop debug (Codex proposes; the orchestrator supplies the real
failing values and verifies) with uncertain per-round yield, NOT the
clean structural wins of the earlier campaign (bind-groups, MSAA,
storage, copies, feature/limit truthfulness). This is the genuine
long tail of Tier-2 GLES conformance.

## glTextureView flexible views DONE (2026-07-06) — cube solved the Dawn way

Per user directive ("how dawn solves them? ... same thing on cubes"),
studied Dawn's opengl backend (`TextureGL.cpp`) and matched its
flexible-views strategy instead of cataloguing cube as a gap.

**Key discovery:** the crocus target actually reports **OpenGL ES 3.2**
with `glTextureView` + `texture_cube_map_array` available (an EGL+GLES
probe confirmed `GL_VERSION="OpenGL ES 3.2 Mesa 23.2.1"`,
texture_view=YES). The earlier "ES 3.1 has no glTextureView" assumption
was wrong; the reverted `textureBindingViewDimension` approach was
unnecessary.

Implementation (Codex, HAL-only): adapter caps `supports_texture_view` /
`supports_cube_map_array`; manually-loaded `glTextureView` proc (EGL +
WGL); bind path creates a transient GL texture view aliasing the base
`TEXTURE_2D_ARRAY` storage for cube / cube-array / array-layer subrange /
stencil-only / color-format-reinterpret bindings, matching Dawn's
`RequiresCreatingNewTextureView` / `CreateView`. No
`textureBindingViewDimension` hint needed (CTS never sets it — its Dawn
oracle uses flexible views). Fallback: retain the prior `HalError` when
glTextureView is unavailable (true ES-3.1 Tier-2 gap). block-67 updated.

**Stale-test discovery (root cause of the "cube returns 0" symptom):**
6 GLES HAL compute-pass unit tests were passing `binding_remaps:
Vec::new()`. Since the multi-bind-group commit 85b4880, an empty remap
means `flat_binding()` returns None → the SSBO/storage-texture binding is
silently skipped → the shader writes nothing → the test reads stale
garbage. They passed in no-GPU CI (device skip) but had been **failing
silently on hardware since 85b4880**. Fixed all 6 with correct
`HalGlesBindingRemap` entries (test-only). This harness bug — not the
cube code — produced the misleading `[1,0,0,...]` / all-zero readbacks.

Verified on crocus (ES 3.2): `cargo test -p yawgpu-hal --features gles`
= **172/172 pass** (was 166/172 with the 6 stale failures);
`submit_compute_pass_samples_cube_view_from_2d_array_texture_view`
samples all 6 cube faces correctly. Workspace Noop green, gles clippy
clean.

**CTS cube delta MEASURED (crocus, workers=2, commit 2ba7e96):**
`textureSample:sampled_3d_coords` (the cube cluster, 27,405 subcases):
pass=6237 **fail=0** crash=0, skip=21168. Broken out by dim:
- **dim="cube": 2673 pass / 0 fail** (was the dominant fail cluster —
  previously HalError "cube texture views require a cube-compatible
  texture" / sampled 0).
- dim="3d": 3564 pass / 0 fail (unchanged, already correct).
- The 21,168 skips are legitimate unsupported-format skips (r16/rg16/
  rgba16 (s)norm — not GLES formats), identical across both dims; NOT
  failures.
Cube sampling is now conformant on the textureSample builtin. The rest
of the ~13.7k cube estimate spans the sibling texture builtins
(textureSampleGrad / textureSampleLevel / textureGather with dim="cube"),
which take the same glTextureView bind path — a full texture-builtin
re-sweep will quantify the total.

## Cube-array pipeline creation FIXED (2026-07-06) — GLSL ES 3.2 when cube-array used

After plain-cube landed, a cube-family sweep across all sampling builtins
showed plain cube (dim="cube") fully passing (0 fail everywhere) but
**cube-array (dim="cube-array") ALL-failing (~21.6k, 0 pass)** — every fail
was "render/compute pass requires a valid render pipeline", i.e. pipeline
CREATION failure, not sampling correctness.

Root cause (probed on hardware): the yawgpu-tint shim hardcoded the GLSL
writer to ES 3.1 (`options.version = Version()` → `#version 310 es`).
`samplerCubeArray` (and the isampler/usampler/shadow/image variants) is
illegal in `#version 310 es` without `GL_EXT_texture_cube_map_array`, and
Tint's GLSL printer emits the type but never the extension → compile error
"illegal use of reserved word 'samplerCubeArray'" → error-pipeline. Driver
probe: v310-noext FAILS, v310+ext OK, v320 OK. Dawn passes the real GL
context version to Tint (ShaderModuleGL.cpp:437), so on ES 3.2 it emits
`#version 320 es` where cube-array is core.

Fix (yawgpu-tint shim only, no dawn submodule edit, zero blast radius on
non-cube-array shaders): `yawgpu_tint_generate_glsl` now inspects the
lowered IR for a `kCubeArray` texture dimension and, only then, sets
`options.version` to ES 3.2; all other shaders stay ES 3.1. Chosen over
full device-version plumbing to avoid bumping every shader to 320.

Verified on crocus — all 8 cube-array clusters now pass, 0 fail (was all-
fail): textureSample 594, textureSampleLevel 1782, textureSampleGrad 1782,
textureSampleBias 594, textureGather 6804, textureSampleCompare 1440,
textureSampleCompareLevel 4320, textureGatherCompare 4320 = **21,636
subcases FAIL→PASS**. yawgpu-tint 59/59, workspace Noop green.

### Remaining cube tail (next)
- **depth-cube (non-array)**: textureSampleLevel:depth_3d_coords 810,
  textureGather:depth_3d_coords 135, textureGather:depth_array_3d_coords
  270 — mix of pipeline-creation fails (samplerCubeShadow + explicit-LOD /
  shadow-lod) and depth-sample-returns-0 (the depth-shadow data/binding
  bug, shared with depth-2d).
- Non-cube long tail unchanged: 2d-array component/layer correctness;
  depth-2d shadow sampling returns 0 (~24k).

## Post-cube texture-builtin residual — FULL characterization (2026-07-06)

After plain-cube + cube-array landed, a texture-builtin re-sweep (crocus,
workers=2, commit 64151a9) shows the sampling correctness tail is
essentially gone — the old "textureGather 23.5k / 2d-array component-layer"
clusters now PASS wholesale (textureSampleLevel:sampled_array_2d 14256/0,
textureGather:sampled_2d 13608/0, textureSampleGrad 38313/0,
textureGatherCompare 46800/0, textureSampleCompare 16560/0,
textureSampleCompareLevel 49680/0 — all 0 fail). Those were fixed earlier
in the campaign (multi-bind-group / view work); the stale characterization
in 89c7182 predated a rebuild.

Remaining texture-builtin fails, fully categorized by root cause:

1. **Raw depth read (Tint GLSL-backend gap, catalogued in block-67)** —
   the LARGEST residual: textureSample 885, textureSampleLevel 2610,
   textureGather 3105 = ~6.6k, ALL depth-format (depth16unorm /
   depth24plus(-stencil8) / depth32float(-stencil8)), zero non-depth fails.
   sampler2DShadow-for-non-compare. Fix = Tint depth modelling (host-risky).

2. **Vertex-stage storage images (GLES hardware limit)** — storage-texture
   textureLoad/Dimensions/NumLayers pipeline-creation fails are dominated by
   stage="v" (e.g. storage_textures_2d_array 768 of 1056 fails are vertex
   stage). GLES 3.1 does not guarantee image load/store in the vertex stage
   (`GL_MAX_VERTEX_IMAGE_UNIFORMS` is commonly 0, as on crocus); such
   pipelines cannot link. Legitimate Tier-2 hardware gap — should be a clean
   HAL rejection, not a surfaced pipeline error. Catalogue in block-67.

3. **Non-required storage formats (GLES spec limit)** — rg32uint/sint/float
   storage images (144 each) are not in the GLES 3.1 required
   image-format list; imageLoad/Store on them is unsupportable. Catalogue.

4. **1D textures (GLES has no 1D)** — texture_1d / texture_storage_1d
   scattered fails (textureLoad:storage_textures_1d 88, textureNumLevels
   texture_1d "expected 1 got 0", textureDimensions 1d). GLES has no
   TEXTURE_1D / image1D; WebGPU 1D must be emulated as height-1 2D. Broader
   emulation gap — separate slice if pursued.

5. **texture-metadata for 1D** — textureNumLevels 123 / textureNumSamples 12
   "metadata query mismatch ... texture_1d expected 1 got 0". The metadata
   UBO returns 0 levels for 1D (tied to #4's 1D handling). Small.

6. **layered storage-view subrange** — storage_textures_2d_array 416
   "storage texture views must bind a whole layered view or one layer"
   (queue.rs:839). A HAL restriction on partial layered storage views; may
   be over-strict vs. glBindImageTexture(layered=false, layer=N) per-layer.
   Investigate if pursued.

7. **multisampled/arrayed textureLoad value** — textureLoad "expected bits"
   arrayed 192 / multisampled 96 / sampled_2d 48. Value mismatches; needs
   per-case investigation.

Net: cube (~40k across the family) + cube-array (21,636) are the two big
structural wins this session. The remaining ~10k is (1) the catalogued
Tint depth gap (~6.6k) and (2) scattered GLES hardware/spec limits + 1D
emulation + a few small value bugs — the genuine best-effort Tier-2 tail,
each an independent investigation with no single large lever left.

## Texture-metadata UBO FIXED (2026-07-06, commit c06e516) — textureNumLevels/NumSamples

`textureNumLevels`/`textureNumSamples` returned 0 for a queried-but-not-
sampled texture: `bind_texture_metadata_ubo` built the UBO from the
combined-sampler list and bailed when there were no samplers (a pure-query
shader has none). Now the shim exposes Tint's `ubo_contents` slot layout
({offset, count, binding}) over the FFI, core maps each post-remap binding
back to WGSL group/binding, and the queue fills each slot at its Tint
offset with the bound texture's value (sample count for MS, else view mip
level count) — mirroring Dawn's UpdateTextureBuiltinsUniformData.
Verified crocus: textureNumLevels 123/123, textureNumSamples 12/12 (135
FAIL->PASS). 12 files; hal 172/172, core 455, workspace green.

### Refined remaining texture-builtin residual (post-metadata)
- Raw depth read (Tint gap, catalogued): textureSample 885 / SampleLevel
  2610 / Gather 3105 + depth textureLoad ~240 (texelFetch on depth uses the
  same sampler-shadow modelling). ~6.8k. Host-risky Tint fix.
- Vertex-stage storage images + rg32 formats (GLES limits, catalogued):
  the bulk of textureDimensions 443 / textureNumLayers 114 / storage
  textureLoad. Not fixable on this hardware.
- Layered storage-view subrange (queue.rs:839) 416 — HAL may be over-strict,
  but storage in vertex stage also blocks many; low tractable yield.
- Non-depth textureLoad value: multisampled 96 / sampled_2d 48 (format-
  specific: rgba8(u/s)norm, rgba16float, r32float 4 each) / storage_3d 16 —
  ~160 scattered small value bugs, each a separate per-format/MS investigation.
Net after this session: no single large lever remains; the tail is the
catalogued Tint depth gap + GLES hardware limits + ~160 scattered per-format
value bugs.

## FINAL shader,execution re-sweep (2026-07-06, yawgpu 15a9ddb) — aggregate

Full `shader,execution` re-sweep on crocus (workers=2, raw, one process at
a time) on the fully-fixed library. Aggregated from all gles-crocus-se-*.jsonl:

| metric | a94ab06 (pre) | 15a9ddb (post) | delta |
|---|---:|---:|---:|
| pass  | 217,273 | **308,373** | +91,100 |
| skip  | 516,424 | 516,424 | 0 |
| fail  | 100,965 | **10,129** | **−90,836 (~90%)** |
| crash | 2 | **0** | −2 |

The −90,836 comes from the texture-builtin batches expr-call-5 (48,428 →
7,022) and expr-call-6 (52,261 → 2,835): cube (glTextureView flexible
views), cube-array (GLSL ES 3.2), and texture-metadata UBO. crash 2 → 0
(the textureDimensions stencil-only cases no longer segfault in this
re-sweep). webgpu-native-cts README GLES table updated to these numbers
(shader/execution row swept on 15a9ddb; other three areas still a94ab06).

Residual 10,129 = catalogued Tier-2 boundaries: raw depth read ~6.8k (Tint
sampler2DShadow modelling), GLES hardware/spec limits (vertex-stage storage
images, rg32 formats, no 1D), ~100 multisampled textureLoad. No clean
large lever remains.

## Next campaign (user-approved 2026-07-07) — prioritized TODOs

Planned from the Windows session; **execution happens on the Linux CTS
host** (crocus, workers=2, one GPU process at a time) with the established
per-slice loop: unit gates → target-gles release rebuild → targeted CTS
file(s) → full-area re-measure → commit. User decisions taken during
planning: (a) the raw-depth-read Tint gap is ATTEMPTED via a shim-side IR
pre-transform (not left catalogued); (b) a Windows WGL/NVIDIA CTS sweep is
explicitly NOT in this campaign (future second-driver validation).

- **P1 — Re-sweep api,validation + api,operation (measurement first).**
  Published numbers are from a94ab06, which predates glTextureView
  flexible views (2ba7e96), cube-array GLSL ES 3.2 (64151a9), and the
  texture-metadata UBO (c06e516) — texture_view (210), render_pipeline
  (2,699) and parts of the command_buffer tail plausibly moved. Raw run,
  update this doc + the cts README table, re-cluster the command_buffer
  tail from fresh JSONL (feeds P3b). No code change; re-baselines all
  items below.
- **P2 — Depth raw-read shim transform (~6.8k, largest single lever).**
  Custom IR pass in `yawgpu-tint/shim/tint_shim.cpp`, inserted in the
  existing pre-writer transform pipeline (the shim already runs
  SingleEntryPoint/SubstituteOverrides and inspects lowered IR for
  kCubeArray): for each depth texture used ONLY by non-comparison
  builtins, rewrite its type to `texture_2d<f32>`, rewriting call results
  (scalar depth read = `.x`; textureGather shape unchanged). Mixed
  compare+non-compare use of one texture: out of scope initially, keep
  current behaviour + catalogue. Spec: supersede the block-67 catalogue
  entry. Tests: yawgpu-tint GLSL-output assertions (sampler2D for
  non-compare, sampler2DShadow still for compare) + real-EGL HAL sampling
  test. Verify on crocus: textureSample 885 / SampleLevel 2610 / Gather
  3105 / depth textureLoad ~240; comparison clusters must stay 0-fail.
  Risk note: shim-only rebuild — the host hang risk applies to full Tint
  rebuilds, not shim recompiles.
- **P3 — command_buffer copy-correctness tail (~19.3k, largest count).**
  3a: piece-5 resumption — FIRST verify checker semantics in
  webgpu-native-cts harness.cpp:601-646 (the saved replica flags a
  failure inside row padding at a byte offset that cannot be texel data,
  so the replica's expected-buffer stride math is suspect), then extract
  exact CTS subcase params (--list-cases) and replicate one verbatim
  before touching production code (WIP patch:
  webgpu-native-cts/transcripts/slice4b-piece5-slicestride-wip.patch).
  3b: re-cluster the remaining tail (T2T format-conversion families
  suspected) from the P1 sweep. Discipline: one path change at a time,
  full command_buffer re-run per change, beat-the-checkpoint (19,331).
- **P4 — MSAA per-sample behavioural correctness (sample_mask, ~2.6k+).**
  Characterize failing mask/alpha-to-coverage subcase patterns first,
  then fix the GLES MSAA path (glSampleMaski / alpha-to-coverage state);
  real-EGL unit tests per fixed behaviour.
- **P5 — Missing feature: 1D texture emulation as height-1 2D** (Dawn
  precedent). Map D1 textures to GL_TEXTURE_2D height=1 (storage 1D to
  image2D coords); confirm Tint's GLSL 1d emission shape in the shim
  before handoff. Unlocks storage_textures_1d 88 + texture_1d
  load/dimensions/levels clusters. Update block-67 matrix 1D rows.
- **P6 — Small over-strict / investigable batch (~800 total, independent
  items):** layered storage-view subrange (gles/queue.rs:839, 416 —
  try glBindImageTexture(layered=GL_FALSE, layer=N) for single-layer
  views); maxBindingsPerBindGroup off-by-one (~40 — subtract reserved
  binding points from the GL-queried limit); memory_sync "binding size
  exceeds GLES limit" (~230 — verify which GL limit and comparison);
  depth32float-stencil8 copy bytes_per_row (57 — spec-check; if the rule
  lives in core, Tier-independence applies: fix only if wrong for ALL
  backends, never relax core for GLES); indexed-indirect nonzero
  index-buffer offset (54+72 — investigate emulation, else close as
  catalogued).
- **P7 — Hygiene (fold into any slice):** clean HAL rejection for
  vertex-stage storage images (instead of surfaced GL link error);
  fix stale block-67 entry ">1 bind group — DEFERRED" (landed in
  85b4880); optional hand-written GL repro for the Mesa
  textureSize-on-stencil-mode crash (suspected→confirmed, F-126
  precedent).

Explicitly deferred (not this campaign): Windows WGL/NVIDIA CTS sweep;
stencil-aspect T2B compute-path readback (908); Unorm8x4Bgra shader
swizzle for hosts without EXT/ARB_vertex_array_bgra; ~160 scattered
textureLoad value bugs (multisampled 96 / sampled_2d 48 / storage_3d 16);
the latent Drop-inside-context deadlock hazard (watch item only).

Campaign end: clean 4-area sweep on one commit → update the cts README
GLES table (currently mixed a94ab06/15a9ddb baselines).

### Dawn-reference findings for P2 / P3a (2026-07-07, from the full Dawn checkout)

Investigated how Dawn handles both problems (checkout: `../../C/dawn`;
line numbers are from that checkout — re-verify against the pinned
`third_party/dawn` submodule before coding, versions may differ).

**P2 — Dawn does NOT solve raw depth reads on GL.** Tint's GLSL printer
unconditionally appends "Shadow" for any `DepthTexture`
(`printer.cc:993-995`) and `TexturePolyfill` injects a fixed comparison
ref of `0.0` for textureSample/SampleLevel/Gather on depth
(`texture_polyfill.cc:766-768, 849-896`) — the sampler's
kSampler-vs-kComparisonSampler kind is never consulted. Dawn avoids the
broken path at the API layer instead: its GL backend only supports
**Compatibility mode** (`PhysicalDeviceGL.cpp:554-556`), and Compat
validation REJECTS pipelines using `texture_depth_*` with a
non-comparison sampler (`Pipeline.cpp:142-146`, driven by the WGSL
inspector flag `has_depth_texture_with_non_comparison_sampler`,
`inspector.cc:673-679`). Consequences for the P2 slice:
- There is no Dawn code to port; the shim IR pass goes beyond Dawn.
- BUT Tint contains the exact rewrite machinery as a copyable template:
  `TexturePolyfill` already rewrites depth → `texture_2d<f32>` (+ `.x`
  swizzle on loads) for the sampler-less `textureLoad` path
  (`texture_polyfill.cc:343-347, 606-674`). The shim pass extends that
  pattern to depth textures whose only sampled uses are non-comparison
  builtins. Feasibility upgraded: type-replacement + call-site fix-up
  precedent exists in-tree.
- If the shim pass fails, cataloguing is well-defended: even Dawn cannot
  express this on GL and forbids it in Compat mode.

**P3a — Dawn has a definitive strategy to mirror.** On GLES, snorm (and
depth/stencil/rgb9e5/float16/32 per toggles,
`PhysicalDeviceGL.cpp:428-500`) T2B copies NEVER use glReadPixels: the
frontend rewrites them at encode time into the shared compute blit
(`ShouldUseTextureToBufferBlit`, `CommandEncoder.cpp:1094-1160` →
`BlitTextureToBuffer.cpp`). The blit's load-bearing details — the direct
checklist for our r8snorm/3d rows_per_image>height family:
- dst offset math: z-slice stride = `bytesPerRow * rowsPerImage`
  (`BlitTextureToBuffer.cpp:224-235`; sub-4-byte formats add a
  `shift = (offset%4)/bytesPerTexel` variant, lines 274-335).
- Padding-byte protection #1: every partial u32 (row start/end,
  offset%4≠0) is masked read-modify-write
  (`(original & mask) | (encoded & ~mask)`, e.g. lines 426-428) — whole
  words in the interior are written directly.
- Padding-byte protection #2: for non-compact copies
  (`blocksPerRow != copyWidth || rowsPerImage != copyHeight`) Dawn
  allocates a 4-byte-aligned intermediate storage buffer and PRE-COPIES
  the entire destination region into it before the dispatch, then copies
  back — so row/image padding bytes always survive (lines 1165-1240,
  1372-1377). Our compute-T2B fallback (slice 4) should be audited
  against exactly these three points.
- Direct glReadPixels path (renderable formats only): row stride via
  `GL_PACK_ROW_LENGTH = blocksPerRow`; NO pack IMAGE_HEIGHT on ES —
  slice stride is manual: per-layer `glFramebufferTextureLayer` +
  `offset += blocksPerRow * rowsPerImage * byteSize`
  (`CommandBufferGL.cpp:1033-1050`).
- B2T upload (relevant to reverted piece-4): Dawn does NOT row-by-row
  loop; it uses `GL_UNPACK_ROW_LENGTH` + `GL_UNPACK_IMAGE_HEIGHT`
  (unpack side exists in ES 3.0) with a single glTexSubImage3D
  (`CommandBufferGL.cpp:1851-1896`); row-by-row is reserved for the
  WriteTexture-only case `bytesPerRow % texelByteSize != 0`.
