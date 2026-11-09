# Backlog — remaining TODOs and missing features, by priority and difficulty

**Snapshot date: 2026-11-02** (tree at `48067a3`; supersedes the
2026-09-23 snapshot at `2c6ea6f`, whose progress log is kept below as
history). This is a cross-cutting inventory, not a plan: it collects
the open / deferred / partial items recorded across `specs/blocks/`,
`specs/tracking/`, `README.md`, and the source, then ranks them. Items
the docs mark DONE / RESOLVED / COMPLETE are omitted; the previous
snapshot's closed rows are listed once under "Closed since the previous
snapshot".

Re-take this snapshot rather than editing rows in place when the state
changes; per-item ledgers stay in their own tracking docs.

## Method

- Every Block 108-120 status line was read. Their open items, the 2026-10
  code review ledger, and the rows of the previous snapshot were carried in.
- Checked against the source at this tree: A3, A6, A7, A10, B2, B3, C1, C2, C4, D2,
  D3, D8, D13, E1, E6, E7, E10, the `TODO` markers, and the export set.
- **Carried from the previous snapshot without a source check:** B6, B7, D1, D4 to
  D7, D9, E2 to E5, E8, E9. Re-confirm these when their work is planned.
- This pass did not repeat the full-doc grep or re-read every Phase Review table.
- Headline counts (2026-11-02): 3 `TODO`s (two `tiled 2.4` Transient arms,
  one GLES `HalError: Clone` note), 0 `FIXME`/`unimplemented!`/`todo!`; 463
  `#[ignore]` tests (331 in `yawgpu/tests`, 132 in `yawgpu-hal`); 202 of 202
  `webgpu.h` symbols exported, plus Dawn's `wgpuCommandEncoderWriteBuffer`.
- CTS state: Metal, Linux native Vulkan, and (2026-11-03, at `d7ff142`) Windows native
  Vulkan all `fail=2 crash=0`. Both failures
  are cases Dawn also fails (Block 119 R9). GLES on Linux NVIDIA is raw
  `fail=28,810 crash=0`.

**Difficulty scale:** S = under a day, M = several days, L = a week or
more. "Manual" = blocked on hardware or a host this machine cannot
reach.

**Priority rationale:** silently-wrong behaviour that the CTS cannot
catch ranks above a missing standard API, which ranks above
platform/presentation coverage, then perf/refactor, then doc drift. By user
direction (2026-11-02), three groups rank below all of that, in this order: CI
(C4), binding arrays (B2), and Tier-2 GLES gaps (D, plus A3, A7, C3). The letters
below are stable IDs, not the rank.

---

## Closed since the previous snapshot (`2c6ea6f` → `48067a3`)

| Prev # | Item | Closed by |
|---|---|---|
| A6 | Metal rejects a non-default `multisample.mask` | Stale row: core has baked the mask into MSL since `462b4d8`; `threading-audit.md` now says so |
| — | Subgroup size control | Block 108, see F2 for its owed Mac notes |
| — | 2026-10 code review: CRITICALs and MAJORs | Blocks 109-112, except G2 (D12) |
| — | 2026-10 bug-class audit | Blocks 113-115 |
| — | Tint double IR lowering; Vulkan per-submit churn | Blocks 116, 117 |
| — | Dawn bump to v20261030 | Block 118 |
| — | CTS findings F-161 to F-164 | Block 119 |
| — | Tint driver workarounds from Block 118 | Block 120, see A12 |
| E1 | Block 97 S4 re-measure | Run 7, 2026-11-02 |
| E13 | P-005, work-done flushed pending writes | Block 121, 2026-11-02 |
| E14 | P-006, staging recycle per write; `compute_unique` explained | Block 122, 2026-11-02 |
| A15 | Metal MSL math mode differed from Dawn | Block 123, 2026-11-02 |
| A16 | Metal `preserveInvariance` always on | Block 124, 2026-11-02 |
| — | F-151 real-GPU confirmation | Metal CTS case passes, 2026-11-02 |
| A12 | Block 120 on native Vulkan | Windows native Vulkan e2e and forced CTS, 2026-11-02 |
| A18 | Vulkan buffers were not zero-initialized | Block 129, 2026-11-03 |
| A22 | Valid render passes failed the submit (Block 130 regression) | Block 132, 2026-11-03 |
| A23 | `WGPUAdapterInfo` reported constants (vendor/device ids 0) | Block 135, 2026-11-03 |
| E6 | Vulkan limits: NVIDIA storage cap, color bytes per sample, fragment combined output resources | Block 134, 2026-11-03 |
| A10 | A pass that failed to lower was dropped silently | Block 130, 2026-11-03 |
| A13 | Dawn's vendor-gated Tint workarounds were unset | Block 131, 2026-11-03 |
| A14 | Device-lost reason at teardown. The premise was wrong: yawgpu already destroyed on last release. The real gap was spontaneous delivery, and the bench exit code was a bench bug | Block 130, 2026-11-03 |
| F1 | `specs/blocks/README.md` indexed 12 of the block files | Full index, 2026-11-04 |
| F2 | Block 108 Mac follow-up | Metal compile covered; MoltenVK `computeFullSubgroups = true`, e2e 3/3, 2026-11-04 |
| F3 | Block 109 sync-validation owed | Satisfied by the Windows run at `decfd69` (contains the R9 fix), 2026-11-04 |
| A20 | Spontaneous callbacks ready at registration waited for `ProcessEvents` | Block 137, 2026-11-04 |
| A19 | Releasing the instance while a device was alive cancelled every callback | Block 138, 2026-11-04 |
| E12 | Remaining 2026-10 review MINORs | Block 140, 2026-11-04 (WaitAny backoff; m5 already done; the rest triaged into A24, A25, or closed) |
| E7 | Metal subgroup size range and f16 review MINORs | Block 140, 2026-11-04 (Dawn's ranges, `subgroup-size-control` on Apple GPUs) |
| A26 | Metal `subgroups` gate and size range differed from Dawn | Block 141, 2026-11-05 (`Apple6 || Mac2`; range always reported) |
| B8 | `texture-compression-unaligned` and `atomic-vec2u-min-max` were not advertised | Block 141, 2026-11-05 (Metal, Vulkan, Noop; the standard feature list now matches Dawn on the M2) |
| A25 | Metal textures were `Shared` | Block 142, 2026-11-05 (`Private`; `Memoryless` only for transient attachments on Apple2) |
| A24 | `destroy()` kept GPU memory until the last handle dropped | Block 143, 2026-11-05 (buffers, textures, and query sets release the backend object at `destroy()`; the HAL frees it when the last submission using it completes) |
| A27 | External texture planes were not lazily zero-initialized | Block 144, 2026-11-05 (planes get the same lazy clears and initialization marks as sampled texture views) |
| A28 | External texture planes were bound as whole textures | Block 145, 2026-11-05 (each plane is a `HalBoundTexture` view; Metal binds the view) |
| D14 | README Linux NVIDIA GLES table was older than the suite's | README updated to the suite's 2026-11-01 sweep at `9977e34` (fail 28,810), 2026-11-05 |
| E9 | Block 102 m12 cosmetic deferrals | Block 147, 2026-11-05 (test-module imports first, test-only helpers moved into test modules, `PassTimestampWrites` serves both pass kinds) |
| E10 | GLES copied `HalError` by hand | Block 147, 2026-11-05 (`HalError` derives `Clone`) |
| E8 | Shader passthrough had no Phase Review; SPIR-V input was not gated by the instance feature | Block 148, 2026-11-08 (review fixes, incl. Metal compute and render binding slots for WGSL too; `ShaderSourceSPIRV` required) |
| A29 | Subpass pipelines rejected external textures (the row's "dropped" premise was wrong: Block 146 rejected them at pipeline creation) | Block 149, 2026-11-09 (supported on Metal and Vulkan; the Metal subpass path now writes the immediates block last) |
| A30 | Depth formats passed as color targets with `writeMask` 0 and no shader output | Block 149, 2026-11-09 (color targets must be color-renderable, as Dawn) |

---

## Priority A — correctness (silent-wrong, invisible to the CTS)

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| A17 | **Native hardware owed for Blocks 126 and 127.** The Windows NVIDIA run took the feature-present paths. Still unexercised: the `frag_depth` shader clamp on a device without `VK_EXT_depth_clip_enable`, and the default compute subgroup size and invocation clamp on a device whose subgroup sizes vary (Intel). | Manual | `specs/blocks/126-vulkan-frag-depth-clamp.md`, `specs/blocks/127-vulkan-subgroup-parity.md` Verification |
| A21 | Intel Gen9 subgroups gate and `polyfill_subgroup_broadcast_f16`. Dawn hides Subgroups and SubgroupSizeControl on Intel Gen9 unless `EnableSubgroupsIntelGen9` is set, and that toggle enables the polyfill. This needs the PCI device id on both backends. | S-M | `specs/blocks/131-dawn-vendor-gated-tint-workarounds.md` Out of scope |
| A3 | **Lazy zero-init GLES clear paths (Block 104 S4)** — core marks and un-marks on every backend, but the GLES HAL has no `ClearTexture` lowering for the Stage-2 eligibility set (sampled reads of uninitialized depth/stencil, compressed and MSAA textures). Tier-2 catalogue item; passes today only where the driver hands back zeroed memory. | M | `specs/blocks/104-lazy-zero-init-stage2.md` S4; `specs/blocks/67-gles-backend.md` mapping matrix |
| A7 | GLES `SetImmediates` is recorded and bounds-checked but never read by the draw path — a silent gap. Unreachable today because `max_immediate_size` is 0 and core rejects first; must become an explicit `HalError` if the limit is ever raised. | S | `specs/blocks/67-gles-backend.md:282`, `yawgpu-hal/src/gles/adapter.rs` |
| A11 | Block 107 F2 (documented, not a bug): `wgpuSurfacePresent` is the one consumer core does not lazily clear for, so a surface texture Discarded then presented shows `DONT_CARE` contents on Vulkan. Matches Metal and Dawn; listed so nobody re-adds the eager clear. | — | `specs/blocks/107-vulkan-discard-epilogue-shared-layout.md` R1 |

## Priority B — missing standard WebGPU surface

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| B2 | **Binding arrays**: `bindingArraySize > 1` rejected outright in core. Dawn also rejects it unless `AllowUnsafeAPIs` is on, supports it then only for sampled textures, and Tint's `sized_binding_array` is experimental, so today's rejection matches default Dawn (`BindGroupLayoutInternal.cpp:362-367`). | M–L | `yawgpu-core/src/bind_group_layout.rs` |
| B3 | **External textures on GLES.** Metal and Vulkan support them (Blocks 36, 146); GLES returns a `HalError`, because it has no lowering for the multiplanar plane and params bindings. | L | `specs/blocks/36-external-textures.md`; `specs/blocks/146-vulkan-external-textures.md` |
| B6 | Tier-2 read-write storage format breadth ambiguous against the current spec (yawgpu grants `R32Float`, `RGBA16Float`, `RGBA32Float`). Needs a spec check more than code. | S | `specs/tracking/format-completeness-audit.md:23-27` |
| B7 | Multi-threading correctness beyond what the ported tests exercise. | L | `specs/SPEC.md:53` |

## Priority C — platform / presentation coverage

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| C1 | **Android `ANativeWindow` surface on Vulkan is a stub** — returns `SwapchainCreationFailed("Android native window surface not implemented")`. | M | `yawgpu-hal/src/lib.rs` |
| C2 | **Xlib / Wayland / XCB surface sources are accepted by validation but inert** — no `VkSurfaceKHR` is created for them (only `CAMetalLayer` and Win32 HWND reach a Vulkan surface). Linux has no windowed presentation on any backend. | M each | `yawgpu/src/ffi/mod.rs`, `yawgpu-hal/src/vulkan/mod.rs`; `specs/blocks/85-windows-surface.md:269-279` |
| C3 | GLES Linux windowed presentation; the headless `EGL_PLATFORM_DEVICE_EXT` cascade must be skipped for window surfaces when it is added. | M | `yawgpu-hal/src/gles/egl.rs`; `specs/blocks/67-gles-backend.md:500-509` |
| C4 | **CI covers only default and `tiled`.** Feature-gated clippy (`vulkan` / `metal` / `gles` / `shader-passthrough`) never runs in CI, so lint rot reaches contributors first; no real-GPU CI for the 463 manual tests; no Windows example-build job. | M (infra) | `.github/workflows/ci.yml`; `specs/tracking/toolchain-clippy-1-98.md:110-126`; `specs/blocks/85-windows-surface.md:276-278` |

## Priority D — GLES Tier 2 residue

Latest raw CTS: Linux NVIDIA `fail=28,810 crash=0` (Block 119). Each row is an
independent investigation.

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| D12 | **GL context never released after `with_current_context`** (review G2), the last open MAJOR of the 2026-10 review. A second thread fails. Needs an ownership change, so it gets its own block. | M | `specs/blocks/111-code-review-hal-majors.md:152` |
| D13 | `khronos-egl` `make_current` can panic inside `Drop`, which breaks the no-panics principle. | S | `yawgpu-hal/src/gles/device.rs:212, 230` |
| D1 | **MSAA per-sample behaviour** — `multisample.mask` / `@builtin(sample_mask)` and alpha-to-coverage do not gate per-sample colour/depth writes (~1,510 fails). Documented as the next deep-debug slice. | L | `specs/tracking/cts-gles-sweep-0705.md:1206-1218` |
| D2 | **`maxBindingsPerBindGroup` off-by-one** — the limit is a plain `min` of GL maxima with no reservation for Tint's texture-metadata UBO. Mechanism confirmed, not fixed. | S | `specs/blocks/67-gles-backend.md:809-825` |
| D3 | Query sets: timestamp not advertised although `EXT_disjoint_timer_query` exists; `submit_resolve_query_set` writes zeroes from a software vector rather than real GL query objects. Block 102's core path (conversion pass, `HalQueryKind::Timestamp`) is ready for a GLES arm. | M | `yawgpu-hal/src/gles/queue.rs`; `specs/blocks/67-gles-backend.md:279` |
| D4 | Stencil-aspect texture-to-buffer readback (~900 fails); a compute-image path could lift it. | M | `specs/blocks/67-gles-backend.md:798-808` |
| D5 | 1D textures / `texture_storage_1d` — height-1 2D emulation (Dawn precedent), not started. | M | `specs/blocks/67-gles-backend.md:250, 910-913` |
| D6 | Texture views on hosts without `glTextureView` (copy-based emulation); texture-to-texture copy without `GL_EXT_copy_image`. | M–L / M | `yawgpu-hal/src/gles/queue.rs` |
| D7 | Cube / cube-array colour attachments; framebuffer fetch via `EXT_shader_framebuffer_fetch`; compressed-format advertisement. | S–M / M / M | `specs/blocks/67-gles-backend.md:254, 265`; `specs/tracking/format-completeness-audit.md:34-37` |
| D8 | Latent self-deadlock: `Drop` impls of GLES buffer / pipeline / sampler / texture inners acquire `with_current_context`; dropping the last `Arc` inside a context closure re-deadlocks. A guard is cheap. | S | `specs/tracking/cts-gles-sweep-0705.md:136-145` |
| D9 | `unorm8x4-bgra` renders R/B swapped on hosts without `EXT/ARB_vertex_array_bgra` (execution-only divergence; shader swizzle emulation deferred). | M | `specs/blocks/67-gles-backend.md:786-790` |
| D10 | **Manual verification owed**: ANGLE re-confirmation after the Tint migration. Windows WGL passed 38 of 39 e2e at the Block 118 pin; `e2e_gles_process_exit` crashes on every run because F-153's exit guard is armed only on the EGL path. | Manual | `specs/tracking/tint-integration-refactor.md:219-221`; `specs/blocks/118-dawn-v20261030-bump.md` |

Permanent / catalogued Tier-2 rejections (not backlog, listed for
completeness): vertex-stage storage images, `rg32*` storage formats,
cube-array without ES 3.2, `ClampToBorder`, `baseVertex != 0`,
indexed-indirect with non-zero index offset, per-target divergent
blend/write-mask, dual-source blend factors, `unclippedDepth`, stencil
reference `> i32::MAX`, `first_instance` indirect, external textures,
`tiled`, `shader-passthrough`. See `specs/blocks/67-gles-backend.md`
"WebGPU × GLES mapping matrix" and "CTS-confirmed Tier-2 catalogue".

## Priority E — performance, refactor, vendor extensions

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| E2 | Residual gaps vs Dawn with no structural lever: `write_buffer_then_wait` 1.53×, `bindgroup/create_destroy` 1.33×, `render_draw` 224 vs 99 ns. | Diffuse | `specs/tracking/perf-dawn-baseline.md:435-446` |
| E3 | Deferred cache layers: Metal `MTLLibrary`-by-source cache (measure-first); whole-module Tint IR clone (needs an upstream Tint API). A real `VkPipelineCache` was measured and dropped on 2026-11-03 (Block 136): Dawn's starts empty in standalone runs, and after the Block 136 fix yawgpu creates pipelines faster than Dawn on Windows NVIDIA. | M / L | `specs/blocks/95-shader-compile-cache.md:35-41`; `specs/blocks/136-msvc-tint-optimization.md`; `yawgpu-tint/src/lib.rs` |
| E11 | Vulkan `VkRenderPass` / `VkFramebuffer` / `VkDescriptorPool` caches, left out of Block 117. Run 7 cannot decide it: no bench case isolates per-pass creation, so add one that submits many passes without waiting. | M | `specs/blocks/117-vulkan-submit-recycling.md` |
| E4 | refactor-dedup deferrals: double `Arc<core::X>` in FFI Impl structs, Metal derived-view cache + per-draw scratch reuse, unified `HalRenderStreamState` interpreter, GLES draw-time dirty tracking + binding index maps, subpass attachment validation dedupe; **+ Block 107 F5**: a `RenderBindings<'_>` struct for the Vulkan render descriptor call chain (`update_render_descriptor_sets` `too_many_arguments`, `bind_textures` / `pass_textures` side by side). | M each | `specs/tracking/refactor-dedup.md:38-56, 98-101, 191-196`; `specs/blocks/107-vulkan-discard-epilogue-shared-layout.md` F5 |
| E5 | `tiled`: depth-format transient attachments, MSAA subpass input (Dawn-deferred; 2-arg `inputAttachmentLoad` parked on a Dawn resolver issue), queries inside subpass passes, GLES mapping, `SubpassAttachmentResource::Transient` arm TODOs (the two remaining `TODO(tiled 2.4)` markers).; a forced-flag e2e on the subpass path (Block 120). Subpass passes do no usage-scope validation for bind-group resources (`subpass.rs` never records a usage scope), so a resource bound and also used as an attachment of the pass is not rejected as in a regular render pass (Block 149 Phase Review m4). | M / L / M / L | `specs/blocks/55-tiled-rendering.md:45-53, 501-507, 639-648`; `yawgpu-core/src/subpass.rs:244`, `yawgpu-hal/src/command.rs:779` |

## Priority F — doc drift

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| — | None open. | | |

---

## Decided, not backlog

- Zero-count indirect dispatch on Mesa ANV Haswell wedges the machine. The user
  decided on 2026-07-05 not to implement the conditional-rendering workaround, and
  the affected CTS files stay quarantined on that host. Block 119 R5d adds a second
  route to the same hazard. See `specs/tracking/cts-full-sweep-0704-native-vulkan.md`.

---

## Recommended order

1. The Block 124 survey's larger items. SPIR-V 1.4 is done (Block 133). Extended dynamic
   state is off by default in Dawn, so yawgpu already matches; `VkPipelineCache` was
   measured and dropped (E3, Block 136).
2. **C2 / C1** surfaces as separate initiatives.
3. Lowered by user direction: **C4** feature-gated clippy in CI, then **B2**
   binding arrays, then **D12** and the rest of the GLES rows.

---

## Progress log

Entries below the 2026-09-23 snapshot line are history carried over from the `d6f865b` snapshot; row numbers in them refer to that snapshot's tables. New entries are appended at the end.

### 2026-09-22 — A4, A2, B1, A1 executed (Blocks 99–102)

| Item | Result |
|---|---|
| A4 | **Done.** Block 99: `6783728` (HAL), `55162d3` (e2e). Metal `timestamp_query` / `float32_filterable` and Vulkan `timestamp_query` / `depth32float_stencil8` / `rg11b10ufloat_renderable` / `bgra8unorm_storage` / `float32_filterable` now follow Dawn's device queries; the rest are documented literals. |
| A2 | **Done.** Block 100: `2f97d3d`. Encoder writes are snapshotted and lowered in command order through the staging pool. |
| B1 | **Done.** Block 101: `45d1942`. 202/202 header names resolve. |
| A1 | **Done (S1–S3), S4 in progress.** Block 102: `57c64e7` (core + Noop) + the Metal/Vulkan HAL commit that follows it. e2e 4/4 on M2 Metal and MoltenVK. |

New findings recorded while executing (each with evidence in the block spec):

- **Metal end-of-encoder counter sampling fails after a blit encoder** (Apple8 / macOS 26): Dawn's `endOfEncoderSampleIndex` form errors the command buffer; yawgpu samples at `startOfEncoderSampleIndex` instead. Watch when Dawn or macOS changes (Block 102 R2).
- **MoltenVK cannot translate a struct load from a storage buffer** in Tint's robustness-clamped SPIR-V (`no matching constructor for initialization of 'Timestamp'`); the conversion shader uses a flat `array<u32>` (Block 102 R5). Any future internal shader must avoid whole-struct loads from `var<storage>`.
- **A9 (new, S) — fixed `288887a`:** Metal `encode_compute_buffer_sizes` / the render equivalent pass `setBytes` with `N * 4` bytes for `tint_storage_buffer_sizes`, but Tint declares `uint4[ceil(N/4)]` (16-byte elements): the Metal API validation layer reports `argument tint_storage_buffer_sizes[0] ... has space for 4 bytes, but argument has a length(16)` on every pipeline that uses `arrayLength`. Harmless on hardware (only the first lanes are read) but an API-validation error; pad the byte slice to a multiple of 16. Found by running the timestamp e2e under `METAL_DEVICE_WRAPPER_TYPE=1`. Evidence: `yawgpu-hal/src/metal/encode.rs` `msl_buffer_sizes`; `third_party/dawn/src/tint/lang/msl/writer/raise/raise.cc:129`.
- **A5 correction:** the Vulkan HAL *does* barrier around compute passes since F-106 (`c723a82`); Block 102 widened those scopes rather than adding new ones. A5's per-subresource layout item stands.
- **Metal API validation layer is worth a periodic run**: `METAL_DEVICE_WRAPPER_TYPE=1 MTL_DEBUG_LAYER=1` on the e2e suites surfaced A9 immediately; the CTS runs do not enable it.

### Phase Review — Blocks 99–102 + A9 (2026-09-22, fresh-context reviewer over `d6f865b..HEAD`)

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| C1 | CRITICAL | Metal timestamp-resolve serialization took its shared-event value at encode time on a device-wide counter; `submit_copies` encodes outside `submission_lock`, so two concurrent submits can commit in reverse value order — the lower value neither waits nor keeps the event monotonic, silently defeating the serialization | **Fixed** — per-submission `MTLSharedEvent` (values `1..k` inside one command buffer), no cross-submission comparison; spec R2 updated |
| M1 | MAJOR | `CommandEncoder::write_buffer` copied the payload before validation (unbounded allocation / abort on an absurd `size` before the validation error); FFI `usize→u64` check vacuous on 64-bit | **Fixed** — validate first, copy on success; FFI rejects `size > isize::MAX` |
| m1 | MINOR | Spec 102 params examples contradicted the formula and code | **Fixed (spec)** — `(42666, 10)` / `(42666, 9)` |
| m2 | MINOR | Block 100 staging-failure branch untested (Noop cannot fail allocation) | **Fixed** — `#[cfg(test)]` fault-injection seam on `PendingWriteBatch` + test (or documented if the seam did not fit; see the fix commit) |
| m3 | MINOR | `debug_assert!` guarded dropped copies in release | **Fixed** — explicit internal error |
| m4 | MINOR | A timestamp conversion pass that fails to lower would be dropped silently (raw ticks left in place) | **Deferred with rationale** — the internal pipeline is non-error by construction and its bind group binds a submit-validated buffer, so `hal_compute_pass_execution` cannot return `None` for it today; surfacing an error requires threading a result through the whole lowering walk. Tracked here; revisit if a second internal pass appears |
| m5 | MINOR | Metal `create_query_set` allocated the sample buffer before the sampling-mode check and bumped the allocation counter on failure | **Fixed** |
| m6 | MINOR | Vulkan occlusion resolves paid the new transfer→compute barrier | **Fixed** — gated on `HalQueryKind::Timestamp` |
| m7 | MINOR | 2 ms calibration sleep on every Metal adapter enumeration | **Fixed** — lazy `OnceLock` on first `timestamp_period()`; spec R1 amended |
| m8 | MINOR | `create_internal_bind_group` caller obligation undocumented | **Fixed (docs)** |
| m9 | MINOR | Spec 101 cited a non-existent `yawgpu/ffi/yawgpu.h`; only 2 of 19 vendor exports asserted | **Fixed** — spec corrected, test asserts every vendor export |
| m10 | MINOR | `#[must_use]` on 2 of 6 sibling recorders | **Fixed** — removed for uniformity |
| m11 | MINOR | A9 pad-then-truncate round trip | **Fixed** — pad at the two `setBytes` sites |
| m12 | MINOR | (a) "S1 placeholder" wording; (b) tests inserted above `use super::*`; (c) `#[cfg(test)] pub(crate)` shim; (d) `RenderPassTimestampWrites` name reused for compute | (a) **Fixed**; (b)(c)(d) **Deferred** — cosmetic; (d) is a pre-existing type reuse the render/compute descriptors share, renaming is a cross-crate API change outside this phase |

Reviewer also confirmed (no finding): chunk retirement against the consuming submission, no lock re-entrancy in the lowering walk, timestamp written-query clamping, conversion params/WGSL byte-equivalent to Dawn `QueryHelper.cpp`, no new panic paths in library code, 202/202 header symbols exported.

### 2026-09-22 (later) — B5, B4, A3 in progress

- **B5 done (Block 103)**: surface capabilities come from the HAL (Metal = Dawn's list; Vulkan = driver query, deduplicated per format). Follow-ups from the MoltenVK validation run, pre-existing: swapchain view usage ⊄ image usage (VUID 02662), present semaphore destroyed while pending (VUID 05149), zero `currentExtent` accepted (VUID 01689). Fix as one small Vulkan task after Block 104 S3.
- **B4 closed as stale** (`62338d8`).
- **A3 (Block 104)**: S1 core landed (`60744d8`); S2 Metal / S3 Vulkan HAL clears in flight; e2e files `e2e_{metal,vulkan}_lazy_init.rs` written.
- **A3 (Block 104) S1–S3 done**: core `60744d8`, Metal `99d786b`, Vulkan `9aa1da3`; e2e 6/6 on both GPUs, validation layers clean; CTS Metal 224,036/0, MoltenVK unchanged vs the `a2907ca` baseline (same 9 documented artifacts). Vulkan validation follow-ups (incl. the A5 "aspect-narrowed barrier" S item — now fixed by `barrier_aspect_mask`) in `974a818`. GLES clear paths (S4) remain Tier-2 catalogue items.

### Phase Review — Blocks 103 + 104 (2026-09-22, fresh-context reviewer over `8d2b602..0259324`)

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| C1 | CRITICAL | `CAMetalLayer.setDisplaySyncEnabled` is macOS-only (`API_UNAVAILABLE(ios, …)`); objc2 does not cfg-gate it, so every `wgpuSurfaceConfigure` would abort on iOS with "unrecognized selector" | **Fixed** — `#[cfg(target_os = "macos")]` as Dawn `SwapChainMTL.mm`; iOS `cargo check` still compiles (review-only catch); spec 103 R4 |
| C2 | CRITICAL | A depth-stencil attachment view narrowed to one aspect of a combined texture (`stencil8`/`StencilOnly` of `depth24plus-stencil8`) was accepted; the lazy-init lowering took the aspects from the *texture*, so `depthReadOnly` marked depth initialized while the HAL stream (format `Stencil8`) never cleared it → a later depth copy read uninitialized memory | **Fixed** — Dawn's "must encompass all aspects of its texture's format" rule in `validate_depth_stencil_attachment` (same order as Dawn) + marks derived from the attachment view's format (defence in depth); unit tests both layers; CTS `api,validation,render_pass` 16,626/0, `encoding` 34,231/0; spec 104 R4 |
| M1 | MAJOR | Metal single-aspect clear of a combined depth-stencil texture set only one attachment; Dawn's `MetalUseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats` toggle (Intel / pre-GCN4 AMD) exists because that silently does not clear | **Fixed** — the other aspect is attached with `Load`/`Store` unconditionally; `ClearStrategy::RenderPass` carries `Option<MTLLoadAction>` per aspect, unit-tested; unreproducible on the M2 (Dawn-toggle class, see [[feature-table-vs-dawn-device-query]]); spec 104 R6 |
| M2 | MAJOR | Vulkan swapchain images (Block 103: usage = requested bits only) had no `TRANSFER_DST`, but `ClearTexture` of a lazy-init-eligible surface texture uses `vkCmdClearColorImage` (VUID 01198 / 00002) — reachable via `getCurrentTexture` → `copyTextureToBuffer` before any pass | **Fixed** — `swapchain_image_usage` ORs in `TRANSFER_DST` when `supportedUsageFlags` offers it (requested bits still validated first); `clear_kind` returns `HalError` on an image without it; new e2e `*_surface_texture_reads_zero_before_first_render_pass` (MoltenVK + Metal) green under the validation layers; spec 103 R1/R4, 104 R6 |
| M3 | MAJOR | Metal force-added `MTLTextureUsageRenderTarget` to every depth/stencil and MSAA texture (undocumented Dawn divergence; `Shared` + `RenderTarget` not universally allocatable on non-Apple-silicon Macs) | **Fixed** — usage widening removed; non-renderable depth/stencil clears take Dawn's blit path with `MTLBlitOption{Depth,Stencil}FromDepthStencil` (new `ClearStrategy::AspectBlit`, per-aspect bytes/texel incl. `Depth16Unorm` = 2); MSAA without `RenderTarget` is a `HalError` (unreachable by validation); new shared e2e `lazy_init_non_renderable_depth_stencil_copies_read_zero` (d32f / s8 / d24s8 / d32s8 both aspects) green on Metal + MoltenVK; spec 104 R6 |
| m1 | MINOR | Dead `viewFormats`-null branch in `wgpuSurfaceConfigure` | **Fixed** — removed; FFI test drives the real entry point |
| m2 | MINOR | HAL capability-query failure at configure dispatched as `Validation` with a `HalError` text | **Fixed** — `ErrorKind::Internal`; unit test on the dispatch decision (the mismatch path is unreachable on a Noop-only build); spec 103 R3 |
| m3 | MINOR | `native_surface_format` maps unmapped formats to `Undefined`, which a `format: Undefined` config would then match | **Fixed** — explicit `Undefined` rejection + membership scan skips unmapped formats; regression test with `R8Unorm` |
| m4 | MINOR | `VulkanSwapchainInner::drop` issued a second device-wide `vkDeviceWaitIdle` after the surface teardown's | **Fixed** — `teardown_waited` once-flag shared by teardown and drop |
| m5 | MINOR | Attachment marks were recorded before the pass was known to lower | **Fixed** — clears + marks staged, applied only when `hal_render_pass_execution` returns `Some`; overlay borrow now immutable during planning; test verified red-before-fix |
| m6 | MINOR | Tiled subpass path ignored the `Discard` un-mark rule | **Fixed** — `unmark_discarded_subpass_attachment` per colour / depth / stencil store op (resolve targets untouched); `tiled` test verified red-before-fix; spec 104 R4 |
| m7 | MINOR | `MetalSurface::capabilities()` only tested on a real device | **Fixed** — constant list split into `capability_list()`, GPU-free test asserts Dawn's macOS list |
| m8 | MINOR | Mixed lock-poisoning style in `ffi/surface.rs` | **Fixed** |
| m9 | MINOR | ~900 lines duplicated between `e2e_metal_lazy_init.rs` and `e2e_vulkan_lazy_init.rs` | **Fixed** — shared `yawgpu/tests/common/lazy_init.rs` via `#[path]`, scenarios take `(RealBackend, YAWGPU_INSTANCE_BACKEND_*)`; wrappers are 8 lines each |
| m10 | MINOR | `intersect_image_view_usage` one-expression wrapper + test | **Fixed** — inlined, VUID comment kept |

Reviewer also confirmed (no finding): `InitOverlay` pointer keying is pinned by the texture clones; intra-submission ordering (read-after-partial-write, Discard-then-Load) correct; compute-pass bind-group snapshots visit every bound view; compressed Vulkan clear offsets/alignment; `create_attachment_image_view` usage intersection; read-only depth + forced `Clear` safe on both HALs; Metal caps == Dawn's list; FFI capability array alloc/free symmetry; no new panic paths.

Re-verification after the fixes: `cargo test --workspace` 1093/0, clippy default/metal/vulkan/tiled/gles clean, fmt clean, iOS `cargo check` ok; Metal HAL `--ignored` 56/56 (clear tests 5/5 under `MTL_DEBUG_LAYER`; the full suite aborts under the debug layer on the pre-existing `u64::MAX` OOM buffer test — run it without the layer), MoltenVK HAL `--ignored` 53/53 validation-clean; e2e Metal lazy-init 7/7 + surface 6/6 (debug layer), MoltenVK lazy-init 7/7 + surface 5/5 (Khronos layer, 0 messages); CTS Metal `api,validation,render_pass` 16,626/0, `api,validation,encoding` 34,231/0 (+ the 8-tree `api,operation` set, see the Block 104 status line).


### 2026-09-22 (later) — A5 done (Block 105), Win32 surface e2e

- **Win32 surface e2e** (`97e1713`): `e2e_vulkan_surface.rs` runs on Windows native Vulkan through a hidden Win32 window (5/5, validation-clean); Block 103 status updated.
- **A5 done (Block 105)**: Vulkan image layouts are tracked per `(mip, layer)` (`yawgpu-hal/src/vulkan/layout.rs`); every copy / clear / present / attachment / bound-view transition names its exact subresources; the tiled subpass path now transitions bound textures (0704-sweep "Known related gaps" 2 closed) and handles its depth-stencil attachment correctly (Phase Review M1). Three new real-Vulkan e2e cases (`e2e_vulkan_layouts.rs`) were red under the Khronos layer before the fixes. Ledger: `specs/tracking/vulkan-subresource-layout.md`.
- **New rows A8** (3D attachment `storeOp: Discard` epilogue clear range, pre-existing since Block 104) and **A9** (sampled + read-only storage of one subresource → `GENERAL` vs `SHADER_READ_ONLY_OPTIMAL` descriptor), both from the Block 105 Phase Review.
- Next (agreed): D11 Vulkan-only items on this host — clip-distances Vulkan execution e2e, texture-component-swizzle depth path on Vulkan, native Windows immediates CTS sweep. ETC2/ASTC probes stay hardware-blocked.

### 2026-09-22 (later) — D11 Vulkan items closed (Block 106)

- `e2e_vulkan_clip_distances.rs` (port of the Metal e2e) and `e2e_vulkan_texture_component_swizzle.rs` (colour remap, depth swizzle composed over `(d, 0, 0, 1)`, identity depth, feature gate) added and green on the native NVIDIA driver under the validation layer.
- native Vulkan CTS (this host, yawgpu `4d5bc52` release `--features vulkan`, CTS `2f0fb9f`, raw, 2026-09-22): `shader,execution,shader_io,vertex_builtins:outputs,clip_distances` 8/0, `capability_checks,features,clip_distances` 368/0, `shader,validation,extension,clip_distances` 4/0, `api,operation,texture_view,texture_component_swizzle` 32,832 pass / 19,494 skip (all skips = compressed formats this GPU does not expose; every depth/stencil-format subcase passes: depth16unorm 1,197, depth24plus 1,197, depth32float 1,197, depth24plus-stencil8 1,539, depth32float-stencil8 1,539, stencil8 342) / 0 fail, `capability_checks,features,texture_component_swizzle` 855/0, `encoding,programmable,pipeline_immediate` 181/0, `encoding,cmds,setImmediates` 378/0; summary pass=34,626 skip=19,495 fail=0 crash=0.
- Tracking docs for Blocks 68 / 71 / 94 updated; D11 now lists only the ETC2 / ASTC probes.

### 2026-09-22 (Mac) — Block 105 MoltenVK re-confirmation, portability-subset fixes

- **Block 105 re-confirmed on MoltenVK** (`daf967e`): 10 CTS trees 274,884 pass / 9 fail = identical to the pre-Block-105 run (documented artifacts only); HAL `--ignored` 54/0; all `e2e_vulkan_*` 106/0 under the Khronos validation layer. Ledger updated.
- **Two pre-existing `VK_KHR_portability_subset` gaps** surfaced by running every Vulkan e2e under the layer on MoltenVK, both fixed:
  - `VUID-VkImageCreateInfo-imageView2DOn3DImage-04459`: `TYPE_2D_ARRAY_COMPATIBLE` was set on every 3D image (since F-043); now only for 3D colour-attachment images (Dawn rule) — `daf967e`.
  - `VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465`: the portability extension was enabled but `VkPhysicalDevicePortabilitySubsetFeaturesKHR` never queried/chained, so every portability feature stayed off; now queried and chained at device creation, and `texture-component-swizzle` is gated on `imageViewFormatSwizzle` on portability devices (unchanged on MoltenVK, which reports it true).
- `e2e_vulkan_clip_distances` execution case skips on macOS (MoltenVK cannot lower `ClipDistance`; Block 106 verified it on native Vulkan).

### 2026-09-23 — F doc-drift cleanup done; A8 + A9 in progress (Block 107)

- **F done** in one docs commit: `perf-dawn-baseline.md` (S3 done via `88cfe58`, S4 open), `specs/blocks/README.md` (60/70 file names, later-block note), `workflow.md` (git + Conventional-Commits convention), `refactor-dedup.md` (GLES hosts), `cts-coverage.md` ×2 (F-133 residual under-validations re-measured under Tint on this host — `statement,loop`+`statement,for` 108/0, `short_circuiting_and_or` 896/0; the 1363 "known core gaps" list annotated closed with the 2026-09-21 sweep evidence). `HANDOFF.md` does not exist; the README GLES section already has the NVIDIA table.
- **A8 + A9 → Block 107 — DONE** (`specs/blocks/107-vulkan-discard-epilogue-shared-layout.md`; S1 + S2 `3f809b1`, S3 = Phase Review fixes in the next commit): R1 drops the eager Vulkan Discard clear (core's Block 104 un-mark + lazy clear covers zero-on-next-read, as on Metal); R2 gives every sampled binding of an image that also has a storage binding in the same pass `GENERAL` on both the transition and the descriptor, decided by one image-wide predicate. Findings while executing: (a) Dawn tracks 3D initialization per mip, so Discard on one depth slice lazily zeroes the whole mip — the A8 row's "data loss for the other slices" was oracle behaviour, only the invalid clear range was a bug; (b) Phase Review F1: a pairwise range-overlap rule is not transitive across sampled views of one image (VUID-00344 reappears with two sampled views) — Dawn's per-texture rule adopted; (c) present is the one consumer core does not lazily clear for (documented, matches Metal/Dawn). e2e: `e2e_vulkan_layouts` 7/7 under the Khronos layer, 0 lines; CTS on this host: storeOp/storeop2/3d_texture_slices/storage_texture/resource_init/resource_usages/command_buffer (170,202)/rendering — fail 0. Deferred: F5 (`RenderBindings` struct for the render descriptor call chain) → E4.


### 2026-09-23 — snapshot re-taken at `2c6ea6f`

- Tables rebuilt after Blocks 99–107 and the F cleanup; closed rows moved to "Closed since the previous snapshot". New rows: A10 (Block 102 m4), A11 (Block 107 F2, documented), E9 (Block 102 m12), E10 (GLES `HalError: Clone` TODO); Block 107 F5 folded into E4. Headline counts re-measured (3 `TODO`, 375 `#[ignore]`, Vulkan `supports_*` fully query-driven, Metal 10 doc-noted literals).

### 2026-09-23 — D11 closed (ETC2 / ASTC probes on MoltenVK)

- The previous snapshot called E8/E9 "hardware-blocked", but the M2 / MoltenVK host exposes `textureCompressionETC2` and `textureCompressionASTC_LDR`, and yawgpu advertises all three compression features on it. `e2e_vulkan_texture_compression` E1–E10 run 10/10 under `VK_LAYER_KHRONOS_validation` with 0 VUID lines; E8 (ETC2 RGB8 / EAC R11 / ASTC 4x4, 8x8, 12x12 multi-block round-trips) and E9 (3D ASTC per-slice round-trip) executed, not self-skipped. Docs: Block 73 status, Block 106 "Out of scope", the texture-compression ledger (new slice 4 + run section). Priority D now has no runnable-on-this-host row; the remaining "Manual" rows (D10) need Windows ANGLE / WGL or the NVIDIA Linux host.

### 2026-09-26 — native-Vulkan CTS skip audit; Block 108 opened

- Re-ran `webgpu:api,*` + `webgpu:shader,execution,*` on this host (yawgpu `2c6ea6f` release `--features vulkan`) with `--output` and bucketed every skip by message: 381,430 of 443,719 are ASTC / ETC2 / EAC (not exposed by desktop NVIDIA); the rest are structural (view-compat pairs, limit == default, over-limit, spec-forbidden stage/format combos), C-API N/A, or harness-side gaps (default device without the needed feature, WGSL language-feature query not wired). Counts match the 2026-09-25 README sweep exactly.
- The one yawgpu-implementable skipped feature is `subgroup-size-control` (`WGPUFeatureName_SubgroupSizeControl = 0x17`, in the pinned header, never advertised). New row **B8 → Block 108** (`specs/blocks/108-subgroup-size-control.md`): Vulkan-only per Dawn (Metal unsupported), `VK_EXT_subgroup_size_control` + `computeFullSubgroups`, core validation of `@subgroup_size` (x-multiple, range, `maxComputeWorkgroupSubgroups`), `REQUIRE_FULL_SUBGROUPS` / `ALLOW_VARYING_SUBGROUP_SIZE` lowering, plus a webgpu-native-cts port update (S4).

### 2026-11-02 — snapshot re-taken at `48067a3`

- Tables rebuilt after Blocks 108-120. A6 closed as stale. New rows: A12, A13, D12, D13, D14, E11, E12, F1 to F3. GLES moved to last by user direction. The Method section lists which rows were not re-checked against the source.
- The Haswell indirect wedge moved out of the list: it was confirmed in 2026-07 and the user decided against a workaround, so Block 119 R5d's "unconfirmed" was wrong and is corrected.
- F-151 confirmed on real hardware; Block 113 status, `SPEC.md` Phases 13, 14, and 16, and `threading-audit.md` corrected.
- Later the same day: C4, B2 and GLES ranked last by user direction; B2 records that default Dawn rejects binding arrays too.
- E1 done as Run 7 in `perf-dawn-baseline.md`; it opened E13 (P-005) and A14, and E11 now needs a bench case first.
- E13 done by Block 121; Run 8 opened E14 (P-006) and carried the `compute_unique` check into it.
- E14 done by Block 122 (the cause was staging recycling, not the blit encoder); it opened A15.
- A15 done by Block 123 (relaxed math pragma, red-checked on M2); it opened A16.
- A16 done by Block 124. A survey of Dawn's backend setup found 13 more differences; the flag-level ones are Block 125 and the `frag_depth` clamp is Block 126.
- Blocks 125 and 126 done (Tint option parity part 1; Vulkan `frag_depth` clamp, which also fixed F-139 on MoltenVK). Native Vulkan runs owed for both.
- Block 127 done (Vulkan reconvergence and default compute subgroup size). Native Vulkan runs owed for Blocks 125-127.
- Blocks 121 and 125-127 ran on Windows native Vulkan at `d7ff142`: e2e 157/157 and HAL ignored 75/75 with 0 VUIDs; CTS `webgpu:*` `fail=2 crash=0`, every area equal to the 2026-11-01 Windows baseline except 5 more `shader,execution` passes where the F-165 crash did not fire. The native-Vulkan owed items for Blocks 121 and 125 are closed. A17 opened for the device-specific paths of Blocks 126 and 127.
- Block 128 done (Vulkan robustness2 and pipeline robustness, as Dawn: the image clamp is removed with `robustImageAccess2`, the runtime-sized storage clamp with `robustBufferAccess2`). On MoltenVK, CTS showed no new failures and the image path ran. The buffer path needs native NVIDIA, where F-141 should also be checked. Remaining survey items: SPIR-V 1.4, extended dynamic state, `VkPipelineCache` (E3), `strip_all_names`.
- Block 128 ran on Windows native Vulkan at `5ecaafc`: e2e 160/160 and HAL ignored 76/76 with 0 VUIDs, buffer and image robustness paths both taken; CTS `webgpu:*` `fail=3 crash=1`. 2 are the R9 cases, the crash is F-165, and the third is a pre-existing `copyTextureToTexture` failure traced to unzeroed Vulkan buffer memory (A18, opened).
- A18 done by Block 129: Vulkan `create_buffer` zero-fills the whole mapping on the host. `copyTextureToTexture:*` 0 failures in 3 of 3 runs (was 2-3 in every run); Windows native Vulkan `webgpu:*` `fail=2 crash=0`. Metal check owed to a Mac host.
- Block 129 Metal check (R3) on the M2 at `019bd3b`: `e2e_metal_buffer_zero_init` 2/2; CTS `copyTextureToTexture:*` and `resource_init,buffer:*` 0 failures in 3 of 3 runs each. No Metal block opened.
- Block 130 done (A10, A14): a pass that fails to lower now aborts the submit with an internal error; `AllowSpontaneous` device-lost callbacks fire inside destroy and release, as Dawn. The A14 premise was wrong (yawgpu already destroyed on last release); the bench now calls `ProcessEvents` before instance release and exits 0. A19 and A20 opened. Metal e2e 161/0; MoltenVK e2e 159/1 (known).
- Block 131 done (A13): Dawn's vendor-gated Tint workarounds are set (8 MSL options for AMD, Intel, and Apple9; 12 SPIR-V options for AMD Mesa below 25.3, Qualcomm, ARM, Imagination, Intel Mesa, Windows AMD, Android Qualcomm, and SwiftShader), with a driverID query that checks the extension name only, as Dawn does. No gate fires on our hosts, so verification there was forced. Native NVIDIA forced CTS owed. A21 opened.
- Blocks 130 and 131 ran on Windows native Vulkan at `f3a63e9`: e2e 162/162 and HAL 78/78, default and with Block 131's 12 SPIR-V options forced. Forced CTS `webgpu:*` `fail=8,254 crash=0`: 7,820 from `texture_sample_compare_2d_polyfill` (Tint's polyfill; Dawn comparison not available), 432 a Block 130 regression that also fails unforced (A22, opened), 2 the R9 cases.
- A22 done by Block 132: render bind groups are resolved at draw time. CTS `pipeline_bind_group_compat:*` 432 failures to 0; Windows native Vulkan `webgpu:*` `fail=3 crash=0` (R9 cases and one intermittent `memory_model,weak`). Metal owed to a Mac host.
- Block 132 Metal check on the M2 at `ba0b289`: CTS `pipeline_bind_group_compat:*` 432 failures before (the `f3a63e9` library) to 0 after; Metal e2e 162/0; CTS `api,operation` fail=0.
- Block 133 done: Vulkan SPIR-V 1.4 as Dawn (`VK_KHR_spirv_1_4` + `VK_KHR_shader_float_controls`; 1.3 on Android ARM and Adreno 7xx). Windows native Vulkan e2e 163/163, HAL 79/79, 0 VUIDs; CTS `webgpu:*` `fail=2 crash=5`. The crashes are F-165, which SPIR-V 1.4 makes far more frequent on NVIDIA (bufferView/bufferArrayView workgroup f16 aliasing: 4-12 crashes per run vs 0 with 1.3). Kept as Dawn by user decision. Block 134 (E6, Vulkan limits) specified next.
- E6 done by Block 134: Vulkan adapter limits follow Dawn (NVIDIA storage binding size 2 GiB - 4, `maxColorAttachmentBytesPerSample` 32 x attachments, `maxFragmentCombinedOutputResources` redistribution, `NormalizeLimits` clamps). Windows native Vulkan CTS `webgpu:*` `fail=2 crash=0`, `api,validation` +182 passes. A23 opened (adapter info vendorID/deviceID are 0).
- A23 done by Block 135: `WGPUAdapterInfo` reports vendor and device ids, adapter type, driver description, and Dawn's `gpu_info.json` vendor and architecture names (generated table, `tools/gen_gpu_info.py`). Windows NVIDIA: `nvidia` / `blackwell` / `0x10de:0x2d04` / discrete. Metal owed to a Mac host.
- Blocks 133-135 on the M2 at `d3e574b`: Metal clippy clean (Block 135's Metal code was first compiled here), workspace 1,375/0, Metal ignored 162/0, and `wgpuAdapterGetInfo` on Metal identical to Dawn `v20261030` in every field. MoltenVK e2e (reference only) 155/1, the known `imageView2DOn3DImage`.
- Block 136 done: the Windows Tint shim was built without optimization (the `cmake` crate drops MSVC's `/O2` with the VS generator). Restored per configuration in `shim/CMakeLists.txt`. Windows NVIDIA: `shader/create_unique` 432 µs to 36 µs (Dawn 44 µs), `pipeline/compute_unique` 1,279 µs to 307 µs (Dawn 495 µs). CTS `webgpu:*` unchanged apart from known intermittents. E3's `VkPipelineCache` dropped; extended dynamic state stays off as Dawn's default.
- F1-F3 closed (doc drift): full `specs/blocks/README.md` index; Block 108 Mac follow-up recorded (MoltenVK `computeFullSubgroups = true`); Block 109 sync-validation owed item satisfied by `decfd69`. A20 widened into Block 137: Dawn fires every `AllowSpontaneous` callback that is ready at registration inside the call (probe on the M2).
- A20 done by Block 137: every `AllowSpontaneous` callback that is ready at registration fires inside the call, older ready spontaneous callbacks first, as Dawn's `TrackEvent`; `FailedCreation` message carries the cause. Probe matches Dawn on Metal; workspace 1,391/0; Metal e2e 162/0; CTS `api,*` fail=0. A19 probe: after instance release yawgpu cancels every callback and stops reporting uncaptured errors, while Dawn's device keeps working.
- A19 done by Block 138: the instance stays in use while it, an adapter, or a device has an external reference; callbacks are cancelled only when none is left. After the instance release a device keeps reporting errors, error scopes, and spontaneous callbacks, as Dawn. Probe matches Dawn on Metal apart from `AllowProcessEvents` callbacks that Dawn never fires (yawgpu cancels them at the last release). Workspace 1,398/0; Metal e2e 162/0; CTS `api,*` fail=0.
- Block 139 done: the Tint shim's Apple minimum OS follows the Rust target (rustc's default or the set variable); it was the host SDK version (27.2) on macOS and iOS. iOS device and simulator cross-builds and clippy clean; Android cross-builds (vulkan, gles, vulkan+tiled+passthrough) and clippy clean with the README recipe now exporting `NDK_SYSROOT`.
- E12 and E7 done by Block 140: `wgpuInstanceWaitAny` backs off instead of spinning; Metal reports Dawn's subgroup range and advertises `subgroup-size-control` on Apple GPUs; unread Vulkan 16-bit flags removed. A24, A25, A26, and B8 opened. CTS Metal `api,*` and `shader,execution,*` fail=0 (+7 passes from subgroup-size-control).
- A26 and B8 done by Block 141 (2026-11-05): Metal `subgroups` follows Dawn's `Apple6 || Mac2` and always reports the size range; `texture-compression-unaligned` and `atomic-vec2u-min-max` advertised on Metal, Vulkan, and Noop. The M2 standard feature list now matches Dawn. CTS Metal `api,*` fail=0; `shader,execution,*` fail=71, all zero-value `atomicStoreMax`/`atomicStoreMin` cases that Dawn fails identically.
- A25 done by Block 142 (2026-11-05): Metal textures are `Private`, `Memoryless` only for transient attachments on Apple2 devices, as in Dawn. No CPU path touched textures, so the change was local. CTS Metal `api,*` and `shader,execution,*` fail=0, identical to Block 141.
- A24 done by Block 143 (2026-11-05): `destroy()` releases the backend object of a buffer, texture, or query set; Metal and Vulkan free it once the submissions that use it complete. External texture planes now join the usage scope and the submit-time destroyed check, as in Dawn. Metal `currentAllocatedSize` stays flat over 16 destroyed 80 MiB iterations (it kept 1,280 MiB before). CTS Metal `api,*` and `shader,execution,*` fail=0. A27 opened.
- Windows native Vulkan at `c7d9acd` (2026-11-05): `cargo test --workspace` did not compile on Windows. Two test state structs from Blocks 137 and 138 declared `status: u32`, but bindgen's C enum type is `i32` under MSVC. Fixed in the tests with a status type alias. Then workspace 1,426/0, Vulkan HAL ignored 82/82, Vulkan e2e 171/171 with 0 VUIDs. Block 141's `atomic-vec2u-min-max` readback, which skips on MoltenVK, ran and passed on NVIDIA; Block 143's destroy e2e passed.
- A27 done by Block 144 (2026-11-05): external texture planes are lazily zero-initialized like any sampled texture view, in compute, render, bundle, and tiled passes. A Metal e2e that discards a plane and samples it reads zero (it read the stored colour before). CTS Metal `api,*` and `shader,execution,*` fail=0. A28 opened (plane views with a non-zero base mip or layer).
- A28 done by Block 145 (2026-11-05): external texture planes are lowered as full texture-view descriptions and Metal binds a view of each, so a plane view of mip 1 or of one layer of a 2D array samples that subresource (it sampled mip 0 / layer 0 before). CTS Metal `api,*` and `shader,execution,*` fail=0.
- Blocks 144 and 145 on Windows native Vulkan at `ebd97f8` (2026-11-05): workspace 1,434/0, clippy default and `vulkan,tiled` clean, Vulkan HAL ignored 82/82, Vulkan e2e 171/171 with 0 VUIDs. External textures were still Metal-only at that commit; the Vulkan rejection e2e passed.
- Block 146 done (user request): external textures on Vulkan. Each `texture_external` expands to plane0 at its binding and plane1 and a params uniform above the layout's largest binding; Tint's multiplanar transform gets a group-aware map for SPIR-V. Windows NVIDIA: shared external-texture e2e 6/6, Vulkan e2e 176/176 with 0 VUIDs, CTS `webgpu:*` `fail=2 crash=2` (known) with about 3,000 subcases moving from skip to pass. The suite's F-111 `xfail` entries can be dropped. Metal run of the shared e2e owed to a Mac host. Also fixed: Block 141's int64-atomics HAL test compared extension names by pointer and failed on Windows.
- Block 146 on Metal and MoltenVK at `fa447be` (2026-11-05): workspace 1,436/0, the shared external-texture e2e 6/6 on Metal and on MoltenVK (no validation output), Metal e2e 174/174, CTS Metal `api,*` and `shader,execution,*` fail=0, identical to Block 145.
- Doc drift fixed (2026-11-05): B3 narrowed to GLES now that Vulkan has external textures (Block 146); README Linux NVIDIA GLES table updated to the suite's 2026-11-01 sweep (D14).
- Block 147 done (user request, E9 and E10): `use` items first in 26 test modules, test-only `#[cfg(test)]` helpers moved into their test modules, `RenderPassTimestampWrites` renamed `PassTimestampWrites`, `HalError` derives `Clone`. No behaviour change; workspace 1,437/0. The GLES `HalError: Clone` `TODO` is gone, so two `TODO`s remain (the `tiled 2.4` arms).
- Block 148 done (user request, E8): the Block 33 Phase Review ran on today's code. Fixed: render passthrough validates the descriptor; Metal compute pipelines use compute slots even when layout entries are also visible to render stages, and Metal render passes no longer bind compute-only entries at slot 0 (both silent-wrong for WGSL too, shown by a new Metal e2e that fails before); backend mismatch is a validation error; MSL workgroup limits; `ShaderSourceSPIRV` is advertised and required for SPIR-V. A29 and A30 opened.
- Blocks 147 and 148 on Windows native Vulkan at `c4cbb5b` (2026-11-08): workspace 1,441/0, clippy default, `vulkan,tiled`, `vulkan,shader-passthrough` and `gles` clean, Vulkan HAL lib 386/0 and ignored 82/82, Vulkan e2e 178/178 with 0 VUIDs, including the SPIR-V passthrough and compute binding-slot e2e.
- Block 149 done (user request, A29, A30, and the examples' MoltenVK acquire failure): subpass pipelines support external textures on Metal and Vulkan; color targets must be color-renderable; the Vulkan surface reports suboptimal per acquire, as Dawn; the macOS examples framework sets the layer's `contentsScale` and the examples accept `SuccessSuboptimal`, so every windowed example runs on MoltenVK.
- Block 149 on Windows native Vulkan at `be17a0d` (2026-11-09): workspace 1,443/0, clippy default, `vulkan,tiled`, `vulkan,shader-passthrough` and `gles` clean, Vulkan HAL ignored 83/83, Vulkan e2e 181/181 with 0 VUIDs (tiled external texture 3/3 on the native subpass path, surface 10/10). Every C example built and ran on Vulkan under the Khronos layer without errors; the windowed ones ran 8 s each.
