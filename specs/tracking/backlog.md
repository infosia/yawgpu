# Backlog — remaining TODOs and missing features, by priority and difficulty

**Snapshot date: 2026-09-23** (tree at `2c6ea6f`; supersedes the
2026-09-22 snapshot at `d6f865b`, whose progress log is kept below as
history). This is a cross-cutting inventory, not a plan: it collects
every open / deferred / partial item recorded across `specs/blocks/`,
`specs/tracking/`, `README.md`, and the source (comment markers,
`HalError` rejections, hard-coded feature tables, the C-ABI symbol set),
then ranks them. Each row was verified against the source or the doc
that records it. Items the docs mark DONE / RESOLVED / COMPLETE are
omitted; the previous snapshot's closed rows (A1, A2, A3 S1–S3, A4, A5,
A8, A9, B1, B4, B5, D11 Vulkan items, all of F) are listed once under
"Closed since the previous snapshot".

Re-take this snapshot rather than editing rows in place when the state
changes; per-item ledgers stay in their own tracking docs.

## Method

- Docs: all `specs/blocks/*.md` (now 41, through Block 107), all
  `specs/tracking/*.md`, `specs/reference/*.md`, `README.md`,
  `DESIGN.md`, grepped for deferred / follow-up / open / not yet /
  unsupported / xfail / residual; every Phase Review table since
  `d6f865b` re-read for "Deferred" dispositions.
- Source: `yawgpu`, `yawgpu-core`, `yawgpu-hal`, `yawgpu-tint` for
  `TODO`/`FIXME`/`unimplemented!`/`todo!`, `HalError` returns for
  validated WebGPU operations, `supports_*` bodies, `#[ignore]` reasons,
  and the `wgpu*` export set against `webgpu.h`.
- Headline counts (2026-09-23): 3 `TODO`s (two `tiled 2.4` Transient
  arms, one GLES `HalError: Clone` note), 0 `FIXME`/`unimplemented!`/
  `todo!`; 375 `#[ignore]` tests (265 in `yawgpu/tests` e2e, 110 in
  `yawgpu-hal`; all manual real-GPU gates, none marked known-failing);
  GLES `HalError` catalogue unchanged since the previous snapshot (no
  GLES mapping commits since `d6f865b`); 202 of 202 `wgpu*` symbols
  exported (Block 101); hard-coded `supports_*`: Metal 10 literal `true`
  (all Dawn-unconditional, doc-noted since Block 99), Vulkan 0 literals
  (every entry is a device query), GLES 18 `false` + 1 optimistic `true`
  (`depth32float_stencil8`).

**Difficulty scale:** S = under a day, M = several days, L = a week or
more. "Manual" = blocked on hardware or a host this machine cannot
reach.

**Priority rationale:** silently-wrong behaviour that the CTS cannot
catch ranks above advertised-but-non-functional, which ranks above a
missing standard API, which ranks above platform/presentation coverage,
which ranks above Tier-2 GLES gaps, then perf/refactor, then doc drift.

---

## Closed since the previous snapshot (`d6f865b` → `2c6ea6f`)

| Prev # | Item | Closed by |
|---|---|---|
| A1 | `timestamp-query` advertised but non-functional | Block 102 (`57c64e7`, `ddec607`, Phase Review `061aab3`); e2e 4/4 Metal + MoltenVK, CTS 0 fail |
| A2 | `wgpuCommandEncoderWriteBuffer` discarded `data` | Block 100 (`2f97d3d`) |
| A3 (S1–S3) | Lazy zero-init Stage 2, core + Metal + Vulkan | Block 104 (`60744d8`, `99d786b`, `9aa1da3`, Phase Review `251a4cc`/`c83bbbf`) — GLES clear paths remain, see A3 below |
| A4 | Hard-coded `supports_*` tables | Block 99 (`6783728`, `55162d3`) |
| A5 | Vulkan per-texture image-layout tracking | Block 105 (`a00daf9`..`2583b87`), MoltenVK re-confirm `daf967e` |
| A8 | Vulkan 3D attachment `storeOp: Discard` epilogue clear range | Block 107 R1 (`3f809b1`) — eager clear removed, core lazy clear covers it |
| A9 | Vulkan sampled + storage-bound subresource layout mismatch | Block 107 R2 (`3f809b1`, `2c6ea6f`) — image-wide `GENERAL` rule |
| B1 | `wgpuGetProcAddress` | Block 101 (`45d1942`), 202/202 |
| B4 | ~50 CTS `#[ignore]`s | Stale (in-repo ports deleted in `a9218a0`), `62338d8` |
| B5 | Fixed surface capabilities | Block 103 (`a2907ca`), Win32 e2e `97e1713` |
| D11 (Vulkan) | clip-distances / swizzle depth / immediates on native Vulkan | Block 106 (`4d5bc52`, `6c8b417`) |
| F | 8 doc-drift rows | `aaef70c` |

Also fixed on the way, never a backlog row: Metal `tint_storage_buffer_sizes`
`setBytes` length (`288887a`), `VK_KHR_portability_subset` feature chaining
+ `TYPE_2D_ARRAY_COMPATIBLE` narrowing (`fdef682`, `daf967e`).

---

## Priority A — correctness (silent-wrong, invisible to the CTS)

Every Tier-1 row from the previous snapshot is closed. What remains is
Tier-2 or unreachable by validation today.

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| A3 | **Lazy zero-init GLES clear paths (Block 104 S4)** — core marks and un-marks on every backend, but the GLES HAL has no `ClearTexture` lowering for the Stage-2 eligibility set (sampled reads of uninitialized depth/stencil, compressed and MSAA textures). Tier-2 catalogue item; passes today only where the driver hands back zeroed memory. | M | `specs/blocks/104-lazy-zero-init-stage2.md` S4; `specs/blocks/67-gles-backend.md` mapping matrix |
| A6 | Metal rejects a non-default `multisample.mask` with `HalError` (pinned `objc2-metal` does not expose `sampleMask`); Vulkan applies it. Accepted as a Tier-1 limitation, still a divergence between the two Tier-1 backends. | S–M | `specs/tracking/threading-audit.md:33-36` |
| A7 | GLES `SetImmediates` is recorded and bounds-checked but never read by the draw path — a silent gap. Unreachable today because `max_immediate_size` is 0 and core rejects first; must become an explicit `HalError` if the limit is ever raised. | S | `specs/blocks/67-gles-backend.md:282`, `yawgpu-hal/src/gles/adapter.rs` |
| A10 | Block 102 m4: a timestamp conversion pass that fails to lower would be dropped silently (raw ticks left in the destination). Unreachable by construction today (internal pipeline is non-error, bind group binds a submit-validated buffer); revisit when a second internal compute pass appears. | S (needs a result threaded through the lowering walk) | `specs/tracking/backlog.md` Phase Review Blocks 99–102, m4 |
| A11 | Block 107 F2 (documented, not a bug): `wgpuSurfacePresent` is the one consumer core does not lazily clear for, so a surface texture Discarded then presented shows `DONT_CARE` contents on Vulkan. Matches Metal and Dawn; listed so nobody re-adds the eager clear. | — | `specs/blocks/107-vulkan-discard-epilogue-shared-layout.md` R1 |

## Priority B — missing standard WebGPU surface

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| B2 | **Binding arrays**: `bindingArraySize > 1` rejected outright in core. | M–L | `yawgpu-core/src/bind_group_layout.rs` |
| B3 | **External textures are Metal-only.** Vulkan is deterministically rejected in core (the HAL would need the multiplanar plane/params bindings, not just driver acceptance of Tint's SPIR-V); GLES is a `HalError`. | L / L | `yawgpu-core/src/compute_pipeline.rs`; `specs/blocks/36-external-textures.md:165-174` |
| B6 | Tier-2 read-write storage format breadth ambiguous against the current spec (yawgpu grants `R32Float`, `RGBA16Float`, `RGBA32Float`). Needs a spec check more than code. | S | `specs/tracking/format-completeness-audit.md:23-27` |
| B7 | Multi-threading correctness beyond what the ported tests exercise. | L | `specs/SPEC.md:53` |

## Priority C — platform / presentation coverage

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| C1 | **Android `ANativeWindow` surface on Vulkan is a stub** — returns `SwapchainCreationFailed("Android native window surface not implemented")`. | M | `yawgpu-hal/src/lib.rs` |
| C2 | **Xlib / Wayland / XCB surface sources are accepted by validation but inert** — no `VkSurfaceKHR` is created for them (only `CAMetalLayer` and Win32 HWND reach a Vulkan surface). Linux has no windowed presentation on any backend. | M each | `yawgpu/src/ffi/mod.rs`, `yawgpu-hal/src/vulkan/mod.rs`; `specs/blocks/85-windows-surface.md:269-279` |
| C3 | GLES Linux windowed presentation; the headless `EGL_PLATFORM_DEVICE_EXT` cascade must be skipped for window surfaces when it is added. | M | `yawgpu-hal/src/gles/egl.rs`; `specs/blocks/67-gles-backend.md:500-509` |
| C4 | **CI runs only the default (Noop) gates.** Feature-gated clippy (`vulkan` / `metal` / `gles` / `tiled` / `shader-passthrough`) never runs in CI, so lint rot reaches contributors first; no real-GPU CI for the 375 manual tests; no Windows example-build job. | M (infra) | `.github/workflows/ci.yml`; `specs/tracking/toolchain-clippy-1-98.md:110-126`; `specs/blocks/85-windows-surface.md:276-278` |

## Priority D — GLES Tier 2 residue

Campaign state: raw CTS fail 30,408 → 6,261 (crocus / Haswell host, now
gone); README also carries the NVIDIA Linux (ES 3.2) table. The
remaining tail is fragmented; each row is an independent investigation.
No GLES mapping commits have landed since the previous snapshot.

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| D1 | **MSAA per-sample behaviour** — `multisample.mask` / `@builtin(sample_mask)` and alpha-to-coverage do not gate per-sample colour/depth writes (~1,510 fails). Documented as the next deep-debug slice. | L | `specs/tracking/cts-gles-sweep-0705.md:1206-1218` |
| D2 | **`maxBindingsPerBindGroup` off-by-one** — the limit is a plain `min` of GL maxima with no reservation for Tint's texture-metadata UBO. Mechanism confirmed, not fixed. | S | `specs/blocks/67-gles-backend.md:809-825` |
| D3 | Query sets: timestamp not advertised although `EXT_disjoint_timer_query` exists; `submit_resolve_query_set` writes zeroes from a software vector rather than real GL query objects. Block 102's core path (conversion pass, `HalQueryKind::Timestamp`) is ready for a GLES arm. | M | `yawgpu-hal/src/gles/queue.rs`; `specs/blocks/67-gles-backend.md:279` |
| D4 | Stencil-aspect texture-to-buffer readback (~900 fails); a compute-image path could lift it. | M | `specs/blocks/67-gles-backend.md:798-808` |
| D5 | 1D textures / `texture_storage_1d` — height-1 2D emulation (Dawn precedent), not started. | M | `specs/blocks/67-gles-backend.md:250, 910-913` |
| D6 | Texture views on hosts without `glTextureView` (copy-based emulation); texture-to-texture copy without `GL_EXT_copy_image`. | M–L / M | `yawgpu-hal/src/gles/queue.rs` |
| D7 | Cube / cube-array colour attachments; framebuffer fetch via `EXT_shader_framebuffer_fetch`; compressed-format advertisement. | S–M / M / M | `specs/blocks/67-gles-backend.md:254, 265`; `specs/tracking/format-completeness-audit.md:34-37` |
| D8 | Latent self-deadlock: `Drop` impls of GLES buffer / pipeline / sampler / texture inners acquire `with_current_context`; dropping the last `Arc` inside a context closure re-deadlocks. A guard is cheap. | S | `specs/tracking/cts-gles-sweep-0705.md:136-145` |
| D9 | `unorm8x4-bgra` renders R/B swapped on hosts without `EXT/ARB_vertex_array_bgra` (execution-only divergence; shader swizzle emulation deferred). | M | `specs/blocks/67-gles-backend.md:786-790` |
| D10 | **Manual verification owed**: ANGLE re-confirmation after the Tint migration, Windows WGL/NVIDIA sweep, catalogue re-sweep on the NVIDIA Linux host (README Haswell table stays until then). | Manual | `specs/tracking/tint-integration-refactor.md:219-221`; `specs/blocks/67-gles-backend.md:141, 269` |
| D11 | **DONE 2026-09-23** — ETC2/ASTC probes E8/E9 executed on Apple M2 via MoltenVK: `e2e_vulkan_texture_compression` 10/10 under the Khronos layer, 0 VUID lines, no self-skip (ledger `texture-compression-vulkan.md` "ETC2 / ASTC probes on MoltenVK"). Vulkan items were closed by Block 106. Original text: only the ETC2/ASTC probes E8/E9 remained (desktop NVIDIA does not expose those families; needs Android Vulkan or MoltenVK on Apple Silicon). | S (Mac) | `specs/tracking/texture-compression-vulkan.md:94-96, 119` |

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
| E1 | Block 97 **S4** (re-measure + CTS re-run, record a Run section). S3 is done (`88cfe58`); the perf ledger now says so. | S | `specs/tracking/perf-dawn-baseline.md` |
| E2 | Residual gaps vs Dawn with no structural lever: `write_buffer_then_wait` 1.53×, `bindgroup/create_destroy` 1.33×, `render_draw` 224 vs 99 ns. | Diffuse | `specs/tracking/perf-dawn-baseline.md:435-446` |
| E3 | Deferred cache layers: real `VkPipelineCache`; Metal `MTLLibrary`-by-source cache (measure-first); whole-module Tint IR clone (needs an upstream Tint API). | M / M / L | `specs/blocks/95-shader-compile-cache.md:35-41`; `yawgpu-tint/src/lib.rs` |
| E4 | refactor-dedup deferrals: double `Arc<core::X>` in FFI Impl structs, Metal derived-view cache + per-draw scratch reuse, unified `HalRenderStreamState` interpreter, GLES draw-time dirty tracking + binding index maps, subpass attachment validation dedupe; **+ Block 107 F5**: a `RenderBindings<'_>` struct for the Vulkan render descriptor call chain (`update_render_descriptor_sets` `too_many_arguments`, `bind_textures` / `pass_textures` side by side). | M each | `specs/tracking/refactor-dedup.md:38-56, 98-101, 191-196`; `specs/blocks/107-vulkan-discard-epilogue-shared-layout.md` F5 |
| E5 | `tiled`: depth-format transient attachments, MSAA subpass input (Dawn-deferred; 2-arg `inputAttachmentLoad` parked on a Dawn resolver issue), queries inside subpass passes, GLES mapping, `SubpassAttachmentResource::Transient` arm TODOs (the two remaining `TODO(tiled 2.4)` markers). | M / L / M / L | `specs/blocks/55-tiled-rendering.md:45-53, 501-507, 639-648`; `yawgpu-core/src/subpass.rs:244`, `yawgpu-hal/src/command.rs:765` |
| E6 | Vulkan limits: `maxFragmentCombinedOutputResources` redistribution, NVIDIA 2 GB storage cap, `max_color_attachment_bytes_per_sample` not queried. | S | `yawgpu-hal/src/vulkan/mod.rs`; `specs/tracking/adapter-limits.md:39-40, 90` |
| E7 | Metal subgroup size range hard-coded `(32, 32)`; f16 review MINORs (write-only `VulkanDeviceInner` flags, duplicate feature query). | S | `specs/tracking/subgroups.md:106-108`; `specs/tracking/shader-f16.md:177-183` |
| E8 | `shader-passthrough`: B5 Phase Review outstanding; gating SPIR-V passthrough behind `WGPUInstanceFeatureName_ShaderSourceSPIRV` deferred. | S / S | `specs/blocks/33-shader-passthrough.md:257-270` |
| E9 | Block 102 m12 cosmetic deferrals: tests inserted above `use super::*`; `#[cfg(test)] pub(crate)` shim; `RenderPassTimestampWrites` reused for compute passes (renaming is a cross-crate API change). | S | `specs/tracking/backlog.md` Phase Review Blocks 99–102, m12 |
| E10 | GLES `HalError` is not `Clone`; the GLES module keeps a `TODO` to derive it upstream once every variant allows it. | S | `yawgpu-hal/src/gles/mod.rs:43` |

## Priority F — doc drift

None open. The previous snapshot's eight rows were closed in `aaef70c`
(see the 2026-09-23 progress entry). Re-check on the next snapshot.

---

## Recommended order

1. ~~**D11** ETC2/ASTC probes E8/E9 on the M2 under MoltenVK~~ — done
   2026-09-23 (see the progress log).
2. **B6 + D2 + D8** — all S, root cause known; one coding-agent handoff
   (a spec check, a limit reservation, a re-entrancy guard).
3. **C4** feature-gated clippy in CI — infra only, stops lint rot from
   reaching the next contributor.
4. **A6** Metal `multisample.mask` — the last Tier-1 vs Tier-1
   divergence (`objc2-metal` bump or a manual selector).
5. **B2** binding arrays, then **C1 / C2** surfaces as separate
   initiatives.
6. **D** (GLES, starting with D3 since Block 102 made it a HAL-only
   arm) and **E** as separate initiatives; E1 whenever a perf change
   lands.

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
