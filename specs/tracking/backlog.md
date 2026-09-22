# Backlog — remaining TODOs and missing features, by priority and difficulty

**Snapshot date: 2026-09-22** (tree at `d05495bc`). This is a
cross-cutting inventory, not a plan: it collects every open / deferred /
partial item recorded across `specs/blocks/`, `specs/tracking/`,
`README.md`, and the source (comment markers, `HalError` rejections,
hard-coded feature tables, the C-ABI symbol set), then ranks them. Each
row was verified against the source or the doc that records it. Items
the docs mark DONE / RESOLVED / COMPLETE are omitted.

Re-take this snapshot rather than editing rows in place when the state
changes; per-item ledgers stay in their own tracking docs.

## Method

- Docs: all 40 `specs/blocks/*.md`, all `specs/tracking/*.md`,
  `specs/reference/*.md`, `README.md`, `DESIGN.md`, grepped for
  deferred / follow-up / open / not yet / unsupported / xfail / residual.
- Source: `yawgpu`, `yawgpu-core`, `yawgpu-hal`, `yawgpu-tint` for
  `TODO`/`FIXME`/`unimplemented!`/`todo!`, `HalError` returns for
  validated WebGPU operations, `supports_*` bodies, `#[ignore]` reasons,
  and the `wgpu*` export set against `webgpu.h`.
- Headline counts: 4 `TODO`s, 0 `FIXME`/`unimplemented!`/`todo!`,
  290 `#[ignore]` tests (all manual real-GPU gates, none known-failing),
  17 GLES `HalError` sites for validated operations, 201 of 202
  `wgpu*` symbols exported, hard-coded `supports_*`: Metal 12 `true`,
  Vulkan 6 `true`, GLES 18 `false` + 1 `true`.

**Difficulty scale:** S = under a day, M = several days, L = a week or
more. "Manual" = blocked on hardware or a host this machine cannot
reach.

**Priority rationale:** silently-wrong behaviour that the CTS cannot
catch ranks above advertised-but-non-functional, which ranks above a
missing standard API, which ranks above platform/presentation coverage,
which ranks above Tier-2 GLES gaps, then perf/refactor, then doc drift.

---

## Priority A — correctness (silent-wrong, invisible to the CTS)

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| A1 | **`timestamp-query` is advertised on Metal and Vulkan but has no implementation.** `HalQueryKind` has only `Occlusion`; a `Timestamp` query set gets no HAL object (`hal: None`), so `writeTimestamp` / pass `timestampWrites` reach nothing and `resolveQuerySet` silently skips the copy (destination buffer left untouched). The CTS never checks timestamp values, so the sweeps stay green. | M | `yawgpu-core/src/device.rs:176`, `yawgpu-core/src/queue.rs:1012`, `yawgpu-hal/src/lib.rs:646`; adverts at `yawgpu-hal/src/metal/mod.rs:385`, `yawgpu-hal/src/vulkan/mod.rs:491` |
| A2 | **`wgpuCommandEncoderWriteBuffer` discards `data`** — still the P6.2 validation-only implementation, a silent no-op on real backends. Either stage the bytes into a buffer copy at submit or return a device error. | S–M | `yawgpu/src/ffi/encoder.rs:269`; `specs/tracking/execution-gap-audit.md:31-34` |
| A3 | **Lazy zero-init Stage 2 not started.** Eligibility is single-sample, uncompressed, colour-only; sampled (non-storage) reads of an uninitialized texture, depth/stencil, compressed and MSAA textures are never cleared. Passes today only because GPUs happen to hand back zeroed memory. | M | `yawgpu-core/src/queue.rs:1402` (TODO), `yawgpu-core/src/texture.rs:484`; `specs/tracking/tint-migration-plan.md:798-800` |
| A4 | **Hard-coded `supports_*` tables.** Metal: 12 unconditional `true` (weakest: `timestamp_query`, never checks `MTLCounterSamplingPoint`). Vulkan: 6 unconditional `true` (`timestamp_query` should check `timestampComputeAndGraphics`/`timestampPeriod`; `depth32float_stencil8` needs a `vkGetPhysicalDeviceFormatProperties` probe). GLES: `depth32float_stencil8` is the only optimistic `true`. Block 72 fixed two entries; these remain. | S each | `yawgpu-hal/src/metal/mod.rs:357-442`, `yawgpu-hal/src/vulkan/mod.rs:417-498`, `yawgpu-hal/src/gles/adapter.rs:170` |
| A5 | **Vulkan image-layout tracking is per-texture, not per-subresource** (one `AtomicU8`); per-mip/per-layer divergence cannot be represented. Related: the `tiled` subpass path lacks the sampled-texture layout transition `encode_render_pass` performs; aspect-narrowed copies emit a single-aspect whole-image barrier (VUID-03320 class). | L / M / S | `specs/tracking/cts-full-sweep-0704-native-vulkan.md:412-434` |
| A6 | Metal rejects a non-default `multisample.mask` with `HalError` (pinned `objc2-metal` does not expose `sampleMask`); Vulkan applies it. Accepted as a Tier-1 limitation, still a divergence. | S–M | `specs/tracking/threading-audit.md:33-36` |
| A7 | GLES `SetImmediates` is recorded and bounds-checked but never read by the draw path — a silent gap. Unreachable today because `max_immediate_size` is 0 and core rejects first; should become an explicit `HalError` if the limit is ever raised. | S | `specs/blocks/67-gles-backend.md:282`, `yawgpu-hal/src/gles/adapter.rs:820` |

## Priority B — missing standard WebGPU surface

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| B1 | **`wgpuGetProcAddress` is not implemented** — the only `wgpu*` symbol declared in `webgpu.h` and absent from the library (201/202). Blocks any consumer that resolves entry points by name. | M (mechanical ~200-entry table; must track the header) | `yawgpu/ffi/webgpu-headers/webgpu.h`; `nm` diff per `webgpu-h-export-gap-abi-diff` |
| B2 | **Binding arrays**: `bindingArraySize > 1` rejected outright in core. | M–L | `yawgpu-core/src/bind_group_layout.rs:37` |
| B3 | **External textures are Metal-only.** Vulkan is deterministically rejected in core (the HAL would need the multiplanar plane/params bindings, not just driver acceptance of Tint's SPIR-V); GLES is a `HalError`. | L / L | `yawgpu-core/src/compute_pipeline.rs:128-146`; `specs/blocks/36-external-textures.md:165-174` |
| B4 | **~50 remaining CTS `#[ignore]`s**, of which three are genuine core gaps: vertex-buffer draw OOB `lastStride`, dual-source-blending validation, storage-texture format/access in render auto-layout. The rest is test wiring (`create_pipeline_at_over` matrices, encoder matrices gated on the eager-setBindGroup gap, bundle `maxColorAttachments`, required-limit `validate`). | S–M each | `specs/tracking/cts-coverage.md:175-201` |
| B5 | **Surface capabilities are fixed constants**: formats BGRA8Unorm / RGBA8Unorm only (no sRGB, no rgba16float), present mode Fifo only, alpha Opaque only. The Vulkan HAL already maps Immediate / Mailbox / FifoRelaxed; the FFI membership check rejects them before the HAL is reached. | S–M | `yawgpu/src/ffi/mod.rs:1229-1236`, `yawgpu-hal/src/vulkan/surface.rs:746-749` |
| B6 | Tier-2 read-write storage format breadth ambiguous against the current spec (yawgpu grants `R32Float`, `RGBA16Float`, `RGBA32Float`). Needs a spec check more than code. | S | `specs/tracking/format-completeness-audit.md:23-27` |
| B7 | Multi-threading correctness beyond what the ported tests exercise. | L | `specs/SPEC.md:53` |

## Priority C — platform / presentation coverage

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| C1 | **Android `ANativeWindow` surface on Vulkan is a stub** — returns `SwapchainCreationFailed("Android native window surface not implemented")`. | M | `yawgpu-hal/src/lib.rs:382` |
| C2 | **Xlib / Wayland / XCB surface sources are accepted by validation but inert** — no `VkSurfaceKHR` is created for them (only `CAMetalLayer` and Win32 HWND reach a Vulkan surface). Linux has no windowed presentation on any backend. | M each | `yawgpu/src/ffi/mod.rs:1238-1247`, `yawgpu-hal/src/vulkan/mod.rs:131-182`; `specs/blocks/85-windows-surface.md:269-279` |
| C3 | GLES Linux windowed presentation; the headless `EGL_PLATFORM_DEVICE_EXT` cascade must be skipped for window surfaces when it is added. | M | `yawgpu-hal/src/gles/egl.rs:53`; `specs/blocks/67-gles-backend.md:502-509` |
| C4 | **CI runs only the default (Noop) gates.** Feature-gated clippy (`vulkan` / `metal` / `gles` / `tiled` / `shader-passthrough`) never runs in CI, so lint rot reaches contributors first; no real-GPU CI for the 290 manual tests; no Windows example-build job. | M (infra) | `.github/workflows/ci.yml`; `specs/tracking/toolchain-clippy-1-98.md:110-126`; `specs/blocks/85-windows-surface.md:276-278` |

## Priority D — GLES Tier 2 residue

Campaign state: raw CTS fail 30,408 → 6,261 (crocus / Haswell host, now
gone). The remaining tail is fragmented; each row is an independent
investigation.

| # | Item | Difficulty | Evidence |
|---|---|---|---|
| D1 | **MSAA per-sample behaviour** — `multisample.mask` / `@builtin(sample_mask)` and alpha-to-coverage do not gate per-sample colour/depth writes (~1,510 fails). Documented as the next deep-debug slice. | L | `specs/tracking/cts-gles-sweep-0705.md:1206-1218` |
| D2 | **`maxBindingsPerBindGroup` off-by-one** — the limit is a plain `min` of GL maxima with no reservation for Tint's texture-metadata UBO. Mechanism confirmed, not fixed. | S | `specs/blocks/67-gles-backend.md:729-745` |
| D3 | Query sets: timestamp not advertised although `EXT_disjoint_timer_query` exists; `submit_resolve_query_set` writes zeroes from a software vector rather than real GL query objects. | M | `yawgpu-hal/src/gles/queue.rs:171-201`; `specs/blocks/67-gles-backend.md:279` |
| D4 | Stencil-aspect texture-to-buffer readback (~900 fails); a compute-image path could lift it. | M | `specs/blocks/67-gles-backend.md:718-721` |
| D5 | 1D textures / `texture_storage_1d` — height-1 2D emulation (Dawn precedent), not started. | M | `specs/blocks/67-gles-backend.md:250, 831-834` |
| D6 | Texture views on hosts without `glTextureView` (copy-based emulation); texture-to-texture copy without `GL_EXT_copy_image`. | M–L / M | `yawgpu-hal/src/gles/queue.rs:648, 4007` |
| D7 | Cube / cube-array colour attachments; framebuffer fetch via `EXT_shader_framebuffer_fetch`; compressed-format advertisement. | S–M / M / M | `yawgpu-hal/src/gles/queue.rs:1618, 1316`; `specs/tracking/format-completeness-audit.md:34-37` |
| D8 | Latent self-deadlock: `Drop` impls of GLES buffer / pipeline / sampler / texture inners acquire `with_current_context`; dropping the last `Arc` inside a context closure re-deadlocks. A guard is cheap. | S | `specs/tracking/cts-gles-sweep-0705.md:136-145` |
| D9 | `unorm8x4-bgra` renders R/B swapped on hosts without `EXT/ARB_vertex_array_bgra` (execution-only divergence; shader swizzle emulation deferred). | M | `specs/blocks/67-gles-backend.md:704-710` |
| D10 | **Manual verification owed**: ANGLE re-confirmation after the Tint migration, Windows WGL/NVIDIA sweep, catalogue re-sweep on the NVIDIA Linux host (README GLES table is still the Haswell snapshot). | Manual | `specs/tracking/tint-integration-refactor.md:219-221`; `specs/blocks/67-gles-backend.md:647-659` |
| D11 | **Hardware-blocked Vulkan verifications**: clip-distances execution (MoltenVK cannot lower `ClipDistance`), texture-component-swizzle depth path, ETC2/ASTC probes E8/E9, native Windows immediates sweep. | Manual | `specs/tracking/clip-distances.md:23-31`, `texture-component-swizzle.md:13`, `texture-compression-vulkan.md:96-98`, `immediates-block94.md:105-115` |

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
| E1 | Block 97 **S4** (re-measure + CTS re-run, record a Run section). **S3 is effectively done**: `HalCopy` carries only `RenderPassCommandStream`, no legacy per-draw `HalRenderPass` variant remains, and GLES replays the stream (`88cfe58`). The perf ledger's "S3 and GLES legacy still open" line is stale. | S | `yawgpu-hal/src/command.rs:154-173`; `specs/tracking/perf-dawn-baseline.md:447-449` |
| E2 | Residual gaps vs Dawn with no structural lever: `write_buffer_then_wait` 1.53×, `bindgroup/create_destroy` 1.33×, `render_draw` 224 vs 99 ns. | Diffuse | `specs/tracking/perf-dawn-baseline.md:435-446` |
| E3 | Deferred cache layers: real `VkPipelineCache`; Metal `MTLLibrary`-by-source cache (measure-first); whole-module Tint IR clone (needs an upstream Tint API). | M / M / L | `specs/blocks/95-shader-compile-cache.md:35-41`; `yawgpu-tint/src/lib.rs:4856` |
| E4 | refactor-dedup deferrals: double `Arc<core::X>` in FFI Impl structs, Metal derived-view cache + per-draw scratch reuse, unified `HalRenderStreamState` interpreter, GLES draw-time dirty tracking + binding index maps, subpass attachment validation dedupe. | M each | `specs/tracking/refactor-dedup.md:38-56, 98-101, 191-196` |
| E5 | `tiled`: depth-format transient attachments, MSAA subpass input (Dawn-deferred; 2-arg `inputAttachmentLoad` parked on a Dawn resolver issue), queries inside subpass passes, GLES mapping, `SubpassAttachmentResource::Transient` arm TODOs. | M / L / M / L | `specs/blocks/55-tiled-rendering.md:45-53, 501-507, 639-648`; `yawgpu-core/src/subpass.rs:244`, `yawgpu-hal/src/command.rs:745` |
| E6 | Vulkan limits: `maxFragmentCombinedOutputResources` redistribution, NVIDIA 2 GB storage cap, `max_color_attachment_bytes_per_sample` not queried. | S | `yawgpu-hal/src/vulkan/mod.rs:1089`; `specs/tracking/adapter-limits.md:39-40, 90` |
| E7 | Metal subgroup size range hard-coded `(32, 32)`; f16 review MINORs (write-only `VulkanDeviceInner` flags, duplicate feature query). | S | `specs/tracking/subgroups.md:106-108`; `specs/tracking/shader-f16.md:177-183` |
| E8 | `shader-passthrough`: B5 Phase Review outstanding; gating SPIR-V passthrough behind `WGPUInstanceFeatureName_ShaderSourceSPIRV` deferred. | S / S | `specs/blocks/33-shader-passthrough.md:257-270` |

## Priority F — doc drift (one cleanup commit)

- `specs/tracking/perf-dawn-baseline.md:447-449` — "S3 and GLES legacy path still open" (see E1).
- Repo-root `HANDOFF.md` — the texture-formats-tier2 report it carries was fixed in Block 72; reads as open.
- `specs/blocks/README.md:22-33` — lists `60-backends.md` / `70-surface-query-errorscope.md`, which exist as `60-real-backends.md` / `70-finalize.md`.
- `specs/reference/workflow.md:151` — "The repo is not yet a git repository".
- `specs/tracking/cts-coverage.md:2093-2115` — naga-era "intentionally not fixed" under-validations; re-measure under Tint.
- `specs/tracking/refactor-dedup.md:53` — "GLES real-GPU runs are Windows ANGLE only"; a Linux EGL host now exists.
- `specs/tracking/cts-coverage.md:1351-1358` — "known core gaps" list partly stale (inter-stage matching already implemented, per `:195`).
- `README.md:753-772` — GLES table is the Haswell/crocus snapshot (re-sweep blocked, see D10).

---

## Recommended order

1. **A1** timestamp-query — the only standard feature that is advertised yet non-functional. Follows the Block 62–71 shape: 3 slices (Metal counter sample buffers → Vulkan `vkCmdWriteTimestamp` + resolve copy → e2e on both), then CTS re-confirm.
2. **A4** hard-coded `supports_*` — small, certain wins; do `timestamp_query` and `depth32float_stencil8` on both Tier-1 backends first.
3. **A2** `wgpuCommandEncoderWriteBuffer`.
4. **B1** `wgpuGetProcAddress`.
5. **A3** lazy zero-init Stage 2.
6. **B4** the three genuine core gaps behind the remaining CTS ignores.
7. **B5** surface capabilities (cheap, and unblocks real presentation use).
8. Then C (platform surfaces, CI) and D (GLES) as separate initiatives; F alongside any of the above.

---

## Progress log

### 2026-09-22 — A4, A2, B1, A1 executed (Blocks 99–102)

| Item | Result |
|---|---|
| A4 | **Done.** Block 99: `1c597cd` (HAL), `3c502dd` (e2e). Metal `timestamp_query` / `float32_filterable` and Vulkan `timestamp_query` / `depth32float_stencil8` / `rg11b10ufloat_renderable` / `bgra8unorm_storage` / `float32_filterable` now follow Dawn's device queries; the rest are documented literals. |
| A2 | **Done.** Block 100: `aed7ded`. Encoder writes are snapshotted and lowered in command order through the staging pool. |
| B1 | **Done.** Block 101: `6482313`. 202/202 header names resolve. |
| A1 | **Done (S1–S3), S4 in progress.** Block 102: `8fe363a` (core + Noop) + the Metal/Vulkan HAL commit that follows it. e2e 4/4 on M2 Metal and MoltenVK. |

New findings recorded while executing (each with evidence in the block spec):

- **Metal end-of-encoder counter sampling fails after a blit encoder** (Apple8 / macOS 26): Dawn's `endOfEncoderSampleIndex` form errors the command buffer; yawgpu samples at `startOfEncoderSampleIndex` instead. Watch when Dawn or macOS changes (Block 102 R2).
- **MoltenVK cannot translate a struct load from a storage buffer** in Tint's robustness-clamped SPIR-V (`no matching constructor for initialization of 'Timestamp'`); the conversion shader uses a flat `array<u32>` (Block 102 R5). Any future internal shader must avoid whole-struct loads from `var<storage>`.
- **A9 (new, S):** Metal `encode_compute_buffer_sizes` / the render equivalent pass `setBytes` with `N * 4` bytes for `tint_storage_buffer_sizes`, but Tint declares `uint4[ceil(N/4)]` (16-byte elements): the Metal API validation layer reports `argument tint_storage_buffer_sizes[0] ... has space for 4 bytes, but argument has a length(16)` on every pipeline that uses `arrayLength`. Harmless on hardware (only the first lanes are read) but an API-validation error; pad the byte slice to a multiple of 16. Found by running the timestamp e2e under `METAL_DEVICE_WRAPPER_TYPE=1`. Evidence: `yawgpu-hal/src/metal/encode.rs` `msl_buffer_sizes`; `third_party/dawn/src/tint/lang/msl/writer/raise/raise.cc:129`.
- **A5 correction:** the Vulkan HAL *does* barrier around compute passes since F-106 (`c723a82`); Block 102 widened those scopes rather than adding new ones. A5's per-subresource layout item stands.
- **Metal API validation layer is worth a periodic run**: `METAL_DEVICE_WRAPPER_TYPE=1 MTL_DEBUG_LAYER=1` on the e2e suites surfaced A9 immediately; the CTS runs do not enable it.
