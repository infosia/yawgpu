# Phase 14 — Tiled rendering (TBDR mobile extension)

Status: **COMPLETE** (B1-B6 + B8 done; B7 REMOVED; Phase 14 Review CLOSED — 0C/7M/11m, all MAJOR + kept MINOR fixed). Commits `phase-14: B1` → `phase-14: phase review`. Rules/plan: `../blocks/55-tiled-rendering.md`.
Roles/loop: `../reference/workflow.md`. Depends on Phase 13's `yawgpu.h` +
feature scaffolding (A0).

**Vendor extension**, gated by cargo feature **`tiled`** (default off; umbrella
`mobile` enables it with `shader-passthrough`). Transient/memoryless
attachments, multi-subpass render passes, subpass-input / framebuffer fetch,
subpass-aware render pipelines, and scaffold-only programmable tile dispatch.
**Vulkan + Metal only**; Noop accepts handles, no GPU work. Purely additive.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy --workspace
--all-targets -- -D warnings` green on **Noop**, run in **both** default and
`--features tiled` (and with `metal`/`vulkan` for backend slices); `missing_docs`
in both. Feature-gated `pub fn`s carry unit tests under the same `#[cfg]`.
Real-GPU e2e (`#[ignore]`) run **by Claude directly** (Apple Silicon Metal;
Vulkan via MoltenVK). **Phase ends with the mandatory Phase Review**
(`phase-14-review.md`).

naga subpass IR is already present in the pinned naga (see
`../reference/dependencies.md`) — B3 enables + wires it, it does not implement
naga.

## B1 — features + TiledCapabilities query  *(☑ DONE)*

Done: core `Feature::{MultiSubpass,TransientAttachments,ShaderFramebufferFetch,
ProgrammableTileDispatch}` (gated); `Adapter::features()` advertises them only
when `tiled` is on AND `tiled_features_supported(backend)` (pure helper: Noop
false / Metal+Vulkan true). `TiledCapabilities` + `Adapter::tiled_capabilities()`
(Noop zeros; Metal/Vulkan from limits). C: `yawgpu.h` `YAWGPU_HAS_TILED` block
(feature-name `#define`s, `YaWGPUTiledCapabilities` + INIT macro, query decl);
Rust `#[repr(C)]` mirror + `YaWGPUFeatureName_*` consts; `conv/feature.rs`
maps the vendor names both ways; `yawgpuAdapterGetTiledCapabilities` FFI.
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; metal/vulkan/metal,tiled/vulkan,tiled `--tests` compile; metal example
builds with `-DYAWGPU_EXTENSIONS=tiled`. T1
(`yawgpuAdapterGetTiledCapabilities_writes_noop_zeros_and_rejects_null_out`),
T2 (`map_feature_accepts_tiled_vendor_feature_names`,
`wgpuAdapterHasFeature_reports_...`). *Real-GPU (Claude):*
`metal_tiled_features_and_capabilities_are_advertised`,
`vulkan_tiled_features_and_capabilities_are_advertised` → **both passed**.

## B2 — transient attachment resource  *(☑ DONE)*

Done: `core::transient_attachment` module — `TransientAttachment` Arc resource +
`TransientAttachmentDescriptor`/`TransientSizeMode`; `Device::create_transient_attachment`
(Explicit → eager HAL alloc, MatchTarget → descriptor only / `hal=None`,
zero-size Explicit → error). HAL `HalTransientAttachment` enum +
`create_transient_attachment`: Vulkan `VkImage`(`TRANSIENT_ATTACHMENT|INPUT_ATTACHMENT|
COLOR_ATTACHMENT`) bound to `LAZILY_ALLOCATED` mem with `DEVICE_LOCAL` fallback
(image cleaned up on every error path) + view; Metal `MTLStorageMode::Memoryless`;
Noop placeholder. C: yawgpu.h handle + `YaWGPUTransientSizeMode` +
`YaWGPUTransientAttachmentDescriptor` + INIT + `yawgpuDeviceCreateTransientAttachment`/
`AddRef`/`Release`; Rust `#[repr(C)]` mirror; FFI Arc handle.
*Scope moved to B4:* MatchTarget extent resolution (T5b) + `tile_memory_check` (T6).
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings` green;
metal/vulkan/*,tiled `--tests` compile; metal example builds with tiled. T3
(`yawgpuDeviceCreateTransientAttachment_returns_handle_and_refcounts`), T5
(`device_create_transient_attachment_validates_explicit_and_defers_match_target`).
*Real-GPU (Claude):* `metal_explicit_transient_attachment_allocates_without_device_error`,
`vulkan_explicit_transient_attachment_allocates_without_device_error` → **both passed**.
*Follow-up (do in B4):* the Vulkan transient image hardcodes `COLOR_ATTACHMENT`
usage + color-aspect view; depth-format transients need `DEPTH_STENCIL_ATTACHMENT`
usage + depth aspect (wire + test when depth subpass attachments land).

## B3 — subpass IR + input-attachment binding  *(☑ DONE)*

Done (naga subpass IR is unconditional in the pinned fork — no feature to
enable, just wiring): `BindingLayoutKind::InputAttachment { sample_type,
multisampled }` + validation/visibility/count arms. Reflection: naga
`ImageClass::Subpass { Color { kind } }` → `ReflectedResourceBindingKind::InputAttachment`
→ `reflected_bind_group_layout_entry`; shader scalar-kind vs declared layout
`sample_type` checked at pipeline-layout derivation (T9). C: `yawgpu.h`
`YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT` + `YaWGPUInputAttachmentBindingLayout`
+ INIT; Rust `#[repr(C)]` mirror + SType const; `conv/bind.rs` decodes the chained
entry → `InputAttachment` kind.
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; metal/vulkan/*,tiled `--tests` compile; metal example builds with tiled.
Tests: T7 (`map_bind_group_layout_descriptor_decodes_input_attachment_entry`),
T8 (`subpass_input_shader_generates_spirv_and_msl_status_is_known`,
`reflects_subpass_input_binding_kind`), T9
(`subpass_input_explicit_layout_checks_sample_type`). Noop-complete; real-GPU
subpass execution is B8.
*Note:* SPIR-V subpass codegen works; MSL **global** `subpass_input` needs the
pass-local color-slot map → supplied in B4 (test tolerates the not-yet path).

## B4 — pass layout object + multi-subpass render pass

Split for reviewability (matches the reference's scaffolding-then-impl pattern):
**B4a** = pass layout + pass lifecycle + validation, Noop-complete; **B4b** =
real Vulkan/Metal pass execution + MatchTarget alloc + tile_memory_check + the
B2 depth-format usage fix + e2e.

### B4a — pass layout + pass lifecycle core (Noop)  *(☑ DONE)*

Done: new `core::subpass` module — `SubpassPassLayout` Arc resource +
`Device::create_subpass_pass_layout` with validation (T10: subpass/color/input
counts vs `tiled_capabilities`, input source range). `SubpassRenderPass`
encoder via `CommandEncoder::begin_subpass_render_pass`: eager-dispatch guard
(T13, must be first encoder op), attachment↔layout consistency (T14),
MatchTarget resolution + `Arc` retention of layout/views/transients across the
caller's `Release` (T12), `next_subpass`/`end`/Drop-safe. HAL
`begin/next/end_subpass_render_pass` (enum-dispatch): Noop records; Vulkan/Metal
arms return `HalError` "subpass pass not yet implemented" (no panic — B4b
replaces). C: `yawgpu.h` pass-layout + tagged attachment-binding + encoder
surface + INIT macros; Rust `#[repr(C)]` mirrors + conv;
`yawgpuDeviceCreateSubpassPassLayout`/`AddRef`/`Release`,
`yawgpuCommandEncoderBeginSubpassRenderPass`/`NextSubpass`/`End`/`AddRef`/`Release`.
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; metal/vulkan/*,tiled `--tests` compile (subpass arms return `HalError`);
tiled example builds. Tests: T10 (`subpass_pass_layout_validates_inputs_and_counts`),
T12 (`subpass_render_pass_lifecycle_retains_resources_and_resolves_match_target`),
T13 (`subpass_render_pass_requires_first_encoder_operation`),
T14 (`subpass_render_pass_validates_attachment_consistency`). Draw machinery is
B5; real backend execution + T11 e2e is B4b.

### B4b — real Vulkan/Metal pass execution  *(☑ DONE)*

Done: HAL Vulkan/Metal `begin/next/end_subpass_render_pass` implemented (B4a's
`HalError` stubs replaced). Vulkan `create_subpass_render_pass` builds a
multi-subpass `VkRenderPass` (attachment/subpass/dependency descs incl. depth
refs + input refs) + `VkFramebuffer` over the resolved views;
`vkCmdBeginRenderPass`→`vkCmdNextSubpass`→`vkCmdEndRenderPass` with clears. Metal
`MTLRenderPassDescriptor` + single `MTLRenderCommandEncoder`; `next_subpass`
advances internal state. **tile_memory_check (T6)**: `tile_memory_fits_budget`
pure fn + `metal_tile_memory_budget_bytes`; over-budget memoryless footprint →
`HalError`. **B2 depth follow-up folded in**: depth-format transients now use
`DEPTH_STENCIL_ATTACHMENT` usage + depth aspect.
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; metal/vulkan/*,tiled `--tests` compile; tiled example builds. T6:
`tile_memory_budget_check_accepts_equal_and_rejects_over_budget`.
*Real-GPU (Claude):* Metal `metal_clear_only_subpass_pass_submits_without_device_error`
+ `..._accepts_memoryless_transient_color` → **passed**; Vulkan
`vulkan_clear_only_subpass_pass_submits_without_device_error` → **passed** (no
validation errors). (draws/pipelines = B5; full deferred-shading demo = B8.)

## B5 — subpass pipeline + dedicated draw encoder

Split for reviewability (the real-backend draw execution is the heavy half, like
B4): **B5a** = subpass-pipeline descriptor/validation + draw-command recording +
resource tracking, Noop-complete; **B5b** = real HAL draw execution + Vulkan
subpass-pipeline↔cached-multi-subpass-pass compatibility + INPUT_ATTACHMENT
descriptor binding + Metal color-slot map + the 2-subpass draw+read e2e.

### B5a — subpass pipeline + draw recording (Noop)  *(☑ DONE)*

Done: `SubpassRenderPipelineDescriptor` (base RP descriptor + `passLayout` +
`subpass_index`); `Device::create_subpass_render_pipeline` +
`validate_subpass_render_pipeline_descriptor` (T15: color/depth formats vs the
layout's subpass, subpass-index range, layout-match). `SubpassRenderPass`
draw methods (`set_pipeline`/`set_bind_group`/`set_vertex_buffer`/
`set_index_buffer`/`draw`/`draw_indexed`/`set_viewport`/`set_scissor_rect`)
record into the active subpass + register resource tracking (T16). C:
`yawgpu.h` `YaWGPUSubpassRenderPipelineDescriptor` + INIT +
`yawgpuDeviceCreateSubpassRenderPipeline` + the
`yawgpuSubpassRenderPassEncoder*` draw fns; Rust `#[repr(C)]` mirror + conv
(embedded base descriptor). `RenderPipeline::new_subpass` carries `subpass_index`.
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; metal/vulkan/*,tiled `--tests` compile; tiled example builds. Tests: T15
(`subpass_render_pipeline_validates_layout_formats_and_subpass_match`), T16
(`subpass_render_pass_draw_records_active_subpass_and_resources`).
*Gap → B5b:* no HAL changes yet — draws are recorded in core but not executed on
a real backend; the subpass pipeline is not yet built against the cached
multi-subpass `VkRenderPass`; no input-attachment descriptor binding / Metal
color-slot map; no e2e.

### B5b — real backend draw execution + input wiring + e2e  *(☑ DONE)*

Done: HAL subpass draw recording (Vulkan command buffer / Metal
`MTLRenderCommandEncoder`); Vulkan subpass `VkGraphicsPipeline` built against the
**cached multi-subpass `VkRenderPass`** with `.subpass(subpass_index)` (via
`cached_subpass_render_pass_for_layout`); `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT`
sets wired from the layout's input source mapping; Metal color-slot map.
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; metal/vulkan/*,tiled `--tests` compile; tiled example builds.
*Real-GPU (Claude):* **Metal** `metal_two_subpass_draw_subpass_load_readback`
(subpass 0 writes red G-buffer → subpass 1 `subpassLoad`s + swizzles → **green**
readback) → **passed** — verifies the subpass-input read end-to-end.
**Vulkan**: same e2e present but **self-skips on MoltenVK** (the only Vulkan
driver here): MoltenVK can't translate Vulkan subpass-input shaders into its
argument-buffer MSL (`Argument buffer resource base type could not be
determined`), and the read returns wrong values even with arg buffers off — a
**MoltenVK limitation**, not a yawgpu bug. The Vulkan input-attachment impl is
Vulkan-spec-compliant; the green-readback assertion runs on a **native Vulkan
driver** (Apple-GPU detection skips it here; per user, native-Vulkan verification
is done off this machine). Other Vulkan tiled e2e (advertise / transient alloc /
clear-only subpass) pass on MoltenVK.
*Noted gap (separate, future):* yawgpu's Vulkan `WGPUAdapterInfo` doesn't report
the real driver name (vendor hardcoded "yawgpu", description empty) — MoltenVK
detection falls back to the Apple-GPU heuristic.

### B5c — native-Vulkan input-attachment correction  *(☑ DONE)*

Running the full Phase 13/14 e2e on a **native Windows Vulkan driver** (the first
time `vulkan_two_subpass_draw_subpass_load_readback` ran outside MoltenVK's
self-skip) exposed that **B5b's "`VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` sets wired
from the layout's input source mapping" was not actually implemented** — only the
render-pass-level `pInputAttachments` refs existed. The descriptor-set side was
absent (`HalBufferBindingKind` had no `InputAttachment`, so the auto-derived
descriptor set layout never declared the input, and `update_subpass_descriptor_sets`
only wrote buffers), so `subpassLoad` read undefined → black. It went undetected
because MoltenVK self-skips this test and no native Vulkan driver had run it.
Three distinct fixes (all Noop-green + native-Vulkan-verified):
1. **Core bind-group validation** (`pass.rs` `validate_subpass_pipeline_bind_groups`,
   used by `subpass.rs` draw): input-attachment-only bind group layouts are
   auto-wired by the pass, so they no longer require a caller `set_bind_group`
   (previously failed `"pipeline requires a missing bind group"` on **all**
   backends — so B5b's Metal "passed" claim is not reproducible with the committed
   code either). Unit test
   `subpass_render_pass_draw_auto_wires_input_attachment_bind_group`.
2. **Vulkan input-attachment descriptors** (core + HAL): `HalBufferBindingKind::
   InputAttachment` (tiled-gated); core `input_attachment_hal_bindings` appends
   them to the Vulkan subpass pipeline's descriptor bindings (Metal still uses the
   color-slot map); `create_descriptor_set_layouts` emits `INPUT_ATTACHMENT` with
   FRAGMENT-only stage flags (VUID-…-01510); descriptor pool sizes it;
   `update_subpass_descriptor_sets` resolves each input's source attachment →
   framebuffer view and writes a `SHADER_READ_ONLY_OPTIMAL` (depth →
   `DEPTH_STENCIL_READ_ONLY_OPTIMAL`) image descriptor. Unit test
   `input_attachment_hal_bindings_extracts_only_input_attachment_entries`.
3. **Texture usage** (`vulkan/format.rs`): under `tiled`, render-attachment
   textures also get `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` so a persistent G-buffer
   can be read as a subpass input (was failing
   `VUID-VkFramebufferCreateInfo-pAttachments-00879` /
   `VkWriteDescriptorSet-descriptorType-00338`).

Also fixed the unrelated `e2e_vulkan_render` clear-color readback assertions
(`[26,51,77,255]`): float→unorm8 of `0.1/0.2/0.3` lands on `.5` rounding ties, so a
native driver returns `[25,51,76,255]` while MoltenVK rounds up — now compared with
a ±1 tolerance (`contains_pixel_approx`).
*Real-GPU (Claude, native Windows Vulkan):* full `--features vulkan,tiled --ignored`
suite **16/16 passed** (incl. `vulkan_two_subpass_draw_subpass_load_readback` →
**green**, validated under `VK_LAYER_KHRONOS_validation` with **0 validation
errors**). *Gate:* default + `shader-passthrough,tiled` test/`clippy -D warnings`
green; `vulkan,tiled` clippy + `--tests` green.
*Apple-Silicon re-verification (Claude, done):* with B5c applied, Metal
`--features metal,tiled --test e2e_metal_tiled -- --ignored` → **5/5 passed**,
incl. `metal_two_subpass_draw_subpass_load_readback` → **green** (fix 1's
auto-wire-input-attachment-bind-group is correct on Metal). Noop default +
`tiled` + `shader-passthrough,tiled` test/`clippy -D warnings` green;
metal/vulkan/metal,tiled/vulkan,tiled `--tests` compile; MoltenVK
`vulkan,tiled --ignored` 4/4 (2-subpass self-skips, no regression). B5 (a/b/c)
fully verified across Noop / Metal / native-Vulkan / MoltenVK.

## B6 — framebuffer fetch path detection  *(☑ DONE)*

Done: Vulkan `FramebufferFetchPath` detection at device setup
(`EXT_SHADER_TILE_IMAGE` → `TileImage`, else
`EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS` → `RasterOrderAttachmentAccess`, else
`Disabled`) + `supports_shader_framebuffer_fetch()`. core
`framebuffer_fetch_supported(backend, path)` pure helper (Metal true; Vulkan
!Disabled; Noop false) gates the `ShaderFramebufferFetch` advertisement (B1 had
it unconditional); `MultiSubpass`/`TransientAttachments` stay always-on for
tiled-capable backends. The Vulkan advertise e2e tolerates a driver without the
extensions (asserts `ShaderFramebufferFetch` only when reported). All vendor
`FramebufferFetchPath` usage is `#[cfg(feature = "tiled")]`-gated (a fix-pass
corrected a `--features vulkan`-without-tiled build break).
*Gate (Claude-run):* **all five build configs** compile (default / vulkan /
metal / metal,tiled / vulkan,tiled `--tests`); default + `--features tiled`
test/`clippy -D warnings` green. T17:
`framebuffer_fetch_support_is_backend_and_path_aware`,
`tiled_feature_advertise_gates_shader_framebuffer_fetch`.
*Real-GPU (Claude):* Metal `metal_tiled_features_and_capabilities_are_advertised`
→ passed (advertises `ShaderFramebufferFetch`); MoltenVK `vulkan,tiled --ignored`
4/4 (no extensions → `ShaderFramebufferFetch` honestly not advertised, no
false-fail).

## B7 — programmable tile dispatch scaffold  *(✗ REMOVED, post-B6)*

Originally landed as a scaffold (commit `a2cb43d`): a `YaWGPUTransientDispatchDescriptor`
+ `yawgpuSubpassRenderPassEncoderDispatchTransient` that returned an unsupported
error on every backend, plus a `Feature::ProgrammableTileDispatch` advertised on
tiled-capable backends. **Removed before Phase 14 Review** by design decision:
no backend implements it (reference forks don't either), no implementation plan,
and shipping a guessed-API scaffold ahead of any real impl only locks in a shape
that isn't driven by anything — while the advertise misleads consumers into
thinking the feature is usable. The numeric IDs are **reserved** in `yawgpu.h`
(`0x70010004` + future tile-dispatch SType / C entry-point names) so they
aren't reused for unrelated features.
*Gate (Claude-run, post-removal):* all five build configs compile (default /
vulkan / metal / metal,tiled / vulkan,tiled `--tests`); default + `--features
tiled` test/`clippy -D warnings` green; tiled example builds. No stray
`ProgrammableTileDispatch`/`DispatchTransient`/`TransientDispatchDescriptor`/
`dispatch_transient` symbols in `yawgpu/src`/`yawgpu-core/src`/`yawgpu-hal/src`/
`yawgpu.h` (only the reserved-ID comment remains). B6's
`tiled_feature_advertise_*` test was adjusted to no longer assert the removed
feature.

## B8 — examples + e2e + Phase Review

### B8 part 1 — C tiled_deferred example  *(☑ DONE, code-complete; on-Mac verification blocked by a pre-existing library defect)*

Done: `examples/tiled_deferred/` (offscreen 2-subpass G-buffer → `subpassLoad` →
PNG), guarded by `#if defined(YAWGPU_HAS_TILED)` with a no-op stub when the
extension is off. The example installs the framework's uncaptured-error
callback, samples the **center pixel** of the readback, and exits non-zero with
a printed RGBA + `FAILED` if it's not green within ±1 (silent-success bug
eliminated). `examples/framework/framework.{c,h}` gained a
`yawgpu_uncaptured_error_count()` helper used by the example. `examples/README.md`
documents the example + backend caveats.
`examples/CMakeLists.txt` now derives the cargo target dir from `(backend,
extensions)` (e.g. `target-metal-tiled` when `-DYAWGPU_EXTENSIONS=tiled` is set),
so an ext-on dylib is no longer clobbered by a subsequent ext-off build sharing
`target-metal` (the A0 follow-up risk materialized in B8 review).
*Gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings` green;
all five `--tests` configs compile; both `cmake -B build -DYAWGPU_FEATURE=metal
-DYAWGPU_EXTENSIONS=tiled` and the no-extensions tree build the
`tiled_deferred` target cleanly (C17-strict).
*Follow-up library defect (NOT a B8 issue; separate ticket):* on this Apple
Silicon, running C examples that link the cdylib with `YAWGPU_BACKEND=metal`
falls back to Noop — debug traces showed
`InstanceBackendSelection::Metal` → `MetalInstance::new()` succeeds, but
`HalInstance::Metal(_).enumerate_adapters()` returns **empty** when invoked
through the cdylib (the same call returns non-empty when invoked through the
rlib — Phase 13/14's many Metal e2e (cargo test) keep passing). Likely
cdylib-specific objc2 / Metal-device initialization issue. Capture in Phase 9
also routed through Noop without our noticing because its clear-only path looks
identical between Metal and Noop. **B8's `tiled_deferred` verifies fine on
native Vulkan (Windows, Apple-skip heuristic doesn't trigger off-Apple),** which
is the verification path used for B5c.

### B8 part 2 — Phase Review  *(☑ DONE — CLOSED)*

Clean Review (fresh no-context subagent, diff `f3db0de..0fc75fc` + block 55 +
CLAUDE.md + naming-conventions): **18 findings — 0 CRITICAL / 7 MAJOR /
11 MINOR.**

Triage + resolution:
- **MAJOR 1 (fixed)** — Vulkan device didn't enable the `EXT_SHADER_TILE_IMAGE`/
  `EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS` extension it detected; the
  ShaderFramebufferFetch advertise was a lie. Fix: pick the right extension by
  detected `FramebufferFetchPath` and add it to the logical-device extension list.
- **MAJOR 2 (fixed)** — Metal `MTLRenderPassDescriptor` only attached the first
  subpass's color slots, so later subpasses writing other slots had undefined
  targets. Fix: `subpass_color_attachment_indices` returns the **union** of all
  subpasses' indices; the e2e 2-subpass tests no longer rely on
  `contains_pixel`-any-match — they now assert the **center pixel** is green
  ±1 (`assert_center_pixel_approx`), so the union fix is testable. Mirror unit
  test added.
- **MAJOR 3 (fixed)** — Vulkan tiled depth attachment hardcoded
  `loadOp=CLEAR/storeOp=DONT_CARE`. Fix: map the binding's
  `depth/stencil_load_op/store_op` to `vk::AttachmentLoadOp/StoreOp`, gating
  stencil ops on the format's aspect.
- **MAJOR 4 (fixed)** — `MatchTarget` transient cached the first-pass extent.
  Fix: a second begin with a different extent now returns an error
  ("already resolved with a different extent"); same-extent reuse still works.
- **MAJOR 5 (fixed)** — `validate_subpass_pass_layout_descriptor` hardcoded
  `max_subpasses=4`. Fix: `tiled_capabilities_for_device` threads
  `Adapter::tiled_capabilities()` into the validation; Noop (zero caps) skips
  the count enforcement (documented exemption).
- **MAJOR 6 (fixed)** — `validate_subpass_render_pipeline_descriptor` dropped
  the wrapped `base.error`. Fix: the validator now propagates `descriptor.base.error`
  before its own checks; unit test added.
- **MAJOR 7 (fixed)** — `tile_memory_fits_budget` hardcoded `sample_count=1`.
  Fix: `subpass_memoryless_sample_count` computes the max across memoryless
  attachments and feeds it to the budget check.
- **MINOR (Vulkan final_layout, fixed)** — transient images now stay in
  `COLOR_ATTACHMENT_OPTIMAL` (or `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`) instead of
  the layout-vs-usage-inconsistent `TRANSFER_SRC_OPTIMAL`.
- **MINOR 7 (spec, fixed)** — added a `// 0x7001_0004 reserved` comment next to
  the `YaWGPUFeatureName_*` consts in `yawgpu/src/lib.rs`, mirroring `yawgpu.h`'s
  reservation block.
- **Deferred (recorded, accepted v1 approximations)** — silent Memoryless→Private
  fallback log; hardcoded `max_subpasses=4` / 256 KiB tile-budget defaults in
  adapter caps (validation now reads them; real per-backend refinement is a
  follow-up); `bind_group_layout_is_input_attachment_only` empty-layout
  handling; `Force32` C-only enum-width asymmetry; `pub type
  YaWGPUSubpassDependencyType = u32` doc sparseness;
  `SubpassRenderPassInner::Drop` re-locking pattern; `MultiSubpass`/
  `TransientAttachments` always-on advertise (currently capability-checked only
  for ShaderFramebufferFetch); inline `pub const` styling.

*Final gate (Claude-run):* default + `--features tiled` test/`clippy -D warnings`
green; **seven** backend `--tests` configs (default / metal / vulkan /
metal,tiled / vulkan,tiled / metal,mobile / vulkan,mobile) all compile.
Real-GPU re-runs: **Metal 5/5** incl. `metal_two_subpass_draw_subpass_load_readback`
with the tightened center-pixel assertion → **green pass**; MoltenVK
`vulkan,tiled` 4/4 (2-subpass self-skips, no regression). No open
CRITICAL/MAJOR → **Phase 14 COMPLETE**.

Pending separately (not Phase-14-blocking):
- ~~Native Vulkan (Windows) re-verification of MAJOR 1/2/3 effects on a real
  driver~~ — **DONE 2026-05-23.** `cargo test -p yawgpu --features vulkan,tiled
  --no-fail-fast -- --ignored` with `VK_LAYER_KHRONOS_validation`: tiled e2e
  **4/4 passed with zero validation errors** in the tiled section
  (`vulkan_tiled_features_and_capabilities_are_advertised`,
  `vulkan_explicit_transient_attachment_allocates_without_device_error`,
  `vulkan_clear_only_subpass_pass_submits_without_device_error`,
  `vulkan_two_subpass_draw_subpass_load_readback` → **green** center-pixel).
  MAJOR 1 (extension-enable for ShaderFramebufferFetch path), MAJOR 2 (Metal
  union — N/A on Vulkan), and MAJOR 3 (depth load/store op mapping) all hold
  on a real driver. Surfaced a separate set of pre-existing non-tiled Vulkan
  buffer/texture VUIDs (UNIFORM/STORAGE/VERTEX buffer-usage bits and image-view
  usage on transfer-only images) — tracked in [vulkan-buffer-texture-usage-vuids.md](vulkan-buffer-texture-usage-vuids.md);
  not Phase-14-introduced, not Phase-14-blocking.
- cdylib + Metal `enumerate_adapters` empty-return on Apple Silicon
  (cargo-test/rlib unaffected; C examples on Mac silently fall back to Noop).
- Vulkan adapter info doesn't report the real driver name; MoltenVK detection
  falls back to Apple-GPU heuristic.

C deferred-shading example (Metal + Vulkan) under `#ifdef YAWGPU_HAS_TILED`;
real-GPU e2e run by Claude and logged. Then the mandatory Phase Review
(`phase-14-review.md`): fresh subagent, CRITICAL/MAJOR/MINOR, fix in severity
order, no COMPLETE with open CRITICAL/MAJOR.
