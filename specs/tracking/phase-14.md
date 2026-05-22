# Phase 14 — Tiled rendering (TBDR mobile extension)

Status: **PLANNED**. Rules/plan: `../blocks/55-tiled-rendering.md`.
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

## B3 — subpass IR + input-attachment binding  *(☐ TODO)*

Enable naga subpass features under `tiled`; `YaWGPUInputAttachmentBindingLayout`
chained on BGL entry (`group`+`binding`); `BindingType::InputAttachment` through
bind-group-layout / pipeline-layout; pass-local auto-wiring (no caller bind
group/view); scalar-kind check.
*Accept:* T7 (noop) + T9 unit-tested; T8 e2e.

## B4 — pass layout object + multi-subpass render pass  *(☐ TODO)*

`YaWGPUSubpassPassLayout` Arc resource (the single compat source of truth;
Vulkan caches a compatible `VkRenderPass` on it). Then
`BeginSubpassRenderPass`/`NextSubpass`/`End` + encoder handle (refcount):
attaches the supplied views/transient handles to the layout (the `Transient`
branch carries the handle directly — no index table); Vulkan `VkFramebuffer`
over the views + input-attachment descriptor sets from the layout; Metal
single-encoder state machine; eager-dispatch ordering guard; layout-consistency
+ capability-limit checks; pass-lifetime retention of views/transients/layout.
*Accept:* T10, T13, T14 unit-tested; T11 (noop), T12 (noop) unit-tested;
T11/T12 e2e.

## B5 — subpass pipeline + dedicated draw encoder  *(☐ TODO)*

`yawgpuDeviceCreateSubpassRenderPipeline` referencing the **same**
`passLayout`+`subpassIndex` (Vulkan compat `VkRenderPass` from the layout, Metal
forward); the dedicated draw machinery
(`yawgpuSubpassRenderPassEncoderSet*/Draw*/...`) with resource tracking;
pipeline↔pass layout-match check at draw.
*Accept:* T15 (noop), T16 (noop) unit-tested; both e2e.

## B6 — framebuffer fetch path detection  *(☐ TODO)*

Vulkan `FramebufferFetchPath` (`TileImage`/`RasterOrderAttachmentAccess`/
`Disabled`) detection + `ShaderFramebufferFetch` advertise; Metal implicit.
*Accept:* T17 e2e.

## B7 — programmable tile dispatch scaffold  *(☐ TODO)*

`yawgpuSubpassRenderPassEncoderDispatchTransient` wired through C/core/HAL,
returns unsupported on every backend.
*Accept:* T18 unit-tested (returns unsupported on all backends).

## B8 — examples + e2e + Phase Review  *(☐ TODO)*

C deferred-shading example (Metal + Vulkan) under `#ifdef YAWGPU_HAS_TILED`;
real-GPU e2e run by Claude and logged. Then the mandatory Phase Review
(`phase-14-review.md`): fresh subagent, CRITICAL/MAJOR/MINOR, fix in severity
order, no COMPLETE with open CRITICAL/MAJOR.
