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

## B1 — features + TiledCapabilities query  *(☐ TODO)*

Vendor `WGPUFeatureName` constants + `yawgpuAdapterGetTiledCapabilities`
backed by per-backend limits; Noop returns zeros.
*Accept:* T1, T2 unit-tested (noop) + e2e advertise check.

## B2 — transient attachment resource  *(☐ TODO)*

`YaWGPUTransientAttachment` Arc resource + descriptor; Vulkan
`LAZILY_ALLOCATED`+`TRANSIENT_ATTACHMENT|INPUT_ATTACHMENT` (fallback normal),
Metal `Memoryless` (fallback `Private`+warn); Metal `tile_memory_check`.
*Accept:* T3, T5 unit-tested (noop); T4, T6 e2e.

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
