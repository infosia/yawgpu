# Phase 14.x review (post-cascade extensions)

## Why this exists

Phase 14 closed twice already: at `5616518` (original B-slice review) and
at `97e1818` (cascade re-open review for the silent-skip + Metal subpass-input
lowering bug). After `97e1818` a chain of Phase 14.x extensions landed to
support the flagship 3-subpass deferred-rendering example port (see
`phase-14.md` → "Phase 14.x extensions"). Per `specs/reference/workflow.md`
every batch of new code ends with a mandatory **fresh, no-context** Phase
Review before re-COMPLETE. This is that review record.

## Scope

Range: `97e1818..6cd881e` (8 commits of code; tracking commits ignored).

Files reviewed:
- `yawgpu-core/src/{bind_group,render_pipeline,subpass,format,texture,queue,command_encoder}.rs`
- `yawgpu-hal/src/{descriptors,format,lib,command}.rs`
- `yawgpu-hal/src/metal/{mod,pipeline,encode,test_helpers,device}.rs`
- `yawgpu-hal/src/vulkan/{mod,pipeline,encode,test_helpers}.rs`
- `examples/tiled_deferred/{main.c,math.h,gbuffer.wgsl,lighting.wgsl,composite.wgsl,CMakeLists.txt}`
- `specs/blocks/55-tiled-rendering.md`

## Findings

### CRITICAL

(none)

### MAJOR

- **[M1] Vulkan-feature test build broken.** `yawgpu-hal/src/vulkan/test_helpers.rs::render_descriptor`
  was not updated when `b94d780` added the `depth_stencil` field to
  `HalRenderPipelineDescriptor`. `cargo check -p yawgpu-hal --features vulkan
  --tests` failed with `error[E0063]: missing field 'depth_stencil'`. The
  per-crate `-p yawgpu --features vulkan,tiled --tests` gate the original
  commit relied on doesn't include `yawgpu-hal`'s lib tests, so the breakage
  slipped past acceptance.
- **[M2] Library-side `.expect()` panic.** `yawgpu-hal/src/metal/pipeline.rs::create_noop_depth_stencil_state`
  used `.expect("no-op MTLDepthStencilState creation cannot fail")` on Apple's
  nullable `newDepthStencilStateWithDescriptor:`. CLAUDE.md principle 3
  ("no panics in library code; FFI boundary only") forbids this; the sibling
  `create_depth_stencil_state` routed the same call through
  `shader_error(...)` correctly.
- **[M3] Metal subpass pipelines for non-depth subpasses skipped `setDepthAttachmentPixelFormat`
  propagation while the encoder had a depth attachment bound.** Per Apple
  the pipeline and encoder depth-attachment pixel formats must match; the
  `tiled_deferred` lighting/composite pipelines (which set
  `descriptor.depth_stencil = None`) left `depthAttachmentPixelFormat` at
  `Invalid`. The `af1bdd2` no-op `MTLDepthStencilState` fix neutralised
  test/write but not the format-compatibility check. Worked in release but
  undefined behavior under `MTL_DEBUG_LAYER=1`.

### MINOR

- ~~**[m1]** Third bind-group test
  (`validate_bind_group_descriptor_rejects_explicit_view_for_input_slot`)
  exits via the count check, not via the `(_, InputAttachment)` arm in
  `validate_bind_group_entry`; the arm is uncovered by direct test.~~
  **Closed 2026-05-25** — rewrote the layout / entry combination so
  the count check passes and the explicit input-attachment binding
  reaches the `(_, BindingLayoutKind::InputAttachment { .. })` arm.
- ~~**[m2]** `format_has_depth_aspect` / `format_has_stencil_aspect` duplicated
  in `metal/pipeline.rs`, `metal/encode.rs`, `vulkan/encode.rs`. Could be a
  single `pub(crate)` helper in `yawgpu-hal/src/format.rs`.~~
  **Closed 2026-05-25** — centralized both helpers in
  `yawgpu-hal/src/format.rs` and routed Metal / Vulkan callers through
  the shared definitions.
- ~~**[m3]** Metal silently drops `HalDepthStencilState.depth_bias*`;
  `create_depth_stencil_state` only reads compare / write / stencil_*.
  Metal needs `setDepthBias(...)` on the encoder, not the pipeline state —
  spec text in 55-tiled-rendering.md is misleading. The deferred demo
  doesn't use depth bias so the gap doesn't bite, but the contract drift
  is real.~~
  **Closed 2026-05-25** — `MetalRenderPipeline` now retains the
  depth-bias triple and both render-pass encoder bind sites apply it
  with `setDepthBias_slopeScale_clamp`.
- ~~**[m4]** New public HAL types (`HalDepthStencilState`, `HalStencilFaceState`,
  `HalStencilOperation`) + the `HalTextureFormat::Rgba16Float` variant lack
  direct unit tests at the HAL crate. CLAUDE.md principle 1; weak case since
  they're pure data shapes (exercised via Noop pipeline creation in
  yawgpu-core).~~
  **Closed 2026-05-25** — added inline HAL unit tests for the
  depth-stencil data shapes, stencil operations, `Rgba16Float`, and
  centralized aspect helpers.
- ~~**[m5]** Vulkan render-pass attachment description sets
  `stencil_load_op = CLEAR` and `stencil_store_op = STORE` unconditionally on
  depth attachments, including depth-only formats (`Depth32Float`,
  `Depth24Plus`, `Depth16Unorm`). Vulkan ignores these for formats without
  a stencil aspect, but the validation layer flags `BestPractices` warnings.~~
  **Closed 2026-05-25** — non-tiled Vulkan render-pass creation now
  gates depth and stencil load/store ops independently using the shared
  aspect helpers.
- **[m6]** `MetalRenderPipeline::depth_stencil_state` typed `Option<...>` but
  after `af1bdd2` always `Some(...)`; the encoder's `as_deref()` branches
  were dead.

## Triage

All three MAJOR findings are kept. All MINOR findings are kept (no false
positives). m6 was fixed alongside M1-M3 (it fell out of the same
change); m1–m5 were deferred at re-COMPLETE and then closed in a
2026-05-25 follow-up polish slice.

## Fixes

Landed in commit `a2d2ddd` ("phase-14.x: fix Phase 14.x review M1/M2/M3
(+ m6 cleanup)"):

- **M1**: Add `depth_stencil: None` to
  `yawgpu-hal/src/vulkan/test_helpers.rs::render_descriptor`. Verified
  `cargo check -p yawgpu-hal --features vulkan --tests` is clean.
- **M2**: `create_noop_depth_stencil_state` returns
  `Result<Retained<...>, HalError>`. Threaded `?` through
  `create_render_pipeline`.
- **M3**: `MetalDevice::create_subpass_render_pipeline` now synthesizes a
  no-op `HalDepthStencilState` carrying `pass_layout.depth_stencil_attachment.format`
  when the caller's `descriptor.depth_stencil` is `None` but the pass layout
  declares depth-stencil. The pipeline's `depthAttachmentPixelFormat` now
  matches the encoder's bound depth attachment per Apple's docs; the
  resulting `MTLDepthStencilState` is the no-op state (depthCompare=Always,
  depthWrite=false, stencil disabled) from M2's fallback path.
- **m6** (incidental): drop the `Option` wrapper on
  `MetalRenderPipeline::depth_stencil_state`; simplify the encoder bind
  call sites; update the struct's doc comment.

## Gate (post-fix)

All green on this M2 (2026-05-24, sandbox off):

- `cargo check -p yawgpu-hal --features vulkan --tests`: clean.
- `cargo clippy -p yawgpu-hal --features vulkan --all-targets -- -D warnings`: clean.
- `cargo clippy -p yawgpu-hal --features metal --all-targets -- -D warnings`: clean.
- `cargo clippy --workspace --all-targets --features yawgpu/tiled -- -D warnings`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo test -p yawgpu-core --features tiled --lib`: 98 passed.
- `cargo test -p yawgpu --features metal,tiled --test e2e_metal_tiled -- --ignored`: 5/5
  (existing 2-subpass smoke unchanged).
- `MTL_DEBUG_LAYER=1 ./tiled_deferred --verify` on Metal: exit 0, center
  pixel `(130, 60, 57, 255)`, **zero Metal validation errors** — confirms
  M3 closes the depth-attachment format-compatibility gap.

## Re-COMPLETE

No CRITICAL/MAJOR findings remain open. All MINOR findings are now closed:
m1/m2/m3/m4/m5 in the 2026-05-25 follow-up polish slice, and m6 alongside
M1-M3.

**Phase 14 (with all Phase 14.x extensions) re-stands COMPLETE at `a2d2ddd`.**
The deferred MINOR polish round is closed.

## Post-COMPLETE — clippy::too_many_arguments on `create_subpass_render_pipeline` *(☑ DONE 2026-05-25)*

Surfaced during the Phase 14.x MINOR closeout in `c135c9b`:
`cargo clippy -p yawgpu --features "metal tiled" --all-targets -- -D
warnings` was failing on
`MetalDevice::create_subpass_render_pipeline`'s 8-arg signature
(limit 7). The eight parameters each represent an orthogonal
concern (shader source, two entry-point names, the base render
descriptor, bindings, the pass-level layout, the subpass index)
and mirror the shape `HalSubpassRenderPipelineDescriptor` carries
from `yawgpu-core` — folding them into a struct would just
re-spell the same eight values without simplifying anything.
Closed by annotating the site with
`#[allow(clippy::too_many_arguments)]` and a brief comment.
`--features "metal tiled"` clippy is now green on this M2.
