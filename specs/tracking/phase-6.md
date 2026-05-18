# Phase 6 — Command encoding & passes

Status: **in progress** (P6.1 active). Rules: `../blocks/50-commands.md`.
Roles/loop: `../reference/workflow.md`. Gate (permanent): `cargo test
--workspace` + `cargo clippy --workspace --all-targets -- -D warnings`
green on Noop. **Phase ends with the mandatory Phase Review**
(`tracking/phase-6-review.md`).

Largest phase; 9 slices. First builds the encoder/pass state machine +
deferred-error model everything else hangs off. Carries P2/P3/P5
deferrals.

## P6.1 — Encoder/pass lifecycle + deferred-error model  *(☑ DONE)*

Done: core `CommandEncoder`(state machine Recording/Finished + open-pass
token + first-error + debug-group depth)/`CommandBuffer`/`RenderPass
Encoder`/`ComputePassEncoder`; deferred-error model (record-first-error,
`finish()` ⇒ open-pass/imbalance/first-error → error CommandBuffer,
first-match-wins; pass errors forwarded to parent;
`record_command_guard` seam for P6.2+). C1–C5,C36,C63,C85,C86 with
Dawn-matching timing (C2/C5/C63 at Finish; C1/C85 immediate; C3/C4 at
pass call). FFI 4 handles + create/finish/begin*/end/debug-group/
Release/AddRef. C1–C5/C36/C63/C85/C86 ported in
`yawgpu/tests/command_encoder_lifecycle_validation.rs` (9), gate green
(34 binaries). Committed `phase-6: P6.1`. **Deferred-error seam ready
for P6.2+.**

#### (original detail)

`WGPUCommandEncoder`/`WGPUCommandBuffer`/`WGPURenderPassEncoder`/
`WGPUComputePassEncoder` handle types; core encoder state machine
(Recording/Finished/Errored), command recording + first-error,
`Finish` ⇒ device error + error CommandBuffer; `BeginCompute/Render
Pass` (skeleton — full descriptor validation is P6.3/P6.4), pass `End`;
C1–C5, C36, C63, C85/C86; debug-group balance at Finish. **Foundational
— establishes the deferred-error machinery for all later slices.**

## P6.2 — Buffer copies / clear / encoder WriteBuffer  *(☑ DONE)*

Done: `CommandEncoder::{copy_buffer_to_buffer,clear_buffer,write_buffer}`
via `record_buffer_command` (hooks the P6.1 deferred-error model:
guard→finished=immediate / pass-open & validation failure→first-error,
surfaced at `finish()`). `validate_copy_buffer_to_buffer` (C8 usage,
C10 error/destroyed, C7 4-byte align, C6 checked bounds, **C9
same-buffer-always-error for size>0** via `Buffer::same` Arc::ptr_eq —
matches Dawn `CopyWithinSameBuffer`), `validate_clear_buffer`
(WHOLE_SIZE), `validate_encoder_write_buffer` (C83/C84). FFI 3 encoder
fns. C6–C10/ClearBuffer/C83/C84 ported in
`yawgpu/tests/command_buffer_copy_validation.rs` (6), gate green
(35 binaries). Committed `phase-6: P6.2`. (Submit-time buffer state →
P6.9.)

#### (original detail)

## P6.3 — Texture copies (B2T/T2B/T2T)  *(☑ DONE)*

Done: `validate_texel_copy_layout` generalized with
`require_bytes_per_row_alignment` (P3.4 queue=false, P6.3 buffer-copy=
true) + `label` — **shared, not duplicated**; P3.4
`queue_write_texture_validation` unregressed. `CommandEncoder::
{copy_buffer_to_texture,copy_texture_to_buffer,copy_texture_to_texture}`
via the deferred-error helper; `Texture::same`;
`texture_formats_copy_compatible` (equal | sRGB pair, C18); C11 256-
align, C12–C17 usage/sample/depth-stencil/bounds, C19–C22 T2T
usage/OOB/aspect/sample-count/same-texture, C79 aspect. FFI 3 fns +
conv `WGPUTexelCopyBufferInfo`/`…TextureInfo`. C11–C22/C79 ported in
`yawgpu/tests/command_texture_copy_validation.rs` (4), gate green
(36 binaries). Committed `phase-6: P6.3`.

#### (original detail)

## P6.4 — RenderPass descriptor  *(☑ DONE)*

Done: core `RenderPassDescriptor`/`RenderPassColorAttachment`/
`…DepthStencilAttachment`; `begin_render_pass(desc)` validates via the
deferred-error model (record-first-error → `finish()`).
`validate_render_pass_descriptor` + `validate_color_attachment`
(C24 RenderAttachment, C27 color-renderable, C32 load/store set, C33
clearValue finite), `validate_depth_stencil_attachment` (C27 aspect,
C28/C32 per-aspect ops, C33 depth clear finite∈[0,1]),
`validate_render_attachment_common` (C26 size via `render_extent`, C29
layer==1, C31 sample_count), `validate_resolve_target` (C30); C23
≥1-attachment, C24 count≤max, C25 sparse. C34/C35 accepted opaquely
(Defer→P8). conv decodes the descriptor + attachments. C23–C33 ported
in `yawgpu/tests/render_pass_descriptor_validation.rs` (5);
`command_encoder_lifecycle` strengthened to real descriptors (still 9,
unregressed). Gate green (37 binaries). Committed `phase-6: P6.4`.

#### (original detail)

## P6.5 — Pass draw/dispatch state + dynamic state  *(☑ DONE)*

Done: `PassEncoderState` extended (bound render/compute pipeline,
`BTreeMap<u32,BoundBindGroup>` incl. dynamic offsets, vertex-buffer
slot set, index buffer) + `record_pass_command` seam (reuses P6.1
`pass_command_guard` → parent `record_first_error`, surfaced at
`finish()`). Render `set_pipeline/set_bind_group/set_vertex_buffer/
set_index_buffer/draw/draw_indexed/set_viewport/set_scissor_rect/
set_blend_constant/set_stencil_reference`; compute `set_pipeline/
set_bind_group/dispatch_workgroups`. `validate_render_draw_state`
(C37 pipeline set + not-error, C41 declared vertex buffers set, C42
index buffer set), `validate_compute_dispatch_state`,
`validate_pipeline_bind_groups` (C38 missing group, C39
`bind_group_layouts_compatible` = Arc-identity for default/auto BGLs
**(P41)** / entry-list eq for explicit, error-BG reject),
`validate_dynamic_offsets` (C40 count == dynamic BGL entries, uniform/
storage alignment, buffer+binding bounds), `validate_viewport` (C56
finite + w/h≥0 + 0≤min≤max≤1), C57 scissor add-overflow, C58 blend
finite, C59 stencil no-op; compute DispatchWorkgroups each axis ≤
`max_compute_workgroups_per_dimension`. Helpers: `BindGroupLayout::
same`, `BindGroup::layout/entries`, `RenderPipeline::
required_vertex_buffer_count`. FFI: 10 `wgpuRenderPassEncoder*` + 3
`wgpuComputePassEncoder*` fns + `dynamic_offsets_slice`/
`map_index_format`/`map_color`. C37–C42/C56–C59/dispatch/P41 ported in
`yawgpu/tests/pass_state_validation.rs` (5; P41 uses distinct WGSL so
the two auto pipelines don't dedup ⇒ meaningful). Gate green (38
binaries, clippy clean). Committed `phase-6: P6.5`.
**Deferred → P6.6:** C43–C55 (index/vertex buffer usage/OOB/format/
align, draw count/OOB, indirect). Scissor attachment-size clamp →
P6.9/P7.

#### (original detail)

C37–C42, C56–C59, compute dispatch limits, P41 draw-time.

## P6.6 — Index/Vertex buffer + draw OOB + indirect  *(NEXT)*
C43–C55.

## P6.7 — RenderBundle  *(after P6.6)*
C65–C74.

## P6.8 — Debug markers  *(after P6.7)*
C60–C64 (across pass/bundle; encoder C63 done in P6.1).

## P6.9 — Resource usage tracking + submit  *(after P6.8; closes slices)*
C75–C82 (aliasing, read/write conflict, attachment+sample, submit-time
mapped/destroyed buffer). Then Phase Review.

## Phase 6 exit criteria

- C1–C33, C36–C84 covered by ported Rust tests green on Noop; C34/C35
  + occlusion-query Defer→P8; real-GPU Defer→P7; gate clean; CI green.
- `dawn-test-mapping.md`: the Phase-6 command/pass/bundle/copy/draw/
  usage rows ☑; `PipelineAndPassCompatibilityTests` ☑ (now incl.
  render-pass part); the P2/P3 deferred rows (`QueueSubmit…`,
  `WriteBufferTests`, `TextureSubresourceTests`) ☑.
- One commit per slice (`phase-6: <slice> — <short>`).
- **Mandatory Phase 6 Review** before COMPLETE; logged in
  `tracking/phase-6-review.md`.
