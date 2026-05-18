# Phase 5 — Render / Compute Pipeline

Status: **in progress** (P5.0 active). Rules: `../blocks/40-pipeline.md`.
Roles/loop: `../reference/workflow.md`. Gate (permanent): `cargo test
--workspace` + `cargo clippy --workspace --all-targets -- -D warnings`
green on Noop. **Phase ends with the mandatory Phase Review**
(`tracking/phase-5-review.md`).

Largest phase; 7 slices. First widens naga + builds the reflection seam
(de-risk, mirrors P4.0). Carries Phase-4 deferreds S8/S35(pipeline)/
S39–S44 + W4.

## P5.0 — naga capability widening + reflection helper  *(☑ DONE)*

Done: `Validator` capability widened to `Capabilities::SHADER_FLOAT16`
only (justified: Dawn Phase-5 `enable f16; override` shaders); 6
`pub(crate)` reflection helpers on the validated module
(`entry_points`, `compute_workgroup_size` incl. override keys,
`entry_point_io`, `resource_bindings` + static-use, `fragment_builtins`,
`overrides`) returning owned `Reflected*` structs (no naga types
leaked); 7 smoke tests. P4 shader tests unregressed. Gate green (27
binaries, yawgpu-core 14 tests). Committed `phase-5: P5.0`.

#### (original detail)

Widen the `shader_naga` `Validator` `Capabilities` to what Phase-5 WGSL
needs (P0a: `f16`/`ShaderF16`, …; data-driven, justify each). Add
reflection helpers on the validated naga `Module` (P0b): entry points
+stage, `@workgroup_size`, `var<workgroup>` total bytes, `@location`
IO+scalar/vector type class, `@group/@binding`+access kind, `@builtin`
outputs, overrides (name/`@id`/type/default). `#[cfg(test)]` smoke per
helper. De-risks the reflection surface before pipelines build on it.

## P5.1 — ComputePipeline  *(☑ DONE)*

Done: core `ComputePipeline`/`ComputePipelineDescriptor`/
`ComputePipelineLayout{Auto,Explicit}`/`PipelineConstant`;
`validate_compute_pipeline_descriptor` → `resolve_*` (P1
`resolve_compute_entry`, P5 `resolve_pipeline_constants` unique/by-id-or-
name/uninitialized, P6 finite+type-range, P2 `resolve_compute_workgroup`
override-applied + per-axis/product limits, P3 storage limit, P4
`validate_compute_pipeline_layout` Auto-ok / explicit S35 compat:
group/binding present + Compute visibility + type/minBindingSize);
error-pipeline. `ShaderModule` keeps `ValidatedWgslModule`;
`PipelineLayout::bind_group_layouts()`. FFI `WGPUComputePipelineImpl` +
create/Release/AddRef; conv null-layout⇒Auto. P1–P6 ported in
`yawgpu/tests/compute_pipeline_validation.rs` (7), gate green
(28 binaries). Committed `phase-5: P5.1`.

#### (original detail)

`wgpuDeviceCreateComputePipeline`(+release/AddRef); P1–P6. Auto/explicit
layout (S35 compute part).

## P5.2a — RenderPipeline (entry/presence/primitive/MS/bias)  *(☑ DONE)*

Done: core `RenderPipeline`/`RenderPipelineDescriptor`/
`RenderPipelineLayout{Auto,Explicit}` + vertex/fragment/primitive/
multisample/depthStencil snapshots; `resolve_render_pipeline_descriptor`
(P7/P8 entry resolution, P9 presence+targetCount, P18 strip topology,
P21 finite bias, P22 non-zero-bias triangle-only, P25 count∈{1,4}, P26
a2c⇒count4, P27 a2c vs `@builtin(sample_mask)` via reflection);
error-pipeline; deferred rules (vertex attrs/color/depth-aspect) NOT
implemented. FFI `WGPURenderPipelineImpl` + create/Release/AddRef; conv
render descriptor + null layout/depthStencil/fragment. P7–P27(subset)
ported in `yawgpu/tests/render_pipeline_validation.rs` (8), gate green
(29 binaries). Committed `phase-5: P5.2a`.

## P5.2b — RenderPipeline (fragment/color/depthStencil-aspect)  *(☑ DONE)*

Done: `FormatCaps` extended (`is_blendable`/`has_alpha`/`output_class`/
`color_components` + `.blendable()`/`.alpha()` ctors — refines the
carried W5 area for the tested formats);
`validate_depth_stencil_aspects` (P19/P20/P23),
`validate_fragment_depth_output` (P24), `validate_color_targets`
(P28/P29/P30/P31/P33/P35), `validate_fragment_output_compat` (P32),
`validate_render_pipeline_layout` (P36 vertex+fragment stage binding
compat), `validate_render_constants` (P37 per-stage). P19–P37 ported in
`render_pipeline_validation.rs` (now 14), gate green (29 binaries).
Committed `phase-5: P5.2b`. **Render pipeline section complete.**

#### (original detail)

P19,P20,P23,P24,P28,P29–P37 (depth/stencil aspect & depthCompare/Write,
frag_depth, color target renderable/blendable/writeMask/output-type/
bytesPerSample, a2c-vs-target, render overridable constants).

## P5.2 — RenderPipeline  *(split a/b)*

`wgpuDeviceCreateRenderPipeline`(+release/AddRef); P7–P9, P18–P37.
P5.2a entry/presence/primitive/multisample/depth-bias
(P7,P8,P9,P18,P21,P22,P25,P26,P27); P5.2b fragment/color/depthStencil
(P19,P20,P23,P24,P28,P29–P37). (P28 alpha-to-coverage-vs-target moved to
P5.2b since it needs the color-target parsing.)

## P5.3 — VertexState  *(☑ DONE)*

Done: core `VertexFormat`/`VertexFormatInfo` table (byte_size +
Float/Sint/Uint class over the WGPUVertexFormat enum, unknown
conservative); `RenderPipelineVertexState` carries decoded buffer
layouts; `validate_vertex_state` (P10 buffer count, P11 attr count, P12
stride %4 & ≤limit, P13 offset align min(4,size)+range, P14 location
unique, P15 location <limit, P16 format-vs-shader scalar class via
`vertex_inputs` reflection, P17 every shader `@location` input covered;
extra attributes allowed). conv decodes `WGPUVertexState.buffers[]`.
P10–P17 ported in `yawgpu/tests/vertex_state_validation.rs` (7), gate
green (30 binaries). Committed `phase-5: P5.3`.

#### (original detail)

P10–P17 (counts, stride/offset, location, format-vs-shader, coverage).

## P5.4 — GetBindGroupLayout + layout/shader compat  *(☑ DONE)*

Done: `BindGroupLayout` gains `is_default`; `derive_bind_group_layouts`
aggregates statically-used bindings across stages (visibility OR,
minBindingSize merge, P42 sample-type Float/UnfilterableFloat from
Sample/Load), group-count vs `max_bind_groups`, marks derived BGLs
default; compute+render pipelines store derived `Vec<Arc<BGL>>`.
`validate_pipeline_layout_descriptor` rejects default BGLs (P40). FFI
`wgpuRender/ComputePipelineGetBindGroupLayout` via
`get_pipeline_bind_group_layout` with a per-pipeline cached handle vec
(P38 OOB⇒error-BGL; P41 same handle per pipeline+index, distinct across
pipelines). P38–P42 ported in
`yawgpu/tests/get_bind_group_layout_validation.rs` (6), gate green
(31 binaries). Committed `phase-5: P5.4`. **P41 draw-time
cross-pipeline incompatibility → Defer→P6** (recorded).

#### (original detail)

Auto-layout reflection; `wgpuRender/ComputePipelineGetBindGroupLayout`;
P36, P38–P42, S35 explicit-layout compat.

## P5.5 — Object caching  *(☑ DONE)*

Done: per-`WGPUDeviceImpl` `Mutex<HashMap<…>>` caches for ShaderModule/
PipelineLayout/Compute/RenderPipeline; structural cache keys using
sub-object handle identity + normalized constants (`f64::to_bits`,
sorted) + full render descriptor; `cache_handle` dedups (AddRef-returns
the same `Arc` ⇒ identical C pointer); **error objects bypass the
cache** (fresh each call). P43–P50 ported in
`yawgpu/tests/object_caching_validation.rs` (6) + error-not-deduped;
gate green (32 binaries). Committed `phase-5: P5.5`.

> Review note: P5.4's `default_bind_group_layout_identity_is_pipeline_
> bound` test was correctly adjusted (pipeline_b now uses
> `compute_uniform_different()`) so the two pipelines no longer dedup
> under the new cache — keeping the P41 "distinct pipelines ⇒ distinct
> default BGL handles" assertion meaningful.
>
> Divergence (code-commented): yawgpu dedups by descriptor-structural
> key with sub-object handle identity — satisfies the `webgpu.h`
> identity ObjectCaching asserts; Dawn's deeper content-equal internal
> dedup beyond `==` is out of scope.

#### (original detail)

Per-device descriptor-keyed dedup (handle identity); P43–P50.

## P5.6 — Async pipeline creation  *(NEXT — closes slices, then Phase Review)*

`wgpuDeviceCreateRender/ComputePipelineAsync` via future machinery; P51.
Then Phase Review.

## Phase 5 exit criteria

- P1–P50 covered by ported Rust tests green on Noop (P51 async too);
  gate clean; CI green. Defer→P6/P7 noted.
- `dawn-test-mapping.md`: `RenderPipelineValidationTests`,
  `ComputeValidationTests`, `VertexStateValidationTests`,
  `ObjectCachingTests`, `GetBindGroupLayoutValidationTests`,
  `OverridableConstantsValidationTests` ☑;
  `PipelineAndPassCompatibilityTests` ◐ (pipeline-create part; pass→P6).
- One commit per slice (`phase-5: <slice> — <short>`).
- **Mandatory Phase 5 Review** before COMPLETE; logged in
  `tracking/phase-5-review.md`.
