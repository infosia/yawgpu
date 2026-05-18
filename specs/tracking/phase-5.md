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

## P5.2b — RenderPipeline (fragment/color/depthStencil-aspect)  *(NEXT)*

P19,P20,P23,P24,P28,P29–P37 (depth/stencil aspect & depthCompare/Write,
frag_depth, color target renderable/blendable/writeMask/output-type/
bytesPerSample, a2c-vs-target, render overridable constants).

## P5.2 — RenderPipeline  *(split a/b)*

`wgpuDeviceCreateRenderPipeline`(+release/AddRef); P7–P9, P18–P37.
P5.2a entry/presence/primitive/multisample/depth-bias
(P7,P8,P9,P18,P21,P22,P25,P26,P27); P5.2b fragment/color/depthStencil
(P19,P20,P23,P24,P28,P29–P37). (P28 alpha-to-coverage-vs-target moved to
P5.2b since it needs the color-target parsing.)

## P5.3 — VertexState  *(after P5.2)*

P10–P17 (counts, stride/offset, location, format-vs-shader, coverage).

## P5.4 — GetBindGroupLayout + layout/shader compat  *(after P5.3)*

Auto-layout reflection; `wgpuRender/ComputePipelineGetBindGroupLayout`;
P36, P38–P42, S35 explicit-layout compat.

## P5.5 — Object caching  *(after P5.4)*

Per-device descriptor-keyed dedup (handle identity); P43–P50.

## P5.6 — Async pipeline creation  *(after P5.5; closes slices)*

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
