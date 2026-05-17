# Phase 5 — Render / Compute Pipeline

Status: **in progress** (P5.0 active). Rules: `../blocks/40-pipeline.md`.
Roles/loop: `../reference/workflow.md`. Gate (permanent): `cargo test
--workspace` + `cargo clippy --workspace --all-targets -- -D warnings`
green on Noop. **Phase ends with the mandatory Phase Review**
(`tracking/phase-5-review.md`).

Largest phase; 7 slices. First widens naga + builds the reflection seam
(de-risk, mirrors P4.0). Carries Phase-4 deferreds S8/S35(pipeline)/
S39–S44 + W4.

## P5.0 — naga capability widening + reflection helper  *(ACTIVE)*

Widen the `shader_naga` `Validator` `Capabilities` to what Phase-5 WGSL
needs (P0a: `f16`/`ShaderF16`, …; data-driven, justify each). Add
reflection helpers on the validated naga `Module` (P0b): entry points
+stage, `@workgroup_size`, `var<workgroup>` total bytes, `@location`
IO+scalar/vector type class, `@group/@binding`+access kind, `@builtin`
outputs, overrides (name/`@id`/type/default). `#[cfg(test)]` smoke per
helper. De-risks the reflection surface before pipelines build on it.

## P5.1 — ComputePipeline  *(after P5.0)*

`wgpuDeviceCreateComputePipeline`(+release/AddRef); P1–P6. Auto/explicit
layout (S35 compute part).

## P5.2 — RenderPipeline  *(after P5.1; split a/b if large)*

`wgpuDeviceCreateRenderPipeline`(+release/AddRef); P7–P9, P18–P37.
P5.2a vertex/primitive/multisample (P7–P9,P18,P21,P22,P25–P28);
P5.2b fragment/color/depthStencil (P19,P20,P23,P24,P29–P37).

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
