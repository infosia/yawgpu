# Phase 4 — ShaderModule / BindGroupLayout / BindGroup / PipelineLayout

Status: **in progress** (P4.0 active). Rules: `../blocks/30-shader-
binding.md`. Roles/loop: `../reference/workflow.md`. Gate (permanent):
`cargo test --workspace` + `cargo clippy --workspace --all-targets --
-D warnings` green on Noop. **Phase ends with the mandatory Phase
Review** (`reference/workflow.md` → "Phase Review";
`tracking/phase-4-review.md`).

5 slices. First wires `naga`. Deferred S8/S35(pipeline)/S39–S44 → P5.

## P4.0 — Wire naga (git+rev) + WGSL parse smoke  *(☑ DONE)*

Done: workspace `naga = { git=infosia/wgpu, rev=216627076… }`;
`yawgpu-core` dep `features=["wgsl-in"]`; `shader_naga::
parse_and_validate_wgsl` (`naga::front::wgsl::parse_str` +
`Validator::new(ValidationFlags::all(), Capabilities::empty())`); 2
smoke tests (valid ok / invalid err). Cargo.lock pins naga 29.0.3 @ the
git SHA. Build resolves the git dep, gate green (23 binaries). naga API
shape confirmed for P4.1. Committed `phase-4: P4.0`.

#### (original detail)

Add `naga` as a workspace git+rev dependency (pin in
`reference/dependencies.md`); `yawgpu-core` depends on it. A throwaway
`#[cfg(test)]` smoke proving `naga::front::wgsl::parse_str` +
`naga::valid::Validator` accept a trivial valid WGSL and reject a
trivial invalid one. Cargo.lock records the SHA. **De-risks the
dependency before P4.1 builds on it.**

## P4.1a — ShaderModule create  *(☑ DONE)*

Done: `shader_naga` returns `{module,info}`; core `ShaderModule`/
`ShaderModuleSource` + `create_shader_module` (WGSL via naga, SPIR-V
opaque words, `validate_wgsl_module_limits` = S4 duplicate `@id` +
S3 binding≥1000), error-module + stored diagnostic. conv chain-walk
(S1 no-source, S2 duplicate-source, SPIRV null-guard). FFI
`WGPUShaderModuleImpl` + create/Release/AddRef. S1–S7,S11 ported in
`yawgpu/tests/shader_module_validation.rs` (6, 0 ignored — pinned naga
enforces S4/S6/S7/S11). Gate green (24 binaries). Committed
`phase-4: P4.1a`.

## P4.1b — ShaderModule GetCompilationInfo  *(☑ DONE)*

Done: `WGPUShaderModuleImpl._instance`; `PendingCallback::Compilation
Info` (holds `Arc<core::ShaderModule>`); `wgpuShaderModuleGet
CompilationInfo` via the future machinery — fires once, status Success,
1 Error message (StringView into the held diagnostic) for an error
module / `messageCount==0` for a valid one; WaitAnyOnly honored. conv
status/type helpers. S9 ported in `shader_module_validation.rs` (now 9;
+3), gate green (24 binaries). Committed `phase-4: P4.1b`. **ShaderModule
section complete** (S1–S7,S9,S11; S8→P5, S10 N/A).

## P4.1 — ShaderModule  *(superseded by P4.1a/P4.1b)*

`wgpuDeviceCreateShaderModule` (WGSL via naga, SPIR-V accept; S1–S5,
S6/S7/S11 rejected-direction divergence), `wgpuShaderModuleGet
CompilationInfo` via the future machinery (S9), release; error-module +
stored diagnostic. May split P4.1a (create+WGSL/SPIRV S1–S5) / P4.1b
(S6/S7/S11 + S9). S10 N/A.

## P4.2 — BindGroupLayout  *(☑ DONE)*

Done: core `BindGroupLayout`/`BindGroupLayoutEntry`/`BindingLayoutKind`
+ enums; `validate_bind_group_layout_descriptor` (S12 unique/<1000,
S15 dynamic-buffer limits, S17 ms⇒2D, S18 storage format/dim, S19
arraySize>1, S20 raw visibility, S21 per-stage counts via
`visible_stages`, S22 ≤1000); conv `map_bind_group_layout_*`
(BindingNotUsed sentinels ⇒ S13 present_count≠1, S14/S16 invalid enum
via `set_first_error`); error-layout model. FFI
`WGPUBindGroupLayoutImpl` + create/Release/AddRef. S12–S22 ported in
`yawgpu/tests/bind_group_layout_validation.rs` (7), gate green
(25 binaries). Committed `phase-4: P4.2`.

#### (original detail)

`wgpuDeviceCreateBindGroupLayout`; S12–S22 (reuse P1.2a Limits, P3.1b
FormatCaps; S19 rejected-direction).

## P4.3 — BindGroup  *(NEXT)*

`wgpuDeviceCreateBindGroup`; S23–S33, S35(BG part). S34 N/A.

## P4.4 — PipelineLayout  *(after P4.3; closes slices)*

`wgpuDeviceCreatePipelineLayout`; S36–S37. Then Phase Review.

## Phase 4 exit criteria

- S1–S7,S9,S11,S12–S33,S35(BG),S36–S37 covered by ported Rust tests
  green on Noop; gate clean; CI green. S10/S34 N/A; S8/S35(pipeline)/
  S39–S44 Defer→P5.
- `dawn-test-mapping.md`: `ShaderModuleValidationTests` ☑,
  `BindGroupValidationTests` ☑, `MinimumBufferSizeValidationTests` ◐
  (BG part; pipeline part→P5), `UnsafeAPIValidationTests` ☑ (R18–R20
  rejected-direction; divergence), `WGSLFeatureValidationTests` ◐,
  `OverridableConstantsValidationTests`/`GetBindGroupLayoutValidation
  Tests`/`ImmediateDataTests` ◐/☐ Defer→P5.
- One commit per slice (`phase-4: <slice> — <short>`).
- **Mandatory Phase 4 Review** (fresh no-context; CRITICAL/MAJOR fixed)
  before COMPLETE; logged in `tracking/phase-4-review.md`.
