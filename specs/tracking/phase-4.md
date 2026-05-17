# Phase 4 — ShaderModule / BindGroupLayout / BindGroup / PipelineLayout

Status: **in progress** (P4.0 active). Rules: `../blocks/30-shader-
binding.md`. Roles/loop: `../reference/workflow.md`. Gate (permanent):
`cargo test --workspace` + `cargo clippy --workspace --all-targets --
-D warnings` green on Noop. **Phase ends with the mandatory Phase
Review** (`reference/workflow.md` → "Phase Review";
`tracking/phase-4-review.md`).

5 slices. First wires `naga`. Deferred S8/S35(pipeline)/S39–S44 → P5.

## P4.0 — Wire naga (git+rev) + WGSL parse smoke  *(ACTIVE)*

Add `naga` as a workspace git+rev dependency (pin in
`reference/dependencies.md`); `yawgpu-core` depends on it. A throwaway
`#[cfg(test)]` smoke proving `naga::front::wgsl::parse_str` +
`naga::valid::Validator` accept a trivial valid WGSL and reject a
trivial invalid one. Cargo.lock records the SHA. **De-risks the
dependency before P4.1 builds on it.**

## P4.1 — ShaderModule  *(after P4.0)*

`wgpuDeviceCreateShaderModule` (WGSL via naga, SPIR-V accept; S1–S5,
S6/S7/S11 rejected-direction divergence), `wgpuShaderModuleGet
CompilationInfo` via the future machinery (S9), release; error-module +
stored diagnostic. May split P4.1a (create+WGSL/SPIRV S1–S5) / P4.1b
(S6/S7/S11 + S9). S10 N/A.

## P4.2 — BindGroupLayout  *(after P4.1)*

`wgpuDeviceCreateBindGroupLayout`; S12–S22 (reuse P1.2a Limits, P3.1b
FormatCaps; S19 rejected-direction).

## P4.3 — BindGroup  *(after P4.2)*

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
