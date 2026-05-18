# Phase 6 — Command encoding & passes

Status: **in progress** (P6.1 active). Rules: `../blocks/50-commands.md`.
Roles/loop: `../reference/workflow.md`. Gate (permanent): `cargo test
--workspace` + `cargo clippy --workspace --all-targets -- -D warnings`
green on Noop. **Phase ends with the mandatory Phase Review**
(`tracking/phase-6-review.md`).

Largest phase; 9 slices. First builds the encoder/pass state machine +
deferred-error model everything else hangs off. Carries P2/P3/P5
deferrals.

## P6.1 — Encoder/pass lifecycle + deferred-error model  *(ACTIVE)*

`WGPUCommandEncoder`/`WGPUCommandBuffer`/`WGPURenderPassEncoder`/
`WGPUComputePassEncoder` handle types; core encoder state machine
(Recording/Finished/Errored), command recording + first-error,
`Finish` ⇒ device error + error CommandBuffer; `BeginCompute/Render
Pass` (skeleton — full descriptor validation is P6.3/P6.4), pass `End`;
C1–C5, C36, C63, C85/C86; debug-group balance at Finish. **Foundational
— establishes the deferred-error machinery for all later slices.**

## P6.2 — Buffer copies / clear / encoder WriteBuffer  *(after P6.1)*
C6–C10, ClearBuffer, C83/C84 (B53–B57).

## P6.3 — Texture copies (B2T/T2B/T2T)  *(after P6.2)*
C11–C22, C79 (256-align bytesPerRow path; FormatCaps block/aspect).

## P6.4 — RenderPass descriptor  *(after P6.3)*
C23–C33 (C30/C31 multisample resolve); C34/C35 Defer→P8.

## P6.5 — Pass draw/dispatch state + dynamic state  *(after P6.4)*
C37–C42, C56–C59, compute dispatch limits, P41 draw-time.

## P6.6 — Index/Vertex buffer + draw OOB + indirect  *(after P6.5)*
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
