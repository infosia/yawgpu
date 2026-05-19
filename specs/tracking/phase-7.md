# Phase 7 — Real backends

Status: **in progress** (P7.0 active). Rules/plan:
`../blocks/60-real-backends.md`. Roles/loop:
`../reference/workflow.md`.

**Roadmap divergence (approved):** SPEC roadmap lists Phase 7 as
"Vulkan→Metal"; we bring up **Metal first, then Vulkan** because the
dev platform is macOS (Metal native; no MoltenVK/Vulkan-SDK needed for
on-machine real-GPU verification). Vulkan (P7.6) reuses the identical
HAL contract. SPEC.md Phase-7 row annotated accordingly.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy
--workspace --all-targets -- -D warnings` green on **Noop**
(real-backend code is build-only in CI). Per slice **also**: `cargo
build -p yawgpu --features metal` (later `--features vulkan`) +
clippy with the feature. **Real-GPU end2end** (`cargo test --features
metal -- --ignored`) is run **manually by the user**, reported, and
logged here per slice. **Phase ends with the mandatory Phase Review**
(`tracking/phase-7-review.md`).

Methodology shift vs Phases 0–6: not validation-rule porting —
execution bring-up verified by gated Dawn `end2end` Basic/Compute/Copy
ports. Validation stays in `yawgpu-core`; backends only execute
already-validated work; driver failure → `HalError` → device error,
never panic.

## P7.0 — Bring-up scaffolding + gating harness  *(active)*

De-risk the real-backend surface before any GPU code (mirrors
P4.0/P5.0). `metal` crate dependency wired (cfg-gated, not in
default); `yawgpu-test` gpu-gated helper (real-adapter probe →
`#[ignore]`/skip); `wgpuCreateInstance` backend selection (Noop
default, opt-in real backend); the empty `metal` HAL module reshaped
to the real contract signatures (still returning `BackendUnavailable`
until P7.1). Acceptance: `cargo build -p yawgpu --features metal` +
clippy clean; `cargo test --workspace` Noop gate unchanged (real
tests skip/ignore cleanly with no adapter).

## P7.1 — Metal Instance/Adapter/Device/Queue  *(after P7.0)*
Real `MTLDevice`/command queue; adapter enumerate; empty submit.
Port `BasicTests` (creation/empty-submit subset).

## P7.2 — Metal Buffer + writeBuffer/submit + B2B  *(after P7.1)*
`MTLBuffer` alloc + map/staging + readback; B2B copy. Port
`BufferTests`/`CopyTests` buffer subset.

## P7.3 — Metal Texture/Sampler + B2T/T2B/T2T  *(after P7.2)*
`MTLTexture`/view/sampler; blit copies. Port `CopyTests` texture
subset.

## P7.4 — Metal Shader (naga→MSL) + compute dispatch  *(after P7.3)*
WGSL→MSL via naga MSL backend; compute pipeline + dispatch +
storage-buffer readback. Port `ComputeDispatchTests` (basic).

## P7.5 — Metal render pipeline + render pass draw  *(after P7.4)*
Render pipeline; render pass load/store + draw; color readback.
Port `BasicTests` render / minimal draw.

## P7.6 — Vulkan bring-up (mirror P7.1–P7.5)  *(after P7.5)*
`ash` + MoltenVK on macOS; naga→SPIR-V; same HAL contract; reuse the
ported end2end tests parametrized by backend feature.

## Phase 7 exit criteria

- Metal + Vulkan fill their HAL enum arms; `yawgpu-core` validation
  unchanged & still green on Noop; per-slice `--features` build +
  clippy clean.
- Ported `end2end` Basic/Compute/Copy pass on real GPU (Metal on this
  machine; Vulkan as available) — user-run, logged here per slice.
- One commit per slice (`phase-7: <slice> — <short>`).
- **Mandatory Phase 7 Review** before COMPLETE; logged in
  `tracking/phase-7-review.md`.
