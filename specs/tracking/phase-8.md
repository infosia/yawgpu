# Phase 8 — Finalize (Query / ErrorScope / DeviceLost / Toggle / Surface / MultipleDevice)

Status: **in progress** (P8.0 active). Rules: `../blocks/70-finalize.md`.
Roles/loop: `../reference/workflow.md`. **Final phase.** Back to the
Noop validation TDD loop (codex-driven, CI-green); no GPU needed (real
Metal/Vulkan execution was proven in Phase 7). Gate (permanent):
`cargo test --workspace` + `cargo clippy --workspace --all-targets --
-D warnings` green on Noop. **Phase ends with the mandatory Phase
Review** (`tracking/phase-8-review.md`).

Closes the remaining WebGPU validation surface + the Phase-1/6
deferrals (C34/C35, encoder WriteTimestamp/ResolveQuerySet, R15/R16,
R21). Reuses the Phase-1 `PendingCallback` future plumbing and the
error-object/deferred-error model.

## P8.0 — ErrorScope  *(active)*
ES1–ES5: push/pop, filter match, nested stack, scope-vs-uncaptured-
callback routing, async pop (future) + device-lost interaction.
Foundational (other slices' errors interact with scopes).
`ErrorScopeValidationTests`.

## P8.1 — QuerySet creation  *(after P8.0)*
QS1–QS4: `CreateQuerySet` type/count, timestamp-feature gating,
GetType/GetCount/Destroy, error QuerySet. `QuerySetValidationTests`/
`QueryValidationTests`.

## P8.2 — Query in commands  *(after P8.1)*
QC1–QC5: C34/C35 (RenderPass occlusion/timestamp — deferred from
P6.4), encoder WriteTimestamp / ResolveQuerySet (deferred from P6),
occlusion begin/end pairing. Deferred-error model.

## P8.3 — DeviceLost  *(after P8.2)*
DL1–DL4: DeviceDestroy → lost + GetLostFuture; post-lost ops return
errors (no panic); pending callbacks resolve lost/aborted; completes
the Phase-1 `device_lost_validation.rs` (no regression).
`DeviceLostValidationTests`.

## P8.4 — Toggle / UnsafeAPI (R21)  *(after P8.3)*
TG1/TG2: map only stable-webgpu.h-analog toggle/feature rules; the
`AllowUnsafeAPIs`-class ✗ N/A (recorded divergence, rejected-
direction only). `ToggleValidationTests`, `UnsafeAPIValidationTests`.

## P8.5 — MultipleDevice (R15/R16)  *(after P8.4)*
MD1/MD2: cross-device object use ⇒ validation error (owning-device
`Arc::ptr_eq`); audit per-object device-identity checks.
`MultipleDeviceTests`.

## P8.6 — Surface (descriptor/arg validation, Noop)  *(after P8.5)*
SF1–SF3: CreateSurface descriptor decode, Configure/GetCurrentTexture
validation vs capabilities; real presentation ✗ N/A on Noop
(recorded). Then Phase 8 Review.

## Phase 8 exit criteria

- ES/QS/QC/DL/TG/MD/SF ported tests green on Noop (N/A recorded);
  C34/C35 + encoder query ops closed; gate clean; CI green.
- `dawn-test-mapping.md` Phase-8 rows ☑/✗-N/A; block-50 C34/C35 ☑.
- One commit per slice (`phase-8: <slice> — <short>`).
- **Mandatory Phase 8 Review** before COMPLETE
  (`tracking/phase-8-review.md`). COMPLETE ⇒ core WebGPU validation
  conformant on Noop + real Metal/Vulkan execution proven (Phase 7).
