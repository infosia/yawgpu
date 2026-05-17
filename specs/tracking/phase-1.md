# Phase 1 — Instance / Adapter / Device + Future

Status: **in progress** (P1.1 active). Rules & async spec: `../blocks/
00-foundation.md`. Roles/loop: `../reference/workflow.md`.

Decomposed into 4 slices. Each = one HANDOFF, codex implements, Claude
reviews + commits. Acceptance for every slice: `cargo test --workspace`
green on Noop **and** `cargo clippy --workspace --all-targets -- -D warnings`
clean (Phase-0 gate is now permanent).

Deferred rules R15, R16, R18–R21 are NOT in Phase 1 (need later-phase
resources); they stay tracked in the block spec and `dawn-test-mapping.md`.

---

## P1.1 — Real Future / callback-mode state machine  *(☑ DONE)*

Done: A1–A7 implemented, `poll_all` removed, `TimedWaitAny` gate, tests in
`yawgpu/tests/future_modes.rs` (5) green; gate clean. Committed
`phase-1: P1.1`.

Replaces the Phase-0 `FutureRegistry::poll_all` stub with the spec state
machine A1–A7 (`../blocks/00-foundation.md`).

- yawgpu-core `FutureRegistry`: futures hold a mode + Pending/Complete +
  fired flag; completion is **not consumed** (A1, A7). API to: register
  with `WGPUCallbackMode`, mark complete, `process_events()` (fire complete
  AllowProcessEvents/AllowSpontaneous, A6), `wait_any(ids, timeout0?)`
  returning a status + which ids completed (A5), respecting that
  `WaitAnyOnly` only fires in wait_any (A2/A3/A4).
- yawgpu FFI: rework `wgpuInstanceWaitAny` (Error/Success/TimedOut per A5,
  `count==0`⇒TimedOut, `count>0 && futures==null`⇒Error, set each
  `WGPUFutureWaitInfo.completed`) and `wgpuInstanceProcessEvents` (A6).
  Keep the existing pending-callback dispatch but drive it by the new
  registry instead of the blanket `poll_all`.
- Instance `TimedWaitAny` feature: model an instance feature set; with
  `timeoutNS>0` and the feature absent ⇒ `Error`. For Noop, ops are
  complete at registration so timeout>0 behaves like poll when the feature
  IS present.
- Tests (in `yawgpu/tests/`): port the behavioural expectations —
  WaitAnyOnly fires only in WaitAny; AllowProcessEvents fires in
  ProcessEvents; repeated WaitAny on a completed future keeps returning
  Success (`FuturesTests::WaitAnySameFuture`); WaitAny timeout0 poll;
  count==0 ⇒ TimedOut; null futures ⇒ Error.

Exit: state machine A1–A7 covered by tests; old `poll_all` removed; gate
green. Update block R-less async items A1–A7 to ☑ and remove the carried
review note.

---

## P1.2 — Adapter/Device limits & features

Split into two slices. Limit/feature model = block 00 "Synthetic Noop
adapter limit/feature model".

### P1.2a — Limits  *(☑ DONE)*

Done: 32-field `Limits` model with Dawn `v1` default constants (verified
against `Limits.cpp`), Maximum/Alignment/AlwaysMax classification, UNDEFINED
sentinel→default in conv, `requiredLimits` validation + effective-limit
computation, `wgpuAdapterGetLimits`/`wgpuDeviceGetLimits`. R1–R4,R14 ported
in `yawgpu/tests/limits_validation.rs` (5), gate green. Committed
`phase-1: P1.2a`.

### P1.2b — Features + core/compat  *(☑ DONE)*

Done: `Feature`/`FeatureLevel`/`FeatureSet`, supported set + implication
closure, core/compat default resolution, `featureLevel` in RequestAdapter,
`requiredFeatures` validation, `wgpuAdapter/DeviceGetFeatures/HasFeature`,
`wgpuSupportedFeaturesFreeMembers` (boxed-slice round-trip). R6,R7,R10–R13
ported in `yawgpu/tests/features_validation.rs` (8), gate green. Committed
`phase-1: P1.2b`. R9 deferred to P1.3.

#### (original outline)

`WGPUSupportedFeatures` (webgpu.h:2931), `WGPUFeatureLevel` (625),
`WGPURequestAdapterOptions.featureLevel` (4138). Noop adapter records the
requested feature level; synthetic supported set = {CoreFeaturesAndLimits,
RG11B10UfloatRenderable, TextureFormatsTier1, TextureFormatsTier2} with the
implication closure (Tier2⇒Tier1⇒RG11B10UfloatRenderable). Implement
`wgpuAdapterGetFeatures/HasFeature`, `wgpuDeviceGetFeatures/HasFeature`,
`wgpuSupportedFeaturesFreeMembers`, `requiredFeatures` validation +
resolution, and `featureLevel` handling in `wgpuInstanceRequestAdapter`.
Port **R6, R7, R10–R13** (R10–R13 via `HasFeature(CoreFeaturesAndLimits)`
per the recorded Divergence). **R9** → defer to P1.3.

## P1.3 — Device lifecycle & device-lost  *(ACTIVE)*

Device-lost channel per block 00 "Device-lost channel" design. Implement
`wgpuDeviceDestroy`, capture `WGPUDeviceDescriptor.deviceLostCallbackInfo`,
fire device-lost via the future/pending-callback machinery. Port **R5**
(request-device-Error → device-lost(FailedCreation), ordering + per-mode
timing) and **R8** reframed (destroy ⇒ device-lost(Destroyed), idempotent,
last-ref implicit destroy, ProcessEvents-after-destroy safe). **R9** is N/A
(Dawn-only extension, dropped — recorded divergence).

## P1.4 — Labels for Device & Queue  *(outline)*

Descriptor `label` + `SetLabel`/`GetLabel` for Device and Queue; a
`#[doc(hidden)]` testing label getter mirroring Dawn's
`GetObjectLabelForTesting`. Port **R17a**. (Other objects' labels = R17b,
ported with their phases.)

---

## Phase 1 exit criteria

- A1–A7 + R1–R8, R10–R14, R17a covered by ported Rust tests, green on Noop.
- R9 done or explicitly deferred with reason.
- `cargo clippy --workspace --all-targets -- -D warnings` clean; CI green.
- `dawn-test-mapping.md`: `DeviceValidationTests` ☑ (resource-dependent
  cases noted deferred), `LabelTests` ◐ (Device/Queue done).
- One commit per slice (`phase-1: <slice> — <short>`).
