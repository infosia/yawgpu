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

## P1.1 — Real Future / callback-mode state machine  *(ACTIVE)*

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

## P1.2 — Adapter/Device limits & features  *(outline; detail when active)*

Synthetic Noop adapter gains an explicit supported-limits + feature set and
a core-vs-compat mode. Implement `wgpuAdapterGetLimits/GetFeatures/
HasFeature`, `wgpuDeviceGetLimits/GetFeatures/HasFeature`, and RequestDevice
`requiredLimits`/`requiredFeatures` validation. Port DeviceValidationTests
**R1–R4, R6, R7, R10–R14** (and **R9** if cheap). May split into P1.2a
(limits R1–R4,R14) / P1.2b (features + core/compat R6,R7,R10–R13).

## P1.3 — Device lifecycle & device-lost  *(outline)*

`wgpuDeviceDestroy` + device tick no-op after destroy (**R8**); a
device-lost callback channel distinct from the uncaptured-error sink with
the request-device→device-lost ordering and per-mode timing (**R5**); GAHB
feature gate (**R9**) if not done in P1.2.

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
