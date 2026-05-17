# Phase 2 — Buffer + Queue

Status: **in progress** (P2.1 active). Rules: `../blocks/10-buffer-queue.md`.
Roles/loop: `../reference/workflow.md`. Gate (permanent): `cargo test
--workspace` + `cargo clippy --workspace --all-targets -- -D warnings`
green on Noop.

4 slices. Deferred B39–B41/B53–B57 (→P6), B58 (→P4) are out of Phase 2.

## P2.1 — Buffer creation / reflection / lifetime  *(☑ DONE)*

Done: `NoopBuffer`/`HalBuffer`, core `Buffer` (Arc, usage bitflags,
map-state, error/destroyed flags) + `validate_buffer_descriptor`,
error-buffer model (reflects size/usage), `wgpuDeviceCreateBuffer`/
`Destroy`/`Unmap`/`GetSize`/`GetUsage`/`GetMapState`/`Release`/`AddRef`.
B1–B6,B32–B36,B38 ported in `yawgpu/tests/buffer_creation_validation.rs`
(8), gate green. Committed `phase-2: P2.1`. B37 callback part → P2.2.

#### (original detail)

`WGPUBuffer` handle + core `Buffer` (Arc, host `Vec<u8>` backing for later
map). `wgpuDeviceCreateBuffer` with usage/size/mappedAtCreation validation
routed through the device error sink (invalid descriptor ⇒ error +
**error-buffer** handle that still reflects size/usage). `wgpuBufferDestroy`
(idempotent), `wgpuBufferUnmap` (safe on any state), `wgpuBufferGetSize/
GetUsage/GetMapState`. `mappedAtCreation` ⇒ Mapped state (no async).
Port **B1–B6, B32–B36, B38** (B37 callback part → P2.2).

## P2.2 — Buffer map async state machine  *(☑ DONE)*

Done: core `MapMode`/`MapAsyncStatus`/`PendingMap`; `begin_map` (B7–B16
sync validation), `resolve_pending_map`/`abort_pending_map`;
`wgpuBufferMapAsync` reusing `PendingCallback::BufferMap` (future
machinery), `WGPU_WHOLE_MAP_SIZE`, `map_map_mode` (B13/B14), validation ⇒
device-error + callback `Error`; unmap/destroy/Drop-before-drain ⇒
`Aborted` (callback holds own `Arc<Buffer>` clone); once-only via
`callback_fired`. B7–B24,B37 ported in `yawgpu/tests/
buffer_map_validation.rs` (9), gate green (47 tests). Committed
`phase-2: P2.2`.

#### (original detail)

`wgpuBufferMapAsync` (+`WGPUBufferMapCallbackInfo`) reusing the Phase-1
future/`PendingCallback` machinery; Pending/Mapped transitions;
Unmap/Destroy/drop-before-result ⇒ `Aborted`; validation ⇒ `Error`;
once-only fire incl. reentrancy. Port **B7–B24, B37**.

## P2.3 — GetMappedRange / GetConstMappedRange  *(NEXT)*

Const vs non-const rules, bounds/offset/whole-map-size, destroyed ⇒ NULL.
Port **B25–B31**.

## P2.4 — Queue writeBuffer / onSubmittedWorkDone  *(after P2.3)*

`wgpuQueueWriteBuffer` arg/usage/state validation; `wgpuQueue
OnSubmittedWorkDone` via future machinery; minimal `wgpuQueueSubmit` arg
checks. Port **B42–B52**. Closes Phase 2.

## Phase 2 exit criteria

- B1–B38 (non-deferred), B42–B52 covered by ported Rust tests green on
  Noop; gate clean; CI green.
- `dawn-test-mapping.md`: `BufferValidationTests` ☑ (deferred submit cases
  noted), `QueueWriteBufferValidationTests` ☑,
  `QueueOnSubmittedWorkDoneValidationTests` ☑,
  `QueueSubmitValidationTests` ◐ (arg-only; commands→P6),
  `MinimumBufferSizeValidationTests` ☐ Defer→P4.
- One commit per slice (`phase-2: <slice> — <short>`).
