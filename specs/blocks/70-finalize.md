# Block 70 — Finalize: Query / ErrorScope / DeviceLost / Toggle / Surface / MultipleDevice

Phase 8 (final). Closes the remaining WebGPU validation surface on the
**Noop** backend (CI-green, same TDD loop as Phases 0–6 — real GPU is
done in Phase 7 and not required here). Rules from Dawn
`QuerySetValidationTests`/`QueryValidationTests`,
`ErrorScopeValidationTests`, `DeviceLostValidationTests`,
`ToggleValidationTests`, `UnsafeAPIValidationTests` (R21 part),
`MultipleDeviceTests` (R15/R16), plus the Phase-6 deferred **C34/C35**
(occlusion/timestamp query in RenderPass) and the deferred encoder
`WriteTimestamp`/`ResolveQuerySet`, and surface descriptor validation.
Status: ☐ ◐ ☑ ✗(N/A).

## Surface (webgpu.h)

`wgpuDevicePushErrorScope`/`PopErrorScope` (+ future);
`wgpuDeviceCreateQuerySet` (`WGPUQuerySetDescriptor`) → `WGPUQuerySet`
(`GetType`/`GetCount`/`Destroy`/`SetLabel`/`Release`/`AddRef`);
`wgpuCommandEncoderWriteTimestamp`/`ResolveQuerySet`; RenderPass
`occlusionQuerySet` + `timestampWrites`; `wgpuDeviceDestroy`,
`wgpuDeviceGetLostFuture`; toggle/feature gating;
`wgpuInstanceCreateSurface` (`WGPUSurfaceDescriptor`) → `WGPUSurface`
(`Configure`/`Unconfigure`/`GetCapabilities`/`GetCurrentTexture`/
`Present`/`SetLabel`/`Release`/`AddRef`).

## Design decisions

- **Error-object + deferred-error model unchanged.** ErrorScope is a
  per-device stack of `{filter, captured}`; a dispatched device error
  is delivered to the innermost matching open scope instead of the
  uncaptured-error callback; `PopErrorScope` returns (via the
  future/callback machinery) the captured error or none; popping with
  no scope ⇒ instant error. Reuses the Phase-1 `PendingCallback`
  future plumbing.
- **QuerySet** = an Arc handle with `{type: Occlusion|Timestamp,
  count}`. Timestamp queries gated by the `timestamp-query` feature
  (HasFeature, mirrors the R10–R13 reframing). Noop holds no real
  query results — validation only; resolve/writeTimestamp validate
  args + buffer usage (`QueryResolve`) and route through the P6
  deferred-error model.
- **C34/C35** (deferred from block 50): now QuerySet exists —
  RenderPass `occlusionQuerySet` must be type Occlusion;
  `timestampWrites` query set type Timestamp + in-range indices;
  validated at `BeginRenderPass`/`Finish` per Dawn timing.
- **DeviceLost**: `wgpuDeviceDestroy` transitions the device to lost;
  the lost future/callback resolves once; subsequent object creation
  / operations return errors (or error objects) per Dawn, never
  panic. Completes the Phase-1 `device_lost_validation.rs` stub.
- **Toggle / UnsafeAPI (R21)**: yawgpu has no canonical webgpu.h
  toggle struct; the `AllowUnsafeAPIs`-class rules are **non-canonical
  divergences** (recorded, rejected-direction only, consistent with
  the Phase-1/Phase-4 `AllowUnsafeAPIs` and Phase-6 `firstInstance`
  divergence handling). Implement only what maps to a stable
  webgpu.h feature/limit; mark the rest ✗ N/A with rationale.
- **MultipleDevice (R15/R16)**: using an object from device A with
  device B (e.g. a bind group / pipeline / buffer in B's encoder) ⇒
  validation error. `Xxx::same`/device-identity checks
  (`Arc::ptr_eq` on the owning `Device`).
- **Surface**: on Noop with no windowing, surface is **descriptor/
  argument-validation only** — `CreateSurface` decodes the descriptor
  (chained window source), `Configure` validates format/usage/size
  vs reported capabilities, `GetCurrentTexture` on an unconfigured
  surface ⇒ error. Real presentation is out of scope (no swapchain
  on Noop; revisit with a real backend + window later — recorded).

## Rules (grouped → slices)

### P8.0 ErrorScope (`ErrorScopeValidationTests`)
- **ES1** `PushErrorScope(filter)` then `PopErrorScope` returns the
  captured error matching `filter` (Validation/OutOfMemory/Internal)
  or none. ☑ (P8.0)
- **ES2** scopes nest as a stack; an error goes to the innermost
  matching open scope; non-matching scopes pass it outward. ☑ (P8.0)
- **ES3** `PopErrorScope` with no open scope ⇒ error (callback
  status). ☑ (P8.0)
- **ES4** an error captured by a scope does NOT reach the device
  uncaptured-error callback; unmatched errors do. ☑ (P8.0)
- **ES5** pop is async (future/callback) — reuses Phase-1 plumbing;
  device-lost interaction. ☑ (P8.0)

> P8.0: core `ErrorScope` gained `filter: ErrorFilter` + captured
> error; `push_error_scope(filter)`; `pop_error_scope() ->
> Result<Option<DeviceError>, PopErrorScopeError::EmptyStack>`;
> `dispatch_error` walks `scopes.iter_mut().rev()` (innermost→outer),
> delivers to the first filter-matching scope (first-match-wins,
> `return`s so it bypasses the uncaptured callback — ES4), unmatched
> ⇒ existing uncaptured path unchanged. FFI
> `wgpuDevicePushErrorScope(filter)` (invalid filter ⇒ validation
> error) + async `wgpuDevicePopErrorScope` via
> `PendingCallback::PopErrorScope` (reuses register_callback/poll/
> WaitAny; EmptyStack ⇒ error status; lost device ⇒ Success/NoError
> like other pending callbacks); `conv` enum maps. Ported in
> `error_scope_validation.rs` (10). Gate green (54 binaries, clippy
> clean; uncaptured/`assert_device_error!` path unregressed).

### P8.1 QuerySet creation (`QuerySetValidationTests`/`QueryValidationTests`)
- **QS1** `CreateQuerySet` type ∈ {Occlusion, Timestamp}; count > 0
  and ≤ max (4096 per Dawn). ☐
- **QS2** Timestamp type requires the `timestamp-query` feature
  (HasFeature); absent ⇒ error + error QuerySet. ☐
- **QS3** `GetType`/`GetCount` reflect the descriptor; `Destroy`
  idempotent; double/destroyed-use rules. ☐
- **QS4** invalid descriptor ⇒ device error + error QuerySet
  (never panic). ☐

### P8.2 Query in commands (deferred C34/C35 + encoder query ops)
- **QC1/C34** RenderPass `occlusionQuerySet` must be a valid
  Occlusion query set. ☐
- **QC2/C35** RenderPass `timestampWrites` set must be Timestamp;
  begin/end indices in range, distinct. ☐
- **QC3** `CommandEncoderWriteTimestamp` query set Timestamp +
  index in range (deferred-error model). ☐
- **QC4** `CommandEncoderResolveQuerySet` range in set; destination
  buffer `QueryResolve` usage + offset 256-aligned + size bounds. ☐
- **QC5** occlusion query begin/end pairing within a render pass
  (Dawn pairing rules). ☐

### P8.3 DeviceLost (`DeviceLostValidationTests`)
- **DL1** `DeviceDestroy` ⇒ device lost; `GetLostFuture` resolves
  once with reason=Destroyed. ☐
- **DL2** operations after lost: creation returns error objects /
  device errors, no panic; idempotent destroy. ☐
- **DL3** pending callbacks (map, work-done, pop-error-scope)
  resolve with the appropriate lost/aborted status. ☐
- **DL4** completes/ô supersedes the Phase-1
  `device_lost_validation.rs` stub (no regression). ☐

### P8.4 Toggle / UnsafeAPI R21 (`ToggleValidationTests`,
`UnsafeAPIValidationTests`)
- **TG1** map only the toggle/feature-gated rules that have a stable
  webgpu.h feature/limit analog. ☐
- **TG2** the `AllowUnsafeAPIs`-class rules: ✗ N/A (non-canonical;
  record the divergence, rejected-direction only). ☐

### P8.5 MultipleDevice (`MultipleDeviceTests`, R15/R16)
- **MD1** an object created by device A used with device B ⇒
  validation error (bind group/pipeline/buffer/etc. via owning-
  device `Arc::ptr_eq`). ☐
- **MD2** the existing per-object device-identity checks are
  consistent and complete for the ported cases. ☐

### P8.6 Surface (descriptor/arg validation only — Noop)
- **SF1** `CreateSurface` decodes the chained window descriptor;
  null/invalid ⇒ error/error surface. ☐
- **SF2** `Configure` device/format/usage/size vs
  `GetCapabilities`; `GetCurrentTexture` unconfigured ⇒ error. ☐
- **SF3** real presentation/swapchain ✗ N/A on Noop (recorded;
  revisit with a real backend + window).

## Phase 8 exit criteria

- ES/QS/QC/DL/TG/MD/SF covered by ported Rust tests green on Noop
  (N/A items recorded with rationale); C34/C35 + encoder
  WriteTimestamp/ResolveQuerySet closed; gate clean; CI green.
- `dawn-test-mapping.md`: `QuerySetValidationTests`/`QueryValidation
  Tests`, `ErrorScopeValidationTests`, `DeviceLostValidationTests`,
  `ToggleValidationTests`, `UnsafeAPIValidationTests`,
  `MultipleDeviceTests` rows ☑/✗-N/A; the block-50 C34/C35 row ☑.
- One commit per slice (`phase-8: <slice> — <short>`).
- **Mandatory Phase 8 Review** before COMPLETE; logged in
  `tracking/phase-8-review.md`. Phase 8 COMPLETE ⇒ the core WebGPU
  validation surface is conformant on Noop with real Metal/Vulkan
  execution proven (Phase 7).
