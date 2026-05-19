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
  and ≤ max (4096 per Dawn). ☑ (P8.1)
- **QS2** Timestamp type requires the `timestamp-query` feature
  (HasFeature); absent ⇒ error + error QuerySet. ☑ (P8.1)
- **QS3** `GetType`/`GetCount` reflect the descriptor; `Destroy`
  idempotent; double/destroyed-use rules. ☑ (P8.1)
- **QS4** invalid descriptor ⇒ device error + error QuerySet
  (never panic). ☑ (P8.1)

> P8.1: added real `Feature::TimestampQuery` (in `supported_features`
> ⇒ requestable; conv both-way map, no `Other` fallthrough) so QS2 is
> a non-vacuous gate. core `QuerySet` Arc handle
> {kind:QueryType(Occlusion|Timestamp|Unknown),count,is_error,
> is_destroyed,label} + `Device::create_query_set` →
> `validate_query_set_descriptor` (count>0 & ≤4096 MAX_QUERY_COUNT,
> Timestamp⇒feature, Unknown⇒error; QS4 device error + error
> QuerySet). FFI `WGPUQuerySetImpl` promoted to a real
> `Arc<core::QuerySet>` handle + CreateQuerySet/GetType/GetCount/
> Destroy(idempotent)/SetLabel/Release/AddRef; conv
> `WGPUQuerySetDescriptor`/`WGPUQueryType` maps. Ported in
> `query_validation.rs` (8). Gate green (55 binaries, clippy clean;
> features/limits unregressed).

### P8.2 Query in commands (deferred C34/C35 + encoder query ops)
- **QC1/C34** RenderPass `occlusionQuerySet` must be a valid
  Occlusion query set. ☑ (P8.2)
- **QC2/C35** RenderPass `timestampWrites` set must be Timestamp;
  begin/end indices in range, distinct. ☑ (P8.2)
- **QC3** `CommandEncoderWriteTimestamp` query set Timestamp +
  index in range (deferred-error model). ☑ (P8.2)
- **QC4** `CommandEncoderResolveQuerySet` range in set; destination
  buffer `QueryResolve` usage + offset 256-aligned + size bounds. ☑ (P8.2)
- **QC5** occlusion query begin/end pairing within a render pass
  (Dawn pairing rules). ☑ (P8.2)

> P8.2: core `RenderPassDescriptor` +`occlusion_query_set` +
> `timestamp_writes(RenderPassTimestampWrites{query_set,beginning_
> index,end_index})`; `conv::map_render_pass_descriptor` decodes
> them (UNDEFINED/null⇒None). `validate_render_pass_descriptor`
> (P6.4 deferred path) ⇒ C34 occlusionQuerySet non-error/destroyed +
> kind Occlusion; C35 timestamp set + ≥1 index + indices<count
> (begin≠end). `CommandEncoder::{write_timestamp,resolve_query_set}`
> via `record_buffer_command` (QC3 Timestamp+index<count; QC4
> first+count≤count & count>0, dest QUERY_RESOLVE usage +
> offset%256 + offset+count*8≤size, all checked). `RenderPass
> EncoderState` tracks `occlusion_query_set`/`open_occlusion_query`/
> `used_occlusion_queries` ⇒ QC5 begin needs the set + index<count +
> no nesting + no dup + must be closed by pass End; end requires an
> open query. FFI `wgpuCommandEncoderWriteTimestamp`/
> `ResolveQuerySet` + `wgpuRenderPassEncoderBegin/EndOcclusionQuery`
> + conv. `render_pass_descriptor_validation` adjusted: the former
> opaque-`dangling()` occlusion/timestamp case → a real Occlusion
> `wgpuDeviceCreateQuerySet` (strengthened, still `_ok`; the
> now-invalid dangling-timestampWrites sub-case dropped). Ported in
> `query_validation.rs` (12). Gate green (55 binaries, clippy clean;
> P6 command/pass suites unregressed).

### P8.3 DeviceLost (`DeviceLostValidationTests`)
- **DL1** `DeviceDestroy` ⇒ device lost; `GetLostFuture` resolves
  once with reason=Destroyed. ☑ (P8.3)
- **DL2** operations after lost: creation returns error objects /
  device errors, no panic; idempotent destroy. ☑ (P8.3)
- **DL3** pending callbacks (map, work-done, pop-error-scope)
  resolve with the appropriate lost/aborted status. ☑ (P8.3)
- **DL4** completes/ô supersedes the Phase-1
  `device_lost_validation.rs` stub (no regression). ☑ (P8.3)

> P8.3: `wgpuDeviceGetLostFuture` implemented — registers a
> `PendingCallback::DeviceLost`, completed immediately if already
> lost else queued in `device_lost_futures` (same single loss event
> as the creation-time device-lost callback; `DeviceDestroy` ⇒
> `Destroyed`, idempotent). DL2: `Device::is_lost()` short-circuit
> **prepended** to every create path (buffer/texture/sampler/shader/
> BGL/BG/pipeline-layout/compute+render pipeline/query-set) ⇒ returns
> the `is_error` handle WITHOUT validating or dispatching a device
> error; non-lost path byte-for-byte unchanged; `lost_reason()`
> accessor added. DL3: `PendingCallback::{BufferMap,QueueWorkDone}`
> gained a `device` ref; on loss the matching pending callbacks
> resolve `Aborted`; `PopErrorScope` keeps the P8.0 lost behavior.
> Extended `device_lost_validation.rs` to 7 (4 Phase-1 kept + DL1–DL3).
> Gate green (55 binaries, clippy clean; non-lost create/map/submit
> unregressed).

### P8.4 Toggle / UnsafeAPI R21 (`ToggleValidationTests`,
`UnsafeAPIValidationTests`)
- **TG1** map only the toggle/feature-gated rules that have a stable
  webgpu.h feature/limit analog. ✗ N/A (P8.4)
- **TG2** the `AllowUnsafeAPIs`-class rules: ✗ N/A (non-canonical;
  record the divergence, rejected-direction only). ☑ (P8.4, rejected-dir)

> P8.4 (verify-and-record, zero production code): `ToggleValidation
> Tests` (`QueryToggleInfo`/`OverrideToggleUsage`/`TurnOffVsyncWith
> Toggle`) use Dawn-internal toggle APIs absent from webgpu-headers
> ⇒ **TG1 ✗ N/A** (no toggle/`AllowUnsafeAPIs` subsystem built —
> non-canonical divergence, already recorded in block 00/30). TG2:
> the canonical *rejected-direction* of `UnsafeAPIValidationTests`
> is enforced and now pinned by `unsafe_api_validation.rs` (5) —
> R18 `chromium_disable_uniformity_analysis` ext, R20 static WGSL
> `binding_array`, R19 `bindingArraySize > 1` (0/1 ok) all already
> rejected by Phase-4 shader/BGL validation; R21
> `CommandEncoderWriteTimestamp` already gated on
> `Feature::TimestampQuery` (P8.2 QC3, ok with the feature). No code
> change needed (all rejections pre-existing). Gate green (56
> binaries, clippy clean).

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
