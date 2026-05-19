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

## P8.0 — ErrorScope  *(☑ DONE)*
ES1–ES5 done: core `ErrorScope` filter + captured error;
`push_error_scope(filter)`; `pop_error_scope() ->
Result<…,EmptyStack>`; `dispatch_error` innermost→outer first-
match-wins routing (ES2) that `return`s before the uncaptured
callback (ES4), unmatched ⇒ unchanged uncaptured path. FFI
`wgpuDevicePushErrorScope` + async `wgpuDevicePopErrorScope`
(`PendingCallback::PopErrorScope`, reuses the Phase-1 future
machinery; EmptyStack=ES3 error status; lost-device resolves
Success/NoError) + `conv` enum maps. Ported
`error_scope_validation.rs` (10: ES1 catch, ES2 nested matching/
non-matching bubble, ES3 empty-stack, ES4 caught-vs-uncaptured, ES5
async WaitAny + destroyed-device pop, first-error-kept, all-filters-
map, success-no-error). Gate green (Noop 54 binaries + clippy clean;
`instance_smoke`/`device_lost_validation` unregressed). Committed
`phase-8: P8.0`.

#### (original)
ES1–ES5: push/pop, filter match, nested stack, scope-vs-uncaptured-
callback routing, async pop (future) + device-lost interaction.
`ErrorScopeValidationTests`.

## P8.1 — QuerySet creation  *(☑ DONE)*
QS1–QS4 done: `Feature::TimestampQuery` added (requestable; conv
both-way); core `QuerySet` Arc handle + `Device::create_query_set`
(type/count≤4096, Timestamp⇒feature gate, QS4 error QuerySet, QS3
destroy idempotent); FFI `WGPUQuerySetImpl` real handle +
CreateQuerySet/GetType/GetCount/Destroy/SetLabel/Release/AddRef +
conv maps. Tests `query_validation.rs` (8). Gate green (Noop 55
binaries + clippy clean; features/limits unregressed). Committed
`phase-8: P8.1`.

## P8.2 — Query in commands  *(☑ DONE)*
QC1–QC5 done: RenderPassDescriptor +occlusion/timestamp fields +
conv decode; validate_render_pass_descriptor C34 (occlusionQuerySet
Occlusion) + C35 (timestampWrites Timestamp/indices); CommandEncoder
write_timestamp/resolve_query_set (QC3/QC4, record_buffer_command
deferred, QUERY_RESOLVE+256-align+OOB); RenderPassEncoder occlusion
begin/end pairing (QC5 via pass state). FFI 4 fns + conv.
render_pass_descriptor test strengthened (dangling→real Occlusion
QuerySet). Tests query_validation.rs (12). Gate green (Noop 55
binaries + clippy; P6 command/pass unregressed). Committed
`phase-8: P8.2`.

## P8.3 — DeviceLost  *(☑ DONE)*
DL1–DL4 done: wgpuDeviceGetLostFuture (reuses PendingCallback::
DeviceLost, single loss event, idempotent Destroyed); Device::
is_lost() short-circuit prepended to all create paths (error object,
no device error, non-lost unchanged); BufferMap/QueueWorkDone +device
ref resolve Aborted on loss; PopErrorScope P8.0 unchanged. Extended
device_lost_validation.rs to 7 (4 Phase-1 kept). Gate green (Noop 55
binaries + clippy; non-lost paths unregressed). Committed
`phase-8: P8.3`.

## P8.4 — Toggle / UnsafeAPI (R21)  *(☑ DONE)*
TG1 ✗ N/A (Dawn toggle APIs absent from webgpu-headers; no toggle/
AllowUnsafeAPIs subsystem — recorded divergence). TG2: rejected-
direction of UnsafeAPIValidationTests pinned by new
unsafe_api_validation.rs (5: R18 chromium ext / R19 bindingArraySize>1
/ R20 static binding_array / R21 WriteTimestamp+TimestampQuery) —
all rejections pre-existing (P4 shader/BGL + P8.2 QC3), zero
production code. Gate green (Noop 56 binaries + clippy clean).
Committed `phase-8: P8.4`.

## P8.5 — MultipleDevice (R15/R16)  *(☑ DONE)*
MD1/MD2 done: bind-group resources already rejected (P4); P8.5 adds
owning-Device same() checks at the FFI for the Dawn-exercised ops
(pipeline-layout BGL, compute/render pipeline shader+layout sync+
async, encoder copies, pass set-pipeline/bindgroup/vertex/index,
queueSubmit, queueWriteBuffer) routed via existing path (immediate
for pipeline create; deferred via new record_validation_error for
encoder/pass). Same-device byte-for-byte unchanged. Tests
multiple_device_validation.rs (6). Gate green (Noop 57 binaries +
clippy; same-device suites unregressed). Committed `phase-8: P8.5`.

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
