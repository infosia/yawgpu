# Block 00 — Foundation (instance / adapter / device / future / error sink)

Phases 0–1. Rules extracted from Dawn `DeviceValidationTests`,
`MultipleDeviceTests`, `LabelTests`, `UnsafeAPIValidationTests` and the
WebGPU async model in `webgpu.h`. Source refs are absolute Dawn paths under
`dawn/src/dawn/tests/unittests/validation/`.

## Surface

- `wgpuCreateInstance`, `wgpuInstanceRelease/AddRef`,
  `wgpuInstanceProcessEvents`, `wgpuInstanceWaitAny`, `wgpuGetInstanceLimits`.
- `wgpuInstanceRequestAdapter` (+ `WGPURequestAdapterCallbackInfo`),
  `wgpuAdapterGetInfo`, `wgpuAdapterGetLimits`, `wgpuAdapterGetFeatures`,
  `wgpuAdapterHasFeature`, `wgpuAdapterRelease/AddRef`.
- `wgpuAdapterRequestDevice` (+ `WGPURequestDeviceCallbackInfo`,
  `WGPUDeviceDescriptor` incl. requiredLimits/requiredFeatures, uncaptured-
  error & device-lost callbacks).
- `wgpuDeviceGetLimits`, `wgpuDeviceGetFeatures`, `wgpuDeviceHasFeature`,
  `wgpuDeviceGetQueue`, `wgpuDeviceDestroy`, device tick/poll,
  `wgpuDeviceRelease/AddRef`.
- Labels: descriptor `label` + `wgpuXxxSetLabel` / `GetLabel`.

## Async / Future state machine (yawgpu-core) — ☑ done (P1.1)

`webgpu.h`: `WGPUCallbackMode` 456-480, `WGPUFuture` 2233-2240,
`WGPUFutureWaitInfo` 3862-3883, `WGPUWaitStatus` 1227-1242,
`wgpuInstanceWaitAny` 6419-6423, `wgpuInstanceProcessEvents` 6412-6416.

- **A1** A future is created Pending; it transitions to Complete when its
  async op finishes. Completion is **not consumed** — a completed future may
  be waited on repeatedly and keeps returning Success.
- **A2** `WaitAnyOnly` callbacks fire **only** inside `wgpuInstanceWaitAny`
  (if the future is/*becomes* complete during the call). Never via
  ProcessEvents, never spontaneously.
- **A3** `AllowProcessEvents` callbacks fire for the same reasons as A2
  **plus** inside `wgpuInstanceProcessEvents` when complete.
- **A4** `AllowSpontaneous` callbacks fire for the same reasons as A3 and
  **may** fire as soon as the op completes (for the Noop backend, treat
  completion as immediate at registration; firing still happens at
  WaitAny/ProcessEvents to stay deterministic, but Spontaneous is allowed to
  fire eagerly).
- **A5** `wgpuInstanceWaitAny(count, futures, timeoutNS)`:
  - `Error` if instance invalid, or `count>0 && futures==null`.
  - With `timeoutNS==0` (poll): `Success` if ≥1 listed future is already
    Complete (set its `completed=TRUE`, fire its callback synchronously by
    mode), else `TimedOut`.
  - With `timeoutNS>0`: requires instance feature `TimedWaitAny`; for Noop
    all ops are already complete so behaves like poll. Without the feature
    enabled, calling with `timeoutNS>0` is an error/unsupported.
  - `count==0` ⇒ `TimedOut`.
- **A6** `wgpuInstanceProcessEvents` fires all complete `AllowProcessEvents`
  and `AllowSpontaneous` callbacks; never `WaitAnyOnly`. Returns void.
- **A7** Repeated WaitAny on the same completed future keeps returning
  Success (Dawn `FuturesTests::WaitAnySameFuture`).

Implemented in **P1.1**: `FutureRegistry` now stores per-future mode +
Pending/Complete + fired flag; `process_events()`/`wait_any()` replace the
old `poll_all` blanket stub; instance `TimedWaitAny` feature gates
`timeoutNS>0`. Behaviour A1–A7 covered by `yawgpu/tests/future_modes.rs`.

> Decision recorded: `wgpuCreateInstance(NULL)` ⇒ `TimedWaitAny` enabled
> (lenient default); an explicit descriptor must list
> `WGPUInstanceFeatureName_TimedWaitAny` in `requiredFeatures` or
> `timeoutNS>0` returns `Error`.

## Validation rules

Status: ☐ not started · ◐ partial · ☑ done. "Defer" = needs a resource type
from a later phase; tracked here but ported in that phase, not Phase 1.

### DeviceValidationTests.cpp — implementable in Phase 1

- **R1** RequestDevice with no `requiredLimits` ⇒ Success, device reports all
  default limits (`maxBindGroups==4`). `NoRequiredLimits` :62. ☐
- **R2** `requiredLimits` filled with default values ⇒ Success, device
  reports those defaults. `DefaultLimits` :78. ☐
- **R3** "Higher is better" limits: requested > supported ⇒ RequestDevice
  `Error`/null; requested worse-than-default ⇒ Success but device still
  reports the default (not the worse value). `HigherIsBetter` :96. ☐
- **R4** "Lower is better" limits: requested < supported ⇒ `Error`; analogous
  default-clamping. `LowerIsBetter` :162. ☐
- **R5** On RequestDevice failure: request-device callback fires with
  `Error`/null first; a registered device-lost callback then fires with
  `FailedCreation` — immediately in `AllowSpontaneous`, only on
  `ProcessEvents` in `AllowProcessEvents`. `ErrorTriggersDeviceLost` :232. ☐
- **R6** Requiring `TextureFormatsTier1` implicitly enables
  `RG11B10UfloatRenderable` (`HasFeature` true). :286. ☐
- **R7** Requiring `TextureFormatsTier2` implicitly enables
  `TextureFormatsTier1`. :302. ☐
- **R8** `Device.Destroy()` then device tick ⇒ no-op, returns false.
  `DestroyDeviceBeforeAPITick` :320. ☐
- **R9** `GetAHardwareBufferProperties` without the AHB feature ⇒ validation
  error. :335. ☐ (low priority / platform-niche — may defer within Phase 1)
- **R10** Core-defaulting adapter, explicit `CoreFeaturesAndLimits` ⇒ core
  limits (`maxStorageBuffersInVertexStage>0`). :356. ☐
- **R11** Core-defaulting adapter, no features ⇒ defaults to
  `CoreFeaturesAndLimits`. :377. ☐
- **R12** Compat-defaulting adapter, no features ⇒ compat limits
  (`maxStorageBuffersInVertexStage==0`). :421. ☐
- **R13** Compat-defaulting adapter, explicit `CoreFeaturesAndLimits` ⇒ core
  limits. :400. ☐
- **R14** `maxImmediateSize`: requested < supported ⇒ device gets the
  supported max (always-max limit). `AlwaysMax` :450. ☐

### LabelTests.cpp — Phase 1 subset

- **R17a** Device and Queue: label empty if unset; settable via descriptor
  `label` and `SetLabel`; `GetLabel` round-trips. (Dawn `LabelTest::Queue`
  :*, others.) ☐  — only Device/Queue in Phase 1; other objects = Defer.

### Deferred (need later-phase resources) — tracked, ported later

- **R15** Submitting a `CommandBuffer` from device B to device A's queue ⇒
  validation error. `MultipleDeviceTest::ValidatesSameDevice`. Defer→P6.
- **R16** `CreateComputePipelineAsync` with a cross-device shader module ⇒
  `ValidationError` status. Defer→P5.
- **R17b** Labels for all other creatable objects (Buffer/Texture/…/Pipeline/
  ShaderModule). Port alongside each object's phase.
- **R18** `chromium_disable_uniformity_analysis` shader ext is unsafe unless
  `AllowUnsafeAPIs`. Defer→P4.
- **R19** `BindGroupLayoutEntry.bindingArraySize>1` unsafe unless
  `AllowUnsafeAPIs` (0/1 always ok). Defer→P4.
- **R20** WGSL static `binding_array` unsafe unless `AllowUnsafeAPIs`.
  Defer→P4.
- **R21** `CommandEncoder.WriteTimestamp` needs `TimestampQuery` +
  `AllowUnsafeAPIs`. Defer→P8.

## Error model

Per-device error sink (Phase 0): uncaptured-error callback + error-scope
stack; `dispatch_error` → top scope else uncaptured. Phase 1 adds the
device-lost callback channel (R5) distinct from the uncaptured-error sink.

## Open questions

- Noop adapter: expose one synthetic adapter; needs a configurable
  core-vs-compat mode + a synthetic limit/feature set so R3/R4/R10–R14 are
  testable. Define the synthetic adapter's "supported" limits explicitly.
- `wgpuGetInstanceLimits` / instance `TimedWaitAny` feature: model as an
  instance descriptor feature list.
- Device-lost timing on `wgpuDeviceRelease` vs explicit `Destroy`.

## Review notes (carried)

- Phase-0 `WGPU*Impl` double-`Arc`; `testing_*` hooks → consider `testing`
  feature gate. (See `tracking/phase-0.md`.)
