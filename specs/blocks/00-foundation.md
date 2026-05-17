# Block 00 — Foundation (instance / adapter / device / future / error sink)

Phases 0–1. To be filled from `DeviceValidationTests`,
`UnsafeAPIValidationTests`, `MultipleDeviceTests`, `LabelTests`, plus the
async machinery used by all later phases.

## Surface (to enumerate at Phase 1)

- `wgpuCreateInstance`, `wgpuInstanceRelease`, `wgpuInstanceProcessEvents`,
  `wgpuInstanceWaitAny`, `wgpuGetInstanceLimits`.
- `wgpuInstanceRequestAdapter` (+ `WGPURequestAdapterCallbackInfo`),
  `wgpuAdapterGetInfo`, `wgpuAdapterGetLimits`, `wgpuAdapterRelease`.
- `wgpuAdapterRequestDevice` (+ `WGPURequestDeviceCallbackInfo`),
  `WGPUDeviceDescriptor` (incl. uncaptured-error & device-lost callbacks),
  `wgpuDeviceRelease`.

## Rules (to extract from Dawn)

- _TBD at Phase 1 — one bullet per Dawn `TEST_F` case, e.g. requesting a
  device with unsupported features → error; using an object from device A
  with device B → error (`MultipleDeviceTests`)._

## Async / error model

- Future registry in `yawgpu-core`; callback modes per spec.
- Noop: futures complete on `wgpuInstanceProcessEvents` /
  `wgpuInstanceWaitAny` synchronously → deterministic tests.
- Per-device error sink: uncaptured-error callback + error-scope stack;
  drives `assert_device_error!`.

## Open questions

- Adapter enumeration semantics for Noop (single synthetic adapter?).
- Device-lost timing on `wgpuDeviceRelease`.
