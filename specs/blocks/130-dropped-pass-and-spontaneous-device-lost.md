# Block 130 — Dropped pass lowering and spontaneous device-lost delivery

Status: **COMPLETE (2026-11-03)**, with a regression found afterwards. Phase Review: 0 CRITICAL, 0 MAJOR in code, 1 MAJOR in the spec, 5 MINOR; all resolved, see below. The
Windows native Vulkan CTS found that R1 fails valid submits; see "Windows native
Vulkan" below and backlog A22.
Source: backlog A10 and A14, started by user request on 2026-11-03.
Oracle: Dawn at the pinned `third_party/dawn`: `native/Device.cpp`
(`DeviceBase::WillDropLastExternalRef`, `DeviceBase::Destroy`, `DeviceLostEvent::SetLost`,
`DeviceLostEvent::Complete`), `native/EventManager.cpp` (`SetFutureReady`), and the
`v20261030` Dawn build that the bench links.

## Problem

### A10 — a pass that fails to lower is dropped silently

`append_hal_command_execution` (`yawgpu-core/src/queue.rs`) lowers each submitted command
to a `HalCopy`. Three arms drop their command when lowering returns `None`. There is no
error, and the rest of the submit runs:

- `CommandExecution::ComputePass`, through `hal_compute_pass_execution`
- `CommandExecution::RenderPass`, through `hal_render_pass_execution`
- `CommandExecution::SubpassRenderPass` (`tiled`), through `hal_subpass_render_pass_execution`

`None` means that a submit-validated object has no HAL object. The lowering cannot tell
which one. Today this cannot happen, but nothing enforces it. If it does happen, the
damage is real. Block 119's indirect-dispatch validation is an internal compute pass. If
it is dropped, the user's indirect dispatch reads unvalidated arguments. The submit walk
already has an abort path for staging errors: nothing from the command buffers runs, the
queued writes still flush, the texture-initialization overlay is discarded, and the submit
returns a `DeviceError`. A lowering failure belongs on that path.

### A14 — the premise was wrong; the real gap is spontaneous delivery

Backlog A14 said that yawgpu does not destroy the device when the last reference is
released, so the device-lost callback reports `CallbackCancelled` where Dawn reports
`Destroyed`. That is not what happens. `wgpuDeviceRelease` already calls
`implicit_destroy_on_last_release`, which records `Destroyed`.

A C probe compared the two libraries. It released the device, adapter, and instance in
several orders, with each device-lost callback mode. Reasons: 2 = `Destroyed`,
3 = `CallbackCancelled`.

| Teardown order | Mode | Dawn | yawgpu |
|---|---|---|---|
| device, adapter, instance | `AllowProcessEvents` | never fires | 3, at instance release |
| device, adapter, instance | `AllowSpontaneous` | 2, inside `wgpuDeviceRelease` | 3, at instance release |
| device, `ProcessEvents`, adapter, instance | either | 2 | 2, at `ProcessEvents` |
| adapter, instance, then device | `AllowProcessEvents` | never fires | 3, at instance release |
| adapter, instance, then device | `AllowSpontaneous` | 2, inside `wgpuDeviceRelease` | 3, at instance release |

Three differences follow.

1. **Spontaneous delivery.** Dawn's `SetFutureReady` completes an `AllowSpontaneous`
   event at once, on the thread that made it ready. So Dawn fires the device-lost
   callback inside `wgpuDeviceRelease` and inside `wgpuDeviceDestroy`. yawgpu always
   waits for `wgpuInstanceProcessEvents` or `wgpuInstanceWaitAny`. That is allowed by
   the spec, but it loses the reason when the application releases the instance without
   processing events first. Fixed here.
2. **Unprocessed `AllowProcessEvents` at instance release.** Dawn never fires the
   callback: the device-lost event holds the device, the device holds the adapter, and
   the adapter holds the instance, so the instance is never freed. yawgpu cancels the
   callback with `CallbackCancelled`, which is what `webgpu.h` asks for when an instance
   goes away. yawgpu is right. The bench's exit code 1 comes from this row: the bench
   releases the instance without processing events, and it treats every reason except
   `Destroyed` as fatal. This is fixed in the bench, not the library.
3. **Instance released while a device is alive.** Dawn keeps the instance alive through
   the device. yawgpu cancels the device's callbacks at instance release. After this
   block, yawgpu fires `Destroyed` there only if the device was already lost. Recorded as
   backlog A19, out of scope.

## Rules

R1. **Lowering failure aborts the submit.** When the compute pass, render pass, or
subpass render pass arm cannot lower its command, the submit walk records an internal
error and stops. The submit then takes the existing abort path, as a staging error does:
- nothing from the command buffers runs
- the queued writes issued before the submit still flush, and every chunk is retired
- the texture-initialization overlay is dropped
- `Queue::submit` returns `DeviceError::internal(..)` with a message that names the pass
  kind

The arms that skip on purpose do not change: empty copies, zero-size clears, and the
other early returns.

R2. **Spontaneous device-lost delivery.** When a device becomes lost and its device-lost
callback mode is `AllowSpontaneous`, the callback fires before the call that caused the
loss returns. This covers `wgpuDeviceDestroy`, the last `wgpuDeviceRelease`, and loss
found during a submit. It fires once. It does not fire again from `ProcessEvents`,
`WaitAny`, or instance release.
- The callback runs with no yawgpu lock held. The application may call back into
  yawgpu from it, for example to release objects.
- `AllowProcessEvents` and `WaitAnyOnly` do not change.
- The device pointer passed to the callback does not change from today: null after the
  last release, the device otherwise.
- A `wgpuDeviceGetLostFuture` waiter sees the future as complete.
- Delivery goes to whichever thread first sees the loss. If another thread's
  `wgpuQueueSubmit` or a second `wgpuDeviceDestroy` sees it first, that thread runs the
  callback. The thread that caused the loss may then return before the callback has
  finished. The callback still runs exactly once.
- "Loss found during a submit" covers only losses that `Device::lose` records.
  yawgpu-core does not detect HAL device loss today. So in production the submit path
  delivers only losses started on another thread by destroy or release.

R3. **Bench teardown.** `benches/src/main.cpp` calls `wgpuInstanceProcessEvents` after
`wgpuDeviceRelease` and before the instance is released. Both libraries then report
`Destroyed`, and the yawgpu bench exits 0.

## Tests

- Unit (core, Noop):
  - Make each pass kind fail to lower by giving it a pipeline or bind group with no HAL
    object, built as a test-only state.
  - Then `submit` returns an internal `DeviceError` and none of the pass's commands reach
    the HAL.
  - A queued `write_buffer` issued before the submit still lands.
  - The texture-initialization marks are unchanged.
- Unit (FFI, Noop), `AllowSpontaneous`:
  - The callback fires inside `wgpuDeviceDestroy` with `Destroyed`, and inside the last
    `wgpuDeviceRelease` with `Destroyed` and a null device.
  - It fires exactly once across a later `ProcessEvents` and instance release.
  - A callback that releases the queue does not deadlock.
- Unit (FFI, Noop), `AllowProcessEvents`: the existing behaviour is pinned.
  - Destroy, then `ProcessEvents`, gives `Destroyed`.
  - Release with no `ProcessEvents`, then instance release, gives `CallbackCancelled`.
- The probe's table is re-run against yawgpu. The `AllowSpontaneous` rows where the device
  is released first must match Dawn. The row "adapter, instance, then device" keeps
  `CallbackCancelled` until A19.
- The bench exits 0 on Metal with yawgpu and with Dawn.

## Out of scope

- A19: keep the instance alive while a device is alive, as Dawn does.
- A20: a `FailedCreation` device-lost callback in `AllowSpontaneous` mode still waits for
  `ProcessEvents`. Dawn fires it inside `RequestDevice` (`native/Adapter.cpp`,
  `lostEvent->SetLost`).
- Spontaneous delivery of the other callbacks (`mapAsync`, `onSubmittedWorkDone`, and the
  rest). Dawn fires them from its own completion polling; yawgpu still delivers them at
  `ProcessEvents` or `WaitAny`.

## Slices

- **S1 (coding agent):** R1-R3 with the unit tests.
- **S2 (Claude):** probe re-run, bench, Metal and Vulkan e2e, Phase Review.

## Verification (2026-11-03)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,331 passed, 0 failed |
| fmt, clippy on 8 feature sets | clean |
| Metal e2e, `metal,tiled` | 161 passed, 0 failed |
| Vulkan e2e, `vulkan,tiled` | 159 passed, 1 failed (the known `imageView2DOn3DImage`). Every VUID comes from that test and its leaked objects |
| Probe re-run against yawgpu, Metal | `AllowSpontaneous`, device released first: `Destroyed` inside `wgpuDeviceRelease`, as Dawn. Instance released first: `CallbackCancelled` (A19). `AllowProcessEvents` rows unchanged |
| Bench `--scale 0.02`, Metal | yawgpu exits 0 (was 1); Dawn exits 0 |

The CTS was not rerun. The CTS cannot see R1, because the abort path is unreachable
with valid objects. Its harness does not use `AllowSpontaneous` device-lost callbacks
(inferred).

### Windows native Vulkan (2026-11-03): regression found

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`f3a63e9`, CTS `a5cbf67`. The Khronos validation layer was on for the e2e and HAL runs.
Vulkan e2e 162 passed, 0 failed, 0 VUIDs, and Vulkan HAL ignored tests 78 passed.

The CTS, which this block did not run, finds a regression in R1. Unforced,
`api,validation,encoding,programmable,pipeline_bind_group_compat:*` gives 432 failures
(`bgl_binding_mismatch` 64, `bgl_resource_type_mismatch` 368), all render pass and
render bundle cases, each with the uncaptured error "render pass lowering failed". At
`019bd3b`, before this block, `api,validation` had 0 failures.

The premise "the abort path is unreachable with valid objects" is wrong. The test sets
a pipeline, then a bind group whose layout does not match the pipeline's, and finishes
without a draw. That command buffer is valid. Render lowering
(`hal_render_commands` in `yawgpu-core/src/queue.rs`) resolves each `SetBindGroup`
against the current pipeline's bindings at once, and `hal_bind_group_resources` returns
`None` when a binding the pipeline declares is missing from the group. Before this block
that `None` dropped the whole pass silently. Since this block it fails the whole submit.
The same command order is common and valid with a draw: `setPipeline(A)`,
`setBindGroup(0, groupForB)`, `setPipeline(B)`, `draw`. It is core code, so Metal is
affected too (inferred; not run). Compute passes pass these cases. Backlog A22.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| M1 | The Tests section required the instance-released-first `AllowSpontaneous` row to match Dawn, which contradicts A19 | Spec corrected: that row keeps `CallbackCancelled` until A19 |
| m1 | A concurrent submit or second destroy can take over delivery, so the thread that caused the loss may return before the callback runs | Accepted; exactly-once holds. Stated in R2 |
| m2 | A spontaneous `FailedCreation` callback is still deferred | Out of scope, backlog A20 |
| m3 | `observe_device_loss` upgrades every device's weak owner on the instance | No change: no lock is held, so it is safe |
| m4 | Test asserts inside an `extern "C"` callback abort the binary instead of failing one test | Observations recorded in state and asserted after the call |
| m5 | Submit-time loss delivery is test-only, because core does not detect HAL loss | Stated in R2 |
