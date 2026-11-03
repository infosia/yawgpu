# Block 137 — Spontaneous callbacks that are ready when registered

Status: **COMPLETE (2026-11-04)**. Phase Review: 0 CRITICAL, 1 MAJOR, 3 MINOR; all fixed,
see below.
Source: backlog A20 (Block 130 "Out of scope"), started by user request on 2026-11-04.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`EventManager.cpp` (`EventManager::TrackEvent`: "Handle the event now if it's
spontaneous and ready"; `SetFutureReady`), `Adapter.cpp` (`AdapterBase::CreateDevice`,
`APIRequestDevice`), and the `v20261030` Dawn build that the bench links.

## Problem

Backlog A20 says that a `FailedCreation` device-lost callback in `AllowSpontaneous` mode
waits for `ProcessEvents`, while Dawn fires it inside `wgpuAdapterRequestDevice`. A C
probe linking each library, on the M2 with Metal, shows that A20 is one case of a wider
rule.

Dawn's `EventManager::TrackEvent` completes an `AllowSpontaneous` event at once when the
event is already ready at registration. Most of Dawn's API entry points create an
event that is ready at once, so their spontaneous callbacks fire before the call
returns. yawgpu queues every callback and delivers it only from
`wgpuInstanceProcessEvents` or `wgpuInstanceWaitAny`, except the device-lost delivery
that Block 130 added.

Probe results. "inside" means the callback ran before the API call returned; "PE" means
it ran in the next `wgpuInstanceProcessEvents`.

| Call, all callbacks `AllowSpontaneous` unless stated | Dawn | yawgpu |
|---|---|---|
| `RequestAdapter`, success and no-adapter failure | inside | PE |
| `RequestDevice` success | inside | PE |
| `RequestDevice` failure (required limit too high): request callback, then device-lost (`FailedCreation`, null device) | inside, in that order | PE, in that order |
| `RequestDevice` failure, request `AllowProcessEvents`, device-lost spontaneous | device-lost inside; request at PE | both at PE |
| `RequestDevice` failure, request spontaneous, device-lost `AllowProcessEvents` | request inside; device-lost at PE | both at PE |
| `PopErrorScope` | inside | PE |
| `ShaderModule::GetCompilationInfo` | inside | PE |
| `Buffer::MapAsync` validation error (buffer not mappable) | inside, `Error` | PE |
| `Buffer::MapAsync` valid, no pending submission | inside, `Success` | PE |
| `Queue::OnSubmittedWorkDone`, no pending submission | inside | PE |
| `CreateComputePipelineAsync` (valid) | not within the probe: Dawn compiles on a worker thread | PE |

The failure message also differs. Dawn's `FailedCreation` device-lost message is
`"Failed to create device:\n"` followed by the creation error. yawgpu's is the fixed
`"Device creation failed"`.

The spec allows both behaviours. But an application written against Dawn that relies on
`AllowSpontaneous` never calling `ProcessEvents`, for example a request followed by
a wait on a flag, hangs with yawgpu.

## Rules

R1. **Ready at registration means fired at registration.** When yawgpu registers a
callback whose mode is `AllowSpontaneous` and the callback is already complete, it
fires the callback before the API call returns, on the calling thread. It fires once.
It does not fire again from `ProcessEvents`, `WaitAny`, or instance release.
- This covers every callback registered through `register_callback`, the path for
  callbacks that are complete at registration: `RequestAdapter`, `RequestDevice`, the
  `FailedCreation` device-lost callback, `PopErrorScope`, `GetCompilationInfo`,
  `CreateComputePipelineAsync`, `CreateRenderPipelineAsync`, and the `MapAsync`
  callbacks completed without a submission gate.
- It also covers callbacks registered through `register_submission_callback`
  (`MapAsync` with a gate, `OnSubmittedWorkDone`) when the registration-time poll finds
  the gate already passed or the device lost.
- `AllowProcessEvents` and `WaitAnyOnly` do not change.

R2. **No lock is held while the callback runs**, as in Block 130 R2. The application may
call back into yawgpu from the callback, for example to release the object, call
`ProcessEvents`, or make another request. Every registration site must reach the
registration with no yawgpu mutex guard alive. If a site cannot, the coding agent
reports it rather than firing under the lock.

R3. **Futures stay observable.** The `WGPUFuture` returned by a call whose callback
already fired is complete: `wgpuInstanceWaitAny` on it reports it completed and fires
nothing.

R4. **Order.** Callbacks that one call makes ready fire in registration order. When a
registration fires its own spontaneous callback, it first fires every older
`AllowSpontaneous` callback that is ready but not yet delivered, in registration order.
For example, a work-done callback registered behind a submission that has since
completed fires before a new work-done callback registered on the same queue. In Dawn
the older one would already have fired from its completion polling. Older
`AllowProcessEvents` callbacks keep waiting. For a
failed `RequestDevice`, the request callback fires first and the device-lost callback
second, as Dawn. When only one of the two is spontaneous, that one fires inside the call
and the other waits for `ProcessEvents`, as Dawn.

R5. **`FailedCreation` message.** The `FailedCreation` device-lost message is
`"Failed to create device:\n"` followed by the same error message the request-device
callback receives. The other reasons keep their messages.

R6. **Unchanged.**
- A spontaneous callback that becomes ready later, after GPU work completes (`MapAsync`
  or `OnSubmittedWorkDone` behind a pending submission), is still delivered at
  `ProcessEvents` or `WaitAny`, or by a later registration under R4. Dawn delivers
  those from its own completion polling.
- `RequestAdapter` and `RequestDevice` with `AllowProcessEvents` keep their timing.
- Callback arguments: statuses, the device pointer (null for `FailedCreation`), and
  messages other than R5.

## Tests

- Unit (FFI, Noop), one per kind in R1, each `AllowSpontaneous`:
  - The callback has run when the API call returns, with the same status and arguments
    as today.
  - A later `ProcessEvents`, `WaitAny` on the returned future, and instance release
    fire nothing more. `WaitAny` reports the future completed.
- Unit (FFI, Noop), failed `RequestDevice`: the four mode combinations of R4 give the
  probe's order. The device-lost message starts with `"Failed to create device:\n"` and
  contains the request error message.
- Unit (FFI, Noop): a spontaneous `RequestDevice` callback that releases the device and
  calls `ProcessEvents` inside the callback does not deadlock. Same for a
  `PopErrorScope` callback that pushes and pops another scope.
- Unit (FFI, Noop): `AllowProcessEvents` timing is pinned for `RequestDevice` and
  `PopErrorScope`.
- Observations made inside an `extern "C"` callback are recorded in state and asserted
  after the call (Block 130 m4).
- Claude: re-run the probe against yawgpu on Metal; every "inside" row in the table must
  match Dawn. Metal e2e. MoltenVK e2e as reference.
- CTS (Claude): `webgpu:api,*` on Metal. The harness's own callback modes decide what it
  can see; it must not regress.

## Out of scope

- Spontaneous delivery of callbacks that become ready later (R6). That needs a
  completion thread or polling, as Dawn has.
- A19 (instance lifetime while a device is alive), the next block.

## Slices

- **S1 (coding agent):** R1-R6 with the unit tests.
- **S2 (Claude):** probe re-run, e2e, CTS, Phase Review.

## Verification (2026-11-04)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,391 passed, 0 failed |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, all targets) | clean |
| Probe re-run against yawgpu, Metal, after the Phase Review fixes | every "inside" row of the Problem table matches Dawn, including all four `RequestDevice` failure mode combinations and their order. The `FailedCreation` message is `"Failed to create device:\n"` plus yawgpu's own validation text |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled` | 162 passed, 0 failed (before and after the fixes) |
| Vulkan e2e, `vulkan,tiled` | 155 passed, 1 failed (the known `imageView2DOn3DImage`); every VUID comes from that test |
| webgpu-native-cts, Metal, `webgpu:api,*` (library before the Phase Review fixes) | pass 586,611, skip 104,756, fail 0, crash 0, xfail 2 |

The CTS ran on the library before the fixes. The fixes change only which older ready
spontaneous callbacks a registration also delivers, and tests. The CTS `xpass=4` comes
from the `index_buffer_format_dirtying:*` expectation glob, which also matches that
query's 4 passing subcases. Dawn gives pass 4 / fail 2 on the query, and yawgpu gives
the same at every library from Block 130 to this block.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| M1 | A registration fired only its own future. An older spontaneous submission callback made ready by the same registration-time poll fired later, so a newer work-done callback on a queue could run before an older one | Fixed: older ready `AllowSpontaneous` callbacks fire first, in id order. R4 and R6 amended; the test records the order |
| m1 | The `compilation_info_callback` test helper stopped rejecting a null `info` | Restored |
| m2 | No re-entrancy tests for the handle clones in `MapAsync`, `GetCompilationInfo`, and `RequestAdapter` | Added. A test also pins today's behaviour when a failed `RequestDevice` callback releases the last instance reference: the `FailedCreation` device-lost callback is cancelled. A19 revisits it |
| m3 | `device_lost_validation.rs` did not check the state before `ProcessEvents` for request spontaneous, device-lost `AllowProcessEvents` | Added |

