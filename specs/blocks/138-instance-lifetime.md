# Block 138 — The instance stays in use while its adapters and devices are

Status: **IN PROGRESS (2026-11-04)**.
Source: backlog A19 (Block 130 Problem, point 3), started by user request on 2026-11-04.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`Adapter.h` (`AdapterBase::mInstance` is a `Ref<InstanceBase>`), `Instance.cpp`
(`InstanceBase::DeleteThis`: callbacks are flushed only when the instance object is
destroyed), `EventManager.cpp`, and the `v20261030` Dawn build that the bench links.

## Problem

In Dawn, every adapter holds a reference to its instance, and every device holds its
adapter. Releasing the application's last `WGPUInstance` reference changes nothing
while an adapter or device is still alive. The instance object is destroyed, and its
remaining events are cancelled, only when nothing refers to it.

yawgpu cancels every pending callback as soon as the instance's last external reference
is released (`wgpuInstanceRelease`, `cancel_callbacks`). After that,
`register_pending_callback` cancels each new registration at once, because the
instance's external reference count is 0.

A C probe ran on the M2 with Metal. It created a device, registered a `mapAsync` in
`AllowProcessEvents` mode behind a submission, released the adapter and the instance,
and then kept using the device. Every callback in the "after" rows is `AllowSpontaneous`.

| Step | Dawn | yawgpu (Block 137) |
|---|---|---|
| Instance release | nothing fires | device-lost fires `CallbackCancelled`; the pending `mapAsync` fires `CallbackCancelled` |
| `createBuffer` after the release | works | works |
| `PopErrorScope` after the release, around a validation error | `Success`, `Validation` error | `CallbackCancelled`, no error |
| Uncaptured error after the release | callback fires | does not fire |
| `mapAsync` and `onSubmittedWorkDone` after the release | `Success` (Dawn's completion polling) | `CallbackCancelled` |
| Device release, device-lost `AllowSpontaneous` | `Destroyed`, inside the call | already cancelled at instance release |
| Device release, device-lost `AllowProcessEvents` | never fires (no instance handle remains to process events) | already cancelled at instance release |

So after the application drops its instance, a yawgpu device keeps working, but it
reports nothing. Its error scopes, uncaptured errors, maps, and device-lost callback
are all silently cancelled.

## Rules

R1. **Instance in use.** An instance is in use while at least one of these has an
external reference: the instance itself, an adapter created from it, or a device
created from one of those adapters. An external reference is one held by the
application through the C API, counted by `AddRef` and `Release`. Internal `Arc`s do not
count, including those held by pending callbacks, queues, buffers, and other child
objects.

R2. **Cancellation happens when the instance stops being in use.** It does not happen
when the instance's own reference count reaches 0. At that point, every pending callback
of the instance is cancelled, as today. From then on, a new registration is cancelled at
once, as today. This keeps the `webgpu.h` rule that dropping the instance cancels its
callbacks. It also covers the case Dawn leaks: an `AllowProcessEvents` callback that can
no longer be processed is cancelled when the last device or adapter is released, and is
never left pending.

R3. **While the instance is in use without an instance reference:**
- `AllowSpontaneous` callbacks behave as they do with an instance reference: they fire
  inside the call when ready at registration (Block 137) or at loss (Block 130).
- `AllowProcessEvents` and `WaitAnyOnly` callbacks are registered and stay pending.
  Nothing can process them until R2 cancels them.
- Uncaptured-error callbacks fire as they do with an instance reference.
- `RequestDevice` on a surviving adapter works, with the same callback rules.

R4. **Order at the last release.** When the last device release also ends the
instance's use, the device's own release work happens first. That is, the implicit
destroy and the spontaneous `Destroyed` device-lost callback (Block 130). Then R2
cancels whatever is still pending, including an `AllowProcessEvents` device-lost
callback, which fires `CallbackCancelled`.

R5. **Unchanged.**
- With an instance reference held, nothing changes.
- An instance with no adapter and no device: its last release cancels at once, as today.
- Device-lost behaviour when the device is released while the instance is held (Block
  130).

## Tests

- Unit (FFI, Noop):
  - Release the instance while a device is alive. Nothing is cancelled. Then:
    - a spontaneous `PopErrorScope` around a validation error reports the error;
    - the uncaptured-error callback fires;
    - a spontaneous `mapAsync` on a buffer with no pending work fires `Success`;
    - an `AllowProcessEvents` callback registered now is not cancelled yet.
  - Then release the device. A spontaneous device-lost callback gets `Destroyed` inside
    the call, and after it every pending `AllowProcessEvents` callback is cancelled, in
    registration order. With an `AllowProcessEvents` device-lost callback instead, it
    gets `CallbackCancelled`.
  - Release the instance while only an adapter is alive: a spontaneous `RequestDevice`
    on that adapter succeeds. Releasing the adapter, and then the device, cancels
    pending callbacks only at the last release.
  - `AddRef` on an adapter or device keeps the instance in use until the matching
    `Release`.
  - Today's behaviour stays pinned: an instance with nothing alive cancels at its last
    release.
  - Update the Block 137 test that pins a `FailedCreation` device-lost callback being
    cancelled when the request callback releases the last instance reference. With
    R1 the adapter is still in use there, so the device-lost callback is delivered.
  - Observations made inside an `extern "C"` callback are recorded in state and asserted
    after the call (Block 130 m4).
- Claude: re-run the A19 probe against yawgpu on Metal. Every row must match Dawn,
  except two: the `mapAsync` and `onSubmittedWorkDone` row (it needs completion polling,
  which yawgpu does not have; Block 137 R6), and the `AllowProcessEvents` device-lost
  row (Dawn never fires it, yawgpu cancels it per R2 and R4).
- Claude: Metal e2e; CTS `webgpu:api,*` on Metal.

## Out of scope

- Spontaneous delivery of callbacks that become ready later (Block 137 R6).
- Child objects (buffers, textures, queues, and others) keeping the instance in use.
  In Dawn they keep their device, and so the instance, alive. In yawgpu the last device
  release already destroys the device (Block 109), so their callbacks resolve as aborted
  or lost at that point.

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests.
- **S2 (Claude):** probe re-run, e2e, CTS, Phase Review.
