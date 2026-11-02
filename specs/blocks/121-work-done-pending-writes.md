# Block 121 — `onSubmittedWorkDone` does not flush pending writes

Status: **COMPLETE (2026-11-02)**. Phase Review: 0 CRITICAL, 0 MAJOR, 6 MINOR; all
resolved, see below. S3 moved to backlog E14.
Source: backlog E13, perf finding P-005 (`specs/tracking/perf-dawn-baseline.md`
Run 7). `queue/write_buffer_then_wait` is 205× Dawn on Metal: yawgpu 118 µs, Dawn
574 ns.
Oracle: Dawn at the pinned `third_party/dawn`: `native/Queue.cpp`
(`APIOnSubmittedWorkDone`, `APIWriteBuffer`, `APIWriteTexture`),
`native/ExecutionQueue.cpp` (`GetScheduledWorkDoneSerial`),
`native/DynamicUploader.cpp` (`MaybeSubmitPendingCommands`),
`native/metal/DeviceMTL.mm` and `native/vulkan/DeviceVk.cpp`
(`CopyFromStagingToBuffer`).

## Problem

Block 113 S10 (`7488716`) made `wgpuQueueOnSubmittedWorkDone` flush the queue's
pending writes, then gate the callback on that flush's submission. Each call with a
pending write therefore costs a GPU round trip. The audit row behind S10 assumed
that ignoring pending writes was a bug.

Dawn ignores them. Its staging copies are recorded with `SubmitMode::Passive`, which
leaves `HasPendingCommands()` false. `GetScheduledWorkDoneSerial()` then returns the
last submitted serial, so the callback does not wait for a write that has not been
submitted. Nothing can observe the difference: every path that reads the written
data, which are submit, `mapAsync`, and present, flushes the pending writes first.

Dawn bounds the pending memory instead. After each `writeBuffer` and
`writeTexture`, `MaybeSubmitPendingCommands` submits the pending commands once the
uploads pending submission reach 16 MiB. yawgpu has no such bound. Once R1 removes
the flush from work-done, a loop of writes and work-done waits with no submit would
grow staging without limit.

## Rules

R1. `wgpuQueueOnSubmittedWorkDone` does not flush pending writes. The callback is
gated on the latest recorded submission index at the time of the call, so it fires
once all previously submitted work has completed. This is Dawn's
`GetLastSubmittedCommandSerial` arm. The validation and lost-device behaviour of the
entry point is unchanged, including Block 113 C9 (`Success` on a lost device).

R2. After a successful `writeBuffer` or `writeTexture`, the queue flushes its pending
writes when their staged bytes reach `16 * 1024 * 1024`, Dawn's
`kPendingMemorySubmitThreshold`. Staged bytes are the bytes of staging buffers
holding writes not yet submitted: the pooled chunks in use plus dedicated buffers
for large writes. In-flight and free chunks do not count. A flush error is reported
like any other flush error. The threshold is a named constant. It is Dawn's value,
but Dawn counts the bytes requested by each upload, and yawgpu counts staging
capacity. A write therefore never stays pending longer than in Dawn, and may be
flushed sooner: 33 KiB writes each take a 64 KiB chunk, so they flush after about
8.3 MiB of data.

R2a. A write whose staging fails after its chunk was added leaves that chunk counted
with no copy, and returns before the threshold check. The next successful write or
flush clears it. This is deliberate: the error path does not flush.

R3. Every other flush point is unchanged: submit, including an empty or rejected
submit; `mapAsync`; `wgpuSurfacePresent` (Block 113 S5, which Dawn also does);
`wait_idle`.

R4. Written data stays visible in queue order. A write followed by a submit reads
the written value in that submit's commands. A write followed by `mapAsync` maps the
written value, whether or not `onSubmittedWorkDone` was called in between.

## Tests

- FFI (R1): replace `audit_s10_work_done_flushes_queued_buffer_writes`. After a
  write, `onSubmittedWorkDone` creates no submission, and its callback completes
  without one. A following `mapAsync` maps the written value (R4).
- Core (R2): writes that stage just under the threshold create no submission, and
  the write that reaches it creates one. Cover a dedicated large write too.
- Core (R2): a flush error at the threshold is returned from the write.
- Core and FFI (R3): the existing flush tests for submit, `mapAsync`, and present
  still pass unchanged.
- Real GPU (R4): Metal and Vulkan e2e that write, wait for work-done, then read the
  value back through a submitted copy and through `mapAsync`.
- Bench: `queue/write_buffer_then_wait` on Metal near Dawn's figure, and no
  regression elsewhere.

## Out of scope

- `pipeline/compute_unique`, which moved from 0.97× to 1.12× in Run 7. It is
  investigated separately, as S3.
- Making staging copies themselves cheaper.

## Slices

- **S1 (coding agent):** R1-R3 with the FFI and core unit tests.
- **S2 (Claude):** real-GPU e2e, CTS on Metal, bench, Phase Review.
- **S3:** moved out of this block to backlog E14, with P-006.

## Verification (2026-11-02)

Run on Apple M2, macOS 27.2.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,302 passed, 0 failed |
| fmt, clippy on default, `metal`, `vulkan`, `metal,tiled`, `vulkan,tiled`, `gles` | clean |
| Metal e2e, `metal,tiled,shader-passthrough` | 150 passed, 0 failed, including the 3 new |
| Vulkan e2e, MoltenVK under the Khronos layer | 138 passed, 1 failed (known `imageView2DOn3DImage`); the 3 new pass |
| webgpu-native-cts, Metal, `webgpu:*` | `fail=2 crash=0`; subcase counts equal to Block 120's |
| Bench `queue/write_buffer_then_wait` | 118,168 ns to 1,238 ns (Dawn 574 ns) |

The bench also moved `queue/write_buffer_4kb` from 590 ns to 1,935 ns (Dawn 649 ns).
That case writes 20,000 times without a submit. Before this block nothing flushed
inside the timed loop, so the copies were paid for by the next case. Now the R2 flush
submits about 4,096 copies inside the loop, as Dawn's does. The cost of that submit is
P-006: the Metal HAL opens one blit encoder per queued buffer copy
(`yawgpu-hal/src/metal/queue.rs`), and Dawn reuses one. It is recorded in
`specs/tracking/perf-dawn-baseline.md` and backlog E14. Every other case moved by less
than 8% from Run 7.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Threshold comment and spec said Dawn parity, but the measure differs | Comment and R2 reworded |
| m2 | The threshold e2e cannot see whether a flush happened | Documented in the test; core unit tests pin R2 |
| m3 | No test of R1 after an earlier submission | `block_121_r1_work_done_gates_on_earlier_submission` |
| m4 | A failed stage leaves its chunk counted | Recorded as R2a |
| m5 | A failed threshold flush leaves `write_texture` init marks committed | Pre-existing at every flush point; recorded here, not changed |
| m6 | Doc sentence merged into the previous paragraph | Fixed |
