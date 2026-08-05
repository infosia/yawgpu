# Block 96 — Queue submission performance

Closes findings **P-001** and **P-002** from `specs/tracking/perf-dawn-baseline.md`.
The baseline measurement showed object construction, validation and shader
compilation competitive with Dawn or ahead of it, while the queue/submission
layer ran ~10× behind end-to-end. Both causes are in this block.

The whole block is a **performance** change. No observable WebGPU behaviour may
change: the queue-timeline ordering guarantees, the device-error surface, and
every CTS result must be identical before and after. A rule below that is only
about speed is still a hard requirement, because the slow behaviour it replaces
is what the existing tests were written against.

## Slice A — deferred queue writes (P-002)

`Queue::write_buffer` and `Queue::write_texture` each allocate a fresh HAL
staging buffer and immediately call `submit_copies`, which blocks on GPU
completion. Cost is ~156 µs per call regardless of size.

Staging itself is load-bearing and must stay. The doc comment on
`Queue::write_buffer` records CTS finding **F-074**: writing host data directly
into the destination races with still-executing prior submits on Vulkan. The fix
is to stop *submitting* per write, not to stop staging.

### Design

The queue owns a **pending-write batch**: a bump-allocated staging chunk plus
the `HalCopy`s that read from it. A write copies host bytes into the chunk and
records a copy; nothing is submitted. The batch is flushed by prepending its
copies to the next submission.

- **Q1** `Queue` owns pending-write state guarded by a single lock. It holds the
  recorded `HalCopy`s in issue order, the staging chunks they borrow from, and a
  free list of chunks available for reuse.
- **Q2** A write sub-allocates from the current chunk at an offset aligned to at
  least 4 bytes, growing a new chunk when the current one cannot fit the write.
  A write larger than the chunk size gets a dedicated exact-size buffer. Chunks
  are created with `copy_src` usage only, exactly as the current per-call
  staging buffer is.
- **Q3** Flushing prepends the pending copies, **in issue order**, ahead of the
  copies of the submission that triggers the flush. This is what preserves the
  queue-timeline guarantee that a write is ordered before later-submitted work,
  and it is the entire correctness argument for the slice.
- **Q4** A write issued *after* a submit must not be pulled into that submit. It
  joins the next batch.
- **Q5** After a flush completes, its chunks return to the free list for reuse.
  Chunk reuse must not outlive the GPU's use of the chunk — safe in this slice
  because submission is still synchronous; Slice B moves recycling behind the
  completion signal (**Q17**).
- **Q6** Zero-length writes still validate and then record nothing, matching
  today's early return.

### Flush points

- **Q7** The batch is flushed by every operation that would observe the write
  under the current synchronous implementation. At minimum:
  `Queue::submit` (including the empty-command-buffer path), `Queue::wait_idle`,
  buffer map resolution, and device destruction/drop. The implementer must
  enumerate the call sites of `submit_copies` / `wait_idle` and confirm each
  either flushes or provably cannot observe a pending write.
- **Q8** **Buffer mapping is the sharp edge.** `Buffer::resolve_pending_map`
  reads back from the HAL buffer. A pending write that has not been copied into
  that HAL buffer yet would be invisible, and `mapAsync` would hand the caller
  stale bytes. This is a silent data bug, not an error — it must be covered by a
  test that writes, maps, and asserts the written bytes are read back, with no
  intervening submit.
- **Q9** A destination buffer or texture destroyed between the write and the
  flush must not cause a use-after-free: a recorded copy keeps its destination
  alive. Already-enqueued work completing after `destroy()` matches Dawn.

## Slice B — asynchronous submission (P-001)

`MetalQueue::submit_copies` ends `commit()` + `waitUntilCompleted()`
(`yawgpu-hal/src/metal/queue.rs:162`). Every submit stalls the caller for a GPU
round trip, so no CPU/GPU overlap is possible through yawgpu on Metal.

### The backends already disagree, and one of them has a live bug

**Vulkan does not block.** `submit_copies` creates a fence, submits, and pushes
the submission into a `RetireRing` that waits only when reusing a slot
(`yawgpu-hal/src/vulkan/encode.rs:141-186`, `vulkan/surface.rs:117`), retaining
referenced resources via `collect_retained_resources`. Only `submit_empty` waits.
So the model Slice B needs already exists on Vulkan — Metal is the outlier, and
matching Vulkan's shape is the cheapest correct route.

That asymmetry currently hides a defect. Callback resolution is not uniform:

- `PendingCallback::BufferMap` calls
  `resolve_pending_map_with_gpu_completion(|| device.wait_idle())`
  (`yawgpu/src/ffi/mod.rs:1616-1628`), so a map drains the queue before reading
  back. Coarse, but correct on both backends.
- `PendingCallback::QueueWorkDone` waits for **nothing**
  (`yawgpu/src/ffi/mod.rs:1640`) and is registered with `status: Success`.

On Metal that is masked by the blocking submit. **On Vulkan it is not**:
`wgpuQueueOnSubmittedWorkDone` can already fire before the GPU has finished.
Q14 is therefore a correctness fix that happens to be needed for Slice B, not a
new constraint Slice B introduces — and it must not be deferred, because making
Metal asynchronous removes the accident that currently hides it there too.

### Why this is not a one-line removal

Every asynchronous future in yawgpu completes at registration:
`WGPUInstanceImpl::register_callback` (`yawgpu/src/ffi/mod.rs:560`) calls
`register_pending_callback` then `complete_future` immediately, and
`wgpuQueueOnSubmittedWorkDone` registers with `status: Success` before any work
has run. That is correct **only** because submit blocks — by the time events are
processed the GPU has genuinely finished. Deleting the wait without adding
completion tracking turns every `onSubmittedWorkDone` and `mapAsync` into a lie
that returns stale data.

### Design

- **Q10** Submission returns a monotonically increasing **submission index**.
  The HAL exposes the highest index known to have completed, without blocking,
  and a blocking wait for a given index.
- **Q11** Metal signals completion through the command buffer's completion
  handler; Vulkan through a per-submission fence. Noop completes immediately.
  The index must advance monotonically and never regress.
- **Q12** HAL resources referenced by an in-flight submission stay alive until
  that submission completes. With a synchronous submit nothing could outlive the
  call; that guarantee is now gone and must be replaced by explicit retention.
- **Q13** `Queue` records the index of its most recent submission.
- **Q14** `wgpuQueueOnSubmittedWorkDone` captures the queue's current submission
  index at registration and its future completes only once the HAL reports that
  index complete. Registering with no work submitted completes immediately.
- **Q15** `mapAsync` is gated the same way: the map may not resolve before the
  submissions preceding it have completed. Resolution still happens on
  `ProcessEvents`/`WaitAny`, which now poll the HAL for completion first.
- **Q16** `wgpuInstanceWaitAny` with a timeout must make progress by polling
  completion, not by assuming futures are already complete.
- **Q17** Slice A's chunk recycling (**Q5**) moves behind the completion signal:
  a chunk returns to the free list only once the submission that consumed it has
  completed.
- **Q18** Device destruction, queue drop and `wgpuDeviceDestroy` wait for
  outstanding submissions before releasing resources.

### Behaviour that must not change

- **Q19** `wait_idle` still blocks until everything submitted has finished.
- **Q20** A device error raised by submission is still reported through the
  device error sink, at submit time, not deferred to completion.
- **Q21** Callback ordering guarantees are unchanged: a future registered after
  another for the same queue may not fire before it.

## Verification

- **Q22** Every new public fn carries an inline `#[cfg(test)]` unit test
  (CLAUDE.md principle 1), and the whole suite passes on Noop with no GPU
  (principle 2).
- **Q23** `benches/` is re-run on Metal before and after each slice and the
  numbers appended as a new "Run" section in
  `specs/tracking/perf-dawn-baseline.md`. Slice A is expected to bring
  `frame/10writes_dispatch_submit_wait` to roughly parity with Dawn; Slice B is
  expected to move `submit/compute_1_dispatch` off the ~156 µs floor and to
  decouple it from `submit/compute_wait_idle`, which is the signature that the
  blocking wait is gone.
- **Q24** webgpu-native-cts is re-run on **both** Tier 1 backends (Metal and
  Vulkan/MoltenVK) against the Dawn oracle. A performance change that moves a
  CTS result is a regression, not a trade-off: this block may not change a
  single pass/fail count. Slice B in particular can only be declared done on
  hardware — a completion-tracking bug shows up as flaky readback, which Noop
  cannot detect.
