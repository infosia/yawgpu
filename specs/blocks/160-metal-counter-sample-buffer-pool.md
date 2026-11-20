# Block 160 — Metal timestamp query sets suballocate pooled counter sample buffers

Status: **IN PROGRESS (2026-11-20)**.
Source: webgpu-native-cts finding **F-170**
(`webgpu:api,operation,command_buffer,queries,timestampQuery:many_query_sets:numQuerySets={64,256,65536}`
fails in both stages on Metal with `HAL out of memory: metal: timestamp counter sample
buffer` for every set past the 32nd).

## Problem

Metal allows at most 32 live `MTLCounterSampleBuffer`s per process, and one sample
buffer holds at most 32 KiB, i.e. 4096 timestamp samples. `MetalQuerySet::new`
(`yawgpu-hal/src/metal/query_set.rs`) creates one sample buffer per timestamp query set,
so the 33rd live set fails. Upstream CTS `many_query_sets` exists so implementations
suballocate query sets from larger shared buffers. Oracle:
`third_party/dawn/src/dawn/native/metal/QuerySetMTL.mm`
(`CounterSampleBufferAllocator`).

## Rules

R1. **Device-owned pool.** Each `MetalDevice` owns one counter-sample pool, shared by
all its timestamp query sets and safe under concurrent `create_query_set` and query-set
drop. A pool buffer is an `MTLCounterSampleBuffer` with the device's timestamp counter
set, `MTLStorageModePrivate`, and `max(request, 4096)` samples. Occlusion query sets are
unchanged.

R2. **Allocation.** A timestamp set of `count` queries reserves `max(count, 1)`
contiguous slots (count 0 keeps a valid sample buffer, as today). First fit: scan
existing pool buffers in creation order for the first free run of that length; if none
fits, create a new pool buffer and take its first slots. A failed pool-buffer creation
is `HalError::OutOfMemory` (same `resource` string as today); it never panics.

R3. **Release.** The reservation is held by a shared allocation object that every clone
of the `MetalQuerySet` (including the clones retained by in-flight submissions) points
to. When the last reference drops, its slots are returned; a pool buffer whose occupied
count reaches 0 is released. Returning slots must work even if the device's pool was
already dropped (hold the pool weakly, or otherwise avoid a reference cycle and a
use-after-free). A poisoned lock while returning slots must not panic in `Drop`.

R4. **Sample index = base + query index.** Every place that addresses the sample buffer
by query index adds the allocation's base index:
- `HalCopy::WriteTimestamp` in `metal/queue.rs`, both the stage-boundary
  (`setStartOfEncoderSampleIndex`) and command-boundary
  (`sampleCountersInBuffer:atSampleIndex:withBarrier:`) modes;
- `encode_resolve_query_set` in `metal/encode.rs` (`resolveCounters:inRange:...`).

Range checks keep using the set's own `count`; the base offset is applied after the
check. Expose this through a `MetalQuerySet` method (e.g. `sample_index(query_index)`)
rather than repeating the addition.

R5. **Unchanged.** Resolve serialization (shared event), the timestamp conversion pass,
sampling-mode selection, and `HalQuerySet` public shape are unchanged. Destroying a query
set does not return slots while a submission still retains its HAL object (R3 covers
this by reference counting).

## Slices

S1: R1–R5 in `yawgpu-hal/src/metal/` (`query_set.rs`, `device.rs`, `queue.rs`,
`encode.rs`, and `mod.rs` if a new module is added). Tests:
- the slot allocator as a pure, GPU-free type with inline tests that run in the normal
  (non-`#[ignore]`) test run when the `metal` feature is enabled: first fit, reuse of a
  freed run, a request that does not fit an existing buffer creates a new one, a
  buffer reaching zero occupancy is removed, count 0 reserves one slot, a request of
  4096 fills exactly one buffer, a request larger than any free run in a partially used
  buffer goes to a new buffer;
- `#[ignore = "manual real Metal backend test"]` tests on a real device: 64 live
  timestamp sets of 2 queries all allocate (more than 32), sets share a sample buffer
  with distinct base indices, dropping a set lets a new set reuse its slots, and a
  write-timestamp + resolve through a non-zero base index produces non-zero values in
  the right destination slots (existing queue-level test helpers in `metal/queue.rs`
  show how to submit and read back).

## Verification

- Agent gates (redirect output to a file; report exit codes):
  `cargo test -p yawgpu-hal --features metal --lib`,
  `cargo clippy --workspace --all-targets --features yawgpu-hal/metal -- -D warnings`
  (or the workspace's usual Metal clippy invocation), plus the default
  `cargo clippy --workspace --all-targets -- -D warnings`.
- Claude, on review: the `#[ignore]` Metal tests on the M2; `e2e_metal_*` timestamp
  targets; CTS `webgpu:api,operation,command_buffer,queries,timestampQuery:*` on Metal
  (`many_query_sets` compute must pass for every `numQuerySets`; the 65536 render case
  is expected to match Dawn's F-169 device loss and may need an `xfail`), then check
  whether the exit-time abort reported in F-170 is gone.

## Verification record

S1 (2026-11-20, M2): `cargo test --workspace` 1500/0; fmt and clippy clean;
`yawgpu-hal --features metal` ignored real-device query/timestamp tests 11/11;
`e2e_metal_timestamp_query` 6/6, including 80 live sets with alternating written slots
and re-allocation after release. CTS `timestampQuery:many_query_sets` passes for
`numQuerySets` 8 to 256 in both stages.

`numQuerySets=65536` still fails, for a reason other than the sample-buffer limit:
- `compute`: the command buffer fails (device lost at the queue drain). A probe of
  empty compute passes with begin/end `timestampWrites` succeeds at 49,152 sets and
  fails at 63,488 (31 pool buffers, within the limit). Each pass currently costs three
  encoders (a mock blit encoder per stamp plus the pass); Dawn samples pass
  `timestampWrites` through the pass descriptor's `sampleBufferAttachments`, one encoder
  per pass, and passes this case.
- `render`: the Metal driver aborts the process
  (`Failed to allocate IOGPUDeviceShmem` assertion in `IOGPUMetalDeviceShmem`); Dawn
  loses the device instead (F-169).

Both are tracked as a follow-up block (pass `timestampWrites` through Metal pass
descriptors). The exit-time Rust panic reported with F-170 did not reproduce.

## Known limits

- First-fit over 4096-sample buffers never compacts: a pattern of releases that leaves
  only holes smaller than later requests creates a new sample buffer per request and can
  reach the 32-buffer limit with most slots free. CTS shapes do not trigger it.
- The 32-buffer limit is per process while the pool is per device, so several devices
  share it.
