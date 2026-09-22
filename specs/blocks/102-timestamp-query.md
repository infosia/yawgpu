# Block 102 — `timestamp-query` executes (Metal + Vulkan)

Status: **S1–S3 IMPLEMENTED (2026-09-22)** — S1 `8fe363a` (core + Noop), S2 Metal + S3 Vulkan landed together (see git log); real-GPU e2e 4/4 on M2 Metal and 4/4 on MoltenVK; Khronos validation layer clean on the Vulkan e2e. S4 (CTS re-run + Phase Review) pending. Backlog item **A1** in
`specs/tracking/backlog.md`. Depends on Block 99 (the advertisement
now follows Dawn's device query; on Metal the cached
`metal_device_supports_counter_sampling` decides the sampling mode
below).

## Problem

`timestamp-query` (`WGPUFeatureName_TimestampQuery`) is advertised on
Metal and Vulkan and fully validated, but nothing executes:

- `HalQueryKind` has only `Occlusion`; `Device::create_query_set` gives a
  `Timestamp` set **no HAL object** (`hal: None`,
  `yawgpu-core/src/device.rs`).
- `CommandEncoder::write_timestamp` validates and records nothing.
- Render-pass `timestampWrites` are validated and the set is registered
  as referenced, but no write is recorded at all. Compute-pass
  `timestampWrites` are turned by the **FFI** into two
  `write_timestamp` calls **both at pass begin** (so the "end" stamp,
  once it executes, would precede the dispatches).
- `resolveQuerySet` on a timestamp set silently skips the HAL copy
  (`queue.rs` lowering `let Some(query_set) = resolve.query_set.hal() else { return }`),
  leaving the destination untouched.

The CTS cannot see this (timestamp values are non-deterministic), so the
sweeps stay green while every timestamp client reads garbage.

## WebGPU contract being implemented

- A timestamp query set holds `count` 64-bit slots.
- `writeTimestamp` / pass `timestampWrites` record the GPU timestamp
  **in nanoseconds** into slot `index` at that point of the queue
  timeline.
- `resolveQuerySet(set, first, count, dst, dstOffset)` writes `count`
  `u64` values at `dstOffset` (256-aligned); slots never written resolve
  to `0`.

## Design (Dawn's shape, one shader for every backend)

Backends deliver **raw ticks**; a device-level compute pass converts
the resolved range to nanoseconds in place, exactly as Dawn's
`EncodeConvertTimestampsToNanoseconds` (`QueryHelper.cpp`). The
conversion shader is WGSL compiled through the normal Tint path, so
Metal, Vulkan (and any future backend) share it.

### R1 — HAL surface

- `HalQueryKind::Timestamp` (new variant; `#[non_exhaustive]` stays).
- `HalAdapter::timestamp_period() -> f32` — nanoseconds per tick:
  - Vulkan: `limits.timestampPeriod`.
  - Metal: calibrated once at adapter construction from
    `sampleTimestamps:gpuTimestamp:`: sample `(cpu0, gpu0)`, sleep
    ≥ 2 ms, sample `(cpu1, gpu1)`, `period = (cpu1 − cpu0) as f64 / (gpu1 − gpu0) as f64`
    (`cpuTimestamp` is in nanoseconds — Dawn compares it against
    `NSEC_PER_SEC / 10`). If `gpu1 ≤ gpu0` or `cpu1 ≤ cpu0` (counter
    reset / clamp) fall back to `1.0`. Only computed when
    `supports_timestamp_query()` is true (the API keeps Intel GPUs
    busy — Dawn's `MetalDisableTimestampPeriodEstimation` note).
  - Noop: `1.0`. GLES: `1.0` (never advertised).
- `HalCopy::WriteTimestamp(HalWriteTimestamp { query_set, query_index })`
  — a top-level command, executed in order with the other copies.
- `HalResolveQuerySet` is reused for timestamp sets; `written_queries`
  carries the indices written **in this command buffer** (see R4).

### R2 — Metal HAL

- `MetalQuerySet` becomes an enum-like struct: occlusion keeps the
  visibility `MTLBuffer`; timestamp holds an `MTLCounterSampleBuffer`
  created from `MTLCounterSampleBufferDescriptor` with the device's
  `timestamp` counter set (found by the same case-insensitive name
  match as Block 99), `sampleCount = max(count, 1)`,
  `storageMode = Private`. Creation failure → `HalError`
  (`OutOfMemory`-class when Metal reports an error).
- `MetalDevice` caches `counter_sampling_at_stage_boundary` and
  `counter_sampling_at_command_boundary` (Dawn's two predicates, from
  `MTLDevice.supportsCounterSampling`) plus a 1-byte private "mock
  blit" buffer (Dawn `GetMockBlitMtlBuffer`).
- `WriteTimestamp` encode (Dawn `CommandBufferMTL.mm` `Command::WriteTimestamp`
  at top level):
  - stage-boundary devices (all Apple GPUs): end any open encoder, then
    open a blit encoder from an `MTLBlitPassDescriptor` whose
    `sampleBufferAttachments[0]` has `sampleBuffer = set`,
    **`startOfEncoderSampleIndex = index`,
    `endOfEncoderSampleIndex = MTLCounterDontSample`**; encode
    `fillBuffer(mock, {0,1}, 0)` (Dawn's
    `MetalUseMockBlitEncoderForWriteTimestamp`, default-on: Metal drops
    the sample on an empty encoder); end the encoder.
    *Deviation from Dawn (measured 2026-09-22, M2 / Apple8, macOS 26):*
    Dawn samples `endOfEncoderSampleIndex`. On this device a blit pass
    that requests an end-of-encoder sample **directly after another blit
    encoder** (any buffer copy or clear, even 256 bytes) fails the whole
    command buffer with `kIOGPUCommandBufferCallbackErrorOutOfMemory`,
    while the same descriptor sampling at the start of the encoder works
    in every ordering (`[W0 C W1 R]`, `[C W0 W1 R]`, `[W0 P W1 R]`, …).
    Because the encoder is a 1-byte mock fill, start and end are the same
    instant for the caller.
  - **Resolve serialization (Dawn `MetalSerializeTimestampGenerationAndResolution`,
    crbug.com/372698905), carried on Apple8+:** before a timestamp-set
    resolve the queue encodes `encodeSignalEvent:value:` +
    `encodeWaitForEvent:value:` on a device-owned `MTLSharedEvent`
    (monotonic value). Without it a stamp written after a compute or
    render pass resolves to `0` on the M2 (measured 2026-09-22: the e2e
    `end must follow begin: [t, 0]`). Both deviations together are what
    make every ordering pass; each alone leaves a failing case.
  - otherwise (command boundary): on a blit encoder,
    `sampleCountersInBuffer:atSampleIndex:withBarrier:YES`.
- `ResolveQuerySet` on a timestamp set (blit encoder): zero-fill
  `[dstOffset, dstOffset + count*8)`, then for each `written_queries`
  index `i` (within range, validated like the occlusion path)
  `resolveCounters:inRange:(i,1) destinationBuffer:dst destinationOffset:dstOffset + (i − first)*8`
  — `MTLCounterResultTimestamp` is one `u64`, so the layout matches.
- No change to occlusion behaviour.

### R3 — Vulkan HAL

- `VulkanQuerySet::new(kind)` picks `vk::QueryType::TIMESTAMP` for
  timestamp sets (pool count `max(count, 1)`).
- `WriteTimestamp`: `cmd_reset_query_pool(pool, index, 1)` then
  `cmd_write_timestamp(PipelineStageFlags::ALL_COMMANDS, pool, index)`
  (Dawn `RecordWriteTimestampCmd` at top level; top-level only, so the
  reset needs no render-pass hoisting).
- `ResolveQuerySet` on a timestamp pool reuses the occlusion resolve
  (fill, barrier, `cmd_copy_query_pool_results` with `TYPE_64 | WAIT`
  per written index).
- Buffers created with `query_resolve` usage also get
  `vk::BufferUsageFlags::STORAGE_BUFFER` (Dawn's internal storage usage
  for `QueryResolve`), so the conversion pass can bind the destination.
- `encode_compute_pass` keeps one global memory barrier before and one
  after the dispatch, widened to the **union** of the F-106 scopes it
  already had (`transfer_to_compute_barrier` / `compute_to_transfer_barrier`
  from `c723a82`: indirect / index / vertex-input / vertex / fragment) and
  the scopes this block needs: before — `src TRANSFER | COMPUTE_SHADER`,
  `MEMORY_WRITE → SHADER_READ | SHADER_WRITE` (+ F-106 dst); after —
  `src COMPUTE_SHADER`, `SHADER_WRITE → MEMORY_READ | MEMORY_WRITE`,
  `dst TRANSFER | COMPUTE_SHADER | HOST` (+ F-106 dst). The resolve
  additionally ends with a buffer barrier `TRANSFER_WRITE → SHADER_READ |
  SHADER_WRITE` on the destination range so the conversion pass observes
  the copied results. *(The earlier draft of this rule said the HAL had no
  compute-pass barrier at all; that was wrong — F-106 added one.)*

### R4 — Core

- `Device::create_query_set` creates a HAL object for `Timestamp` sets
  (`HalQueryKind::Timestamp`), error → error query set + message, like
  occlusion.
- `CommandExecution::WriteTimestamp(WriteTimestampCommand { query_set, query_index })`
  recorded by `CommandEncoder::write_timestamp` after the existing
  validation.
- Pass `timestampWrites` are recorded **by core**, not the FFI:
  `begin_compute_pass(timestamp_writes: Option<RenderPassTimestampWrites>)`
  and `begin_render_pass(descriptor)` record
  `WriteTimestamp(beginning_index)` immediately (before the pass's
  commands) and stash `end_index` in `PassEncoderInit`; `end()` records
  `WriteTimestamp(end_index)` **after** the pass's own command. The FFI
  compute-pass begin stops calling `write_timestamp` itself and passes
  the mapped writes to core (device-ownership check stays in the FFI).
- Submit lowering: `WriteTimestamp` → `HalCopy::WriteTimestamp` (error
  set / destroyed set already rejected at submit by the existing
  referenced-query-set checks). `ResolveQuerySet` on a timestamp set →
  `HalCopy::ResolveQuerySet` with `written_queries` = indices of every
  `WriteTimestamp` on that set earlier in the same command buffer
  (mirror of `resolve_written_occlusion_queries`), **followed by** the
  conversion pass (R5). A timestamp set with no HAL object (error set)
  keeps the current skip.
- The written-in-an-earlier-submission limitation is the same one the
  occlusion path has today; record it in the block's "Known
  limitations" (not fixed here).

### R5 — Conversion pass (core, Tint-compiled)

- WGSL, adapted from Dawn's `sConvertTimestampsToNanoseconds`: the
  timestamps buffer is `@group(0) @binding(0) var<storage, read_write>`
  declared as a **flat `array<u32>`** (low word at `2*i`, high word at
  `2*i + 1`) rather than Dawn's `array<Timestamp>` of structs — loading a
  whole struct out of a storage buffer becomes a `device`-to-local struct
  copy in MSL that SPIRV-Cross (MoltenVK) rejects for Tint's
  robustness-clamped SPIR-V (`no matching constructor for initialization
  of 'Timestamp'`, found 2026-09-22); scalar loads translate everywhere —
  and the parameters come from **immediates**
  (`var<immediate> params: TimestampParams` — `count`, `multiplier`,
  `right_shift`; 12 bytes) instead of a uniform buffer. No quantization
  mask (yawgpu does not fingerprint-quantize; equivalent to Dawn with
  `timestamp_quantization` off). The 16-bit-chunk multiply/shift is
  kept verbatim.
- `multiplier` / `right_shift` derive from the period exactly as
  Dawn's `TimestampParams` constructor:
  `upper = ceil(log2(period))`, `right_shift = 16 − min(upper, 16)`,
  `multiplier = (period · 2^right_shift) as u32`. Pure function,
  unit-tested (period 1.0 → (65536, 16); 41.666… → (43690, 10);
  83.333 → (21333, 8)).
- The pipeline is created lazily, once per `Device`, on the first
  timestamp resolve (`OnceLock` on `DeviceInner`), through the ordinary
  `create_shader_module` + `create_compute_pipeline` path with an
  explicit pipeline layout (`immediate_size = 12`, one storage-buffer
  BGL entry). It is never visible to the C API.
- Per resolve, core builds an **internal bind group** binding the
  destination at `(destinationOffset, count*8)`. Internal creation
  skips only the `STORAGE` usage check (Dawn
  `UsageValidationMode::Internal`); every other bind-group rule runs.
  `destinationOffset` is 256-aligned by validation, so the storage
  offset alignment holds.
- Recorded as an ordinary `ComputePassCommand` (pipeline, bind group 0,
  `Direct((count + 7) / 8, 1, 1)`, immediates = params bytes) directly
  after the resolve in the command buffer, so the existing lowering,
  destroyed-resource checks and retention apply unchanged.
- On Noop the compute pass is a no-op and the resolve zero-fills, so
  the observable Noop result stays all-zero; Noop tests assert the
  lowering shape instead.

### R6 — Unchanged

Validation (`validate_timestamp_query_set`, index rules, pass
timestampWrites rules, `resolveQuerySet` rules), `wgpuQuerySet*`
accessors, occlusion queries, GLES (`timestamp-query` stays
unadvertised; a `Timestamp` HAL kind on GLES returns `HalError`).

## Tests

- **Core inline (Noop):** `write_timestamp` records a `WriteTimestamp`;
  compute/render pass `timestampWrites` record beginning before and end
  after the pass command; `resolve_query_set` on a timestamp set lowers
  to `[ResolveQuerySet{written_queries = written indices}, ComputePass{dispatch = ceil(count/8), immediates = params}]`;
  a never-written range lowers with empty `written_queries`; the
  params derivation table; `create_query_set(Timestamp)` yields a HAL
  object on Noop.
- **HAL inline:** Metal — `#[ignore]` real-device: create a timestamp
  set, write two stamps around a blit, resolve into a shared buffer,
  wait, assert `t1 > t0 > 0` and `t1 − t0 < 1 s` in ticks·period;
  `timestamp_period()` is finite and `> 0`. Vulkan — the same on
  MoltenVK; `timestamp_period()` equals `limits.timestampPeriod`;
  `query_resolve` buffers carry `STORAGE_BUFFER`. Pure: the Metal
  sampling-mode selection as a function of the two cached predicates.
- **FFI inline:** `wgpuCommandEncoderBeginComputePass` with
  `timestampWrites` records begin/end in the right order (device-mismatch
  path still errors).
- **Integration (Noop, `yawgpu/tests/timestamp_query.rs`):** the full C
  path: request `timestamp-query`, create set, `WriteTimestamp`, compute
  pass with `timestampWrites`, render pass with `timestampWrites`,
  `ResolveQuerySet`, submit, map — no errors, buffer readable (zeros on
  Noop).
- **Real-GPU e2e (Claude authors + runs):**
  `yawgpu/tests/e2e_metal_timestamp_query.rs`,
  `yawgpu/tests/e2e_vulkan_timestamp_query.rs`: (1) `writeTimestamp`
  before/after a 256×256 compute dispatch, resolve, map: both values
  `> 0`, `end > begin`, `end − begin < 1 s` **in nanoseconds** (proves
  the conversion ran: on MoltenVK `timestampPeriod ≠ 1`); (2) compute
  and render pass `timestampWrites` with only `endOfPassWriteIndex`
  set: the other slot resolves to 0; (3) resolve of a never-written
  range is all zeros; (4) two sets resolved into one buffer at
  different 256-aligned offsets don't clobber each other.
- **CTS:** `webgpu:api,validation,queries,*`,
  `webgpu:api,operation,queries,*` (if present in the pinned suite —
  otherwise `webgpu:api,validation,encoding,cmds,*` and
  `webgpu:api,validation,capability_checks,features,query_types:*`) on
  Metal + MoltenVK: no regression versus the current Run.

## Slices

- **S1 core + Noop** (R1 kinds/period plumbing on HAL enums, R4, R5 with
  the Noop pipeline) — Noop green, integration test green.
- **S2 Metal HAL** (R2 + Metal `timestamp_period`) — Claude runs the
  inline real-device test + e2e on the M2.
- **S3 Vulkan HAL** (R3 + Vulkan `timestamp_period`) — Claude runs on
  MoltenVK.
- **S4** CTS re-run + Phase Review of the cumulative A4/A2/B1/A1 diff.

## Known limitations (documented, not fixed here)

- A query written in an earlier submission and resolved in a later one
  resolves to 0 (same as occlusion today: availability is tracked per
  command buffer, not on the query set).
- Metal timestamp period is a single calibration at adapter creation,
  not Dawn's running Kalman estimate.
- `ChromiumExperimentalTimestampQueryInsidePasses` (in-pass
  `writeTimestamp`) is Dawn-only and out of scope.
