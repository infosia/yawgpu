# Block 100 — `wgpuCommandEncoderWriteBuffer` executes

Status: **IMPLEMENTED (2026-09-22)** — core snapshot + in-order staged copy, FFI reads `data`; Noop unit/integration green; real-GPU e2e (`e2e_metal_encoder_write_buffer.rs`, `e2e_vulkan_encoder_write_buffer.rs`: offset write, write→copy ordering, 100 KiB oversized staging) green on M2 Metal + MoltenVK. Backlog item **A2** in
`specs/tracking/backlog.md`; closes the execution-gap-audit deferral
(`specs/tracking/execution-gap-audit.md` "P6.2 validation-only").

## Problem

`wgpuCommandEncoderWriteBuffer(encoder, buffer, offset, data, size)` has
been validation-only since Phase 6: the FFI validates through
`CommandEncoder::write_buffer` and **discards `data`**. On a real backend
the call succeeds silently and the buffer is never written — the worst
failure class the CTS audits identified (validate-but-don't-act).

Dawn (`CommandEncoder::APIWriteBuffer`, `CommandEncoder.cpp`) validates,
copies `size` bytes of `data` into the command allocator, and replays the
write at submit through its upload path. The write is ordered with the
other commands of the command buffer.

## Behaviour contract

### R1 — Encode-time snapshot

On successful validation the encoder records a
`CommandExecution::BufferWrite(BufferWriteCommand { buffer, offset, data: Vec<u8> })`
holding a **copy** of the `size` bytes. The caller may free or modify
`data` as soon as the call returns. `size == 0` records nothing (validated
no-op, matching `Queue::write_buffer`).

Validation is unchanged (`validate_encoder_write_buffer`: error buffer,
`CopyDst` usage, 4-byte aligned offset and size, in range). Destroyed /
mapped checks stay deferred to submit, as they are for the other buffer
commands (`destroyed_buffer: None, mapped_buffer: None` in the
validation table) — the buffer is registered as a referenced buffer so
submit rejects a destroyed one exactly like `copyBufferToBuffer`.

### R2 — FFI reads `data`

`wgpuCommandEncoderWriteBuffer` passes `std::slice::from_raw_parts(data, size)`
to core after the `usize → u64` check. A null `data` with `size > 0`
routes a validation error to the device error sink
(`"command encoder write buffer data must not be null"`) — it does not
dereference and does not panic. Null with `size == 0` is fine.

### R3 — Submit-time execution, in command order

At submit lowering (`queue.rs`, the `CommandExecution → HalCopy` walk)
a `BufferWrite` becomes a `HalCopy::Buffer` from a queue-owned staging
allocation to the destination, emitted **at the command's position** in
the command buffer's copy list. It must not be hoisted into the
pre-submit pending-write flush: a write recorded after a copy that
reads the same buffer must observe the copy's result, and vice versa.

Staging reuses `PendingWriteBatch::stage` (alignment 4). The chunks a
submission consumes must be retired against **that submission's**
`SubmissionIndex` (or a later one), never an earlier one, so a chunk is
not recycled while the GPU may still read it. The implementer chooses
the mechanics (e.g. stage into a local `PendingWriteSubmission` and
`retire` after the HAL submit returns its index); the invariant is what
is tested.

Staging failure (`HalError`) surfaces as a device error from
`Queue::submit`, the same way a failed queue write does
(`device_error_from_staging`).

### R4 — Noop is observable

On the Noop HAL `HalCopy::Buffer` is executed eagerly into host storage,
so an integration test can map the destination and read the bytes back.

### R5 — Resource-usage and lifetime

The destination buffer participates in the same referenced-buffer
tracking as `copyBufferToBuffer`'s destination: destroyed-at-submit →
validation error at submit; the buffer is retained by the command buffer
until submission completes.

## Tests

- **Inline unit (core, Noop):** `CommandEncoder::write_buffer` records
  a `BufferWrite` with the copied bytes; a `size == 0` call records
  nothing; each validation failure still records the first error and
  records nothing.
- **Inline unit (core queue, Noop):** submitting a command buffer with
  `[BufferWrite, BufferCopy]` yields HAL copies in that order with the
  write's `HalCopy::Buffer` first; the write's source is a `copy_src`
  staging buffer; after the submission completes the staged bytes are in
  the destination (Noop host storage).
- **Inline unit (FFI):** null `data` + `size > 0` dispatches the R2
  message and records nothing; null + `size == 0` is silent.
- **Integration (`yawgpu/tests/command_encoder_write_buffer.rs`, Noop):**
  write 16 bytes at offset 4 of a 32-byte `CopyDst | MapRead` buffer,
  submit, map, assert bytes `[4..20)` equal the data and the rest is
  zero; ordering test: `writeBuffer(A)` then `copyBufferToBuffer(A→B)`
  in one encoder → B holds the written bytes.
- **Real-GPU e2e (Claude authors + runs on the M2):**
  `yawgpu/tests/e2e_metal_encoder_write_buffer.rs` and the Vulkan
  analogue — same two scenarios through the C ABI with readback, plus
  a > 64 KiB write (exercises the dedicated oversized staging path).
- **CTS:** `webgpu:api,validation,encoding,cmds,*` and
  `webgpu:api,operation,command_buffer,*` on Metal + MoltenVK unchanged
  (the CTS never calls `writeBuffer` on an encoder — it is a
  Dawn/wgpu-native extension of `webgpu.h`).

## Out of scope

- `wgpuQueueWriteBuffer` (already staged and ordered — Block 96).
- Changing the validation table.
