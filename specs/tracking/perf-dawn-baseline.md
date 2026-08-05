# Performance baseline vs Dawn

First CPU-overhead measurement of yawgpu against Dawn across the identical
`webgpu.h` C ABI. Harness, method and build instructions: `benches/README.md`.

Until now every comparison against Dawn was a **conformance** comparison
(pass/fail counts in webgpu-native-cts). This is the first time the *cost* of
the same call has been measured on both sides.

## Run 1 — 2026-08-05

| | |
|---|---|
| Machine | Apple M2, macOS 26.5.2 (25F84) |
| Backend | Metal (both) |
| yawgpu | `0e12cd5`, `cargo build -p yawgpu --release --features metal` |
| Dawn | `out/Release`, `libwebgpu_dawn.dylib`, default toggles (validation on) |
| Harness | `benches/`, `--reps 7`, min-of-batches |

`ns/op` is the minimum per-operation time across 7 batches. `ratio` > 1 means
yawgpu is slower.

| case | iters | yawgpu ns | dawn ns | ratio |
|---|---|---|---|---|
| `buffer/create_destroy` | 20000 | 1756 | 1929 | 0.91× |
| `buffer/create_mapped_unmap` | 10000 | 1781 | 4428 | 0.40× |
| `bindgroup/create_destroy` | 20000 | 163 | 133 | 1.23× |
| `queue/write_buffer_4kb` | 20000 | 156238 | 669 | 233.50× |
| `queue/write_buffer_then_wait` | 500 | 147249 | 592 | 248.94× |
| `frame/10writes_dispatch_submit_wait` | 300 | 1604314 | 159279 | 10.07× |
| `shader/create_cached` | 5000 | 132 | 657 | 0.20× |
| `shader/create_unique` | 200 | 16091 | 21154 | 0.76× |
| `pipeline/compute_cached` | 5000 | 66 | 364 | 0.18× |
| `pipeline/compute_unique` | 100 | 84787 | 87555 | 0.97× |
| `pipeline/render_unique` | 60 | 118858 | 135576 | 0.88× |
| `encode/empty_encoder_finish` | 20000 | 159 | 177 | 0.90× |
| `encode/compute_1_dispatch` | 20000 | 727 | 668 | 1.09× |
| `encode/render_pass_empty` | 20000 | 875 | 928 | 0.94× |
| `encode/render_draw` | 200000 | 148 | 42 | 3.49× |
| `submit/empty` | 10000 | 193 | 3123 | 0.06× |
| `submit/compute_1_dispatch` | 5000 | 155875 | 25885 | 6.02× |
| `submit/compute_wait_idle` | 500 | 149310 | 134873 | 1.11× |

### Cases that are not like-for-like

Two rows must not be quoted as wins:

- **`submit/empty` (0.06×)** — `MetalQueue::submit_copies` returns `Ok(())`
  immediately for an empty slice (`yawgpu-hal/src/metal/queue.rs:47`), so yawgpu
  is not doing the work Dawn does, it is skipping it.
- **`queue/write_buffer_4kb` (233×)** — Dawn may defer the upload into the next
  submit, so part of its cost falls outside the timed region. This is precisely
  why `frame/10writes_dispatch_submit_wait` exists; that case drains the queue
  and is the defensible end-to-end figure.

`queue/write_buffer_then_wait` is weaker evidence than it looks: Dawn's 592 ns
suggests `onSubmittedWorkDone` returns without waiting when no submit is
pending, i.e. it did not force the deferred upload through. Treat the `frame/*`
row as the authoritative end-to-end comparison.

## Findings

### P-001 — every `queue.submit` blocks until GPU completion (6× on submit, 10× end-to-end)

`MetalQueue::submit_copies` ends with `command_buffer.commit()` followed by
`command_buffer.waitUntilCompleted()` (`yawgpu-hal/src/metal/queue.rs:162`).
Every `wgpuQueueSubmit` with recorded work therefore stalls the calling thread
for a full GPU round trip.

The evidence is internally consistent: `submit/compute_1_dispatch` (156 µs) and
`submit/compute_wait_idle` (149 µs) are the *same* figure — adding an explicit
drain costs nothing because the submit already drained. On Dawn the two differ
by 5× (26 µs vs 135 µs), which is what an asynchronous submit looks like.

Consequence: no CPU/GPU overlap and no pipelining is possible through yawgpu.
An application cannot record frame N+1 while frame N executes, regardless of how
it is written.

### P-002 — `queue.writeBuffer` allocates a staging buffer and round-trips per call

`Queue::write_buffer` (`yawgpu-core/src/queue.rs:159-201`) allocates a fresh
`copy_src` HAL buffer for every call, writes into it, and issues
`submit_copies` — which then blocks per P-001. Cost is ~156 µs per call
irrespective of size.

`frame/10writes_dispatch_submit_wait` decomposes exactly: 10 × ~150 µs +
~150 µs submit ≈ 1.60 ms measured. The entire 10× end-to-end gap is this one
call plus P-001.

The staging-per-call design is deliberate and load-bearing for correctness — the
doc comment records CTS finding F-074, a Vulkan race where a direct host write
into the destination is observed by still-executing prior submits. A fix must
preserve queue-timeline ordering; the target is a staging ring buffer whose copy
is folded into the next submit, not removal of staging.

### P-003 — each draw snapshots the whole render-pass state (3.5× per draw)

`RenderPass::draw` (`yawgpu-core/src/render_pass.rs:279-315`) pushes one
`RenderPassCommand` per draw carrying a full copy of pass state:
`bind_group_layouts().to_vec()`, plus clones of `attachment_texture_uses`,
`attachment_textures`, `bind_groups`, `vertex_buffers`, `index_buffer`,
`occlusion_query_set` and `immediate_data`, and reloads the attachment list.
That is ~7 heap allocations and a batch of `Arc` traffic per draw, against
Dawn's flat command stream with delta state-setting: 148 ns vs 42 ns.

This is the same structural choice behind the render-pass-per-draw `storeOp`
forcing already noted in the CTS work — the cost is architectural, not a hot
spot to micro-optimise.

### Where yawgpu is ahead

- `shader/create_cached` 0.20× and `pipeline/compute_cached` 0.18× — the
  dedup/compile cache (Block 95) beats Dawn's on a repeated identical request.
- `buffer/create_mapped_unmap` 0.40×.
- Unique (cache-missing) shader and pipeline creation is at parity or slightly
  ahead (0.76×–0.97×) — expected, since both drive the same Tint.
- Pure command recording other than draws is at parity (0.90×–1.09×).

The picture is consistent: **object construction, validation and shader
compilation are competitive; the queue/submission layer is not.**

## Status

Measured only. No fix is in progress; P-001–P-003 are recorded here so the
baseline is reproducible before anything changes. Re-run the harness and add a
new "Run" section rather than editing Run 1 in place.
