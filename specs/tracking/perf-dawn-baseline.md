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

## CTS baseline before Block 96

Taken so a queue-layer change can be proved not to move a single result
(Block 96 **Q24**). Metal, yawgpu `0e12cd5`, `--workers 6`,
`--expectations expectations/yawgpu.txt`, trees `api,operation,{queue,buffers,
memory_sync,command_buffer,resource_init}`:

```
pass=171474 skip=37 warn=0 fail=0 crash=0 xfail=0 xpass=0
```

## Run 2 — 2026-08-05, after Block 96 Slice A (P-002 fixed)

Same machine, same Dawn build, `--reps 9`. `before` is Run 1's yawgpu column.

| case | before ns | after ns | dawn ns | after/dawn |
|---|---|---|---|---|
| `buffer/create_destroy` | 1756 | 1780 | 1929 | 0.92× |
| `buffer/create_mapped_unmap` | 1781 | 1816 | 4428 | 0.41× |
| `bindgroup/create_destroy` | 163 | 164 | 133 | 1.24× |
| `queue/write_buffer_4kb` | 156238 | **636** | 669 | 0.95× |
| `queue/write_buffer_then_wait` | 147249 | **788** | 592 | 1.33× |
| `frame/10writes_dispatch_submit_wait` | 1604314 | **164336** | 159279 | 1.03× |
| `shader/create_cached` | 132 | 140 | 657 | 0.21× |
| `shader/create_unique` | 16091 | 16423 | 21154 | 0.78× |
| `pipeline/compute_cached` | 66 | 67 | 364 | 0.18× |
| `pipeline/compute_unique` | 84787 | 85360 | 87555 | 0.97× |
| `pipeline/render_unique` | 118858 | 118421 | 135576 | 0.87× |
| `encode/empty_encoder_finish` | 159 | 162 | 177 | 0.91× |
| `encode/compute_1_dispatch` | 727 | 787 | 668 | 1.18× |
| `encode/render_pass_empty` | 875 | 842 | 928 | 0.91× |
| `encode/render_draw` | 148 | 147 | 42 | 3.47× |
| `submit/empty` | 193 | 220 | 3123 | 0.07× |
| `submit/compute_1_dispatch` | 155875 | 161238 | 25885 | 6.23× |
| `submit/compute_wait_idle` | 149310 | 151241 | 134873 | 1.12× |

**P-002 is resolved.** `queue.writeBuffer` went 156 µs → 636 ns (246×) and is
now marginally ahead of Dawn. End-to-end, `frame/10writes_dispatch_submit_wait`
went 1.60 ms → 164 µs (9.8×) and sits at 1.03× of Dawn — parity. Nothing else
moved outside noise; P-001 (`submit/compute_1_dispatch`, still 6.2×) and P-003
(`encode/render_draw`, still 3.5×) are untouched by design.

CTS re-run on Metal over the same trees as the baseline:
`pass=171474 skip=37 warn=0 fail=0 crash=0` — **byte-identical**, so Q24 holds
for this slice on Metal.

### A methodology note worth keeping

The first post-fix measurement was taken while the CTS run was still executing
with 6 workers, and reported everything ~30% slower — including
`shader/create_cached` and `buffer/create_destroy`, which the change cannot
touch. That across-the-board shift on untouched cases is the tell. Benchmark
runs must have the machine to themselves; a case that "regressed" alongside
cases the diff never reaches is measuring the load, not the code.

### Known cost carried by Slice A

Recycling a staging chunk requires proving the GPU is done with it. Submission
is asynchronous on Vulkan, so Slice A returns chunks to the reuse pool only
after `wait_idle`, and allocates a fresh 64 KiB chunk otherwise. That costs
roughly 14 µs per submit on Metal — visible as `frame/*` sitting at 1.03×
rather than the 0.94× an unconditional recycle reached. It is deliberate: the
alternative measured earlier made Vulkan submission synchronous, which is a
regression on a Tier 1 backend. Block 96 **Q17** removes the cost by keying
recycling on the completion index Slice B introduces.

## Run 3 — 2026-08-06, after Block 96 Slice B (P-001 fixed)

Same machine, `--reps 9`, both binaries run back to back on an idle machine.
Slice B is B1 (`4cb1fe7`, submission index in the HAL), B3 (`abbf19c`,
callbacks gated on completion) and B2 (Metal submission made asynchronous).

| case | yawgpu ns | dawn ns | ratio |
|---|---|---|---|
| `buffer/create_destroy` | 1789 | 1925 | 0.93× |
| `buffer/create_mapped_unmap` | 1823 | 4409 | 0.41× |
| `bindgroup/create_destroy` | 170 | 130 | 1.31× |
| `queue/write_buffer_4kb` | 661 | 636 | 1.04× |
| `queue/write_buffer_then_wait` | 853 | 598 | 1.43× |
| `frame/10writes_dispatch_submit_wait` | 148131 | 156938 | 0.94× |
| `shader/create_cached` | 137 | 659 | 0.21× |
| `shader/create_unique` | 16379 | 20632 | 0.79× |
| `pipeline/compute_cached` | 66 | 376 | 0.18× |
| `pipeline/compute_unique` | 85583 | 87409 | 0.98× |
| `pipeline/render_unique` | 118759 | 133046 | 0.89× |
| `encode/empty_encoder_finish` | 160 | 177 | 0.90× |
| `encode/compute_1_dispatch` | 753 | 673 | 1.12× |
| `encode/render_pass_empty` | 826 | 918 | 0.90× |
| `encode/render_draw` | 148 | 42 | **3.50×** |
| `submit/empty` | 1682 | 3234 | 0.52× |
| `submit/compute_1_dispatch` | 25173 | 25253 | **1.00×** |
| `submit/compute_wait_idle` | 122626 | 135783 | 0.90× |

**P-001 is resolved.** `submit/compute_1_dispatch` went 155875 ns → 25173 ns
against Dawn's 25253 ns — exact parity, from 6.23×.

The decisive evidence is the *decoupling*, not the absolute number. Run 1
recorded submit (155875) and submit-then-drain (149310) as the same figure,
which is what a blocking submit looks like: adding an explicit drain cost
nothing because the submit had already drained. They are now 25173 and 122626,
a 4.9× gap — the same shape Dawn has always had.

`submit/empty` is no longer a fake win. Run 1's 193 ns came from
`MetalQueue::submit_copies` short-circuiting an empty slice; it now commits a
tracked command buffer like Dawn does and measures 1682 ns against 3234 ns.
The earlier "not like-for-like" caveat on that row no longer applies.

**P-003 is now the only substantial gap left**: `encode/render_draw` at 3.50×,
unchanged and untouched by this block.

CTS over the same trees as the baseline: `pass=171474 skip=37 fail=0 crash=0`
on Metal, identical to the pre-block baseline, with 171k cases exercising the
new asynchronous resource-retention path.

### What B2 changed for consumers

Making Metal submission asynchronous changed one observable characteristic:
a future gated on GPU completion now requires the caller to actually pump the
event loop. `wgpuInstanceWaitAny` with a zero timeout is a poll and may report
TimedOut until the submission finishes.

This surfaced as 72 failing real-Metal e2e tests. The cause was not the library
— `yawgpu-test`'s `wait_for_future` did one `ProcessEvents` plus one zero-timeout
`WaitAny`, which sufficed only because the old Metal submit blocked. It now
loops to a deadline, the way any real consumer and the CTS harness's `pumpUntil`
already did, and all 98 real-Metal e2e tests pass.

Worth remembering as a class: a synchronous implementation lets callers get away
with a single event-loop pass, and every such caller becomes a latent failure
the moment the implementation becomes asynchronous. The failures all showed up
as a status field left at its *initialising* value (`WGPUMapAsyncStatus_Error`),
which reads like an error being raised rather than a callback never firing.

## P-003 is much worse than Run 1 suggested — 2026-08-06

Run 1 recorded P-003 as `encode/render_draw` at 3.5×, and framed it as per-draw
*recording* overhead: `RenderPass::draw` cloning the whole pass state. That
framing was incomplete, and the CPU number understated the problem by an order
of magnitude.

`HalRenderPass` (`yawgpu-hal/src/command.rs:426`) is a **whole-pass-plus-one-draw**
struct: it carries the colour targets, depth-stencil attachment, bind groups,
vertex buffers, viewport, scissor and exactly one optional `draw`. Core emits one
per draw, and no backend merges them — `MetalQueue::submit_copies`
(`yawgpu-hal/src/metal/queue.rs:307`) builds a descriptor, creates a
`renderCommandEncoderWithDescriptor`, encodes, and calls `endEncoding` for
**every** `HalCopy::RenderPass`.

So a WebGPU render pass with N draws becomes N GPU render passes, each with its
own attachment load and store. On a tiled GPU that resolves tile memory once per
draw. `encode/render_draw` never showed this because it stops at
`wgpuCommandEncoderFinish` and never submits.

A new case, `submit/render_100_draws_wait`, submits and drains a 100-draw pass:

| | yawgpu ns | dawn ns | ratio |
|---|---|---|---|
| `submit/render_100_draws_wait` | 5143693 | 252045 | **20.4×** |

5.1 ms against Dawn's 0.25 ms — about 51 µs of GPU-side cost per draw. This is
the real P-003, and the existing render-pass-per-draw `storeOp` forcing noted in
the CTS work is a symptom of the same shape.

Fixing it is a structural change, not a micro-optimisation: the render-pass
representation has to become one pass plus a stream of draw commands carrying
only what changed, across `yawgpu-core` and every backend encoder.

## Run 4 — 2026-08-06, after Block 97 S2 (P-003 fixed)

Same machine, `--reps 9`, both binaries back to back on an idle machine.

| case | yawgpu ns | dawn ns | ratio |
|---|---|---|---|
| `buffer/create_destroy` | 1812 | 1925 | 0.94× |
| `buffer/create_mapped_unmap` | 1803 | 4390 | 0.41× |
| `bindgroup/create_destroy` | 173 | 130 | 1.33× |
| `queue/write_buffer_4kb` | 680 | 633 | 1.07× |
| `queue/write_buffer_then_wait` | 931 | 609 | 1.53× |
| `frame/10writes_dispatch_submit_wait` | 149754 | 156132 | 0.96× |
| `shader/create_cached` | 135 | 658 | 0.20× |
| `shader/create_unique` | 16362 | 20840 | 0.79× |
| `pipeline/compute_cached` | 67 | 357 | 0.19× |
| `pipeline/compute_unique` | 85047 | 87572 | 0.97× |
| `pipeline/render_unique` | 118127 | 133459 | 0.89× |
| `encode/empty_encoder_finish` | 160 | 177 | 0.90× |
| `encode/compute_1_dispatch` | 759 | 667 | 1.14× |
| `encode/render_pass_empty` | 813 | 922 | 0.88× |
| `encode/render_draw` | 48 | 41 | **1.16×** |
| `submit/empty` | 1656 | 3208 | 0.52× |
| `submit/compute_1_dispatch` | 25344 | 25308 | 1.00× |
| `submit/render_100_draws_wait` | 174955 | 189312 | **0.92×** |
| `submit/compute_wait_idle` | 140665 | 140754 | 1.00× |

**P-003 is resolved.** `submit/render_100_draws_wait` went 5143693 ns →
174955 ns, a **29× speedup**, and now sits at 0.92× of Dawn. The CPU-side
`encode/render_draw` went 148 ns → 48 ns, from 3.50× to 1.16×.

The GPU figure is the one that matters: a 100-draw pass is now one GPU render
pass rather than 100, so the attachment load/store happens once.

### CTS

Metal, Block 96 baseline trees plus `api,operation,{rendering,render_pass,
render_pipeline}`: `pass=175738 skip=47 fail=0 crash=0`.

Vulkan/MoltenVK, same trees: `pass=175729 skip=47 fail=9 crash=0`. Those 9 are
**pre-existing** — the same failure set, case for case, at `a2b7d41` (before S2)
and after. Seven are `rendering,3d_texture_slices` reporting "attachment image
view creation failed" and two are `rendering,depth_clip_clamp`, the documented
MoltenVK artifacts. They live in the `rendering` tree, which the Block 96
baseline did not cover, which is why they appear here for the first time.

Taking that baseline mattered: the raw result reads as "S2 broke 9 Vulkan
cases". Diffing the failure sets before and after is what showed the change is
clean. Never attribute a count to a change without the before.

## P-004 — usage-scope validation was superlinear in binds per pass

Reported 2026-08-08 by an external workload (the subscript-gpu encode
benchmark, via `HANDOFF.md`): one render pass doing `setBindGroup` + `draw`
× 1000 cost ~82 µs *per draw* at encode time, with a `sample` profile putting
~97% of samples inside `wgpuRenderPassEncoderSetBindGroup`.

Verified against source: `record_resource_usage_scope_uses`
(`yawgpu-core/src/pass.rs`) cloned the full accumulated scope on every bind and
re-validated it from zero through an all-pairs O(k²) loop — bind k cost O(k²),
a pass with N binds O(N³). None of the existing bench cases could see it:
`encode/render_draw` never calls `setBindGroup` inside the pass, so the usage
scope stayed empty. The same blind-spot lesson as P-003, one layer up — a
bench that skips a per-draw call cannot price that call's accumulation.

New case `encode/render_bind_draw` (setBindGroup + draw × `kDrawsPerPass`,
finish, no submit, ns per draw):

| kDrawsPerPass | before ns/draw | after ns/draw |
|---|---|---|
| 100 | 1501 | 396 |
| 1000 | 81622 | 384 |

54× growth for 10× N before; flat after. The reporter's 82 µs figure at
N=1000 reproduced exactly.

Fix: incremental validation (`LenientUsageScopeIndex`). A bind's uses are
checked against an identity-keyed index of the already-valid scope — buffers
as one access per identity (sound because a valid scope is access-uniform per
buffer), textures as per-identity buckets with the pairwise
overlap/aspect/compatibility rule — plus new-vs-new pairs within the batch,
then appended. Identical texture uses are not re-added to the index, keeping
same-group rebinding O(1) amortized. The `scope_buffer_uses` /
`scope_texture_uses` history vectors still receive every use un-deduped in
original order for their non-validation readers, and the strict compute-path
validators and all error strings are untouched. The render-bundle path was
folded onto the same helper.

## Run 5 — 2026-08-08, after the P-004 fix

| case | yawgpu ns | dawn ns | ratio |
|---|---|---|---|
| `encode/render_bind_draw` | 399 | 100 | **3.99×** |
| `encode/render_draw` | 49 | 41 | 1.18× |
| `submit/render_100_draws_wait` | 178292 | 190882 | 0.93× |
| `submit/compute_1_dispatch` | 24256 | 24400 | 0.99× |
| `frame/10writes_dispatch_submit_wait` | 147904 | 156801 | 0.94× |
| `queue/write_buffer_4kb` | 612 | 673 | 0.91× |

Every other case is within noise of Run 4. The superlinearity is gone; what
remains on `encode/render_bind_draw` is a ~4× constant factor per rebind
(~300 ns: bind-group usage collection plus the per-draw command build), which
is a further candidate but a different, linear problem.

CTS, Run 4 trees: Metal `pass=175738 skip=47 fail=0 crash=0`;
Vulkan/MoltenVK `pass=175729 fail=9` with the failure set identical case for
case to Run 4 (the 9 pre-existing MoltenVK artifacts).

## Run 6 — 2026-08-08, after the P-004 constant-factor pass

The ~4× rebind constant was profiled with `sample` on the release bench.
Of ~3900 in-pass samples: `validate_pipeline_bind_groups` ~740 — building,
comparing and dropping `BTreeMap`s **on every draw** for a verdict that only
changes when the pipeline or a bound group changes; `collect_bind_group_usage`
~500 — `BTreeMap::from_iter` per bind for an immutable bind group; SipHash on
the scope index's `usize` identity keys ~250; per-call batch `HashMap`
construction ~200; `record_referenced_buffers` append growth ~190.

Fixes, in that order: the pipeline↔bind-group compatibility verdict is
memoized behind a dirty flag set by every pipeline/bind-group identity change
(bundle replay included; the error still surfaces at draw time with the same
message, and a failed validation leaves the flag dirty so the next draw errs
again); per-bind-group scope uses are precomputed at bind-group creation, with
only dynamic-offset resolution left per bind (`BufferScopeUse` carries resolved
offset/size, so dynamic bindings add a cheap offset add at bind time); the
scope index uses an in-repo identity hasher (full-entropy address keys,
HashDoS irrelevant); batch maps became reusable scratch cleared on both
success and error; referenced-buffer recording dedups by identity (readers are
membership-only: submit-time destroyed/mapped checks).

| case | run 5 ns | now ns | dawn ns | ratio |
|---|---|---|---|---|
| `encode/render_bind_draw` | 399 | **224** | 99 | 2.27× |
| `bindgroup/create_destroy` | 170 | 240 | 130 | 1.85× |
| `encode/render_draw` | 49 | 52 | 44 | 1.19× |
| `submit/render_100_draws_wait` | 178292 | 199466 | 185985 | 1.07× |

`bindgroup/create_destroy` +70 ns is the deliberate side of the trade: the
per-bind-group precompute moved cost from every bind to one-time creation.
`submit/render_100_draws_wait` looked regressed in the full run (1.07×) but
three isolated back-to-back runs put yawgpu and Dawn both at ~240 µs — the
case is thermally sensitive and the delta is machine state, not code. Same
verdict for `encode/render_draw` (+3 ns).

CTS: Metal `pass=175738 fail=0`; Vulkan/MoltenVK the identical 9-case
pre-existing failure set. All 98 real-Metal e2e tests pass.

What remains of the gap (224 vs 99) is the honest per-command cost: bound-group
clone into the command, usage-scope index probes, and the per-draw command
build. No single >30% lever is visible in the profile any more; further
narrowing would be diffuse micro-work, recorded here as low-priority.

## Run 7 — 2026-11-02, Block 97 S4 (after Blocks 98-120)

| | |
|---|---|
| Machine | Apple M2, macOS 27.2 (26B123) |
| Backend | Metal (both); yawgpu also on MoltenVK |
| yawgpu | `356e125`, `cargo build -p yawgpu --release --features metal,vulkan` |
| Dawn | tag `v20261030.184100` (`97ae1b10ac`), CMake Release, Metal only, default toggles |
| Harness | `benches/`, default reps, two full rounds per implementation, alternating |

The Dawn library differs from Runs 1-6: it is now the release yawgpu's Tint pin is
based on, so both sides compile WGSL with the same Tint. Each figure is the lower of
the two rounds' minimums. The rounds agreed within 8% on every case except yawgpu
`pipeline/compute_cached` (76 and 67 ns).

| case | iters | yawgpu ns | dawn ns | ratio |
|---|---|---|---|---|
| `buffer/create_destroy` | 20000 | 1701 | 1851 | 0.92× |
| `buffer/create_mapped_unmap` | 10000 | 1764 | 4480 | 0.39× |
| `bindgroup/create_destroy` | 20000 | 190 | 128 | 1.48× |
| `queue/write_buffer_4kb` | 20000 | 590 | 649 | 0.91× |
| `queue/write_buffer_then_wait` | 500 | 118168 | 574 | 205.69× |
| `frame/10writes_dispatch_submit_wait` | 300 | 148344 | 164268 | 0.90× |
| `shader/create_cached` | 5000 | 127 | 676 | 0.19× |
| `shader/create_unique` | 200 | 16186 | 20286 | 0.80× |
| `pipeline/compute_cached` | 5000 | 67 | 346 | 0.19× |
| `pipeline/compute_unique` | 100 | 99835 | 88812 | 1.12× |
| `pipeline/render_unique` | 60 | 133397 | 147365 | 0.91× |
| `encode/empty_encoder_finish` | 20000 | 142 | 162 | 0.88× |
| `encode/compute_1_dispatch` | 20000 | 705 | 608 | 1.16× |
| `encode/render_pass_empty` | 20000 | 965 | 861 | 1.12× |
| `encode/render_bind_draw` | 200000 | 153 | 83 | 1.84× |
| `encode/render_draw` | 200000 | 45 | 38 | 1.20× |
| `submit/empty` | 10000 | 1531 | 2537 | 0.60× |
| `submit/compute_1_dispatch` | 5000 | 24570 | 46145 | 0.53× |
| `submit/render_100_draws_wait` | 200 | 174282 | 200097 | 0.87× |
| `submit/compute_wait_idle` | 500 | 137494 | 145048 | 0.95× |

yawgpu on MoltenVK, same build, one round. MoltenVK is reference-only, and Dawn was
not built with Vulkan, so there is no ratio column:

| case | yawgpu MoltenVK ns |
|---|---|
| `buffer/create_destroy` | 2101 |
| `buffer/create_mapped_unmap` | 2151 |
| `bindgroup/create_destroy` | 190 |
| `queue/write_buffer_4kb` | 638 |
| `queue/write_buffer_then_wait` | 122578 |
| `frame/10writes_dispatch_submit_wait` | 165508 |
| `shader/create_cached` | 135 |
| `shader/create_unique` | 17198 |
| `pipeline/compute_cached` | 71 |
| `pipeline/compute_unique` | 159838 |
| `pipeline/render_unique` | 194797 |
| `encode/empty_encoder_finish` | 146 |
| `encode/compute_1_dispatch` | 714 |
| `encode/render_pass_empty` | 900 |
| `encode/render_bind_draw` | 150 |
| `encode/render_draw` | 45 |
| `submit/empty` | 18298 |
| `submit/compute_1_dispatch` | 24419 |
| `submit/render_100_draws_wait` | 186741 |
| `submit/compute_wait_idle` | 144056 |

### Reading

- End to end, yawgpu is now ahead of Dawn: `frame/10writes_dispatch_submit_wait`
  0.90×, `submit/render_100_draws_wait` 0.87×, `submit/compute_1_dispatch` 0.53×.
- Per-command recording is still behind: `encode/render_bind_draw` 1.84× (Run 6:
  2.27×), `bindgroup/create_destroy` 1.48× (1.85×), `encode/render_draw` 1.20×,
  `encode/compute_1_dispatch` 1.16×. These are the diffuse costs Run 6 described.
- `pipeline/compute_unique` moved from 0.97× (Run 1) to 1.12×: yawgpu went from
  85 to 100 µs, and Dawn stayed at 88 to 89 µs. Not investigated. Candidates are the
  newer Tint and the per-pipeline work Blocks 118-120 added.
- `submit/empty` is no longer a fake win and stays under Dawn (0.60×).
- MoltenVK `submit/empty` 18.3 µs and `submit/compute_1_dispatch` 24.4 µs match
  Block 117's 17.4 and 24.1 µs, so nothing has regressed since.
- No case isolates the per-pass `VkRenderPass` / `VkFramebuffer` creation that
  backlog E11 would cache: the `_wait` cases are dominated by GPU latency. E11
  needs a case that submits many passes without waiting before it can be decided.
- **`queue/write_buffer_then_wait` 205.69×** is a regression from Run 4's 1.53×,
  P-005 below.

### P-005 — `onSubmittedWorkDone` flushes pending writes, Dawn does not

yawgpu takes 118 µs, a full GPU round trip. Dawn takes 574 ns. Block 113 S10
(`7488716`) made `onSubmittedWorkDone` flush the queue's pending writes and wait for
them, because the audit row said it "ignores pending writes". Its test is
`audit_s10_work_done_flushes_queued_buffer_writes` in `yawgpu/src/ffi/mod.rs`.

Dawn does not do this. `CopyFromStagingToBuffer` records the copy with
`SubmitMode::Passive` on Metal (`metal/DeviceMTL.mm`) and on Vulkan
(`vulkan/DeviceVk.cpp`), so `HasPendingCommands()` stays false.
`GetScheduledWorkDoneSerial()` then returns the last submitted serial, and the
callback does not wait for the write (`Queue.cpp`, `ExecutionQueue.cpp`). A passive
copy is flushed by the next submit, by a `mapAsync` that needs it, or after 16 MiB
of pending uploads (`DynamicUploader::MaybeSubmitPendingCommands`).

So S10 is stricter than the oracle, and the audit premise was wrong. Dawn's
behaviour is observably safe, because every way to read the written data forces the
flush first. The fix is to restore Dawn's behaviour for `onSubmittedWorkDone` only.
`wgpuSurfacePresent` keeps flushing (S5), as Dawn's present does. Not yet planned as
a block.

### Teardown note

The yawgpu bench binary exits 1 after printing every result: "device lost (reason 3):
Instance released". The bench treats every loss except `Destroyed` as fatal. Dawn
destroys the device when its last external reference is released
(`DeviceBase::WillDropLastExternalRef`), and the lost callback then reports
`Destroyed`. yawgpu does not, so the callback reports `CallbackCancelled` when the
instance is released. The Dawn bench exits 0. This does not affect any figure.

### CTS

Covered by runs of the same library code: Metal `webgpu:*` `fail=2 crash=0`
(Block 120, 2026-11-02); native Vulkan on Windows (Block 120) and on Linux
(Block 119).

## Run 8 — 2026-11-02, after Block 121 (P-005 fixed)

Same machine, Dawn and harness as Run 7; yawgpu two rounds, lower minimum.

| case | run 7 ns | now ns | dawn ns | ratio |
|---|---|---|---|---|
| `queue/write_buffer_then_wait` | 118168 | **1238** | 574 | 2.16× |
| `queue/write_buffer_4kb` | 590 | 1935 | 649 | 2.98× |
| `frame/10writes_dispatch_submit_wait` | 148344 | 145363 | 164268 | 0.88× |

Every other case moved by less than 8%. CTS on Metal: `fail=2 crash=0`, subcase
counts unchanged.

### P-006 — one Metal blit encoder per queued buffer copy

`queue/write_buffer_4kb` did not regress in real terms. It runs 20,000 writes with no
submit. Before Block 121 the pending copies were submitted by the next case, outside
the timed loop, which is the same blind spot Run 1 noted for Dawn. Block 121's 16 MiB
flush now submits about 4,096 copies inside the loop, as Dawn does, so the loop pays
for them: about 1.3 µs per copy. `MetalQueue::submit_copies`
(`yawgpu-hal/src/metal/queue.rs`) opens a new `blitCommandEncoder` for every
`HalCopy::Buffer`. Dawn's `CommandRecordingContext::EnsureBlit` reuses the open blit
encoder. Every submit that carries several queue writes pays this, so it is not
specific to the bench. Backlog E14. Vulkan is not measured for it.

## Run 9 — 2026-11-02, after Block 122

Same machine, Dawn and harness as Run 7; yawgpu two rounds, lower minimum. Rows marked
`*` moved by more than 8% from Run 8.

| case | run 8 ns | now ns | dawn ns | ratio |
|---|---|---|---|---|
| `buffer/create_destroy` | 1773 | 1783 | 1851 | 0.96× |
| `buffer/create_mapped_unmap` | 1800 | 1784 | 4480 | 0.40× |
| `bindgroup/create_destroy` | 203 | 199 | 128 | 1.55× |
| `queue/write_buffer_4kb` | 1935 | 1058 | 649 | 1.63× | *
| `queue/write_buffer_then_wait` | 1238 | 1144 | 574 | 1.99× |
| `frame/10writes_dispatch_submit_wait` | 145363 | 112999 | 164268 | 0.69× | *
| `shader/create_cached` | 131 | 130 | 676 | 0.19× |
| `shader/create_unique` | 16506 | 17226 | 20286 | 0.85× |
| `pipeline/compute_cached` | 72 | 71 | 346 | 0.20× |
| `pipeline/compute_unique` | 102379 | 102152 | 88812 | 1.15× |
| `pipeline/render_unique` | 136649 | 139983 | 147365 | 0.95× |
| `encode/empty_encoder_finish` | 142 | 144 | 162 | 0.89× |
| `encode/compute_1_dispatch` | 744 | 740 | 608 | 1.22× |
| `encode/render_pass_empty` | 944 | 938 | 861 | 1.09× |
| `encode/render_bind_draw` | 153 | 153 | 83 | 1.84× |
| `encode/render_draw` | 46 | 45 | 38 | 1.20× |
| `submit/empty` | 1555 | 1544 | 2537 | 0.61× |
| `submit/compute_1_dispatch` | 24978 | 23911 | 46145 | 0.52× |
| `submit/render_100_draws_wait` | 170877 | 171952 | 200097 | 0.86× |
| `submit/compute_wait_idle` | 136897 | 138186 | 145048 | 0.95× |

`frame/10writes_dispatch_submit_wait` is not an improvement: its two rounds read 113
and 144 µs. That case is dominated by GPU latency and is noisy across runs.

### P-006, corrected

Run 8 attributed `queue/write_buffer_4kb` to one Metal blit encoder per queued copy.
Block 122 coalesced the encoders, and the case stayed at 1,951 ns, so that diagnosis
was wrong. A `sample` profile then showed the cost in
`PendingWriteBatch::recycle_completed`. It ran on every write and rebuilt the whole
in-flight list, which holds hundreds of chunks after a 16 MiB flush. Recycling now
pops only the completed prefix (Block 122 R6), and the case fell to 1,058 ns (1.63×).
The rest is the data copy, staging-buffer allocation once the four-chunk free pool is
exhausted, and the submit. **Profile before naming a cause.** Two plausible readings of
the same number pointed at different layers.

### `pipeline/compute_unique`

See Block 122 S2. The case measures Metal compiler-cache lookups, because its shader
sources repeat across batches, and about 98% of it is in the Metal compiler service.

## Status

P-004 **resolved**, including the constant-factor follow-up (399 → 224 ns,
2.27× of Dawn, no superlinearity). All three findings from Run 1 are **resolved**: P-001 and P-002 in Block 96,
P-003 in Block 97. The remaining gaps against Dawn are small and none is
structural: `queue/write_buffer_then_wait` 1.53×, `bindgroup/create_destroy`
1.33×, `encode/compute_1_dispatch` 1.14×, `encode/render_draw` 1.16×.

Block 97 **S3** is **done** (`88cfe58`) and **S4** is **done** (Run 7,
2026-11-02). Run 7 found **P-005**, fixed by Block 121. Run 8 found **P-006**, fixed by Block 122
(Run 9: `queue/write_buffer_4kb` 2.98× to 1.63×; the cause was staging recycling, not
the blit encoder). The other gaps are per-command recording cost, at 1.84× or below.

## Superseded status notes

**P-001 and P-002 resolved** (Block 96). **P-003 open** — `encode/render_draw`
at 3.50×, caused by `RenderPass::draw` snapshotting the whole pass state per
draw. Re-run the harness and add a new "Run" section rather than editing an
earlier one in place.
