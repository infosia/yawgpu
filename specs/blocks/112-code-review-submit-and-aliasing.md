# Block 112 — whole-codebase review: queue submit serialization + writable-binding aliasing

Status: **COMPLETE (2026-10-25)** — S1 `74e7415`, S2 `e493528`, Phase Review
fixes (R5, m1, m2) in the closing commit. Gate after fixes: `cargo test
--workspace` green, strict clippy + fmt clean, Metal e2e 127/0, Vulkan e2e
under sync validation 115 pass / 1 pre-existing MoltenVK fail, 0
`SYNC-HAZARD`.
Source: `specs/tracking/code-review-2026-10.md` (core MAJOR rows "concurrent
`Queue::submit` lowering vs HAL submit order" and "writable-binding aliasing
check at draw/dispatch missing"). Both are core (`yawgpu-core`), Noop-testable.

| Slice | Rules | Files |
|---|---|---|
| S1 | R1 | `yawgpu-core/src/queue.rs` (+ `device.rs` if the lock lives there) |
| S2 | R2–R4 | `yawgpu-core/src/pass.rs`, `compute_pass.rs`, `render_pass.rs`, `render_bundle.rs` |
| S3 | — | Phase Review |

## S1 — one submission at a time per queue

### R1. Lowering, HAL submit and commit happen under one per-queue lock

`Queue::submit` (`queue.rs`) lowers command buffers under the
`pending_writes` lock (reading texture lazy-init state through
`InitOverlay`), releases it, then calls `hal.submit_copies`, then commits
the overlay's init marks and retires staging. Two threads can therefore
lower in one order and reach the HAL in the other: thread B lowers a
sample of texture T (plans a zero-clear), thread A lowers and submits a
copy into T first, then B's clear wipes A's data — a result no serial
order of the two submits produces. The same gap lets queued writes taken
by one submit reach the HAL after writes flushed by a concurrent
`flush_pending_writes`.

Hold a single per-queue submission mutex from taking the pending writes,
through lowering, `hal.submit_*`, overlay commit and staging retirement;
take it in `flush_pending_writes` and in `write_texture`'s immediate
init-marking path too, so every path that orders work on the HAL queue is
serialized. Validation that does not touch shared queue state may stay
outside the lock. Never call user callbacks while holding it. Document
the lock order relative to the existing `pending_writes`, buffer-state and
device locks in a comment and keep it acyclic (Dawn serializes this path
with its device lock). Test: a Noop unit test that interleaves two
submits on two threads with a HAL hook (or the existing test seam) and
asserts the HAL sees each submission's lowering and its commit as one
unit; plus a deterministic test that the lazy-init clear for a texture
written by an earlier-submitted copy is not emitted.

## S2 — writable-binding aliasing (Dawn `FindStorageBufferBindingAliasing`)

### R2. Usage scope: storage + storage on one resource is compatible

Per the WebGPU usage-scope rules (Dawn `SyncScopeUsageTracker`: a set of
internal usages is valid if it is all read-only **or** a single usage bit),
`storage` (writable storage buffer) combined with `storage` on the same
buffer, and `write-only storage texture` with itself / `read-write storage
texture` with itself on the same subresource, are **compatible** in both
render and compute usage scopes. Today `validate_buffer_usage_scope`
(`pass.rs`) rejects any two `Write` uses of a buffer even with disjoint
ranges, e.g. `storage @[0,256)` + `storage @[256,512)` in one dispatch —
Dawn accepts that. A writable storage use combined with any *other* usage
(uniform, read-only storage, vertex, index, indirect, …) still conflicts.
Write/read combinations already rejected by the CTS-verified texture rules
stay rejected.

### R3. Draw / dispatch reject overlapping writable bindings

At each `draw*` / `dispatchWorkgroups*` (and when a render bundle is
recorded / executed — follow where Dawn validates bundles), after the bind
group compatibility check, reject if two **writable** bindings of the
current pipeline layout's bound groups alias (Dawn
`CommandBufferStateTracker.cpp` `FindStorageBufferBindingAliasing`):

- storage buffers (`type = storage`, size > 0): same buffer and
  overlapping `[offset + dynamicOffset, … + size)` (whole-size resolved to
  the effective size);
- storage textures with `write-only` or `read-write` access: same texture
  and overlapping mip range **and** overlapping array-layer range.

Read-only storage / uniform bindings never participate. The error names
both bindings (group, binding, range). Cache the verdict with the existing
bind-group compatibility cache so it is recomputed only when bind groups,
dynamic offsets or the pipeline layout change.

### R4. Tests and CTS

Unit tests: disjoint storage ranges in one dispatch → ok; overlapping →
error; dynamic offsets that move a range into overlap → error; write-only
storage textures on different mips of one texture → ok, same mip → error;
read-only + writable on the same range → usage-scope error (unchanged);
render draw with two overlapping writable storage buffers → error.
Claude runs webgpu-native-cts on Metal before / after:
`webgpu:api,validation,resource_usages,*`,
`webgpu:api,validation,encoding,programmable,*`,
`webgpu:api,validation,encoding,cmds,*` — no new failures.

## S3 — Phase Review

Fresh no-context reviewer over the block's cumulative diff.

## S3 — Phase Review (2026-10-25)

Fresh reviewer over `78e5bfb..e493528`: **0 CRITICAL / 1 MAJOR / 2 MINOR**;
R2/R3 found at parity with Dawn on every axis. Gates before review:
`cargo test --workspace` green, strict clippy + fmt clean;
webgpu-native-cts on Metal unchanged vs the pre-block baseline
(`resource_usages` 7978/0, `encoding,programmable` 2701/0,
`encoding,cmds` 30613/0), plus `render_pass` 16626/0 and
`api,operation` 228854/0.

| ID | Finding | Disposition |
|---|---|---|
| M1 | submit validation (mapped / destroyed checks, `mark_submitted`) runs before the submission lock: a concurrent `mapAsync` can pass `begin_map`, read `latest_submission_index() = N` and gate on N while the racing submit then becomes N+1 and writes the buffer — the map resolves while the GPU still writes it (pre-existing; breaks R1's goal) | fix (R5): run the map / destroy validation and `mark_submitted` under the submission lock |
| m1 | `write_texture` / `mapAsync` flush now wait for an in-progress (possibly blocking) HAL submit | document in the lock comment (throughput note, not a bug) |
| m2 | no dispatch-level test that read-only + writable storage on one buffer is still a usage-scope error | add the test |

### R5. Submit validation is serialized with map / destroy

The checks that read buffer / texture state shared with `mapAsync` /
`destroy` (mapped-state, destroyed-state, `mark_submitted`) run after the
submission lock is taken, so `mapAsync`'s flush (which takes the same lock)
either completes before the submit's validation (the submit then fails on
a pending-map buffer) or after the submit is issued (the map gates on its
index). Test: deterministic Noop test using the existing hook pattern —
`begin_map` + index read on one side, submit on the other, assert either a
submit validation error or a map gated on the new index.
