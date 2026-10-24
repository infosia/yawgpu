# Block 110 — whole-codebase review: core / FFI validation MAJORs

Status: **PLANNED (2026-10-25)**.
Source: `specs/tracking/code-review-2026-10.md` (core validation + FFI
MAJOR rows, plus three confirmed core MINORs). Every rule here is
Noop-testable; the oracle is Dawn (`third_party/dawn/src/dawn/native/`)
and the WebGPU spec. Claude verified each finding against the code at
`3107bf8` before planning.

| Slice | Rules | Files |
|---|---|---|
| S1 | R1–R3, R8–R10 | `yawgpu-core` encoding / pipeline / layout modules |
| S2 | R4–R7 | `yawgpu-core/src/buffer.rs`, `device.rs`, `yawgpu/src/**` |
| S3 | — | Phase Review |

## S1 — encoding / pipeline validation

### R1. Dynamic offsets follow increasing binding number

`setBindGroup(dynamicOffsets)` assigns offsets to the layout's dynamic
buffer bindings **in increasing `binding` order** (WebGPU spec; Dawn sorts
`BindingIndex` with dynamic buffers ordered by binding number). Today
`validate_dynamic_offsets` (`pass.rs`), `precompute_bind_group_usage`
(`pass.rs`) and `dynamic_offset_for_binding` (`queue.rs`) all walk
`layout.entries()` in **declaration** order, and `BindGroupLayout::new`
does not sort, so a BGL declared `[{binding:1, dynamic}, {binding:0,
dynamic}]` swaps the offsets (wrong alignment check, wrong bounds, wrong
HAL binding).

Fix shape: store entries sorted by `binding` at BGL creation (both
`BindGroupLayout::new` and `new_auto`), so every consumer sees one order;
audit every `entries()` consumer for an order assumption and list them in
the report. (`bind_group_layouts_compatible` already compares by binding
number and is unaffected.) Test: out-of-order BGL with a uniform and a
storage dynamic binding — alignment and bounds are checked against the
right binding, and the lowered HAL offsets land on the right bindings.

### R2. Dynamic offset bound includes the effective binding size

For a dynamic buffer binding whose bind-group entry size is
`WGPU_WHOLE_SIZE`, the bound check is
`offset + dynamicOffset + effectiveSize <= buffer.size` with
`effectiveSize = buffer.size - offset` (spec; Dawn), i.e. any non-zero
dynamic offset fails. Today only `offset + dynamicOffset <= buffer.size`
is checked (`pass.rs` `validate_dynamic_offsets`). Resolve the effective
size once (as `precompute_bind_group_usage` already does) rather than a
second ad-hoc computation. Test: 512-byte buffer, whole-size dynamic
storage binding, offsets 0 (ok) and 256 (error).

### R3. Attachment signatures ignore trailing null color formats

`colorFormats` comparisons for pipeline ↔ pass, bundle ↔ pass and
pipeline ↔ bundle ignore trailing `None` entries (spec "ignoring trailing
nulls"; Dawn compares an attachment bitset). Trim trailing `None` when
each `AttachmentSignature` is built (pass, pipeline, bundle builders), not
at compare time. Tests: pass `[rgba8, null]` + pipeline `[rgba8]` →
`setPipeline` ok; bundle `colorFormats [rgba8, undefined]` executed in a
`[rgba8]` pass → ok; interior nulls still significant.

### R8. Resolve target `RenderAttachment` usage is checked on the view

`validate_*` for `resolveTarget` uses the **view's** usage (as color
attachments already do), not the texture's — Dawn `ValidateCanUseAs(view,
RenderAttachment)`. Test: resolve view created with a usage subset lacking
`RenderAttachment` → error.

### R9. Vertex / instance range math is u64

`firstVertex + vertexCount`, `firstInstance + instanceCount` and the index
range math in draw validation (`pass.rs`) use u64 like Dawn, so
`draw(1, 1, u32::MAX, 0)` with an `arrayStride == 0` vertex buffer is
valid instead of "overflows". Test that case plus a genuine OOB with
stride > 0 still failing.

### R10. `resolveQuerySet` with `queryCount == 0` is valid

Match Dawn `CommandEncoder.cpp` `ValidateResolveQuerySet` (read it; the
review cited the `firstQuery >= count` / `firstQuery + queryCount > count`
checks): a zero-count resolve with an in-range `firstQuery` is valid and
records nothing. Update the existing unit test
`resolve_query_set_with_zero_count_records_no_conversion_pass`, which
currently asserts the error.

## S2 — buffer mapping and FFI conversions

### R4. A stale map callback cannot consume a newer map

`mapAsync → unmap → mapAsync` before the first callback fires must yield
callback 1 = `Aborted` and callback 2 = `Success` with the buffer mapped
(spec). Today `unmap` leaves the aborted `pending_map` in place, the
second `begin_map` overwrites it, and callbacks resolve in future order —
callback 1 takes map 2's entry (`Success`, possibly reading before map 2's
GPU gate) and callback 2 finds nothing and sets the buffer `Unmapped`.

Fix shape: each `begin_map` returns a map serial stored in the pending
callback; `resolve_pending_map(serial)` only consumes a matching entry; a
stale serial resolves `Aborted` and never touches `map_state`. `unmap` /
`destroy` clear the pending entry immediately. Unit test the exact
sequence (the reviewer's probe reproduced the swap).

### R5. A null map callback still completes the map

`wgpuBufferMapAsync` with a null callback must still resolve the pending
map when its future completes (buffer becomes mapped / aborted as usual);
today `resolve_pending_map` runs only inside `if let Some(callback)`
(`ffi/mod.rs` `PendingCallback::fire`). Test: null-callback mapAsync,
process events, `getMapState` = mapped.

### R6. `dstFactor` Undefined is Zero; invalid enum values are errors

`map_blend_component` (`conv/pipeline.rs`) maps `WGPUBlendFactor_Undefined`
per field: `srcFactor` → `One`, `dstFactor` → `Zero` (`webgpu.h`), and
`operation` Undefined → `Add`. Out-of-range values of blend factor / blend
operation / stencil operation / compare function / texture dimension /
component swizzle / optional bool / subpass dependency type no longer fall
back silently: they record a validation error the way `map_primitive_state`
already does (`set_first_error`). List each mapper and its new behaviour in
the report. Tests per mapper (Undefined default + invalid value).

### R7. Host-side map storage only when a map can use it

`Buffer::try_new` allocates and zero-fills a host copy of the whole
buffer for every buffer. Allocate it only when the buffer can be mapped
through it: (`MAP_READ` | `MAP_WRITE` usage, or `mappedAtCreation`) **and**
the HAL buffer exposes no `mapped_ptr`. A `mappedAtCreation` buffer without
MAP usage may allocate lazily and free it on unmap. Audit every
`HostBuffer` consumer (map paths, `mapped_ptr_at`, queue writes, Noop
paths) and list them in the report; out-of-memory still surfaces as
`OutOfMemory` through the existing F-073 path. Tests: a large STORAGE
buffer allocates no host storage; map paths on Noop unchanged.

## S3 — Phase Review

Fresh no-context reviewer over the block's cumulative diff.

## S3 — Phase Review (2026-10-25)

Fresh reviewer over `3107bf8..c60989b`: **0 CRITICAL / 0 MAJOR / 5 MINOR**.
Gates before review: `cargo test --workspace` green; strict clippy + fmt
clean; Metal e2e 124/0; Vulkan e2e (MoltenVK, layer) 115 pass / 1
pre-existing fail (Block 109 ledger), no new VUID.

| ID | Finding | Disposition |
|---|---|---|
| m1 | zero-count `resolveQuerySet` returns before recording the query set / destination buffer references; Dawn still tracks them, so a destroyed one must fail submit | fix (R13) |
| m2 | texture-view enum-conversion error forced through `mip_level_count = Some(0)`; the stored view error message is the mip-count one | fix (R14): `create_view_with_error` like texture / sampler |
| m3 | `PendingMap.outcome` is dead after R4 (always `Success`) | fix: remove the field |
| m4 | host copy of a non-MAP mappedAtCreation buffer survives `destroy()`; `mapped_range` drops the state lock before taking the host lock, so a racing unmap can yield a dangling non-null pointer | fix (R15): free on destroy; hold the state lock across `mapped_ptr_at` or return null when unmapped |
| m5 | `create_*_with_error` carry the error as an argument while descriptors elsewhere carry an `error` field | **deferred** — style only; both shapes exist in core today and unifying them is a refactor outside a fix block |

R13–R15 are fixed in the Phase Review commit.
