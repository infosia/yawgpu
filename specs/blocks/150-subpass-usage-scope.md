# Block 150 — Usage-scope validation for subpass render passes

Status: **IN PROGRESS (2026-11-10)**.
Source: backlog A31 (Block 149 Phase Review m4), started by user request on 2026-11-10.
Oracle: subpass rendering (`tiled`) is a yawgpu extension with no Dawn counterpart. Every
rule below copies what yawgpu's regular render pass already does, which follows Dawn's
render-pass usage tracker (`PassResourceUsageTracker.cpp`, `CommandValidation.cpp`
`ValidateSyncScopeResourceUsage`).

## Problem

`SubpassRenderPass` (`yawgpu-core/src/subpass.rs`) keeps a per-pass `SubpassDrawState`.
It never records a usage scope. `RenderPassEncoder` does: it seeds the scope with the
attachment uses at begin (`CommandEncoder::begin_render_pass` →
`PassEncoderState::set_attachment_texture_uses`), then records every bind group
(`record_bind_group_usage_scope`), vertex buffer, and index buffer
(`record_buffer_usage_scope_use`) at set time.

As a result, a subpass pass accepts each of these, and a regular render pass rejects each
one:

1. A bind group samples a texture that is also a color or depth-stencil attachment of the
   pass (a read/attachment-write conflict).
2. A bind group binds a buffer as writable storage, and the same buffer is also bound as a
   uniform, or as a vertex or index buffer.
3. Two bind groups conflict with each other (for example, read-only storage and writable
   storage on one buffer).
4. Two attachment slots use the same texture subresource.

The backend then runs a pass with an undefined hazard (a feedback loop on the tile memory
on Metal, a layout conflict on Vulkan) instead of producing a validation error.

## Rules

R1. **One usage scope per subpass pass.** A subpass render pass is one usage scope for its
whole lifetime, across every `next_subpass`. It uses the same lenient render-scope rules
as a regular render pass (`LenientUsageScopeIndex` /
`validate_texture_usage_pair_lenient` / `compatible_in_render_scope`). `next_subpass`
clears the bound state but does **not** reset the scope.

R2. **Attachments seed the scope at begin.** After `begin_subpass_render_pass` validation
passes, the scope is seeded with one `TextureScopeUse` per attachment, built the way
`render_pass_attachment_scope_uses` builds them:
- each color attachment's `Persistent` view: `AttachmentWrite`, aspect `COLOR`;
- each color resolve target: `AttachmentWrite`, aspect `COLOR`;
- the depth-stencil attachment's view: `AttachmentWrite` for each aspect the view's format
  has (`DEPTH`, `STENCIL`). Subpass passes have no read-only depth or stencil flags.

A seeding conflict (for example, the same texture subresource in two attachment slots) is
a validation error. It is recorded with `record_first_error` on the parent encoder, as the
regular render pass does. This is a new rejection: such a descriptor was accepted before.

R3. **Bind groups are recorded at `set_bind_group`.** `SubpassRenderPass::set_bind_group`
with `Some(group)` records the group's uses (`collect_bind_group_usage`, the same helper
`record_bind_group_usage_scope` calls) into the pass scope before it stores the binding. A
conflict is a validation error on the parent encoder, and the binding is not stored, as in
`RenderPassEncoder::set_bind_group`. External-texture planes take part through whatever
`collect_bind_group_usage` already records for them. Input-attachment slots are
auto-wired, never in a bind group, so they never enter the scope.

R4. **Vertex and index buffers are recorded at set time.**
`SubpassRenderPass::set_vertex_buffer` (with `Some(buffer)`) and `set_index_buffer` record
a `BufferScopeUse` with access `Read` over `offset..offset + size`, as the regular render
pass does.

R5. **Ownership and release.** The scope's index keys resources by `Arc` address. Its
documented invariant, that every key has an owning handle in the scope history, must hold
for the subpass scope too. When the pass ends (`finish_pass`, also on the `Drop` path),
the scope is released along with the pass's other encoding-time state, so an ended,
leaked `SubpassRenderPass` handle does not pin attachment or bound textures through the
scope (Block 50 D1).

R6. **No duplication of the scope algorithm.** The subpass pass reuses the regular render
pass's scope code. Extracting a small `pub(crate)` tracker type from `pass.rs`, holding
the index and the two history vectors, is in scope. `PassEncoderState` and render bundles
may switch to the new type only if the change is mechanical and their behaviour is the
same. A second copy of the validation logic is out of scope.

## Slices

S1 (single slice): R1–R6 in `yawgpu-core`, with inline unit tests in
`yawgpu-core/src/subpass.rs` (and in `pass.rs` for any extracted type). The tests run on
Noop with the `tiled` feature.

## Verification

- Inline unit tests, one or more per rule: each of the four problem cases is rejected; a
  conflict seen only after `next_subpass` is still rejected (R1); a valid deferred
  two-subpass pass with a mixed input-attachment + uniform group and a sampled texture
  that is not an attachment is still accepted; vertex/index + read-only use of the same
  buffer is accepted, vertex + writable storage is rejected.
- `cargo test -p yawgpu-core --features tiled --lib`,
  `cargo test -p yawgpu --features tiled --test tiled_subpass_ffi`,
  `cargo clippy --workspace --all-targets --features tiled -- -D warnings`.
- Claude: the workspace test, and the real-GPU `e2e_metal_tiled*` and `e2e_vulkan_tiled*`
  suites (MoltenVK) plus the `tiled` examples, to confirm that no valid pass now fails
  R2.
