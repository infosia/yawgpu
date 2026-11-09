# Block 150 — Usage-scope validation for subpass render passes

Status: **COMPLETE (2026-11-10)**. Phase Review: 0 CRITICAL, 0 MAJOR, 6 MINOR. m1 (R2
wording), m2 and m3 (tests), m4 (R8), and m5 (R7) are fixed in S2; m6 is the bookkeeping
below.
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
a validation error. It is recorded as the encoder's first error, and the subpass pass
becomes an error pass, the same as any other begin-time validation failure of a subpass
pass: the HAL pass is not begun and later commands do nothing. (A regular render pass
stays a recording pass after a seeding conflict; the encoder's result is the same.) This
is a new rejection: such a descriptor was accepted before.

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

R7. **The depth-stencil attachment encompasses every aspect of its format.**
`validate_subpass_depth_stencil_attachment` rejects a view whose aspect does not cover
every aspect of its texture's format (for example a `DepthOnly` view of
`depth24plus-stencil8`), with the same check and message wording as the regular render
pass (`command_encoder.rs`, "must encompass all aspects of its texture's format"; Dawn
`ValidateRenderPassDepthStencilAttachment`). R2's seeding reads the view format's
aspects and relies on this.

R8. **An ended pass releases its attachments.** After `end()` (or the `Drop` path), an
ended `SubpassRenderPass` handle no longer holds its color, resolve, or depth-stencil
attachment views: `finish_pass` moves them into the parent command instead of cloning
them (Block 50 D1). The accessors `color_attachment_count` and
`has_depth_stencil_attachment` keep returning the counts the pass began with.

## Slices

S1: R1–R6 in `yawgpu-core`, with inline unit tests in `yawgpu-core/src/subpass.rs` (and in
`pass.rs` for any extracted type). The tests run on Noop with the `tiled` feature.

S2 (Phase Review fixes): R7, R8, and two tests: one buffer bound as both vertex and index
buffer in one pass is accepted; the Drop-path test checks that dropping an unended pass
releases what its scope held.

## Verification

- Inline unit tests, one or more per rule (R1–R8): each of the four problem cases is rejected; a
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

## Phase Review (2026-11-10, fresh-context reviewer over `9df3616..bd1958d`)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| m1 | MINOR | A seeding conflict makes the subpass pass an error pass; R2 said "as the regular render pass does" | R2 reworded to the implemented behaviour |
| m2 | MINOR | The Drop-path test checked nothing scope-specific | S2: the test checks that the scope's handles are released |
| m3 | MINOR | No test bound one buffer as both vertex and index buffer | S2: test added |
| m4 | MINOR | An ended, leaked handle kept its attachment views (cloned into the command) | S2: R8, attachments move into the command |
| m5 | MINOR | A depth-stencil attachment view narrowed to one aspect was accepted | S2: R7, the regular pass's check is shared |
| m6 | MINOR | Block status and backlog row still open | Closed |

## Verification record

At S2: workspace test 1,445/0 (default features); `yawgpu-core --features tiled --lib`
788/0; `tiled_subpass_ffi` 8/0; clippy clean with and without `tiled`. Real GPU on the M2:
Metal `e2e_metal_tiled*` 6/6; MoltenVK `e2e_vulkan_tiled*` 8/8 under the Khronos
validation layer with 0 VUIDs. At S1 also `tiled_deferred --verify` on Metal and Vulkan and
`tiled_msaa --verify` on Vulkan, with the expected center pixels and 0 VUIDs.
