# Block 144 — External texture planes are lazily zero-initialized

Status: **IN PROGRESS**.
Source: backlog A27 (Block 143 Phase Review m7), started by user request on 2026-11-05.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`BindGroup.cpp` (an external-texture binding expands into plane texture bindings),
`CommandBufferStateTracker`/`PassResourceUsageTracker.cpp` (`AddBindGroup`), and the
backends' lazy clears of sampled textures before a pass.

## Problem

Before a submission reads a texture subresource that was never written, yawgpu inserts a
clear so the read returns zero (Blocks 103 and 104). Bindings get their clears from
`append_bound_texture_init_clears` (`yawgpu-core/src/queue.rs`). It handles
`Texture`, `StorageTexture`, and `InputAttachment` bindings and skips
`ExternalTexture` bindings. A texture used only as plane 0 or plane 1 of an external
texture is therefore sampled without its clear, and the shader sees whatever the memory
held: stale contents of a reused allocation, or the contents a `storeOp: discard` left
behind.

Dawn expands an external-texture binding into its plane texture bindings, and clears
them like any other sampled texture.

External textures are implemented on Metal only; Vulkan rejects a pipeline that uses
one (Block F-060), and GLES has no lowering.

## Rules

R1. **Planes are lazily initialized.** For every bound external texture a pass uses,
each plane texture view gets the same lazy-initialization handling as a sampled texture
view: uninitialized subresources the view covers are cleared before the pass, in the
same place and order as for a `Texture` binding. Compute passes, render passes, and
render bundles. One-plane and two-plane formats.

R2. **Reads do not mark.** Sampling a plane does not mark it initialized, as for any
sampled texture.

R3. **Unchanged.** External texture validation, the parameter buffer, every other
binding kind, and every backend.

## Tests

- Unit (core, Noop): a submission that binds an external texture whose plane 0 (and, for
  NV12, plane 1) was never written lowers a `ClearTexture` for that plane before the
  pass, in a compute pass, a render pass, and a render bundle. An initialized plane gets
  no clear. Both planes of a two-plane format are handled.
- Claude, Metal e2e (`e2e_metal_external_texture.rs`): a plane texture is cleared to a
  non-zero colour with `storeOp: store`, then a second pass loads it and ends with
  `storeOp: discard`, which leaves the memory non-zero but the texture uninitialized.
  Sampling it through an RGBA external texture returns zero. Without R1 it returns the
  non-zero colour.
- Claude: the full Metal e2e and every other ignored test; CTS on Metal
  `webgpu:api,*` and `webgpu:shader,execution,*` must not regress.

## Out of scope

- External textures on Vulkan and GLES.
- Lazy initialization of the external texture's internal parameter buffer.

## Slices

- **S1 (coding agent):** R1-R3 with the unit tests.
- **S2 (Claude):** e2e, CTS, Phase Review.
