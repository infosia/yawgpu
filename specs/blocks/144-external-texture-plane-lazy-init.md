# Block 144 — External texture planes are lazily zero-initialized

Status: **COMPLETE (2026-11-05)**. Phase Review: 0 CRITICAL, 0 MAJOR, 3 MINOR; two resolved,
one moved to the backlog (A28), see below.
Source: backlog A27 (Block 143 Phase Review m7), started by user request on 2026-11-05.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`BindGroup.cpp` (an external-texture entry expands into `plane0`/`plane1` sampled
texture bindings), `PassResourceUsageTracker.cpp` (`SyncScopeUsageTracker::AddBindGroup`
tracks them as sampled textures), and `metal/CommandBufferMTL.mm`
(`LazyClearSyncScope` clears every non-attachment texture usage before the pass and marks
it initialized).

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

R2. **The clear initializes.** The lazy clear writes zeros, so it marks the cleared
subresources initialized, exactly as it does for a sampled texture view. A later
submission that samples the same plane needs no clear.

R3. **Unchanged.** External texture validation, the parameter buffer, every other
binding kind, and every backend.

## Tests

- Unit (core, Noop): a submission that binds an external texture whose plane 0 (and, for
  NV12, plane 1) was never written lowers a `ClearTexture` for that plane before the
  pass, in a compute pass, a render pass, and a render bundle. An initialized plane gets
  no clear, and neither does a plane a previous submission already cleared. Both planes
  of a two-plane format are handled.
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

## Verification (2026-11-05)

Run on Apple M2, macOS 27.2.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,431 passed, 0 failed (after the Phase Review test additions) |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough` | clean |
| `metal_external_texture_discarded_planes_read_zero` (new) | passes. Against `4bd397d` (before this block) it fails: the discarded RGBA plane samples as `[64, 128, 191, 255]` instead of zero |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled` | 172 passed, 0 failed |
| webgpu-native-cts, Metal, `webgpu:api,*` | pass 589,574, skip 101,793, fail 0, crash 0; identical to Block 143 |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*` | pass 857,323, skip 25,860, fail 0, crash 0, xfail 70, xpass 2 (F-160); pass + xfail identical to Block 143 |

The CTS and e2e ran before the Phase Review fixes, which add unit tests only.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | The "already initialized" unit case only initialized plane 0, so a bug that handles planes by position could pass | The cases cover no plane, plane 0, and (NV12) plane 1 initialized |
| m2 | No unit test of a plane discarded or written earlier in the same command buffer | Added: a discarding attachment pass before the sampling pass gets exactly one clear between them; a full copy before it gets none |
| m3 | The Metal HAL binds the plane's whole texture, not its view, so a plane view with a non-zero base mip or layer samples the wrong subresource (predates this block) | Backlog A28 |
