# Block 104 — Lazy zero-initialization Stage 2

Status: **SPEC (2026-09-22)**. Backlog item **A3** in
`specs/tracking/backlog.md`. Completes F-138 Stage 1
(`specs/tracking/tint-migration-plan.md` → "F-138 — texture lazy
zero-initialization"; commits `8ebdfa6`, `0449b89`).

## Problem

WebGPU requires every texture subresource to read as zero until it is
written. Stage 1 implemented Dawn-style lazy clears, but only for
textures that are **single-sample, uncompressed and colour-only**
(`texture_lazy_init_eligible`, `yawgpu-core/src/texture.rs`), and only
on the *write* side plus copy reads. Four gaps remain, each a silent
wrong result that the CTS port does not exercise
(`resource_init/texture_zero.spec.cpp` covers two colour formats through
`CopyToBuffer` only):

1. **Reads through bindings** — a `texture_2d` / `texture_depth_2d` /
   `read-only` storage binding of an uninitialized texture is never
   cleared (`queue.rs` `TODO(stage2)` in
   `append_writable_storage_texture_init_clears`, which skips
   `ReadOnly` storage and every sampled binding).
2. **Depth / stencil textures** are ineligible: no clear on copy reads,
   no clear when used as a `loadOp: Load` attachment.
3. **Compressed textures** (BC / ETC2 / EAC / ASTC) are ineligible.
4. **Multisampled textures** are ineligible.

Two related correctness holes surfaced by the survey are in scope too:

5. **`storeOp: Discard` does not un-mark** the attachment; Metal encodes
   `MTLStoreAction::DontCare`, so the next `Load` reads garbage while
   core believes the subresource is initialized (Dawn
   `CommandBuffer.cpp` `LazyClearRenderPassAttachments`: Discard →
   `SetIsSubresourceContentInitialized(false)`).
6. **Marks are applied during lowering, before the HAL submit**: a
   staging or HAL failure leaves subresources marked initialized that
   were never cleared.

Everything passes today only because fresh GPU allocations happen to be
zero on the dev machines.

## Design

Dawn's shape, adapted to yawgpu's lowering walk:

- Track initialization per **mip × layer × aspect**.
- Attachments: instead of emitting a separate clear, rewrite
  `loadOp: Load → Clear(0)` per uninitialized aspect (every backend
  already encodes load-op Clear for colour, depth and stencil, including
  MSAA), and honour `storeOp` when marking. 3D colour attachments keep
  the Stage 1 whole-mip `ClearTexture` (a render pass initializes one
  depth slice only).
- Non-attachment uses (copies, every texture binding kind, storage
  read-only or not): `HalCopy::ClearTexture` before the command, one
  per uninitialized subresource, as Stage 1 does — but the HAL must now
  execute it for every format / sample count.
- Marks are collected during the walk in a per-submit overlay and
  committed to the textures only after the HAL submit succeeds.

## Behaviour contract

### R1 — Core: init state has an aspect axis

`TextureInitState` indexes `mip × layer × aspect` (aspect ∈ {color} or
{depth}, {stencil}, {depth, stencil} per format;
`Texture::is_initialized(mip, layer, aspect)` /
`mark_initialized(mip, layer, aspect)` take a `TextureAspect`-derived
`InitAspect`). `TextureAspect::All` on a combined depth-stencil format
expands to both aspects; on a single-aspect format it is that aspect.
`copy_subresources` reports the aspect(s) the copy touches.

### R2 — Core: eligibility is "everything except transient"

`texture_lazy_init_eligible` returns `true` for every format, sample
count and dimension; only `TRANSIENT_ATTACHMENT` stays out (memoryless
contents are never observable). The Stage 1 unit test
`texture_lazy_init_eligible_rejects_depth_stencil_compressed_and_multisampled`
is inverted. Consequence: `init_state` is allocated for every
non-transient texture (`mips × layers × aspects` bools — a few hundred
bytes at most).

### R3 — Core: reads through bindings clear first

`append_writable_storage_texture_init_clears` becomes
`append_bound_texture_init_clears`: for every bind-group entry whose
layout kind is `Texture { .. }`, `StorageTexture { .. }` (any access),
or (tiled) `InputAttachment { .. }`, the bound view's
`[base_mip, +count) × [base_layer, +count) × aspect(view)` subresources
that are not initialized get a `ClearTexture` before the pass and are
marked (storage write access marks too, as today). Called from the
compute-pass lowering, the render-stream walker (which already recurses
into bundles), and the tiled per-draw path. External-texture bindings
are out of scope (Metal-only vendor path, planes are user-written).

### R4 — Core: attachments rewrite the load op

In the `RenderPass` lowering (`hal_render_color_target`,
`hal_render_depth_stencil_attachment`):

- Colour attachment, 1D/2D view, `load_op == Load`, subresource
  uninitialized → emit `HalRenderLoadOp::Clear` with clear colour
  `[0,0,0,0]` instead of `Load`; no `ClearTexture`. 3D views keep the
  Stage 1 `ClearTexture` of the whole mip (then `Load`).
- Depth-stencil attachment: per aspect, `depth_load_op == Load` and
  depth uninitialized → `Clear` with `clear_depth = 0.0`; same for
  stencil with `clear_stencil = 0`. A read-only aspect that is
  uninitialized is cleared the same way (Dawn does; the read-only flag
  only forbids *user* writes).
- Resolve targets are **marked** initialized (the resolve overwrites
  them) — no clear.
- Marking follows the store op: `Store` → mark initialized, `Discard` →
  mark **uninitialized**, per aspect (`depth_store_op` /
  `stencil_store_op` independently). Colour `Discard` likewise.
- Render bundles executed inside the pass do not change attachment
  marks (they cannot end the pass).

### R5 — Core: marks are committed after a successful submit

The lowering walk records intended marks (initialized / uninitialized)
into a per-submit `InitOverlay` keyed by `(texture id, mip, layer,
aspect)`; `is_initialized` queries during the walk consult the overlay
first (so read-after-partial-write inside one submit still sees the
earlier op's effect). The overlay is applied to the textures **only
when `HalQueue::submit_copies` returns `Ok`**. On a validation error,
staging failure or HAL error the overlay is dropped and no texture
state changes. `Queue::write_texture` (queue-side, not in a command
buffer) keeps its immediate mark since its copy is queued before any
command buffer of the next submit and its staging failure returns
before marking.

### R6 — HAL: `ClearTexture` executes for every format and sample count

`HalTextureClear` is unchanged in shape (`texture`, `format`, `aspect`,
`mip_level`, `base_array_layer`, `array_layer_count`); the contract is
now: zero the given subresource(s) of the given aspect for **any**
format (colour, depth, stencil, combined, compressed) and **any** sample
count. Value is always zero (`0.0` depth, `0` stencil, zero bytes for
compressed, `0` colour).

- **Metal** (`encode_texture_clear`): dispatch on the texture:
  - depth/stencil aspect or `sample_count > 1` → an **empty render
    pass** per (mip, layer): `MTLRenderPassDescriptor` with the
    subresource as the depth / stencil / colour attachment, `loadAction
    = Clear`, `storeAction = Store`, `clearDepth 0` / `clearStencil 0` /
    `clearColor (0,0,0,0)`; `renderCommandEncoderWithDescriptor` +
    `endEncoding` (Dawn `TextureMTL.mm` `ClearTexture`, renderable
    branch). A combined depth-stencil texture with `aspect == All`
    clears both attachments in one pass.
  - compressed format → the existing zero-buffer blit with **block
    math**: `bytes_per_row = blocks_wide · block_bytes`,
    `bytes_per_image = bytes_per_row · blocks_high`, using
    `HalTextureFormat::compressed_block_info`.
  - otherwise → the Stage 1 blit path, unchanged.
- **Vulkan** (`encode_texture_clear`):
  - depth / stencil aspects → `vkCmdClearDepthStencilImage` with
    `{depth: 0.0, stencil: 0}` and `aspectMask` from the requested aspect
    (reuse the discarded-aspect epilogue code), after transitioning to
    `TRANSFER_DST_OPTIMAL`; the F-138 WAW barrier follows as for colour.
  - compressed → `vkCmdCopyBufferToImage` from a zeroed transient
    buffer (block-aligned `bufferRowLength`/`bufferImageHeight` in
    texels, as `encode_buffer_to_texture` already computes), one region
    per layer.
  - colour, any sample count → `vkCmdClearColorImage` as today (valid
    for multisampled images).
- **GLES** (Tier 2): depth / stencil via the temporary-FBO path with
  `glClearBufferfv(GL_DEPTH, ..)` / `glClearBufferiv(GL_STENCIL, ..)`;
  multisample colour via an FBO with the multisample texture attached;
  compressed via `glCompressedTexSubImage2D/3D` of a zeroed block buffer.
  Anything the GL context cannot express stays a catalogued `HalError`
  (Block 67 matrix updated).
- **Noop**: records the copy (unchanged).

### R7 — Unchanged

`Queue::write_texture` / copy-side write clears; the 3D whole-mip
handling; `TRANSIENT_ATTACHMENT` exclusion; validation. GLES CTS
numbers may move (Tier 2, best-effort).

## Tests

- **Core inline (Noop)**: aspect-axis state (`mark`/`is_initialized`
  per aspect, `All` expansion, combined format independence); eligibility
  inversion; `sampled_texture_binding_read_clears_uninitialized_mips_before_pass`
  (view of mips 1..3 of a 5-mip texture → exactly those three
  `ClearTexture`s, then the pass; a second pass emits none); read-only
  storage and input-attachment variants; `depth_texture_copy_execution_submits_aspect_clear`
  (rewrites the Stage 1 negative test); combined depth-stencil copy of
  `DepthOnly` clears only depth and leaves stencil unmarked; colour
  attachment `Load` on uninitialized → the lowered `HalRenderColorTarget`
  has `Clear` + zero colour and no `ClearTexture`; depth/stencil
  attachment per-aspect rewrite; `Discard` un-marks and the next `Load`
  rewrites again; resolve target marked without clear; 3D attachment
  keeps `ClearTexture`; overlay: read-after-partial-write in one submit
  emits one clear; a failing submit (fault-injection seam from Block
  100 m2) leaves every mark untouched.
- **HAL inline** (`#[ignore]` real device, Metal and Vulkan): clear a
  `depth32float`, a `stencil8`, a `depth24plus-stencil8` (`DepthOnly`
  then `StencilOnly`), a 4× MSAA `rgba8unorm`, and a `bc1-rgba-unorm`
  (Metal + Vulkan both advertise BC on this M2 / MoltenVK) subresource
  that was first filled with a canary, then read back (copy, or
  resolve/sample for MSAA / depth) and assert zeros; other mips /
  layers keep the canary. Pure helpers (block-math byte counts,
  aspect→attachment selection) unit-tested on Noop.
- **Real-GPU e2e (Claude)**: `yawgpu/tests/e2e_{metal,vulkan}_lazy_init.rs`
  through the C ABI, mirroring the upstream CTS matrix on a small scale
  with canaries on the initialized subresources: (1) sampled read of an
  uninitialized `rgba8unorm` mip via a compute shader → zeros, canary
  mips intact; (2) `depth32float` `copyTextureToBuffer` → zeros; (3)
  `depth24plus-stencil8` stencil aspect readback → zeros; (4) 4× MSAA
  `rgba8unorm` used as `Load` attachment then resolved → zeros; (5)
  `bc1-rgba-unorm` copy → zero bytes; (6) `storeOp: Discard` then
  sampled read → zeros; (7) a partial write then a sampled read → the
  written texels survive (the F-138 inverse-bug guard, Block 97 R8).
- **CTS**: `webgpu:api,operation,resource_init,*`,
  `webgpu:api,operation,render_pass,storeOp,*`,
  `webgpu:api,operation,rendering,depth,*`,
  `webgpu:api,operation,command_buffer,copyTextureToTexture,*`,
  `webgpu:api,operation,command_buffer,image_copy,*` on Metal + MoltenVK:
  no regression vs the current run (Metal 0 fail). The port's
  `texture_zero` matrix is minimal; extending it upstream in
  webgpu-native-cts is a follow-up, not a gate.

## Slices

- **S1 core** (R1–R5, Noop tests; HAL unchanged — Metal/Vulkan clears of
  the new kinds may fail at submit until S2/S3, so S1 is Noop-gated).
- **S2 Metal HAL** (R6 Metal) + Claude's Metal e2e.
- **S3 Vulkan HAL** (R6 Vulkan) + Claude's MoltenVK e2e.
- **S4 GLES HAL** (R6 GLES; compile-verified here, Linux/Windows real-GPU
  later — Tier 2).
- **S5** CTS re-run + Phase Review.

## Known limitations (documented)

- Cross-queue races on the init state are outside the single-queue
  model (one device queue in yawgpu).
- External-texture planes are not lazily cleared (user-provided).
