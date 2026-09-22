# Vulkan per-subresource layout tracking — Block 105 ledger (backlog A5)

Spec: `specs/blocks/105-vulkan-subresource-layouts.md`. Host: Windows 11,
native NVIDIA Vulkan driver, `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

## Baseline (2026-09-22, `97e1713`)

- `cargo test -p yawgpu --features vulkan,tiled --no-fail-fast -- --ignored`:
  104/104 (+ `shader-passthrough` 2/2), 0 validation-layer lines.
- Problems 1–4 of the spec are latent: no in-tree test exercises a
  divergent-subresource sequence, so the whole-image tracker never
  meets a mismatch. The Block 105 e2e cases are written to fail first
  under the validation layer (red-before-fix is logged per slice below).

## Slice log

| Slice | Commit | Gate | Notes |
|---|---|---|---|
| S1 | (this commit) | workspace 1093/0; clippy vulkan,tiled + fmt clean; Vulkan e2e `--ignored` 109/0, HAL `--ignored` 54/0, 0 validation lines | `layout.rs` tracker (11 unit tests), `transition_image_range` + pure `layout_barriers`, copy/clear/present sites carry exact ranges, same-image copies drop the `GENERAL` split/restore (`transition_copy_subresource` removed), `barrier_aspect_mask` / `transition_image_aspect` removed; render/subpass sites still whole-image via `transition_image` + `set(whole, ..)` (S2/S3). Review nits for the Phase Review: `mod.rs` still duplicates the `IMAGE_LAYOUT_*` constants (only `UNDEFINED` aliased); `_attachment` binding in the depth-stencil pre-pass transition (goes away in S2). |
| S2 | — | — | — |
| S3 | — | — | — |
| S4 | — | — | — |
