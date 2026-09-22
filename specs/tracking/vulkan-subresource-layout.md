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
| S1 | — | — | — |
| S2 | — | — | — |
| S3 | — | — | — |
| S4 | — | — | — |
