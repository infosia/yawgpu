# Block 163 — Subpass attachments run the regular attachment restrictions

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 0 MAJOR.
Source: parameter-propagation audit at `b772306`
(`specs/tracking/threading-audit.md`, "Propagation audit 2026-11"), candidates 01, 02, 07.
Oracle: subpass rendering (`tiled`) has no Dawn counterpart. R1 and R2 copy what yawgpu's
regular render pass already enforces (`command_encoder.rs` `validate_color_attachment`,
`validate_depth_stencil_attachment`, `validate_render_attachment_common`, which follow
Dawn `CommandEncoder.cpp` `ValidateRenderPassColorAttachment` /
`ValidateRenderPassDepthStencilAttachment`). R3 enforces the `yawgpu.h` field contract
(`YaWGPUSubpassInputAttachment.sourceSubpass` — "Producing subpass index").

## Problem

`validate_subpass_color_attachment` and `validate_subpass_depth_stencil_attachment`
(`subpass.rs`) re-implement the usage, format, sample-count, extent and op-presence checks
of the regular pass, but omit two restrictions the regular pass applies to the same view:

- a `TRANSIENT_ATTACHMENT` view must use `Clear` / `Discard` on every present aspect
  (`specs/blocks/55-tiled-rendering.md` "Transient attachment");
- a render attachment view must have the identity component swizzle.

The same view and ops are rejected by `beginRenderPass` and accepted by the subpass
encoder. Only the subpass resolve target is covered, because it calls the shared
`validate_resolve_target`.

Separately, `validate_subpass_pass_layout_descriptor_with_caps` checks that an input's
`source_subpass` is earlier and that `source_attachment` exists in the pass-wide list,
but not that the named subpass writes that attachment. Backends select the input image by
attachment slot alone, so a wrong producer index is accepted and ignored.

## Rules

R1. **Transient load/store.** For a subpass color attachment whose view usage contains
`TRANSIENT_ATTACHMENT`, `load_op` must be `Clear` and `store_op` must be `Discard`. For a
subpass depth-stencil attachment whose view usage contains `TRANSIENT_ATTACHMENT`, the
same holds for the depth ops when the format has a depth aspect and for the stencil ops
when it has a stencil aspect. Violations are validation errors at `begin`.

R2. **Identity swizzle.** A subpass color attachment view and a subpass depth-stencil
attachment view must have the identity component swizzle.

R3. **Input producer writes the source.** For every input attachment of subpass `i`,
the subpass named by `source_subpass` must write the source: its
`color_attachment_indices` contains `source_attachment`, or — for
`DEPTH_STENCIL_ATTACHMENT_INDEX` — its `uses_depth_stencil` is true. Otherwise layout
creation fails with a validation error. The existing "prior subpass" and range checks
keep their precedence (they run first).

R4. **One implementation.** R1 and R2 are not a second copy of the regular pass's checks:
the regular validators' transient-ops and swizzle checks are extracted into helpers that
take what they read (view, ops, label), and both pass types call them. The regular pass's
behaviour, error precedence and messages are unchanged. Subpass messages follow the
existing convention (the regular message with `render pass` → `subpass render pass`).

## Slices

S1: R1–R4 in `yawgpu-core` (`command_encoder.rs`, `subpass.rs`), inline tests under
`--features tiled`.

## Verification

- Inline tests in `subpass.rs`:
  - R1 color: transient view with Clear/Discard accepted; Load/Discard and Clear/Store
    each rejected. R1 depth and R1 stencil: the same three cases per aspect on a transient
    depth-stencil view, varying only that aspect's ops.
  - R2: a non-identity swizzled view rejected as a color attachment and as a
    depth-stencil attachment; the explicit identity swizzle accepted.
  - Parity: for each rejected R1/R2 case, the regular validator rejects the same view and
    ops (the test asserts both, so a future divergence fails).
  - R3: three subpasses writing color slots 0, 1 and 2; subpass 2 reading slot 1 is
    accepted with `source_subpass = 1` and rejected with `source_subpass = 0`; a depth
    input is accepted only when the named subpass has `uses_depth_stencil`.
- Agent gates: `cargo test -p yawgpu-core --features tiled --lib`,
  `cargo test -p yawgpu-core --lib`,
  `cargo test -p yawgpu --features tiled --test tiled_subpass_ffi`, clippy with and
  without `tiled`.
- Claude: workspace test; real-GPU `e2e_metal_tiled*` and `e2e_vulkan_tiled*` and the
  `tiled` examples with `--verify`, to confirm no valid pass is now rejected.

## Verification record

Shared gates for Blocks 163–167 (2026-11-28, M2): `cargo test --workspace` 1514/0, with
`yawgpu/vulkan,yawgpu/tiled` 1871/0; fmt clean; clippy clean for default,
`vulkan,tiled` and `metal,tiled`; `yawgpu-hal --lib` 395/0 (`vulkan,tiled`) and 218/0
(`metal,tiled`); all `e2e_metal_*` with `metal,tiled` 176/0; all `e2e_vulkan_*` with
`vulkan,tiled` on MoltenVK under the Khronos layer 174 passed, 1 failed (the known
`imageView2DOn3DImage` portability case; every VUID comes from that test).

`yawgpu-core --features tiled --lib` 824/0; `tiled_subpass_ffi` 8/0.
`tiled_deferred --verify` passes on Metal and MoltenVK and `tiled_msaa --verify` on
MoltenVK, so the layouts in the examples and e2e tests satisfy R3.

## Known limits

- R3 requires a writer: an attachment that no subpass writes (content from `loadOp =
  Load` only) cannot be a subpass input, and an input must name a subpass that writes the
  source, not merely an earlier one.

### Windows native Vulkan (2026-11-28)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`b22601f`. The Khronos validation layer was on for the HAL ignored, e2e, and example runs.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,513 passed, 0 failed; with `yawgpu/vulkan,yawgpu/tiled` 1,870 passed, 0 failed |
| clippy `-D warnings`, default (workspace), `vulkan,tiled`, `vulkan,shader-passthrough` and `gles` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, `yawgpu-tint`), all targets | clean |
| Vulkan HAL lib tests, `vulkan,tiled` | 395 passed, 0 failed |
| Vulkan HAL ignored tests, `vulkan,tiled` | 83 passed, 0 failed, 0 VUIDs |
| Vulkan e2e (`e2e_vulkan_*`, `vulkan,tiled,shader-passthrough`, `--ignored`) | 177 passed, 0 failed, 0 VUIDs; `e2e_vulkan_tiled` 6 of 6 |
| C examples on Vulkan (`device_info`, `compute`, `capture`; windowed `hello_triangle`, `triangle`, `triangle_passthrough`, `surface_smoke`, `tiled_msaa`, `tiled_deferred` for 8 s each; `tiled_deferred --verify`) | no error, no VUID; every windowed example configures its surface and presents |

## Phase Review

Fresh-agent review of Blocks 163–164: 0 CRITICAL, 0 MAJOR, 3 MINOR for this block.
- R3 narrows which attachments can be inputs — recorded under "Known limits".
- R1 tested on `Depth24PlusStencil8` only — fixed: the parity test also covers
  `Depth24Plus` and `Stencil8`.
- `validate_transient_attachment_ops` takes `InitAspect` to name the aspect — deferred:
  the enum already names exactly the three aspects and no behaviour depends on it.
