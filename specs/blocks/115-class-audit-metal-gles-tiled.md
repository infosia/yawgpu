# Block 115 — bug-class audit: Metal / GLES / tiled

Status: **COMPLETE (2026-10-25)** — wave A `960805c`, F8 rework `2d371e5`,
tiled S3 `9476501`, Phase Review fixes (M1, m1–m5, R-PR4, R-PR5) in the
closing commit. GLES F5 / F6 / F8 real-GPU: `e2e_gles_class_audit.rs` (Linux NVIDIA EGL, 2026-10-25): F5 (3D storage
texture, every z slice), F6 (sRGB view of a unorm texture: clear + draw, MSAA
resolve), F8 (`frag_depth` clamped to the viewport depth range) all pass; F6
and F8 go red with their fix reverted. F5 does not: the pre-fix
`layered = FALSE` binding of an `image3D` is undefined per GL and NVIDIA still
writes every slice, so red-before is not observable on this host; accepted
(the fix is covered by its HAL unit test).
Source: `specs/tracking/class-audit-2026-10.md` (Metal / GLES / tiled
rows). Oracle: Dawn `third_party/dawn/src/dawn/native/{metal,opengl}/`.

| Slice | Rows | Files | Wave |
|---|---|---|---|
| S1 Metal | T5, T8, S8 | `yawgpu-hal/src/metal/**` | A |
| S2 GLES | F5, F6, F8 | `yawgpu-hal/src/gles/**` | A |
| S3 tiled | F3, F4, F7 | `yawgpu-core` subpass / queue lowering, `yawgpu-hal` descriptors + metal + vulkan (tiled paths) | B |
| S4 | Phase Review | | |

## Rules

- **T5** Metal buffer↔texture and texture↔texture copies of compressed
  formats clamp the copy size to the mip's real (unpadded) size at the
  texture edge (Dawn `UtilsMetal.mm` `ComputeTextureBufferCopySplit`).
- **T8** Metal buffer↔texture copies split per row when `bytesPerRow`
  exceeds Metal's per-row limit, and handle single-image copies with
  `bytesPerImage = 0` / last-row cases as Dawn does.
- **S8** Metal `nextDrawable` / drawable texture access inside an
  autoreleasepool.
- **F5** GLES storage-texture binding of a 3D view compares against the
  view's own layer count (1 for 3D) and binds `layered = TRUE`; the
  sampled-path `requires_texture_view_for` uses the same comparison.
- **F6** GLES render attachments and resolve blits honour `view_format` /
  `resolve_view_format` (attach a `glTextureView` when the formats differ,
  or return `HalError` where `glTextureView` is unavailable — catalogued).
- **F8** GLES `frag_depth` clamp to the viewport depth range: implement as
  Dawn GL does — Tint GLSL writer `depth_range_offsets` into the existing
  `tint_immediates` uniform (Dawn `ShaderModuleGL.cpp` `clampFragDepth`),
  uploaded per draw from the viewport. **Review note (wave A):** the first
  implementation wrapped the translated GLSL entry point by text rewriting
  in the HAL; that is replaced in wave B by the Tint option (the shim must
  expose it; `shader_tint.rs` sets it for GLES fragment stages that write
  `frag_depth`).
- **F3** (tiled) subpass attachments carry the view's base mip / layer /
  format to the HAL; Metal sets level / slice; Vulkan builds the view from
  them.
- **F4** (tiled) Vulkan subpasses build resolve attachments for color
  `resolve_target`.
- **F7** (tiled) subpass `setIndexBuffer` / `drawIndexed` lower an index
  buffer into the HAL subpass draw and both HALs execute it (or core
  rejects them, documented in the tiled block).

## Verification (Claude)

T5 / T8: new `e2e_metal_*` cases (12×12 BC1 mip-1 round trip; wide
`bytesPerRow` sub-rectangle) fail on the pre-fix code under
`MTL_DEBUG_LAYER` and pass after. F3 / F4 / F7: tiled e2e on Metal and
MoltenVK. GLES rows: `e2e_gles_class_audit` (see the status note above).

## S4 — Phase Review (2026-10-25)

Fresh reviewer over `960805c`, `2d371e5`, `9476501`: **0 CRITICAL / 1 MAJOR /
5 MINOR**.

| ID | Finding | Disposition |
|---|---|---|
| M1 | Metal compressed texture→texture copies clamp the extent to both mip edges; with mismatched mip sizes the source size becomes neither a block multiple nor edge-clamped (Metal rejects). Dawn `CommandBufferMTL.mm` passes the block-rounded size unclamped for texture→texture | fix: match Dawn; Claude adds an e2e with mismatched compressed mip sizes under `MTL_DEBUG_LAYER` |
| m1 | Metal row-split path keeps `bytesPerRow = max_row` for the last row of a tightly sized buffer (Metal checks `bytesPerRow × rows`) | fix: use the row's own byte count for single-row parts; Claude adds a wide-`bytesPerRow` e2e |
| m2 | tiled Vulkan: resolve-slot sentinel `u32::MAX - slot` collides with the depth-input sentinel `u32::MAX` | fix: explicit resolve flag |
| m3 | tiled Vulkan: depth framebuffer view lost `INPUT_ATTACHMENT` usage (regression from F3; VUID-…-00879 for depth input layouts) | fix: keep input-attachment usage on depth views used as input |
| m4 | GLES F5 / F6 / F8 behaviour (incl. the new `glTextureView`-unavailable `HalError`) not in the 67-gles matrix; ledger Status column empty | fix (Claude, specs) |
| m5 | no GPU coverage for compressed texture→texture, wide `bytesPerRow`, tiled resolve (F4) and tiled indexed draw (F7) | fix: Claude adds e2e (Metal + MoltenVK for tiled) |

### Phase Review fix verification (Claude, real GPU)

- `e2e_metal_class_audit::metal_bc1_texture_to_texture_mismatched_mips_round_trip`
  (first written with a 10×10 BC1 destination — not block-aligned, so the
  texture was an error object and the zeros were the test's own bug; fixed
  to 12×12 mip 1 = 6×6 → 20×20 mip 2 = 5×5, copy size 8×8). With valid
  textures the pre-fix code trips Metal API validation in
  `copyFromTexture:…sourceSize:`; the same copy between equal mip sizes round-trips. No single
  Metal `copyFromTexture` size is valid for both sides here (8 exceeds both
  edges; 5 is neither a block multiple nor the source edge; 6 exceeds the
  destination edge), and Metal drops the copy. **R-PR4:** when the source
  and destination mip edge clamps differ, copy through a temporary buffer
  (texture → buffer clamped to the source edge, buffer → texture clamped to
  the destination edge, same block layout); otherwise keep the single blit.
- `e2e_vulkan_tiled::vulkan_tiled_msaa_subpass_resolve_target_is_written`
  (new F4 e2e): data correct, but `SYNC-HAZARD-READ-AFTER-WRITE` on the
  following copy from the subpass resolve target (prior usage: the
  render-pass final layout transition, no barrier). **R-PR5:** subpass
  resolve targets get the same final / tracked layout handling and outgoing
  dependency as regular-pass resolve targets.

Final gate (Claude): `e2e_metal_class_audit` 3/3 under `MTL_DEBUG_LAYER`
(the mismatched-mip case trips Metal API validation on the pre-fix code);
Metal e2e with `metal,tiled` 134/0 incl. the tiled indexed-draw case;
MoltenVK all `e2e_vulkan_*` with `vulkan,tiled` under sync validation:
0 `SYNC-HAZARD` incl. the new tiled resolve (F4) and indexed (F7) cases,
only the pre-existing MoltenVK 3D-attachment fail; the same BC1
mismatched-mip copy also passes on MoltenVK.
