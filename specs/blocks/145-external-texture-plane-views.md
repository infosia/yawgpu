# Block 145 — External texture planes are bound as their views

Status: **COMPLETE (2026-11-05)**. Phase Review: 0 CRITICAL, 0 MAJOR, 3 MINOR; all resolved,
see below.
Source: backlog A28 (Block 144 Phase Review m3), started by user request on 2026-11-05.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`BindGroup.cpp` (an external-texture entry expands into `plane0`/`plane1` bindings of the
external texture's texture views, `ExternalTextureBase::GetTextureViews`), and
`metal/CommandBufferMTL.mm` (a sampled texture binding binds the view's `MTLTexture`).

## Problem

Lowering an external-texture binding (`yawgpu-core/src/queue.rs`, the
`MetalBindingKind::ExternalTexture` arm) fills `HalBoundExternalTexture` with
`plane0.texture().hal()` and `plane1.texture().hal()`: the planes' whole textures. The
plane views' base mip level, base array layer, format, aspect, and swizzle are dropped.
`HalBoundExternalTexture` carries one `format` and one `dimension`, both taken from
plane 0. The Metal HAL (`encode_compute_external_texture`,
`encode_render_bind_external_texture` in `yawgpu-hal/src/metal/encode.rs`) binds the
whole `MTLTexture` of each plane.

External-texture validation (`yawgpu-core/src/external_texture.rs`) accepts a 2D plane
view with one mip level, at any base mip level, and a 2D view of any layer of a
2D-array texture. For such a view the shader samples mip 0 / layer 0 of the texture
instead of the view's subresource. Since Block 144 the lazy clear covers the view's
subresource, so the sampled subresource can also be uninitialized memory.

Sampled `Texture` bindings do not have this problem: `HalBoundTexture` carries the view
description, and the Metal HAL makes a texture view from it (`metal_texture_view`).

## Rules

R1. **Planes are views.** `HalBoundExternalTexture` describes each plane as a view: the
texture plus that plane's view format, dimension, base mip level, mip level count, base
array layer, array layer count, aspect, and swizzle, as `HalBoundTexture` does for a
sampled texture. Plane 1 has its own description; for a one-plane external texture it
repeats plane 0's. Reuse `HalBoundTexture` (or a shared view-description type) rather
than duplicating its fields.

R2. **Metal binds the views.** The Metal HAL binds each plane through the same
view-creation path as a sampled texture binding (`metal_texture_view`), for compute and
render, for every stage slot it binds today.

R3. **Unchanged.** External texture validation, the parameter buffer and its slots, the
Metal slot assignment, and the Vulkan behaviour (pipelines with external textures are
rejected; the Vulkan HAL only retains the planes' resources until the submission
completes, which must keep working with the new type). The GLES and Noop HALs keep
compiling and retaining what they retain today.

## Tests

- Unit (core, Noop): lowering an external-texture binding whose plane 0 view has
  `base_mip_level = 1`, and one whose plane view is layer 1 of a 2D-array texture,
  produces a `HalBoundExternalTexture` whose plane descriptions carry those values; an
  NV12 external texture carries plane 1's own format.
- Unit (HAL): any pure helper added for the plane view description.
- Claude, Metal e2e (`e2e_metal_external_texture.rs`): an RGBA external texture over a
  view of mip 1 of a two-level texture samples mip 1's colour, not mip 0's; an RGBA
  external texture over a view of layer 1 of a two-layer 2D-array texture samples
  layer 1's colour. Both fail before this block.
- Claude: the full Metal e2e and every other ignored test; the Vulkan e2e under MoltenVK
  (the external-texture rejection test); CTS on Metal `webgpu:api,*` and
  `webgpu:shader,execution,*` must not regress.

## Out of scope

- External textures on Vulkan and GLES.
- Caching Metal texture views across submissions.

## Slices

- **S1 (coding agent):** R1-R3 with the unit tests.
- **S2 (Claude):** e2e, CTS, Phase Review.

## Verification (2026-11-05)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,434 passed, 0 failed |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough` | clean |
| HAL Metal lib and ignored tests | 200 passed; 66 passed |
| `metal_external_texture_samples_plane_view_subresource` (new) | passes for a view of mip 1 of a two-level texture and a view of layer 1 of a two-layer 2D array, through a fragment shader and a compute shader. Against `4fd45fb` (before this block) both paths sample mip 0's colour and fail |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled` | 173 passed, 0 failed |
| All `e2e_vulkan_*`, MoltenVK, Khronos layer | 154 passed, 1 failed (the known `imageView2DOn3DImage` portability case); `vulkan_external_texture_pipeline_reports_internal_error_without_panic` passes, re-run after the Phase Review fixes with no validation output |
| webgpu-native-cts, Metal, `webgpu:api,*` | pass 589,574, skip 101,793, fail 0, crash 0, before and after the Phase Review fixes; identical to Block 144 |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*` | pass 857,321, skip 25,860, fail 0, crash 0, xfail 72 (F-160); pass + xfail identical to Block 144 |

`webgpu:shader,execution,*` and the full MoltenVK run used the library before the Phase
Review fixes, which remove duplicated slot fields and reuse one Metal view for a
one-plane external texture.

### Windows native Vulkan (2026-11-05)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`ebd97f8`. The Khronos validation layer was on for the HAL and e2e runs. External textures
are Metal-only, so on Vulkan this checks that the shared core lowering and the Vulkan HAL
changes break nothing.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,434 passed, 0 failed |
| clippy `-D warnings`, default (workspace) and `vulkan,tiled` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`), all targets | clean |
| Vulkan HAL ignored tests | 82 passed, 0 failed, 0 VUIDs |
| Vulkan e2e (every integration test, `vulkan,tiled,shader-passthrough`, `--ignored`) | 171 passed, 0 failed, 0 VUIDs, including `vulkan_external_texture_pipeline_reports_internal_error_without_panic` |

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Each plane's binding slots were stored twice, in `HalBoundExternalTexture` and in the plane's `HalBoundTexture`, and the Metal HAL read only the outer copy | The outer slot fields are removed; the Metal HAL reads the plane's own slots, and the lowering tests assert them |
| m2 | A one-plane external texture made two identical Metal views per bind | Plane 1 reuses plane 0's view when the texture and view description match (`texture_views_match`, unit-tested) |
| m3 | The e2e exercised only the fragment path | The e2e also loads the plane view in a compute shader |
