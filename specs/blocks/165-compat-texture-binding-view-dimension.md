# Block 165 — Compatibility-mode texture bindings match textureBindingViewDimension

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 0 MAJOR.
Source: parameter-propagation audit at `b772306`
(`specs/tracking/threading-audit.md`, "Propagation audit 2026-11"), candidate 04.
Oracle: pinned Dawn `src/dawn/native/BindGroup.cpp` —
`ValidateTextureBindingViewDimension`, `ValidateCompatibilityModeTextureViewArrayLayer`,
and their call sites in `ValidateSampledTextureBinding` and
`ValidateStorageTextureBinding`; `DeviceBase::HasFlexibleTextureViews`.

## Problem

On a device without `CoreFeaturesAndLimits` (compatibility mode) every texture has a
`textureBindingViewDimension`: the chained `WGPUTextureBindingViewDimension` value, or the
default derived from the texture dimension and layer count. The FFI validates the value
at creation, stores it on `WGPUTextureImpl`, and returns it from
`wgpuTextureGetTextureBindingViewDimension`. `yawgpu-core` never sees it, so bind-group
validation compares the view only with the bind-group layout: a six-layer texture declared
`Cube` is accepted in a `2DArray` binding.

## Rules

R1. **Core owns the resolved dimension.** The resolved compatibility binding view
dimension is stored on the core texture at creation (`None` on a device with
`CoreFeaturesAndLimits`, and on an error texture). The FFI getter reads it from the core
texture; the FFI keeps no second copy. Creation-time validation of the chained value
(valid enum, compatible with the texture dimension and layer count) keeps its current
behaviour and messages.

R2. **Binding dimension check.** On a device without `CoreFeaturesAndLimits`, a sampled
texture binding and a storage texture binding are each a validation error at
`createBindGroup` when the view's dimension differs from the texture's resolved
compatibility binding view dimension.

R3. **Whole-array check.** On the same devices, a sampled or storage texture binding is a
validation error when the view does not cover all array layers of its texture
(`base_array_layer != 0` or `array_layer_count != texture array layers`).

R4. **Core devices unchanged.** A device with `CoreFeaturesAndLimits` runs neither R2 nor
R3. Tier-independent: the rules do not depend on the backend.

R5. **Precedence.** R2 and R3 run after the existing layout-dimension, sample-type and
format checks of their validators, in Dawn's order: R2 then R3 for a sampled binding,
R3 then R2 for a storage binding.

## Slices

S1: R1–R5 in `yawgpu-core` (`texture.rs`, `bind_group.rs`) and `yawgpu`
(`ffi/device.rs`, `ffi/texture.rs`), inline tests.

## Verification

- Inline core tests on a compatibility-level Noop device, six-layer square 2D texture:
  - declared `Cube`: a `Cube` view in a `Cube` layout binding accepted; a `2DArray` view
    in a `2DArray` layout binding rejected (R2);
  - default (`2DArray`): the `2DArray` binding accepted, the `Cube` binding rejected;
  - a `2DArray` view of layers 1..3 on a `2DArray` texture rejected (R3);
  - the storage counterpart with a storage-eligible format: declared `2DArray`, a
    single-layer `2D` view in a `2D` storage binding rejected, the full `2DArray` view
    accepted;
  - the same bindings on a core-level device all accepted (R4).
- Inline FFI test: the getter still returns defaults, the explicit value, and `Undefined`
  for core devices and error textures; a C-ABI bind group with the mismatched view pops a
  validation error.
- Agent gates: `cargo test -p yawgpu-core --lib`, `cargo test -p yawgpu --lib`, clippy.
- Claude: workspace test; CTS runs in core mode and does not exercise these rules.

## Verification record

Shared gates for Blocks 163–167 (2026-11-28, M2): `cargo test --workspace` 1514/0, with
`yawgpu/vulkan,yawgpu/tiled` 1871/0; fmt clean; clippy clean for default,
`vulkan,tiled` and `metal,tiled`; `yawgpu-hal --lib` 395/0 (`vulkan,tiled`) and 218/0
(`metal,tiled`); all `e2e_metal_*` with `metal,tiled` 176/0; all `e2e_vulkan_*` with
`vulkan,tiled` on MoltenVK under the Khronos layer 174 passed, 1 failed (the known
`imageView2DOn3DImage` portability case; every VUID comes from that test).

`yawgpu-core --lib` 746/0; `yawgpu --lib` 338/0.

Surface textures are created with the device's feature set (`Texture::from_hal` takes
it), so their resolved dimension and their format capabilities follow the device, as
those of every other texture do.

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

Fresh-agent review: 0 CRITICAL, 0 MAJOR, 3 MINOR.
- Two HAL-wrapping constructors, one without a caller — fixed: a single
  `Texture::from_hal` that takes the device features.
- No test for the surface-texture dimension — fixed:
  `current_texture_binding_view_dimension_matches_device_feature_level`.
- Surface-texture format capabilities now follow the device features — recorded above.
