# Block 167 — SurfaceColorManagement is rejected explicitly

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 0 MAJOR.
Source: parameter-propagation audit at `b772306`
(`specs/tracking/threading-audit.md`, "Propagation audit 2026-11"), candidate 11.
Oracle: pinned Dawn `src/dawn/native/Surface.cpp` — `ValidateSurfaceDescriptor` returns
"SurfaceColorManagement unsupported."; a `SurfaceConfiguration` carrying the chain fails
chain validation in `SurfaceBase::APIConfigure`.

## Problem

`webgpu.h` defines `WGPUSurfaceColorManagement` (`colorSpace`, `toneMappingMode`) as an
extension of `WGPUSurfaceConfiguration`. `wgpuSurfaceConfigure` never reads the chain: a
request for Display-P3 or extended tone mapping returns no error and presents as sRGB /
standard range. No HAL backend carries a color space or tone-mapping value.

## Rules

R1. **Configure rejects the chain.** `wgpuSurfaceConfigure` with a
`WGPUSurfaceColorManagement` struct anywhere in `config.nextInChain` — whatever its
`colorSpace` and `toneMappingMode` values — is a device validation error
("surface color management is not supported"). It takes the same path as the other
configuration errors: the HAL is not configured with the request and the surface's
configured state follows the existing error behaviour.

R2. **Surface creation rejects the chain.** `wgpuInstanceCreateSurface` with a
`WGPUSurfaceColorManagement` struct in the descriptor chain yields what an invalid
surface descriptor already yields (the existing error-surface path).

R3. **No other configuration changes.** A configuration without the chain behaves as
before. Tier-independent: the rule does not depend on the backend.

## Slices

S1: R1–R3 in `yawgpu` FFI (`ffi/surface.rs`, the surface validation in `ffi/mod.rs`),
inline tests.

## Verification

- Inline FFI tests on a Noop surface: configure without the chain pops `NoError`;
  configure with the chain (SRGB + Standard, and DisplayP3 + Extended) pops a validation
  error and the test HAL capture shows no new configuration; a surface descriptor with
  the chain produces the error surface.
- Agent gates: `cargo test -p yawgpu --lib`, clippy.
- Claude: workspace test; windowed examples still configure and present.

## Verification record

Shared gates for Blocks 163–167 (2026-11-28, M2): `cargo test --workspace` 1514/0, with
`yawgpu/vulkan,yawgpu/tiled` 1871/0; fmt clean; clippy clean for default,
`vulkan,tiled` and `metal,tiled`; `yawgpu-hal --lib` 395/0 (`vulkan,tiled`) and 218/0
(`metal,tiled`); all `e2e_metal_*` with `metal,tiled` 176/0; all `e2e_vulkan_*` with
`vulkan,tiled` on MoltenVK under the Khronos layer 174 passed, 1 failed (the known
`imageView2DOn3DImage` portability case; every VUID comes from that test).

`yawgpu --lib` 338/0. `surface_smoke` on Metal configures and reaches its frame loop.

## Phase Review

Fresh-agent review of Blocks 166–167: 0 CRITICAL, 0 MAJOR, 1 MINOR for this block: on an
error surface and on a failed capability query the existing error is reported before the
color-management check. This is Dawn's order (`IsError` → capabilities → configuration
validation); no change.
