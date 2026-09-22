# Block 106 — Native-Vulkan verifications deferred from Blocks 68 / 71 / 94 (backlog D11, Vulkan items)

Status: **COMPLETE (2026-09-22)** — R1 `e2e_vulkan_clip_distances.rs` 2/2, R2 `e2e_vulkan_texture_component_swizzle.rs` 4/4 (native NVIDIA, validation layer 0 lines); R3 native Vulkan CTS (this host, yawgpu `4d5bc52` release `--features vulkan`, CTS `2f0fb9f`, raw, 2026-09-22): `shader,execution,shader_io,vertex_builtins:outputs,clip_distances` 8/0, `capability_checks,features,clip_distances` 368/0, `shader,validation,extension,clip_distances` 4/0, `api,operation,texture_view,texture_component_swizzle` 32,832 pass / 19,494 skip (all skips = compressed formats this GPU does not expose; every depth/stencil-format subcase passes: depth16unorm 1,197, depth24plus 1,197, depth32float 1,197, depth24plus-stencil8 1,539, depth32float-stencil8 1,539, stencil8 342) / 0 fail, `capability_checks,features,texture_component_swizzle` 855/0, `encoding,programmable,pipeline_immediate` 181/0, `encoding,cmds,setImmediates` 378/0; summary pass=34,626 skip=19,495 fail=0 crash=0; R4 records updated. Backlog **D11** "Hardware-blocked
Vulkan verifications", Vulkan-related items only: clip-distances
*execution* (Block 68), the texture-component-swizzle depth path
(Block 71 "R001") and the native-Windows immediates CTS sweep (Block 94).
The ETC2 / ASTC probes E8 / E9 (`specs/tracking/texture-compression-vulkan.md`)
stay hardware-blocked: desktop NVIDIA does not expose those families in
Vulkan. Host: Windows 11, NVIDIA RTX 5060 Ti, native Vulkan driver,
`VK_LAYER_KHRONOS_validation`; CTS: `../webgpu-native-cts` (`2f0fb9f`).

## Problem

Three verifications were deferred to "native Vulkan hardware" because
MoltenVK could not perform them:

1. **Clip distances** (Block 68): MoltenVK cannot lower the `ClipDistance`
   builtin (`specs/tracking/clip-distances.md`), so the spec's
   `e2e_vulkan_clip_distances.rs` was never written — only the Metal e2e
   and the CTS validation trees exist. Whether a negative clip distance
   actually culls on yawgpu's Vulkan path has not been observed in-repo.
2. **Texture-component-swizzle depth path** (Block 71): the core composes
   every depth/stencil view's swizzle over the WebGPU base `(d, 0, 0, 1)`
   ("R001", `yawgpu-core/src/queue.rs` `texture_view_hal_component_swizzle`)
   and the Vulkan HAL applies `VkComponentMapping` at bind time; the
   CTS-verified evidence is Metal-only. Neither
   `e2e_{metal,vulkan}_texture_component_swizzle.rs` (spec 71 slice 2)
   exists.
3. **Immediates** (Block 94): `specs/tracking/immediates-block94.md` still
   lists "Native-Vulkan (Windows RTX) confirmation on the next user-run
   sweep" as outstanding, although the CTS ledger (`webgpu-native-cts`
   `docs/FINDINGS.md` F-145) records the fix re-run 181/181 on this host on
   2026-07-03 and the 2026-09-21 whole-suite sweep on this host is
   `api/validation` 4 fail (all documented `xfail`, none in the immediates
   trees) / `api/operation` 0 fail. The item is doc drift.

## Behaviour contract

No library code changes are expected. Every rule is a test or a record.

### R1 — `yawgpu/tests/e2e_vulkan_clip_distances.rs`

A Vulkan port of `e2e_metal_clip_distances.rs` (`#![cfg(feature = "vulkan")]`,
`#[ignore]`, `real_backend_skip_reason(RealBackend::Vulkan)`):
`vulkan_adapter_advertises_clip_distances` (the adapter reports
`WGPUFeatureName_ClipDistances`, matching `shaderClipDistance`) and
`vulkan_clip_distances_cull_by_sign` (a device requested with the feature;
a full-screen triangle with `@builtin(clip_distances) = [-1.0]` leaves the
clear colour, `[1.0]` draws red; no device error). Same shader, same
readback as the Metal file; only the instance/backend selection differs.

### R2 — `yawgpu/tests/e2e_vulkan_texture_component_swizzle.rs`

Device requested with `WGPUFeatureName_TextureComponentSwizzle`; views
created with a chained `WGPUTextureComponentSwizzleDescriptor`
(`WGPUSType_TextureComponentSwizzleDescriptor`); reads through a compute
shader `textureLoad` into a storage buffer, copied to a `MAP_READ`
readback buffer:

1. `vulkan_swizzle_remaps_color_channels` — `rgba8unorm` texel
   `(0x10, 0x20, 0x30, 0x40)` written by `writeTexture`; view swizzle
   `(B, G, R, One)`; `textureLoad` on `texture_2d<f32>` →
   `(0x30, 0x20, 0x10, 0xFF)` after `pack4x8unorm` (tolerance ±1 on each
   channel for the unorm round trip).
2. `vulkan_swizzle_composes_over_the_depth_base` — `depth16unorm`
   texture written by `writeTexture` (aspect `DepthOnly`, `u16 0x4000`,
   so `d = 16384 / 65535`; `depth16unorm` is the only 32-bit-or-narrower
   depth format the WebGPU depth-stencil capability table allows as a
   copy destination — `depth32float` is copy-source only, and yawgpu
   correctly rejects a `writeTexture` into it); view swizzle
   `(G, R, A, B)`; bound as `texture_2d<f32>` through an explicit
   `UnfilterableFloat` bind group layout; expected `(0.0, d, 1.0, 0.0)`
   — the WebGPU base `(d, 0, 0, 1)` read through the user swizzle
   (Block 71 R001). Per-channel tolerance `1e-3`.
3. `vulkan_swizzle_identity_on_depth_reads_d001` — same texture,
   identity swizzle (no chain) → `(d, 0.0, 0.0, 1.0)`.
4. `vulkan_non_identity_swizzle_without_feature_is_a_device_error` —
   a device without the feature: `wgpuTextureCreateView` with a
   non-identity swizzle raises a validation error on the uncaptured-error
   sink (Block 71 gate).

### R3 — Native-Vulkan CTS confirmation

The following trees run on this host through `webgpu-native-cts`
(`build-yawgpu/Release/cts.exe --workers 8`, yawgpu built
`--release --features vulkan` from the tree under test), raw (no
expectations), with `fail = 0`, `crash = 0`:

- `webgpu:shader,execution,shader_io,vertex_builtins:outputs,clip_distances:*`
- `webgpu:api,validation,capability_checks,features,clip_distances:*`
- `webgpu:shader,validation,extension,clip_distances:*`
- `webgpu:api,operation,texture_view,texture_component_swizzle:*` (529;
  includes the depth / stencil aspect cases)
- `webgpu:api,validation,capability_checks,features,texture_component_swizzle:*`
- `webgpu:api,validation,encoding,programmable,pipeline_immediate:*`
- `webgpu:api,validation,encoding,cmds,setImmediates:*`

Result recorded in the tracking docs below with the yawgpu / CTS
revisions.

### R4 — Records

- `specs/tracking/clip-distances.md`: Vulkan execution verified on native
  hardware (e2e + CTS execution tree); the MoltenVK limitation note stays.
- `specs/tracking/texture-component-swizzle.md` and Block 71 status: the
  depth R001 path verified on native Vulkan (e2e + CTS); the Metal e2e
  file is still owed (next Mac session; CTS-verified there already).
- `specs/tracking/immediates-block94.md`: "Outstanding" native-Vulkan
  confirmation closed with the F-145 re-run and the 2026-09-21 sweep
  reference, plus this block's re-run.
- `specs/tracking/backlog.md` D11: rewritten to the remaining
  hardware-blocked items (ETC2 / ASTC probes E8 / E9); progress log entry.

## Tests

- **Real-GPU e2e (coding agent writes, Claude runs)**: R1 (2 cases) and R2
  (4 cases) green under the validation layer with zero layer lines; the
  full `--features vulkan,tiled -- --ignored` suite stays green.
- **CTS**: R3 trees, raw `fail = 0`.
- No new library `pub` items, so no new inline unit tests are required
  (CLAUDE.md principle 1 is unaffected).

## Out of scope

- ETC2 / ASTC / ASTC-sliced-3D probes (E8 / E9) — hardware-blocked here.
- E6 Vulkan limits (NVIDIA storage cap, `maxFragmentCombinedOutputResources`,
  `max_color_attachment_bytes_per_sample`) — separate item.
- The Metal swizzle e2e file (Mac session).
