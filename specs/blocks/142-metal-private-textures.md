# Block 142 — Metal textures use private storage

Status: **IN PROGRESS**.
Source: backlog A25, started by user request on 2026-11-05.
Oracle: Dawn at the pinned `third_party/dawn`, `src/dawn/native/metal/TextureMTL.mm`
lines 236-242.

## Problem

`yawgpu-hal/src/metal/texture.rs` picks the storage mode of every texture it creates:
`Memoryless` for transient attachments, `Private` for multisampled textures, and
`Shared` for everything else. Dawn creates every texture `Private`, and uses
`Memoryless` for a transient attachment only when the device supports
`MTLGPUFamilyApple2`.

`Shared` costs three things:

- On Apple GPUs, only `Private` textures get lossless compression, so render targets and
  sampled textures use more memory bandwidth than in Dawn.
- On Intel and AMD Macs, some formats (depth and stencil among them) do not allow
  `Shared`, and `Shared` textures are slower to access from the GPU.
- `Memoryless` exists only on Apple GPUs. yawgpu requests it for every transient
  attachment, so a transient attachment on an Intel or AMD Mac does not get a valid
  storage mode.

yawgpu never reads or writes a texture from the CPU: `writeTexture`, texture readback,
and lazy zero-initialization all go through buffers and blit encoders. Nothing depends
on `Shared`.

## Rules

R1. **Private by default.** Every Metal texture that yawgpu creates uses
`MTLStorageModePrivate`, whatever its sample count, format, dimension, or usage, except
as R2 says.

R2. **Memoryless only on Apple GPUs.** A texture with the transient-attachment usage uses
`MTLStorageModeMemoryless` when the device supports `MTLGPUFamilyApple2`, as Dawn does.
Otherwise it uses `Private`.

R3. **No CPU access.** No code path maps a texture or calls `replaceRegion` or
`getBytes` on it. If the coding agent finds one, it reports it instead of keeping
`Shared` for it.

R4. **Unchanged.** Buffers, query sets, surface textures (owned by `CAMetalLayer`), and
every other backend.

## Tests

- Unit (HAL, Metal, pure): the storage-mode choice as a pure function of "transient" and
  "Apple2 supported": transient with Apple2 → `Memoryless`; transient without Apple2 →
  `Private`; non-transient, single-sample and multisample → `Private`.
- Metal `#[ignore]` (coding agent writes it; Claude runs it on the M2): created textures
  report `MTLStorageModePrivate` for a sampled 2D color texture, a depth texture, a BC
  texture, and a 3D texture, and `Memoryless` for a transient attachment.
- Claude: the full Metal e2e and every other ignored test; CTS on Metal
  `webgpu:api,*` and `webgpu:shader,execution,*` must not regress; a bench comparison
  against the Run 7 baseline for the texture and render cases.

## Out of scope

- Intel and AMD Mac verification. This machine has an Apple GPU only.
- Heaps, aliasing, or `MTLHazardTrackingMode` changes.

## Slices

- **S1 (coding agent):** R1-R4 with the unit test and the ignored test.
- **S2 (Claude):** e2e, CTS, bench, Phase Review.
