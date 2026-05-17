# Phase 3 — Texture / TextureView / Sampler

Status: **in progress** (P3.1a active). Rules: `../blocks/20-texture-
sampler.md`. Roles/loop: `../reference/workflow.md`. Gate (permanent):
`cargo test --workspace` + `cargo clippy --workspace --all-targets --
-D warnings` green on Noop. **Phase ends with the mandatory Phase Review**
(`reference/workflow.md` → "Phase Review").

5 slices. Deferred T54–T56 (→P6) and shader/storage (→P5) are out of
Phase 3.

## P3.1a — Texture creation core + reflection + lifetime  *(☑ DONE)*

Done: `NoopTexture`/`HalTexture` (alloc-counted); core `TextureUsage`/
`TextureDimension`/`TextureFormat`(opaque)/`Extent3d`/`Texture` +
`validate_texture_descriptor` (T1–T4,T6–T19,T23 first-match-wins) +
`max_texture_mips`; error-texture model, idempotent `destroy`; FFI
`wgpuDeviceCreateTexture`/`Destroy`/8 getters/`Release`/`AddRef`; conv
usage/dimension/format/extent maps. Format capability deliberately opaque
(P3.1b). T1–T20(non-format),T23,T25,T57–T65 ported in
`yawgpu/tests/texture_creation_validation.rs` (8), gate green (70 tests).
Committed `phase-3: P3.1a`.

#### (original detail)

`HalTexture`/`NoopTexture` (alloc-counted), core `Texture` (Arc; usage,
dimension, size, format-as-opaque-enum, mip, sampleCount; error/destroyed
flags) + non-format-capability validation. `wgpuDeviceCreateTexture`,
`wgpuTextureDestroy` (idempotent), getters, error-texture model. Port
**T1–T4, T6–T20(non-format), T23, T25, T57–T65**. Format-capability rules
(T5/T20/T21/T22/T24/T52/T53) are P3.1b — for now accept any non-Undefined
format opaquely (T24 itself is P3.1b).

## P3.1b — Format capability table  *(☑ DONE)*

Done: core `FormatCaps`/`FormatAspects` + builder ctors; `TextureFormat
::caps()`/`is_undefined()` populated from Dawn `Format.cpp` for the
Phase-3 set (unknown ⇒ conservative renderable color); format rules
wired into `validate_texture_descriptor` (T24,T5,T20,T21,T22/T52; T53 via
T7). T24/T5/T20/T21/T22/T52/T53 ported in
`yawgpu/tests/texture_format_validation.rs` (7, incl. caps sanity), gate
green (77 tests). Committed `phase-3: P3.1b`. Caps-table approximation
recorded in block 20 (refine P4/P5; flag in Phase 3 Review).

#### (original detail)

Core `TextureFormat` + capability records from Dawn `Format.cpp` (block 20
design). Port **T24, T5, T20, T21, T22, T52, T53**.

## P3.2 — TextureView  *(☑ DONE)*

Done: core `TextureView`/`TextureViewDescriptor`/`TextureViewDimension`/
`TextureAspect`; `TextureFormat::srgb_pair`; `Texture` captures
`view_formats` + `is_view_format_compatible`; `create_view` with
default-view inference + `validate_texture_view_descriptor`
(T26–T33: zero/undefined, overflow-checked ranges, dimension matrix,
sRGB-pair/viewFormats compat, aspect via caps); error-view model. FFI
`WGPUTextureViewImpl` + `wgpuTextureCreateView`/`Release`/`AddRef`; conv
view descriptor/dimension/aspect + viewFormats capture. T26–T33 ported
in `yawgpu/tests/texture_view_validation.rs` (6), gate green (83 tests).
Committed `phase-3: P3.2`.

#### (original detail)

`WGPUTextureView` handle + core view; dimension/format/aspect compat,
range bounds, default-view inference. Port **T26–T33**.

## P3.3 — Sampler  *(☑ DONE)*

Done: `NoopSampler`/`HalSampler`; core `AddressMode`/`FilterMode`/
`MipmapFilterMode`/`CompareFunction`/`SamplerDescriptor`/
`ResolvedSamplerDescriptor` (webgpu.h defaults: lod 0/32, anisotropy 1,
Nearest, ClampToEdge); `validate_sampler_descriptor` (T34/T35 finite,
T36 ≥1, T37 anisotropy>1⇒all-Linear, first-match-wins); error-sampler
model. FFI `WGPUSamplerImpl` + `wgpuDeviceCreateSampler`(NULL⇒defaults)/
`Release`/`AddRef`; conv address/filter/mipmap/compare maps. T34–T39
ported in `yawgpu/tests/sampler_validation.rs` (6, incl. compare-not-an-
error sanity), gate green. Committed `phase-3: P3.3`.

#### (original detail)

`WGPUSampler` handle + core sampler; lod/anisotropy/filter validation,
default sampler. Port **T34–T39**.

## P3.4 — QueueWriteTexture  *(☑ DONE — slices complete; Phase Review next)*

Done: core `Origin3d`/`TexelCopyBufferLayout`/`validate_queue_write_
texture` + `subresource_size` (mip-shift, dim-aware) + Dawn-style
`required_bytes_in_texel_copy` (overflow-checked); T40–T51 (CopyDst/
live/sampleCount/mip, aspect-vs-caps, origin+extent bounds, 2D one
layer, bytesPerRow non-256, rowsPerImage, dataSize, offset overflow).
FFI `wgpuQueueWriteTexture` decodes destination/dataLayout/writeSize
(null struct ⇒ device error), reuses `FormatCaps`; conv `Origin3d`/
`TexelCopyBufferLayout` (`WGPU_COPY_STRIDE_UNDEFINED`⇒None). T40–T51
ported in `yawgpu/tests/queue_write_texture_validation.rs` (7), gate
green (23 binaries). Committed `phase-3: P3.4`.

#### (original detail)

`wgpuQueueWriteTexture` arg/layout/bounds/aspect validation reusing
texture + format-texel info. Port **T40–T51**. Closes Phase 3 (then Phase
Review).

## Phase 3 exit criteria

- T1–T53, T57–T65 covered by ported Rust tests green on Noop; gate clean;
  CI green.
- `dawn-test-mapping.md`: `TextureValidationTests` ☑,
  `TextureViewValidationTests` ☑, `SamplerValidationTests` ☑,
  `QueueWriteTextureValidationTests` ☑, `StorageTextureValidationTests` ◐
  (creation rules only; shader→P5), `TextureSubresourceTests` ☐ Defer→P6.
- One commit per slice (`phase-3: <slice> — <short>`).
- **Mandatory Phase 3 Review** (fresh no-context reviewer; CRITICAL/MAJOR
  fixed) before Phase 3 can be marked COMPLETE; logged in
  `tracking/phase-3-review.md`.
