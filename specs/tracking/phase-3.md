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

## P3.2 — TextureView  *(NEXT)*

`WGPUTextureView` handle + core view; dimension/format/aspect compat,
range bounds, default-view inference. Port **T26–T33**.

## P3.3 — Sampler  *(after P3.2; independent — may reorder)*

`WGPUSampler` handle + core sampler; lod/anisotropy/filter validation,
default sampler. Port **T34–T39**.

## P3.4 — QueueWriteTexture  *(after P3.3)*

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
