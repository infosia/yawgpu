# Block 36 — External textures (`texture_external`)

Surfaced by the WebGPU-CTS port (external-CTS finding **F-060**,
`render_pipeline,misc:external_texture`). A WGSL `texture_external` binding lets
a shader sample a 1–3 plane YUV/RGBA image (e.g. decoded video) with implicit
colour-space conversion. Status legend: ☐ todo ◐ partial ☑ done.

**Hybrid surface, NOT a cargo feature.** Unlike the `tiled` / `shader-passthrough`
vendor blocks, external textures are a **core WebGPU** capability, so the binding
model + validation are **always on** (no opt-in feature, no `#[cfg]` gate). Only
the *creation* of an `WGPUExternalTexture` is a yawgpu **vendor extension**, because
the canonical `webgpu.h` declares the handle/binding types but states external-texture
creation is "extremely implementation-dependent and not defined in this header". This
mirrors wgpu, whose creation API (`Device::create_external_texture(desc, planes)`,
`wgpu/src/api/device.rs:370`) is likewise a non-canonical, create-from-plane-views
method (the JS `GPUDevice.importExternalTexture(video)` is web-only).

`yawgpu.h` defines **`YAWGPU_HAS_EXTERNAL_TEXTURE 1`** (always) so C clients can
`#ifdef`-detect the vendor creation entry point, mirroring `YAWGPU_HAS_SHADER_PASSTHROUGH`.
This is the same "standard-header surface + intentionally ungated" shape as the
`WGPUTextureUsage_TransientAttachment` bit (see Block 55): both live in the base
`webgpu.h`, are honored unconditionally, and are *not* behind a cargo feature — a
deliberate choice, not a missing gate.

**Backend support — Metal-only, matching wgpu.** External-texture *sampling*
(the codegen that lowers `texture_external` to plane textures + a params buffer)
is implemented on **Metal only**, exactly as in wgpu — wgpu's external-texture
shader path is unimplemented on both naga's SPIR-V backend and `wgpu-hal/vulkan`.
naga's SPIR-V backend does not lower `naga::ImageClass::External`, so the **Vulkan**
backend rejects an external-texture pipeline with a clean **`GPUInternalError`**
(never a panic, never a silent fake-`texture_2d` rewrite). The descriptor itself is
*valid* WebGPU (binding model + validation are core and backend-independent — see the
Tier-independent core-validation rule in `CLAUDE.md`); the rejection is a HAL-level
code-generation limitation surfaced honestly as an internal error. This is strictly
better than wgpu, which `unimplemented!()`-panics on the same Vulkan path. GLES
(Tier 2) likewise `HalError`s at execution (catalogue in `67-gles-backend.md`).

## Surface

**Canonical `webgpu.h` (already present):**
- `WGPUExternalTexture` (opaque handle), `wgpuExternalTextureAddRef`/`Release`.
- `WGPUExternalTextureBindingLayout` (chained on a `WGPUBindGroupLayoutEntry`).
- `WGPUExternalTextureBindingEntry` (chained on a `WGPUBindGroupEntry`).
- `WGPUSType_ExternalTextureBindingLayout`.

**yawgpu vendor extension (`yawgpu.h`, new — Slice 2):**
- `yawgpuDeviceCreateExternalTexture(device, *YaWGPUExternalTextureDescriptor, planes…)`
  mirroring wgpu's `ExternalTextureDescriptor` (`wgpu-types/src/texture/external_texture.rs`):
  `format` (`Rgba` | `Nv12` | `Yu12`), `yuv_conversion_matrix: [f32;16]`,
  `gamut_conversion_matrix: [f32;9]`, `src_/dst_transfer_function`,
  `sample_transform: [f32;6]`, `load_transform: [f32;6]`, `width`/`height`, plus the
  plane `WGPUTextureView`s.

**WGSL:** `texture_external` (`naga::ImageClass::External`, gated by
`naga::valid::Capabilities::TEXTURE_EXTERNAL`); `textureLoad`/`textureSampleBaseClampToEdge`.

## Architecture (mirrors wgpu)

> **Tint migration (2026-06-27).** The frontend is now Tint, whose multiplanar
> model differs from naga's: **2 planes** (`plane0`/`plane1`) + a Tint-defined
> `tint_ExternalTextureParams` UBO (`metadata`), wired via
> `tint::Bindings::external_texture = ExternalMultiplanarTexture{metadata, plane0,
> plane1}`. **Slice A DONE** (commit 279f9a4): `texture_external` shaders generate
> multiplanar MSL through Tint (the HAL maps plane0=metal_index, plane1=metal_index+1,
> metadata=ext_params_buffer_slot). The Metal HAL must populate Tint's params layout
> — `numPlanes, doYuvToRgbConversionOnly, yuvToRgbConversionMatrix:mat3x4,
> src/dstTransferFunction:{mode,A..G}, gamutConversionMatrix:mat3x3,
> sample/loadTransform:mat3x2, samplePlane0/1Rect{Min,Max}:vec2f, apparentSize:vec2u,
> plane1CoordFactor:vec2f, ootfParam:vec4f` (see
> `third_party/dawn/src/tint/lang/core/ir/transform/multiplanar_external_texture.cc`)
> — **not** wgpu's 208-byte struct below. The naga description that follows is the
> historical Slice-1 record. Remaining Slices B–D (create API, runtime binding, HAL
> params + e2e) target Tint's layout. Vulkan keeps honest GPUInternalError rejection.

One external texture lowers to **3 plane `texture2d<float>` + 1 `_params` uniform
buffer** (naga `back/msl`, `back/spv`; `ExternalTextureNameKey::{Plane(0..2), Params}`).
The HAL allocates 3 texture slots + 1 buffer slot per external texture and writes them
into naga's `BindTarget::external_texture` at pipeline-layout creation; at bind-group
creation it binds the plane views + params buffer at those slots. `_params` is the
`#[repr(C)]` 208-byte `ExternalTextureParams` (wgpu `device/resource.rs:90-173`):
`yuv_conversion_matrix[16]` + `gamut_conversion_matrix[12]` +
`src_/dst_transfer_function` + `sample_transform[6]` + `load_transform[6]` +
`size[2]` + `num_planes` + pad. Plane counts: `Rgba`=1 (4-component),
`Nv12`=2 (R8 + Rg8), `Yu12`=3 (all R8); planes must be filterable-float, 2D,
non-multisampled, `TEXTURE_BINDING`.

## Rules

Slice 1 (validation/codegen — closes F-060; exercised by inline unit tests +
the F-060 CTS case on real Metal + Vulkan/MoltenVK) — **DONE (commit 3665178;
SPIR-V honest-rejection follow-up)**:

- **R1** ☑ The WGSL frontend compiles `texture_external` (naga `TEXTURE_EXTERNAL`
  capability enabled); a `texture_external` shader does not become an error module.
  *(F-060.)*
- **R2** ☑ `naga::ImageClass::External` reflects to a dedicated external-texture
  binding kind; auto-layout derives an `ExternalTexture` bind-group-layout entry.
- **R3** ☑ An external-texture bind-group-layout entry counts **4 sampled textures
  + 1 sampler + 1 uniform buffer** toward the per-stage binding limits (mirrors wgpu
  `binding_model.rs:497-508`); over-limit is rejected.
- **R4** ☑ Explicit-layout compatibility: a `texture_external` shader binding is
  compatible only with an `ExternalTexture` layout entry (exact), incompatible with
  any other kind.
- **R5** ☑ The FFI `WGPUExternalTextureBindingLayout` (chained) maps to the
  external-texture binding-layout kind in `wgpuDeviceCreateBindGroupLayout`.
- **R6** ☑ **Metal:** MSL codegen lowers the external texture (3 planes + params) with
  the HAL-assigned `BindTarget::external_texture` slots; a render/compute pipeline
  binding a `texture_external` validates and compiles. **Vulkan:** naga's SPIR-V backend
  does not lower `ImageClass::External`, so `generate_spirv` rejects external-texture
  pipelines with a clean `GPUInternalError` (`"external textures are not supported on
  the Vulkan backend"`) — no panic, no fake rewrite — matching wgpu's Metal-only support.
  Regression: `yawgpu/tests/e2e_vulkan_external_texture.rs` (real MoltenVK: asserts the
  internal error fires and no panic occurs); Metal coverage via the F-060 CTS case.

### Tint `tint_ExternalTextureParams` UBO layout (Slice 2 target — replaces the naga 208-byte struct)

The runtime must upload **Tint's 288-byte `tint_ExternalTextureParams_packed_vec3`** host struct
(the layout Tint generates for the host to fill + unpack in-shader), NOT wgpu's 208-byte struct and
NOT the naïve field-order layout (which ignores MSL alignment). AUTHORITATIVE source: the Dawn
golden `third_party/dawn/test/tint/builtins/textureLoad/texture_external_param.wgsl.expected.msl`
(`struct tint_ExternalTextureParams_packed_vec3` with explicit `/* 0xNNNN */` byte offsets). Layout:

| off | field | type | size | notes |
|---|---|---|---|---|
| 0 | `numPlanes` | u32 | 4 | |
| 4 | `doYuvToRgbConversionOnly` | u32 | 4 | |
| 8 | (padding) | — | 8 | mat3x4 needs 16-align |
| 16 | `yuvToRgbConversionMatrix` | float3x4 | 48 | 3 cols × vec4, 12 floats direct |
| 64 | `srcTransferFunction` | TransferFunctionParams | 32 | |
| 96 | `dstTransferFunction` | TransferFunctionParams | 32 | |
| 128 | `gamutConversionMatrix` | **packed_vec3[3]** | 48 | 3 cols × 3 f32, **stride 16** (4B pad/col) |
| 176 | `sampleTransform` | float3x2 | 24 | 3 cols × vec2, stride 8 |
| 200 | `loadTransform` | float3x2 | 24 | 3 cols × vec2, stride 8 |
| 224 | `samplePlane0RectMin` | vec2f | 8 | |
| 232 | `samplePlane0RectMax` | vec2f | 8 | |
| 240 | `samplePlane1RectMin` | vec2f | 8 | |
| 248 | `samplePlane1RectMax` | vec2f | 8 | |
| 256 | `apparentSize` | vec2u | 8 | |
| 264 | `plane1CoordFactor` | vec2f | 8 | |
| 272 | `ootfParam` | vec4f | 16 | |

Total **288 bytes**. `TransferFunctionParams { mode:u32 @0, A,B,C,D,E,F,G:f32 @4..28 }` = 32 bytes.
**Gotcha (cost a real bug):** the gamut matrix is a PACKED-vec3 array (column stride 16, only 12 bytes
used/col), and every field after `doYuvToRgbConversionOnly` is shifted by the 8-byte mat3x4 alignment
pad — the field-declaration order in `multiplanar_external_texture.cc` does NOT give the byte offsets;
only the `_packed_vec3` golden does. Verified end-to-end: an identity YUV matrix on a 2-plane Nv12
external texture round-trips (Y,U,V)→(R,G,B). The Tint binding model is **2 planes** (plane0/plane1)
+ the params UBO; for single-plane `Rgba`, `numPlanes=1` and plane1 may alias plane0.

Slice 2 (resource + create + runtime binding — Metal-only, mirrors wgpu; inline unit tests +
GPU e2e authored by Claude):

- **R7** ☑ `yawgpuDeviceCreateExternalTexture(desc)` validates plane count vs `format`
  (`Rgba`=1, `Nv12`=2), and each plane view's sample type (filterable-float), dimension (2D),
  non-multisampled, `TEXTURE_BINDING` usage; mismatches route to the device error sink.
  (`yawgpu-core/src/external_texture.rs`, `yawgpu/src/ffi/external_texture.rs`.)
- **R8** ☑ Creation builds the **296-byte Tint `tint_ExternalTextureParams`** (above) from the
  descriptor and uploads it to a `UNIFORM | COPY_DST` buffer; the resource is `Arc`-handle managed
  (`planes: ArrayVec<TextureView,2>` + `params: Buffer`), `Drop` releases. Offsets unit-tested
  against the table above.
- **R9** ☑ A `WGPUExternalTextureBindingEntry` in `wgpuDeviceCreateBindGroup` binds the
  external texture; at draw/dispatch plane0 → `metal_index`, plane1 → `metal_index+1` (RGBA aliases
  plane1=plane0), params → `ext_params_buffer_slot` (the Slice-A slot assignment), per-stage indices
  honoured (`bind_group.rs` + `queue.rs hal_bind_resources` + `HalBoundExternalTexture` +
  `metal/encode.rs`). **Metal** only. **Vulkan: deterministically rejected at the core level** (commit
  0985538): although Tint's SPIR-V writer DOES lower `texture_external` (naga could not), whether a
  driver accepts the resulting multiplanar SPIR-V is **GPU-divergent** — native NVIDIA compiled it,
  Mesa rejected it, MoltenVK fails SPIR-V→MSL on the argument-buffer base type. yawgpu's posture is
  wgpu's (external textures are Metal-only), so `select_{render,compute}_shader_source` detects the
  external-texture binding and returns a backend error (`ErrorKind::Internal`, "external textures are
  not supported on the Vulkan backend") **before any SPIR-V reaches a driver** — making the rejection
  device-independent and the `e2e_vulkan_external_texture` contract hold on every GPU. (Re-homing
  external textures onto Vulkan would require setting up the multiplanar plane/params bindings on the
  Vulkan HAL, not just relying on the driver to accept Tint's SPIR-V.)
- **R10** ☑ End-to-end on real **Metal** (`yawgpu/tests/e2e_metal_external_texture.rs`): **Rgba**
  passthrough (`doYuvToRgbConversionOnly=1`) round-trips the source colour, and **Nv12** (2-plane
  R8 luma + RG8 chroma) with an identity `yuvToRgbConversionMatrix` recombines (Y,U,V)→(R,G,B). The
  Nv12 test exposed a real params-layout bug — the 288-byte `tint_ExternalTextureParams_packed_vec3`
  offsets (above) were initially derived from field order and were misaligned; now matched to the
  Dawn golden. (Color-managed Nv12 with real BT.601/709 + transfer/gamut params is a further refinement.)

## Async

None specific — `wgpuDeviceCreateExternalTexture` is synchronous (like other create
fns). Noop accepts the descriptor/planes and performs no GPU work.

## Open questions

- Exact `YaWGPUExternalTextureDescriptor` field encoding for the C ABI (matrices as
  flat `float` arrays mirroring wgpu's column-major layout).
- Whether to expose `Yu12`/`Nv12` planar formats day-1 or start `Rgba`-only and widen.
- GLES (Tier 2) mapping: external textures likely `HalError` (no clean GLES 3.1 path);
  catalogue in `67-gles-backend.md` when GLES bring-up reaches this.
