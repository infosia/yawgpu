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
method (the JS `GPUDevice.importExternalTexture(video)` is web-only). **Vulkan + Metal
are Tier-1**; GLES (Tier 2) may `HalError` at execution (catalogue in
`67-gles-backend.md`).

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
the F-060 CTS case on real Metal + Vulkan/MoltenVK):

- **R1** ☐ The WGSL frontend compiles `texture_external` (naga `TEXTURE_EXTERNAL`
  capability enabled); a `texture_external` shader does not become an error module.
  *(F-060.)*
- **R2** ☐ `naga::ImageClass::External` reflects to a dedicated external-texture
  binding kind; auto-layout derives an `ExternalTexture` bind-group-layout entry.
- **R3** ☐ An external-texture bind-group-layout entry counts **4 sampled textures
  + 1 sampler + 1 uniform buffer** toward the per-stage binding limits (mirrors wgpu
  `binding_model.rs:497-508`); over-limit is rejected.
- **R4** ☐ Explicit-layout compatibility: a `texture_external` shader binding is
  compatible only with an `ExternalTexture` layout entry (exact), incompatible with
  any other kind.
- **R5** ☐ The FFI `WGPUExternalTextureBindingLayout` (chained) maps to the
  external-texture binding-layout kind in `wgpuDeviceCreateBindGroupLayout`.
- **R6** ☐ MSL and SPIR-V codegen lower the external texture (3 planes + params) with
  the HAL-assigned `BindTarget::external_texture` slots; a render/compute pipeline
  binding a `texture_external` validates and compiles on Metal and Vulkan.

Slice 2 (resource + create + runtime binding — full wgpu-parity; inline unit tests +
GPU e2e authored by Claude):

- **R7** ☐ `yawgpuDeviceCreateExternalTexture(desc, planes)` validates plane count vs
  `format`, and each plane's sample type (filterable-float), dimension (2D),
  non-multisampled, `TEXTURE_BINDING` usage; mismatches route to the device error sink.
- **R8** ☐ Creation builds the 208-byte `ExternalTextureParams` from the descriptor and
  uploads it to a `UNIFORM | COPY_DST` buffer; the resource is `Arc`-handle managed
  (`planes: ArrayVec<TextureView,3>` + `params: Buffer`), `Drop` releases.
- **R9** ☐ A `WGPUExternalTextureBindingEntry` in `wgpuDeviceCreateBindGroup` binds the
  external texture; at draw/dispatch the 3 plane views + params buffer are bound at the
  slots from R6 (Metal + Vulkan).
- **R10** ☐ End-to-end: sampling an `Nv12`/`Rgba` external texture in a fragment shader
  yields the colour-space-converted result (GPU readback on Metal + Vulkan).

## Async

None specific — `wgpuDeviceCreateExternalTexture` is synchronous (like other create
fns). Noop accepts the descriptor/planes and performs no GPU work.

## Open questions

- Exact `YaWGPUExternalTextureDescriptor` field encoding for the C ABI (matrices as
  flat `float` arrays mirroring wgpu's column-major layout).
- Whether to expose `Yu12`/`Nv12` planar formats day-1 or start `Rgba`-only and widen.
- GLES (Tier 2) mapping: external textures likely `HalError` (no clean GLES 3.1 path);
  catalogue in `67-gles-backend.md` when GLES bring-up reaches this.
