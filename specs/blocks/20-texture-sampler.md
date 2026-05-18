# Block 20 — Texture / TextureView / Sampler

Phase 3. Rules from Dawn `TextureValidationTests`,
`TextureViewValidationTests`, `SamplerValidationTests`,
`QueueWriteTextureValidationTests`, `StorageTextureValidationTests`,
`TextureSubresourceTests` (under
`dawn/src/dawn/tests/unittests/validation/`). Status: ☐ ◐ ☑
✗(N/A). "Defer→Px" = needs a later-phase resource.

## Surface (webgpu.h)

- `WGPUTextureDescriptor` 4320 (label,usage,dimension,size,format,
  mipLevelCount,sampleCount,viewFormatCount,viewFormats); `WGPUExtent3D`
  2161 (width,height,depthOrArrayLayers); `WGPUTextureUsage` 1361
  (CopySrc1 CopyDst2 TextureBinding4 StorageBinding8 RenderAttachment16
  TransientAttachment32); `WGPUTextureDimension` 1010 (1D/2D/3D);
  `WGPUTextureFormat` (large enum); `WGPUTextureAspect` 999
  (All/StencilOnly/DepthOnly).
- `WGPUTextureViewDescriptor` 4664 (label,format,dimension,baseMipLevel,
  mipLevelCount,baseArrayLayer,arrayLayerCount,aspect,usage);
  `WGPUTextureViewDimension` 1149 (1D/2D/2DArray/Cube/CubeArray/3D);
  `WGPU_MIP_LEVEL_COUNT_UNDEFINED`/`WGPU_ARRAY_LAYER_COUNT_UNDEFINED`.
- `WGPUSamplerDescriptor` 2704 (addressModeU/V/W,magFilter,minFilter,
  mipmapFilter,lodMinClamp,lodMaxClamp,compare,maxAnisotropy);
  `WGPUAddressMode` 364, `WGPUFilterMode` 667, `WGPUMipmapFilterMode` 738,
  `WGPUCompareFunction` 482.
- `WGPUTexelCopyBufferLayout` 3390 (offset,bytesPerRow,rowsPerImage),
  `WGPUTexelCopyTextureInfo` 4263 (texture,mipLevel,origin,aspect).
- `wgpuDeviceCreateTexture` 6328, `wgpuTextureCreateView` 6729,
  `wgpuDeviceCreateSampler` 6318, `wgpuTextureDestroy` 6730,
  `wgpuQueueWriteTexture` 6470, texture/view/sampler reflection getters +
  `Release`/`AddRef`/`SetLabel`.

## Design decision — Format capability model (P3.1b)

`WGPUTextureFormat` is a large enum. yawgpu defines a core `TextureFormat`
with a **capability descriptor** per format, sourced from Dawn's table
`dawn/src/dawn/native/Format.cpp` (analogous to how Limits used
`Limits.cpp`): per format record `{ aspects(color|depth|stencil),
texel_block_size, block_w, block_h, renderable, multisample_capable,
storage_capable, is_compressed, srgb_view_pair }`. Only the formats the
Phase-3 Dawn tests exercise need to be populated initially (uncompressed
color R8/RG8/RGBA8(+Srgb)/R32/RG32/RGBA16/RGBA32 family, depth/stencil
`Stencil8`/`Depth16Unorm`/`Depth24Plus`/`Depth24PlusStencil8`/
`Depth32Float`/`Depth32FloatStencil8`, the non-renderable set
`RG11B10Ufloat`/`RGB9E5Ufloat`/`*Snorm`, and one compressed family for
block-alignment); `Undefined` is a distinct sentinel. Other formats may
map to a conservative default (treated as plain color) — extend as later
phases need.

## Divergences (recorded)

- Dawn-only formats/usages (e.g. `TransientAttachment`, internal-usage,
  Dawn-specific compressed sliced3D) are accepted as opaque bits where
  canonical `webgpu.h` defines them but tests don't exercise them; not
  invented beyond the header.
- `wgpuQueueWriteTexture` `bytesPerRow` only needs to be **≥ the bytes in
  one row of blocks** (`ceil(width/block_w)·block_size`) when copyHeight>1
  or copyDepth>1 — it is **NOT** 256-aligned (the 256-byte alignment is a
  *buffer*-copy rule → P6). Authoritative source is Dawn
  `QueueWriteTextureValidationTests` (e.g. "bytesPerRow=11 invalid since a
  row takes 12 bytes"). T45 reflects this.
- Like buffers: descriptor validation is **first-match-wins** (one device
  error); error-texture/-view/-sampler still reflect descriptor getters
  and allow `Destroy`/`Release` (mirror block 10 B32/B38).
- GetMappedRange-style: invalid view/sampler create ⇒ device error +
  error object handle (not panic).

## Rules

### Texture creation / reflection / lifetime

- **T1** usage non-zero. `UsageNonZero` :88. ☑ (P3.1a)
- **T2** sampleCount ∈ {1,4}. `SampleCount` :107. ☑ (P3.1a)
- **T3** sampleCount>1 ⇒ mipLevelCount==1. :134. ☑ (P3.1a)
- **T4** sampleCount>1 ⇒ dimension==2D. :143. ☑ (P3.1a)
- **T6** sampleCount>1 ⇒ depthOrArrayLayers==1. :172. ☑ (P3.1a)
- **T7** sampleCount>1 ⇒ no StorageBinding. :181. ☑ (P3.1a)
- **T8** sampleCount>1 ⇒ must have RenderAttachment. :190. ☑ (P3.1a)
- **T9** mipLevelCount ≥ 1. :216. ☑ (P3.1a)
- **T10** mipLevelCount ≤ maxMips(size) (per-dim halving). :226. ☑ (P3.1a)
- **T11** dimension==1D ⇒ mipLevelCount==1. :342. ☑ (P3.1a)
- **T12** arrayLayers ≤ `maxTextureArrayLayers`. :360. ☑ (P3.1a) (reuse P1.2a Limits)
- **T13–T15** 1D: width∈[1,max1D], height==1, depthOrArrayLayers==1.
  :388. ☑ (P3.1a)
- **T16–T18** 2D: width/height∈[1,max2D], depthOrArrayLayers≥1, no
  zero-size. :433. ☑ (P3.1a)
- **T19** 3D: all dims∈[1,max3D]. :481. ☑ (P3.1a)
- **T23** RenderAttachment ⇒ dimension==2D. :652. ☑ (P3.1a)
- **T25** `wgpuTextureDestroy` valid (idempotent; error texture ok).
  :556. ☑ (P3.1a)
- **T57–T64** getters (Format/Dimension/Width/Height/DepthOrArrayLayers/
  MipLevelCount/SampleCount/Usage) reflect descriptor (incl. error
  texture). :1125. ☑ (P3.1a)
- **T65** invalid descriptor ⇒ device error + error-texture handle.
  :1172. ☑ (P3.1a)

### Texture creation — format-capability dependent (P3.1b)

- **T24** format != `Undefined`. :671. ☑ (P3.1b)
- **T5** sampleCount>1 ⇒ multisample-capable format (non-renderable set
  forbidden). :156. ☑ (P3.1b)
- **T20** depth/stencil format ⇒ dimension==2D (forbidden 1D/3D). :537.
  ☑ (P3.1b)
- **T21** RenderAttachment ⇒ renderable format. :617. ☑ (P3.1b)
- **T22/T52** StorageBinding ⇒ storage-capable format. :635 /
  StorageTexture :472. ☑ (P3.1b)
- **T53** StorageBinding ⇒ sampleCount==1 (dup of T7, format-table file).
  :792. ☑ (P3.1b)

### TextureView (P3.2)

- **T26** arrayLayerCount > 0. :113. ☑ (P3.2)
- **T27** mipLevelCount > 0 (or UNDEFINED ⇒ inferred). :120. ☑ (P3.2)
- **T28** baseMipLevel+mipLevelCount ≤ texture.mipLevelCount. :173. ☑ (P3.2)
- **T29** baseArrayLayer+arrayLayerCount ≤ texture layers. :134. ☑ (P3.2)
- **T30** view dimension compat with texture dim (1D/2D/2DArray/Cube/
  CubeArray/3D; Cube⇒6 layers, CubeArray⇒6N). :107/192/282/381. ☑ (P3.2)
- **T31** view format compat: allowed **iff** `view.format ==
  texture.format` **or** `view.format` is in the texture's `viewFormats`
  list. There is NO implicit sRGB-pair allowance — the sRGB counterpart
  must itself be listed in `viewFormats` (Dawn `Texture.cpp`
  `ValidateCanViewTextureAs`). :711. ☑ (P3.2; corrected in P3 Review V1)
- **T32** aspect compat with format (Depth/StencilOnly rules). :751/885.
  ☑ (P3.2)
- **T33** `wgpuTextureViewRelease` valid; default-view inference when
  descriptor fields UNDEFINED. :876. ☑ (P3.2)

### Sampler (P3.3)

- **T34/T35** lodMinClamp/lodMaxClamp finite (no NaN/Inf). :38. ☑ (P3.3)
- **T36** maxAnisotropy ≥ 1. :77. ☑ (P3.3)
- **T37** maxAnisotropy>1 ⇒ mag/min/mipmap filters all Linear. :65. ☑ (P3.3)
- **T38** default descriptor (or null) ⇒ valid sampler. :110. ☑ (P3.3)
- **T39** `wgpuSamplerRelease` valid; error sampler handle on invalid. ☑ (P3.3)

### QueueWriteTexture (P3.4)

- **T40** destination usage has CopyDst. :237. ☑ (P3.4)
- **T41** mipLevel < texture.mipLevelCount. :214. ☑ (P3.4)
- **T42** origin+extent ≤ subresource size (overflow-checked). :206. ☑ (P3.4)
- **T43** 2D ⇒ extent.depthOrArrayLayers==1. :226. ☑ (P3.4)
- **T44** dataSize ≥ required bytes (bytesPerRow/rowsPerImage/extent/
  format). :170. ☑ (P3.4)
- **T45** bytesPerRow ≥ `ceil(width/block_w)·block_size` (bytes in one
  block-row) when copyHeight>1 or copyDepth>1; 0/UNDEFINED only if
  height≤1 & depth≤1. NOT 256-aligned (queue path). :248. ☑ (P3.4)
- **T46** rowsPerImage ≥ copyHeight when set & depth>1. :296. ☑ (P3.4)
- **T47** destination sampleCount==1. :337. ☑ (P3.4)
- **T48** destination not destroyed. :349. ☑ (P3.4)
- **T49** DepthOnly write ⇒ format has depth aspect. :489. ☑ (P3.4)
- **T50** StencilOnly write ⇒ format has stencil aspect. :520. ☑ (P3.4)
- **T51** data offset overflow-checked. :321. ☑ (P3.4)

### Deferred

- **T54–T56** subresource lifetime/usage tracking — needs
  `CommandEncoder`/`RenderPass`. Defer→P6.
- Full shader-driven storage-texture access validation — Defer→P5
  (BindGroupLayout/pipeline). Compressed-format copy block alignment
  beyond creation — Defer→P6.

## Review notes (carried)

- P3.1b `FormatCaps` is an **approximation** of Dawn `Format.cpp` for the
  populated set; unknown formats default to renderable color. The
  storage-capable / multisample set for the `*16`/`*32` families is not
  feature-accurate (some are feature-gated in Dawn, e.g. `RGBA16Unorm`).
  Acceptable for Phase 3 (only tested formats' caps must be right);
  refine when StorageTexture/BindGroup lands (P4/P5) and flag in the
  Phase 3 Review.
  - **Update (Phase 5 Review J6):** the *unknown-defined-format* hole is
    resolved — `TextureFormat::caps()` now returns `None` for
    unrecognized format codes (they are rejected, not treated as
    renderable color). The remaining `*16`/`*32` feature-gating
    accuracy for the *populated* set stays a low-priority refinement
    (no current rule depends on it).

## Open questions

- Noop texture: track descriptor + per-subresource map none; host backing
  not needed (no map/getMappedRange for textures). `wgpuTextureDestroy`
  idempotent like buffers.
- maxMips(size): `floor(log2(max over the dimension's relevant extents))
  + 1`, per-dimension halving to 1 (Dawn `Texture.cpp` algorithm).
- Default-view inference rules (format=texture.format, dimension from
  texture dim+layers, mip/array full range) — from webgpu.h
  `WGPUTextureViewDescriptor` `INIT` + spec.
