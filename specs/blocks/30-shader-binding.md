# Block 30 — ShaderModule / BindGroupLayout / BindGroup / PipelineLayout

Phase 4. Rules from Dawn `ShaderModuleValidationTests`,
`WGSLFeatureValidationTests`, `BindGroupValidationTests`,
`GetBindGroupLayoutValidationTests`, `OverridableConstantsValidationTests`,
`ImmediateDataTests`, `MinimumBufferSizeValidationTests` (B58),
`UnsafeAPIValidationTests` (R18–R20) — under
`dawn/src/dawn/tests/unittests/validation/`. Status: ☐ ◐ ☑
✗(N/A). "Defer→Px" = needs a later-phase resource.

## Surface (webgpu.h)

- `WGPUShaderModuleDescriptor` 4193 (+ chained `WGPUShaderSourceWGSL
  {code}` / `WGPUShaderSourceSPIRV {codeSize,code}`);
  `wgpuDeviceCreateShaderModule` → `WGPUShaderModule`;
  `wgpuShaderModuleGetCompilationInfo` (+`WGPUCompilationInfoCallbackInfo`)
  → `WGPUFuture`; `WGPUCompilationInfo{messageCount,messages[]}`,
  `WGPUCompilationMessage{message,type,lineNum,linePos,offset,length}`;
  `wgpuShaderModuleRelease`/`AddRef`.
- `WGPUBindGroupLayoutEntry` 3626 (binding, visibility `WGPUShaderStage`,
  bindingArraySize, one of buffer/sampler/texture/storageTexture):
  `WGPUBufferBindingLayout{type,hasDynamicOffset,minBindingSize}`,
  `WGPUSamplerBindingLayout{type}`,
  `WGPUTextureBindingLayout{sampleType,viewDimension,multisampled}`,
  `WGPUStorageTextureBindingLayout{access,format,viewDimension}`;
  `WGPUBindGroupLayoutDescriptor` 4467 → `wgpuDeviceCreateBindGroupLayout`.
- `WGPUBindGroupEntry` 3564 (binding,buffer,offset,size,sampler,
  textureView); `WGPUBindGroupDescriptor` 4431 (layout,entries) →
  `wgpuDeviceCreateBindGroup`.
- `WGPUPipelineLayoutDescriptor` 2360 (bindGroupLayoutCount,
  bindGroupLayouts, immediateSize) → `wgpuDeviceCreatePipelineLayout`.
- `WGPUConstantEntry` 2131 (key,value). `WGPUShaderStage` (Vertex/
  Fragment/Compute bits). `WGPU_WHOLE_SIZE` for binding size.

## Design decisions

- **WGSL via `naga`** (pinned `infosia/wgpu` rev — see
  `reference/dependencies.md`). `wgpuDeviceCreateShaderModule` parses +
  validates WGSL with `naga::front::wgsl` + `naga::valid::Validator`;
  failure ⇒ device error + a `WGPUCompilationInfo` with the naga
  diagnostic (≥1 Error message). SPIR-V source: accept the bytes
  (optionally `naga::front::spv` parse); deep SPIR-V validation is
  best-effort. Shader reflection (binding numbers, override ids, entry
  IO) comes from the naga `Module`.
- **AllowUnsafeAPIs is non-canonical.** Dawn's toggle is not in
  `webgpu.h`; yawgpu does **not** expose a way to enable it. Therefore
  R18/S6 (`chromium_disable_uniformity_analysis`), R20/S7 (static
  `binding_array` in WGSL), R19/S19 (`bindingArraySize>1`), S11
  (experimental extensions) are validated **only in the rejected
  direction** (always an error). Recorded divergence.
- Error-object model (mirror blocks 10/20): invalid create ⇒ device
  error + an error `WGPUShaderModule`/`BindGroupLayout`/`BindGroup`/
  `PipelineLayout` handle that is `Release`-safe; first-match-wins.
- **naga≠Tint divergence.** naga may accept/reject WGSL differently than
  Dawn's Tint. Ported tests assert *that an error is produced* for
  clearly-invalid WGSL and *success* for clearly-valid WGSL — they do
  NOT pin exact diagnostic text or borderline cases. Borderline
  divergences recorded as encountered.
- `WGPUShaderStage` visibility: unknown bits are not a creation error
  (S20) — only Vertex/Fragment/Compute are meaningful; carry raw bits.

## Rules

### ShaderModule (P4.1)

- **S1** descriptor needs exactly one chained source (WGSL xor SPIRV).
  `NoChainedDescriptor` :309. ☐
- **S2** WGSL and SPIRV (or WGSL+Dawn SPIRV options) mutually exclusive.
  `MultipleChainedDescriptor_*` :316. ☐
- **S3** WGSL parsed+validated by naga; invalid (syntax/semantic/UTF-8/
  missing `@location`/`@group`+`@binding`/binding≥1000) ⇒ error.
  `MissingDecorations`/`MaxBindingNumber` :101/564/589. ☐
- **S4** override `@id(n)` numeric ids unique within module.
  `OverridableConstantsNumericIDConflicts` :545. ☐
- **S5** SPIR-V source accepted (bytes; best-effort validate).
  `CreationSuccess` :101. ☐
- **S6** `chromium_disable_uniformity_analysis` ⇒ error (AllowUnsafeAPIs
  non-canonical → always rejected). UnsafeAPI :52. ☐ (divergence)
- **S7** static `binding_array` in WGSL ⇒ error (same divergence).
  UnsafeAPI :88. ☐
- **S9** `wgpuShaderModuleGetCompilationInfo` async via the future/
  callback-mode machinery; returns `WGPUCompilationInfo` (≥1 Error msg
  for an invalid module, empty/Info for a valid one).
  `GetCompilationMessages` :364. ☐
- **S10** `DawnShaderModuleSPIRVOptionsDescriptor` — Dawn-only chained
  struct; not in `webgpu.h`. ✗ N/A (recorded).
- **S11** experimental WGSL extensions requiring AllowUnsafeAPIs ⇒ error
  (divergence as S6). `ShaderModuleExtensionValidationTest` :973. ☐
- **S8** inter-stage variable location limits — validated at **pipeline**
  creation. Defer→P5.

### BindGroupLayout (P4.2)

- **S12** entry `binding` unique; `0 ≤ binding < 1000`. `BindGroupEntry`
  :1395. ☐
- **S13** exactly one of buffer/sampler/texture/storageTexture set per
  entry (none/too-many ⇒ error). `BindGroupLayoutEntry{TooManySet,
  NoneSet}` :1422. ☐
- **S14** buffer layout: `type∈{Uniform,Storage,ReadOnlyStorage}`,
  `hasDynamicOffset` bool, `minBindingSize` recorded. :1404. ☐
- **S15** dynamic uniform/storage buffer counts ≤
  `limits.maxDynamic{Uniform,Storage}BuffersPerPipelineLayout`. ☐
  (reuse P1.2a Limits)
- **S16** sampler layout `type∈{Filtering,NonFiltering,Comparison}`. ☐
- **S17** texture layout `sampleType`/`viewDimension`/`multisampled`
  well-formed. ☐
- **S18** storageTexture `access∈{WriteOnly,ReadOnly,ReadWrite}`,
  `format` storage-capable (reuse P3.1b `FormatCaps`), `viewDimension`
  not 1D. ☐
- **S19** `bindingArraySize>1` ⇒ error (0/1 ok; AllowUnsafeAPIs
  divergence). UnsafeAPI :67. ☐
- **S20** visibility unknown bits NOT an error (carry raw). :1449. ☐
- **S21** per-stage binding counts ≤ `limits.max{SampledTextures,
  Samplers,StorageBuffers,StorageTextures,UniformBuffers}PerShaderStage`.
  :1486. ☐ (reuse Limits)
- **S22** total entries per group ≤ 1000. `BindGroupLayoutEntryMax`
  :1382. ☐

### BindGroup (P4.3)

- **S23** `entryCount` == layout entry count. `EntryCountMismatch` :172.
  ☐
- **S24** every layout binding present once; no duplicate binding.
  `WrongBindings`/`BindingSetTwice` :184. ☐
- **S25** entry resource kind matches layout entry kind (buffer/sampler/
  textureView; others must be null). :223/277/336. ☐
- **S26** buffer offset aligned to `minUniform/StorageBufferOffsetAlign`
  per type. `BufferOffsetAlignment` :926. ☐ (reuse Limits)
- **S27** effective size (`size==0 ⇒ WHOLE_SIZE = buffer.size-offset`)
  > 0, `offset+size ≤ buffer.size`, `≥ layout.minBindingSize`, ≤
  `maxUniform/StorageBufferBindingSize`. :986–1225. ☐
- **S28** buffer usage: Uniform⇒`Uniform`; Storage/ReadOnlyStorage⇒
  `Storage`. :874. ☐
- **S29** sampled-tex view usage `TextureBinding`; storage-tex view
  usage `StorageBinding`; depth format ⇒ not `Float` sampleType. :536.
  ☐
- **S30** view dimension == layout `viewDimension`. :793. ☐
- **S31** layout `multisampled` ⇔ texture.sampleCount>1. :945. ☐
- **S32** storage-texture view: layerCount==1 (no array storage view).
  :809. ☐
- **S33** sampler/textureView same device as the bind group. ☐
  (cross-device — see R15-family)
- **S34** `bindingArraySize>1` requires all elements — N/A (gated off
  by S19 divergence). ✗
- **S35/B58** BG-creation part: bound buffer effective size ≥ layout
  `minBindingSize`. The shader-declared-vs-layout check is at **pipeline**
  creation. ☐ (BG part here; pipeline part Defer→P5)

### PipelineLayout (P4.4)

- **S36** each `bindGroupLayouts[i]` non-null; `bindGroupLayoutCount` ≤
  `maxBindGroups` (Limits). ☐
- **S37** `immediateSize` ≤ `limits.maxImmediateSize`. ☐ (reuse Limits)
- **S38** downstream pipeline-compat — Defer→P5.

### Deferred → P5 (need Pipeline)

- **S39–S42** overridable-constant key/value/range/uninitialized
  validation (pipeline-constant context). Defer→P5.
- **S43–S44** `Pipeline.GetBindGroupLayout` + reflected-layout
  aggregation. Defer→P5.
- **S8** inter-stage IO location limits. Defer→P5.
- **S35** shader-declared minBindingSize vs layout (pipeline). Defer→P5.

## Open questions

- Compilation-info storage: keep the naga diagnostic on the
  `ShaderModule` so `GetCompilationInfo` can replay it (success ⇒ empty
  or one Info message).
- SPIR-V: parse with `naga::front::spv` for reflection, or accept opaque
  for now (decide in P4.1; opaque acceptable if no Phase-4 test needs
  SPIR-V reflection).
- naga `Validator` capability/feature flags to enable (default set;
  expand as later phases need).
