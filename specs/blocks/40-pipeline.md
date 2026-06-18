# Block 40 — Render / Compute Pipeline

Phase 5. Rules from Dawn `RenderPipelineValidationTests`,
`ComputeValidationTests`, `VertexStateValidationTests`,
`PipelineAndPassCompatibilityTests`, `ObjectCachingTests`,
`GetBindGroupLayoutValidationTests` (deferred S43/S44),
`OverridableConstantsValidationTests` (deferred S39–S42) — under
`dawn/src/dawn/tests/unittests/validation/`. Status: ☐ ◐ ☑
✗(N/A). "Defer→Px" = needs a later-phase resource.

## Surface (webgpu.h)

- `WGPURenderPipelineDescriptor` + `WGPUVertexState`
  (`WGPUVertexBufferLayout`/`WGPUVertexAttribute`/`WGPUVertexFormat`/
  `WGPUVertexStepMode`), `WGPUPrimitiveState`, `WGPUDepthStencilState`
  (`WGPUStencilFaceState`, depthBias*), `WGPUMultisampleState`,
  `WGPUFragmentState` (`WGPUColorTargetState`, `WGPUBlendState`,
  `WGPUColorWriteMask`). `WGPUComputePipelineDescriptor` +
  `WGPUComputeState` (module, entryPoint, constants `WGPUConstantEntry`).
  Pipeline `layout` = explicit `WGPUPipelineLayout` or NULL/Auto.
- `wgpuDeviceCreateRenderPipeline`/`CreateComputePipeline`(+`Async`
  +`WGPUCreate*PipelineAsyncCallbackInfo`,
  `WGPUCreatePipelineAsyncStatus`),
  `wgpuRender/ComputePipelineGetBindGroupLayout`,
  `wgpuRender/ComputePipelineRelease`/`AddRef`.
- Limits: `maxVertexBuffers/Attributes/VertexBufferArrayStride`,
  `maxInterStageShaderVariables`, `maxColorAttachments`,
  `maxColorAttachmentBytesPerSample`, `maxComputeWorkgroupSizeX/Y/Z`,
  `maxComputeInvocationsPerWorkgroup`, `maxComputeWorkgroupStorageSize`,
  `maxComputeWorkgroupsPerDimension` (all P1.2a Limits).

## Design decisions

- **naga reflection.** Pipeline validation reads the naga `Module`
  (from P4.1a `ShaderModule`): entry points + stage (`@vertex`/
  `@fragment`/`@compute`), `@workgroup_size` (incl. override-driven),
  `var<workgroup>` sizes, `@location` IO + types, `@group/@binding`
  resources + access (textureSample vs textureLoad ⇒ Float vs
  UnfilterableFloat), `@builtin(frag_depth|sample_mask)`, overrides
  (name/`@id`/type/default). **W4 widened**: naga `Validator`
  `Capabilities` gains what the Phase-5 WGSL needs (e.g. `f16` for
  `ShaderF16`); compute entries already validate. Capability set is
  data-driven; recorded in P5.0.
- **Auto vs explicit layout.** `layout==NULL` ⇒ pipeline derives an
  auto pipeline-layout/BGLs from reflection; explicit ⇒ validate the
  shader's declared bindings are satisfied by the layout (the deferred
  **S35 pipeline part**: layout `minBindingSize ≥ shader-required`,
  type/visibility match).
- **Object identity / caching.** Dawn dedups; externally observable as
  C-handle identity. yawgpu keeps a per-device descriptor-keyed cache so
  identical `CreateShaderModule`/`PipelineLayout`/
  `Render/ComputePipeline` calls return the **same** handle pointer
  (testable via `==`). **`BindGroupLayout` dedup is intentionally NOT
  implemented** (m2): no P-rule requires it; identical BGL descriptors
  yield distinct handles (a recorded divergence — `PipelineLayout`
  dedup keys on BGL handle identity, so callers reusing one BGL handle
  still dedup). Auto-layout default BGLs are pipeline-bound
  (P40/P41): a default BGL is rejected for `CreatePipelineLayout` and is
  not interchangeable across pipelines. Where exact Dawn identity is an
  internal optimization not observable through `webgpu.h`, record as a
  divergence rather than over-engineer.
- Error-object model + first-match-wins (mirror blocks 10/20/30):
  invalid create ⇒ device error + error pipeline handle, `Release`-safe.
- Async create reuses the Phase-1 future/`PendingCallback` machinery
  (`WGPUCreatePipelineAsyncStatus`); validation identical to sync **but
  a validation failure is reported ONLY via the callback
  `ValidationError` status — it does NOT raise a device/uncaptured
  error** (canonical Dawn behavior; J3 fix). The sync path still
  raises the device error as before.
- **W4/SHADER_FLOAT16 (m4) — being resolved.** Historically the naga
  `Validator` enabled `SHADER_FLOAT16` unconditionally, so `enable f16;`
  WGSL validated even without a device `shader-f16` feature (recorded
  divergence). This is now superseded by the canonical `ShaderF16`
  feature: shader-module creation gates `SHADER_FLOAT16` on the requested
  feature (block 30 **S12**) and the feature is advertised/honoured per
  backend. Full contract + status: `specs/tracking/shader-f16.md`.
- naga≠Tint divergence (as block 30): assert error-vs-success, not exact
  diagnostics; record borderline cases.

## Rules

### P5.0 — naga capability + reflection helper

- **P0a** widen naga `Validator` `Capabilities` to cover Phase-5 WGSL
  (`f16`/`ShaderF16`, …; data-driven). ☑ (P5.0)
- **P0b** `shader_naga` reflection helpers: entry points (+stage),
  `@workgroup_size`, `var<workgroup>` total, `@location` IO+types,
  `@group/@binding`+access, `@builtin` outputs, overrides. ☑ (P5.0) (consumed
  by P5.1+)

### Compute pipeline (P5.1)

- **P1** compute entry resolution (null⇒unique `@compute`; mismatch/
  none/ambiguous⇒error). ComputeValidationTests :63/159. ☑ (P5.1)
- **P2** `@workgroup_size` ≤ `maxComputeWorkgroupSizeX/Y/Z`; product ≤
  `maxComputeInvocationsPerWorkgroup`. :42. ☑ (P5.1)
- **P3** `var<workgroup>` total ≤ `maxComputeWorkgroupStorageSize`. ☑ (P5.1)
- **P4** explicit-or-auto layout accepted; explicit ⇒ shader/layout
  binding compat (S35 pipeline part). GetBGL :390. ☑ (P5.1)
- **P5** overridable-constant key lookup (name/`@id`; mixed/duplicate/
  uninitialized-without-default ⇒ error). Overridable :146/199. ☑ (P5.1)
- **P6** override value finite & representable in WGSL type. Overridable
  :280. ☑ (P5.1)

### Render pipeline (P5.2 — may split a:vertex/primitive/multisample,
b:fragment/color/depthStencil)

- **P7/P8** vertex/fragment entry resolution. RenderPipeline :1508. ☑ (P5.2a)
- **P9** must have fragment OR depthStencil; fragment⇒≥1 target.
  :373/1061. ☑ (P5.2a)
- **P18** `stripIndexFormat` only with strip topology. :1343. ☑ (P5.2a)
- **P19** depth test/write ⇒ format has depth aspect (FormatCaps).
  :214. ☑ (P5.2b)
- **P20** stencil ops ⇒ format has stencil aspect. :214. ☑ (P5.2b)
- **P21** depthBiasSlopeScale/Clamp finite. :90. ☑ (P5.2a)
- **P22** non-zero depth bias only with triangle topology. :140. ☑ (P5.2a)
- **P23** depth-aspect format ⇒ depthCompare & depthWriteEnabled set
  (not Undefined). :1410. ☑ (P5.2b)
- **P24** `@builtin(frag_depth)` ⇒ depthStencil w/ depth aspect. :327.
  ☑ (P5.2b)
- **P25** multisample count ∈ {1,4}. :817. ☑ (P5.2a)
- **P26** alphaToCoverage ⇒ count==4. :1083. ☑ (P5.2a)
- **P27** alphaToCoverage conflicts with `@builtin(sample_mask)` out.
  :1107. ☑ (P5.2a)
- **P28** alphaToCoverage ⇒ a blendable target with alpha. :1156. ☑ (P5.2b)
- **P29** color target format renderable (FormatCaps); Undefined =
  hole. :442. ☑ (P5.2b)
- **P30** blend ⇒ blendable format. :467. ☑ (P5.2b)
- **P31** target format set but no shader `@location(i)` out ⇒
  writeMask must be 0. :424. ☑ (P5.2b)
- **P32** fragment output type matches target format class
  (float/uint/sint, component count). :553. ☑ (P5.2b)
- **P33** Undefined target format ⇒ blend must be null. :395. ☑ (P5.2b)
- **P34** fragment present ⇒ targetCount ≥ 1. :373. ☑ (P5.2b)
- **P35** Σ color-target bytesPerSample ≤
  `maxColorAttachmentBytesPerSample`. ☑ (P5.2b)
- **P36** render auto/explicit layout (as P4). GetBGL :60. ☑ (P5.2b)
- **P37** render overridable constants (vertex+fragment) — as P5/P6. ☑ (P5.2b)

### Vertex state (P5.3)

- **P10** bufferCount ≤ `maxVertexBuffers`. VertexState :181. ☑ (P5.3)
- **P11** Σ attributeCount ≤ `maxVertexAttributes`. :205. ☑ (P5.3)
- **P12** arrayStride %4==0 (or 0) and ≤ `maxVertexBufferArrayStride`.
  :226/240. ☑ (P5.3)
- **P13** attribute offset aligned to min(4,formatSize); offset+size ≤
  stride (stride≠0). :312/327. ☑ (P5.3)
- **P14** `shaderLocation` unique across all attributes. :253. ☑ (P5.3)
- **P15** `shaderLocation` < `maxVertexAttributes`. :294. ☑ (P5.3)
- **P16** vertex format class matches shader input type. :385. ☑ (P5.3)
- **P17** every shader vertex input `@location` covered by a buffer
  attribute (subset of buffers ok). :106. ☑ (P5.3)

### GetBindGroupLayout + layout compat (P5.4) — deferred S43/S44/S35

- **P38** group index in range. GetBGL :1272. ☑ (P5.4)
- **P39** returned BGL aggregates shader bindings across stages
  (visibility OR, minBindingSize max). :60/266. ☑ (P5.4)
- **P40** default (auto) BGL rejected for `CreatePipelineLayout`. :116.
  ☑ (P5.4)
- **P41** default BGLs are pipeline-bound (not interchangeable across
  pipelines). :174. ☑ (P5.4; draw-time cross-pipeline → P6)
- **P42** texture sample-type default from usage (textureSample⇒Float,
  textureLoad⇒UnfilterableFloat). :266. ☑ (P5.4)

### Object caching (P5.5)

- **P43** identical ShaderModule WGSL ⇒ same handle. ObjectCaching
  :205. ☑ (P5.5)
- **P44** identical PipelineLayout (BGL array) ⇒ same handle. :188. ☑ (P5.5)
- **P45–P50** Compute/Render pipeline dedup on module/layout/constants
  ⇒ same handle. :224–486. ☑ (P5.5)

### Async (P5.6)

- **P51** `wgpuDeviceCreateRender/ComputePipelineAsync` via the future
  machinery; `WGPUCreatePipelineAsyncStatus`; same validation as sync.
  ☑ (P5.6)

### Deferred

- RenderPass/RenderBundle pipeline-compat (read-only depth/stencil) —
  `PipelineAndPassCompatibilityTests` :96. Defer→P6.
- Real-GPU multisample/attachment match. Defer→P7.

## Open questions

- Caching scope: descriptor-key equality definition per object; how much
  of Dawn's internal dedup is observable through `webgpu.h` (decide per
  slice; record divergences).
- naga override-driven `@workgroup_size` evaluation (constants applied
  before limit check).
- Auto-layout reflection: building `BindGroupLayout`/`PipelineLayout`
  equivalents from the naga `Module` (group/binding/type/visibility/
  minBindingSize).
