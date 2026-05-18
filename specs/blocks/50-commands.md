# Block 50 — Command encoding & passes

Phase 6 (largest). Rules from Dawn `CommandBufferValidationTests`,
`CopyCommandsValidationTests`, `RenderPassDescriptorValidationTests`,
`RenderBundleValidationTests`, `DynamicStateCommandValidationTests`,
`IndexBufferValidationTests`, `VertexBufferValidationTests`,
`DrawIndirectValidationTests`, `DrawVertexAndIndexBufferOOBValidationTests`,
`ComputeIndirectValidationTests`, `DebugMarkerValidationTests`,
`ResourceUsageTrackingTests`,
`Writable{Buffer,Texture}BindingAliasingValidationTests`,
`TextureSubresourceTests`, plus the P2/P3/P5 deferrals
(B39–B41 submit-with-mapped/destroyed-buffer, B53–B57
CommandEncoder.WriteBuffer, T54–T56 texture subresource, P41 draw-time
cross-pipeline, PipelineAndPassCompat render-pass parts). Status: ☐ ◐ ☑
✗(N/A). "Defer→Px" = needs a later-phase resource.

## Surface (webgpu.h)

`wgpuDeviceCreateCommandEncoder` (+desc), `wgpuCommandEncoderFinish`
(+desc) → `WGPUCommandBuffer`; `BeginRenderPass`
(`WGPURenderPassDescriptor` + `WGPURenderPassColorAttachment` /
`…DepthStencilAttachment`) / `BeginComputePass`
(`WGPUComputePassDescriptor`); `CopyBufferToBuffer`/`…ToTexture`/
`TextureToBuffer`/`TextureToTexture` (`WGPUTexelCopyBufferInfo`/
`…TextureInfo`); `ClearBuffer`; `WriteBuffer` (encoder);
`Insert/Push/PopDebugGroup`; `WriteTimestamp`/`ResolveQuerySet`
(Defer→P8). RenderPassEncoder: `SetPipeline`/`SetBindGroup`(+dynamic
offsets)/`SetVertexBuffer`/`SetIndexBuffer`/`Draw`/`DrawIndexed`/
`DrawIndirect`/`DrawIndexedIndirect`/`SetViewport`/`SetScissorRect`/
`SetBlendConstant`/`SetStencilReference`/`ExecuteBundles`/`End`.
ComputePassEncoder: `SetPipeline`/`SetBindGroup`/`DispatchWorkgroups`/
`DispatchWorkgroupsIndirect`/`End`. `wgpuDeviceCreateRenderBundleEncoder`
(+desc) / `Finish` → `WGPURenderBundle`. `wgpuQueueSubmit`
(`WGPUCommandBuffer[]`).

## Design decisions

- **Deferred-error model (central).** Like Dawn, most encoder/pass
  command errors are recorded and surface at `wgpuCommandEncoderFinish`
  (or `RenderBundleEncoder::Finish`) as a single device error + an
  **error CommandBuffer/RenderBundle** handle (first-match-wins on the
  recorded first error). Pass encoders forward their first error to the
  parent encoder; `End` finalizes the pass. Synchronous-only checks
  (e.g. some descriptor validation at `BeginRenderPass`) may also error
  immediately where Dawn does — match Dawn per-rule.
- **Encoder/pass state machine.** Encoder: Recording → Finished
  (Finish) / Errored. A pass must be `End`ed before `Finish`; no two
  open passes; commands after `End`/`Finish` ⇒ error; double
  `End`/`Finish` ⇒ error. Debug-group push/pop balance enforced at
  `End`/`Finish`.
- **Usage scopes.** A render pass / compute dispatch forms a usage
  scope; track per-(sub)resource read/write to detect
  writable-binding aliasing (C75/C77), read+write conflict (C76),
  attachment+sampled conflict (C78). Submit-time (C80–C82): a command
  buffer referencing a mapped/destroyed buffer ⇒ error at
  `wgpuQueueSubmit` (reuses the P2 buffer map-state/destroyed flags;
  the command buffer holds `Arc`s to referenced resources).
- **Copy bytesPerRow 256-alignment** DOES apply to the
  CommandEncoder buffer↔texture copies (unlike `queueWriteTexture`,
  block 20 divergence). Reuse P3 `FormatCaps` block size/aspects.
- Error-object model + Arc handles as prior blocks. naga≠Tint /
  caching / FormatCaps notes carried.
- Occlusion/timestamp query validation (C34/C35, occlusion query) →
  Defer→P8. Real-GPU execution → Defer→P7.

## Rules (grouped → slices)

### P6.1 Encoder/pass lifecycle
- **C1** commands after `Finish` ⇒ error; double `Finish` semantics.
  `CallsAfterASuccessfulFinish` :238. ☑ (P6.1)
- **C2** open pass not `End`ed before `Finish` ⇒ error.
  `EndedMidRenderPass` :47. ☑ (P6.1)
- **C3** pass `End` twice ⇒ error. `RenderPassEndedTwice` :115. ☑ (P6.1)
- **C4** command on a pass after its `End` ⇒ error.
  `EncodeAfterEndingPass` :446. ☑ (P6.1)
- **C5** two open passes at once ⇒ error. `BeginRenderPassBeforeEnd
  PreviousPass` :213. ☑ (P6.1)
- **C36** ComputePass descriptor optional/minimal. :82. ☑ (P6.1)
- **C63** unbalanced debug groups on the encoder at `Finish` ⇒ error.
  ☑ (P6.1)
- **C85/C86** pass after parent `Finish` ⇒ error; encoder from
  destroyed device safe. ☑ (P6.1)

### P6.2 Buffer copies / clear / encoder WriteBuffer
- **C6–C10** B2B size/OOB, 4-byte align, CopySrc/CopyDst usage,
  same-buffer overlap, error buffers. `CopyCommandTest_B2B` :298. ☑ (P6.2)
- **ClearBuffer** offset/size align & bounds + CopyDst. ☑ (P6.2)
- **C83/C84** encoder `WriteBuffer` 4-byte align + bounds + CopyDst
  (B53–B57). `WriteBufferTests`. ☑ (P6.2)

### P6.3 Texture copies (B2T/T2B/T2T)
- **C11–C17** B2T/T2B bytesPerRow %256, rowsPerImage, bounds, usage,
  depth/stencil 2D-only, no-multisample. :492. ☐
- **C18–C22** T2T format-compat (sRGB), usage/OOB, depth/stencil,
  sample-count match, same-texture subresource. :1747. ☐
- **C79** texture aspect consistency with format. ☐

### P6.4 RenderPass descriptor
- **C23–C27** ≥1 attachment, color count ≤ max & RenderAttachment &
  renderable, sparse, size match, format aspect. :101. ☐
- **C28/C32/C33** depth/stencil store ops, loadOp/storeOp set, clear
  value finite (+depth ∈[0,1]). :304/:1369. ☐
- **C29** attachment view arrayLayerCount==1. :353. ☐
- **C30/C31** multisample resolve target (count1/format/size/usage)
  & all attachments same sampleCount. :1165/:1276. ☐
- **C34/C35** occlusion/timestamp query sets. Defer→P8.

### P6.5 Pass draw/dispatch state + dynamic state
- **C37–C42** SetPipeline-before-draw, bind-group count/compat,
  dynamic-offset count/align/bounds, vertex/index buffer set. ☐
- **C56–C59** SetViewport/ScissorRect finite+bounds,
  SetBlendConstant finite, SetStencilReference. ☐
- compute: SetPipeline-before-dispatch, bind-group compat,
  DispatchWorkgroups size ≤ `maxComputeWorkgroupsPerDimension`. ☐
- P41 draw-time: a BindGroup from pipeline A's default BGL rejected
  with pipeline B (carried Phase-5 deferral). ☐

### P6.6 Index/Vertex buffer + draw OOB + indirect
- **C43–C46** index format valid, Index usage, OOB, offset align,
  matches pipeline strip format. :62. ☐
- **C47–C49** vertex buffer Vertex usage, offset OOB, format. ☐
- **C50–C52** Draw/DrawIndexed vertex/index count vs bound buffer
  size; instanceCount. `DrawVertexAndIndexBufferOOB`. ☐
- **C53–C55** Draw/DrawIndexedIndirect + ComputeIndirect: Indirect
  usage, 4-byte offset align, indirect-args size bounds; firstInstance
  feature gating. ☐

### P6.7 RenderBundle
- **C65–C68** bundle encoder descriptor (≥1 format, count, renderable;
  pipeline color/depth/sample format match). :602/:797. ☐
- **C69–C74** bundle state independence, ExecuteBundles state-clear,
  Finish-twice, ExecuteBundles format/sample match, multi-execute.
  :272/:592/:942. ☐

### P6.8 Debug markers
- **C60/C61/C62/C64** push/pop balance in render/compute pass &
  render bundle; InsertDebugMarker no-op. `DebugMarkerValidationTests`.
  ☐

### P6.9 Resource usage tracking + submit-time buffer state
- **C75/C77** writable buffer/texture binding aliasing in a scope.
  :308. ☐
- **C76** read+write same resource conflict in a pass. ☐
- **C78** attachment + sampled-in-pass conflict (T54–T56). ☐
- **C80–C82** submit: command buffer finished/valid; referenced buffer
  not mapped (B39) / not destroyed (B40/B41). ☐

## Open questions

- CommandBuffer model: store the recorded command list + referenced
  resource `Arc`s + a first-error; `Finish` validates the deferred
  rules; `Submit` runs C80–C82.
- Usage-scope granularity: per-buffer-range / per-texture-subresource
  tracking sufficient for the ported tests.
- How much pass/draw validation is deferred-to-Finish vs immediate —
  decide per rule from the Dawn test (record divergences).
