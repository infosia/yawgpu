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
  depth/stencil 2D-only, no-multisample. :492. ☑ (P6.3)
- **C18–C22** T2T format-compat (sRGB), usage/OOB, depth/stencil,
  sample-count match, same-texture subresource. :1747. ☑ (P6.3)
- **C79** texture aspect consistency with format. ☑ (P6.3)

### P6.4 RenderPass descriptor
- **C23–C27** ≥1 attachment, color count ≤ max & RenderAttachment &
  renderable, sparse, size match, format aspect. :101. ☑ (P6.4)
- **C28/C32/C33** depth/stencil store ops, loadOp/storeOp set, clear
  value finite (+depth ∈[0,1]). :304/:1369. ☑ (P6.4)
- **C29** attachment view arrayLayerCount==1. :353. ☑ (P6.4)
- **C30/C31** multisample resolve target (count1/format/size/usage)
  & all attachments same sampleCount. :1165/:1276. ☑ (P6.4)
- **C34/C35** occlusion/timestamp query sets. Defer→P8.

### P6.5 Pass draw/dispatch state + dynamic state
- **C37–C42** SetPipeline-before-draw, bind-group count/compat,
  dynamic-offset count/align/bounds, vertex/index buffer set. ☑ (P6.5)
- **C56–C59** SetViewport/ScissorRect finite+bounds,
  SetBlendConstant finite, SetStencilReference. ☑ (P6.5)
- compute: SetPipeline-before-dispatch, bind-group compat,
  DispatchWorkgroups size ≤ `maxComputeWorkgroupsPerDimension`. ☑ (P6.5)
- P41 draw-time: a BindGroup from pipeline A's default BGL rejected
  with pipeline B (carried Phase-5 deferral). ☑ (P6.5)

> P6.5 notes / divergences (deferred-error model; surface at
> `wgpuCommandEncoderFinish`):
> - SetVertexBuffer/SetIndexBuffer only **record slot/buffer state**
>   here; full index/vertex buffer rules (usage/OOB/format/align,
>   draw count/OOB, indirect) are **C43–C55 → P6.6**.
> - BGL compatibility: default/auto BGLs use Arc identity
>   (`BindGroupLayout::same`, i.e. the P5.4 pipeline-bound default-BGL
>   identity ⇒ P41); explicit BGLs compare by entry list.
> - SetScissorRect: minimal per-Dawn check = integer add overflow of
>   `x+width`/`y+height`; attachment-size clamping needs the render
>   target extent → **Defer→P6.9/P7** with usage-scope/real-GPU work.
> - C40 dynamic-offset bounds use `Limits.min_uniform/storage_buffer_
>   offset_alignment`; offsets zipped with dynamic BGL entries in BGL
>   entry order (sufficient for the ported cases).

### P6.6 Index/Vertex buffer + draw OOB + indirect
- **C43–C46** index format valid, Index usage, OOB, offset align,
  matches pipeline strip format. :62. ☑ (P6.6)
- **C47–C49** vertex buffer Vertex usage, offset OOB, format. ☑ (P6.6)
- **C50–C52** Draw/DrawIndexed vertex/index count vs bound buffer
  size; instanceCount. `DrawVertexAndIndexBufferOOB`. ☑ (P6.6)
- **C53–C55** Draw/DrawIndexedIndirect + ComputeIndirect: Indirect
  usage, 4-byte offset align, indirect-args size bounds; firstInstance
  feature gating. ☑ (P6.6)

> P6.6 notes / divergences (deferred-error model; surface at
> `wgpuCommandEncoderFinish`):
> - SetIndexBuffer/SetVertexBuffer validate at the Set call (C43–C45/
>   C47–C49): error/destroyed buffer, INDEX/VERTEX usage, offset
>   alignment (index-format size / 4), `WHOLE_SIZE` resolved via
>   `validate_buffer_range`, vertex slot < `max_vertex_buffers`; null
>   vertex buffer requires zero offset+size.
> - Draw OOB (C50–C52) per the pipeline's vertex-buffer layouts:
>   `stepMode==Vertex` bounded by `firstVertex+vertexCount`,
>   `stepMode==Instance` by `firstInstance+instanceCount`,
>   `arrayStride==0` skipped. **Indexed draws skip Vertex-step OOB**
>   (indices are GPU-side) and instead bound the index buffer by
>   `firstIndex+indexCount` + Instance-step buffers — matches Dawn.
>   `baseVertex` not used for bounds (GPU-side).
> - C46 strip format checked at draw only for strip topologies.
> - C53–C55 indirect: run the C37–C42 pre-draw state checks, then
>   INDIRECT usage / 4-byte offset / `offset+args ≤ size`
>   (args = 16 Draw / 20 DrawIndexed / 12 Dispatch). No vertex/index
>   count OOB on indirect (counts are GPU-side).
> - **`firstInstance` feature gating: accepted unconditionally
>   (divergence).** webgpu.h exposes no stable
>   indirect-first-instance toggle in our header; Dawn's
>   `IndirectFirstInstance`-feature path has no canonical webgpu.h
>   analog. Revisit if/when the feature is added (cf. AllowUnsafeAPIs
>   divergence pattern).

### P6.7 RenderBundle
- **C65–C68** bundle encoder descriptor (≥1 format, count, renderable;
  pipeline color/depth/sample format match). :602/:797. ☑ (P6.7)
- **C69–C74** bundle state independence, ExecuteBundles state-clear,
  Finish-twice, ExecuteBundles format/sample match, multi-execute.
  :272/:592/:942. ☑ (P6.7)

> P6.7 notes / divergences:
> - `RenderBundleEncoder` is its own deferred-error root (mirrors the
>   P6.1 model but not a child of `CommandEncoder`); recorded-command
>   errors surface at `wgpuRenderBundleEncoderFinish` as one device
>   error + an error `RenderBundle`. Bundle draw/set commands **reuse
>   the P6.5/P6.6 core validators** (`validate_render_draw_state`,
>   `validate_set_index/vertex_buffer`, `validate_indirect_buffer`).
> - C72/C73 via an `AttachmentSignature` {ordered color formats,
>   depthStencil format, sampleCount}: derived from the render-pass
>   descriptor (`render_pass_attachment_signature`), the bundle
>   encoder descriptor, and `RenderPipeline` (fragment targets / depth
>   / multisample) — `ExecuteBundles` requires bundle == pass; C68
>   requires bundle SetPipeline == bundle encoder descriptor.
> - C69 `ExecuteBundles` clears the render pass's pipeline/bind-group/
>   vertex/index state; C74 same/multiple bundles allowed.
> - **Invalid-descriptor encoder error reported exactly once** (at
>   creation), per core principle 3: a bad descriptor sets an
>   `Errored` lifecycle; `Finish` then returns an error
>   `RenderBundle` **without** re-emitting a device error, and further
>   recorded commands are silently dropped (Phase-review fix).
> - Debug-group **balance** in bundles (C62) → Defer→P6.8 (Insert/
>   Push/Pop are recording no-ops here). Usage-scope/submit → P6.9.

### P6.8 Debug markers
- **C60/C61/C62/C64** push/pop balance in render/compute pass &
  render bundle; InsertDebugMarker no-op. `DebugMarkerValidationTests`.
  ☑ (P6.8)

> P6.8 notes: C60/C61 (render/compute pass balance at `End` +
> pop-underflow) and C63 (encoder, at `Finish`) were already
> implemented in P6.1 via `PassEncoderState.debug_group_depth`;
> C64 (`InsertDebugMarker` is a no-op — arbitrary label, any nesting,
> even outside a group) already held (pass = guard-only; encoder/
> bundle = no-op). P6.8 adds **C62**: `RenderBundleEncoder`
> push/pop now track `debug_group_depth` (pop-underflow ⇒ error;
> unbalanced depth at `finish()` Recording branch ⇒ error) via the
> P6.7 bundle deferred-error model (surfaced at
> `wgpuRenderBundleEncoderFinish`). All four scopes ported in
> `debug_marker_validation.rs`.

### P6.9 Resource usage tracking + submit-time buffer state
- **C75/C77** writable buffer/texture binding aliasing in a scope.
  :308. ☑ (P6.9)
- **C76** read+write same resource conflict in a pass. ☑ (P6.9)
- **C78** attachment + sampled-in-pass conflict (T54–T56). ☑ (P6.9)
- **C80–C82** submit: command buffer finished/valid; referenced buffer
  not mapped (B39) / not destroyed (B40/B41). ☑ (P6.9)

> P6.9 notes / divergences:
> - **Usage scope** validated at each draw/dispatch (deferred-error →
>   `finish()`). Per bound bind-group entry the BGL `kind` classifies
>   access: write = `Buffer{Storage}` / `StorageTexture{WriteOnly|
>   ReadWrite}`; read = `Buffer{Uniform|ReadOnlyStorage}` /
>   `Texture` (sampled) / `StorageTexture{ReadOnly}`; `Sampler`
>   ignored. Buffers keyed by `Buffer::same` + overlapping
>   `[offset,offset+size)` (dynamic offset applied); textures by
>   `Texture::same` via `TextureView::texture()`. write+write
>   (C75/C77) or write+read (C76) ⇒ error; read+read allowed. **C78**
>   render-pass color/depth/resolve attachment textures stashed at
>   `begin_render_pass`; a bound texture equal to an attachment ⇒
>   error.
> - **Deferred:** ExecuteBundles-contributed resource usage is NOT
>   merged into the render pass's usage scope (the bundle still
>   validates its own pipeline/format/state; its internal resource
>   conflicts are out of scope). Documented gap; the ported
>   aliasing/usage/subresource tests bind resources directly in the
>   pass. Revisit alongside real-GPU work.
> - **Submit (C80–C82):** the `CommandEncoder` accumulates the
>   `Arc<Buffer>`s referenced by copies/clear/encoder-WriteBuffer/
>   buffer↔texture copies and every bind-group/vertex/index/indirect
>   buffer recorded in its passes (only on successfully-recorded
>   commands); `finish()` (success path) moves them into the
>   `CommandBuffer` (+ one-shot `submitted` flag). `Queue::submit`
>   first-match-wins: C80 error CB / double-submit (per-CB flag and
>   in-batch duplicate via `Arc::ptr_eq`); C81 referenced buffer
>   `map_state != Unmapped` (B39); C82 referenced buffer
>   `is_destroyed()` (B40/B41). One device error; clean submit
>   dispatches nothing. (Real GPU execution still Defer→P7.)
> - `TexelCopyBufferInfo`/`TexelCopyTextureInfo` changed `&Buffer`/
>   `&Texture` → `Arc<Buffer>`/`Arc<Texture>` so the encoder can
>   retain references for submit tracking (FFI updated accordingly).

## Open questions

- CommandBuffer model: store the recorded command list + referenced
  resource `Arc`s + a first-error; `Finish` validates the deferred
  rules; `Submit` runs C80–C82.
- Usage-scope granularity: per-buffer-range / per-texture-subresource
  tracking sufficient for the ported tests.
- How much pass/draw validation is deferred-to-Finish vs immediate —
  decide per rule from the Dawn test (record divergences).
