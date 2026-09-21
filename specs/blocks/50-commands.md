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
- **CTS finding F-067 (2026-06-10):** two buffer↔texture-copy
  (`copyBufferToTexture`/`copyTextureToBuffer`) validation gaps —
  - **(a) combined depth+stencil aspect.** A buffer copy of a *combined*
    depth+stencil format (`depth24plus-stencil8`, `depth32float-stencil8`) with
    `aspect = All` is never permitted — only a single aspect (depth-only /
    stencil-only) is buffer-copyable. (This is the **inverse** of the T2T rule,
    which *requires* `All` for combined DS — see `validate_texture_to_texture_copy`.)
    `validate_buffer_texture_copy` must reject `aspect == All` when the format has
    both a depth and a stencil aspect. Single-aspect DS formats (`stencil8`,
    `depth16unorm`, `depth32float`) and colour formats are unaffected.
  - **(b) buffer device-mismatch.** The copy's buffer must belong to the same
    device as the command encoder (mirrors the bind-group `same`-device check);
    a mismatched-device buffer is a validation error. Requires threading the
    owning device of each copy resource through the FFI (as `BindGroupResource`
    already carries `device`).
  - **(c) bytesPerRow alignment** for single-aspect DS formats was investigated
    and is **already enforced** (`validate_texel_copy_layout`,
    `require_bytes_per_row_alignment = true`); confirm via the CTS re-run that no
    residual remains after (a)+(b).
  **Resolution notes (2026-06-10):** (a) applies to **both** the copy path
  (`validate_buffer_texture_copy`) and the `writeTexture` path
  (`validate_queue_write_texture`) — the rule is identical. (b) is enforced as a
  **deferred** encoder error (recorded, surfaced at `finish()`), per the WebGPU
  encoder error model; the older eager FFI buffer-device check was removed in its
  favour. Additionally, `wgpuQueueWriteTexture` permits an *arbitrary*
  (non-texel-aligned) `bytesPerRow`/offset, which Vulkan's texel-unit
  `bufferRowLength` cannot represent: `Queue::write_texture` repacks rows into a
  tightly-packed staging layout (`repack_texel_rows`) whenever the caller's
  stride/offset is not texel-block-aligned, and the Vulkan HAL passes
  `bufferRowLength = 0` for single-block-row copies.
  See `specs/tracking/cts-coverage.md` → F-067.

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
  - **CTS finding F-066 (2026-06-10):** `setViewport` bounds were too strict —
    they rejected an in-bounds viewport. The allowed rectangle is **not** clamped
    to `maxTextureDimension2D`; per WebGPU/Dawn the bound is
    `maxViewportBounds = 2 × maxTextureDimension2D`: reject when
    `x < -maxViewportBounds`, `y < -maxViewportBounds`,
    `x + width > maxViewportBounds − 1`, or `y + height > maxViewportBounds − 1`,
    **plus** the separate per-dimension limits `width > maxTextureDimension2D` /
    `height > maxTextureDimension2D` (and the existing non-negative/finite
    checks). See `specs/tracking/cts-coverage.md` → F-066.
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

## Pass-encoder resource retention on `end()` (post-COMPLETE addition)

> **Status: IMPLEMENTED — C1-C3 landed 2026-09-21 (`0d470a1`).** Phase
> Review the same day: 0 CRITICAL, 2 MAJOR, 5 MINOR; both MAJORs were
> defects in this section's own field list and are corrected inline below.
> Measurements and the finding ledger:
> `tracking/pass-encoder-retention.md`.

### Problem (measured 2026-09-21)

`PassEncoderState` holds strong references to everything bound during
encoding — `compute_pipeline`, `render_pipeline`, `bind_groups`,
vertex/index buffers, attachment textures, the occlusion query set, the
usage-scope vectors. `PassEncoderInner::end()` sets `ended = true` and
`mem::take`s `render_commands`, but **clears none of the rest**. So an
ended pass encoder goes on pinning every resource it ever bound, for as
long as its handle lives.

That turns a caller's leaked handle into an unbounded library leak. Found
by webgpu-native-cts, which leaks pass encoders at 174 call sites (no
`finalize()` tracking for them; being fixed in that repo separately).
Measured on one CTS file
(`shader,execution,expression,binary,bitwise:bitwise_or:*`, yawgpu
`d6f865b`, NVIDIA RTX 5060 Ti):

| counter | value |
|---|---:|
| `wgpuDeviceCreateShaderModule` / `…Release` | 476 / 476 |
| `wgpuDeviceCreateComputePipeline` / `…Release` | 476 / 476 |
| `wgpuCommandEncoderBeginComputePass` | 476 |
| `wgpuComputePassEncoderRelease` | **0** |
| `tint::Program` created / destroyed | 442 / **0** |

Peak RSS 636 MB for 36 passing cases, ~449 MB of it live at process exit
in `tint::core::constant::Manager`. The retention chain, confirmed by
refcount instrumentation:

```
leaked WGPUComputePassEncoder handle
  -> PassEncoderState.compute_pipeline : Arc<ComputePipeline>
  -> ComputePipelineInner._shader_module : Arc<ShaderModule>
  -> ShaderModuleInner._source : Box<ReflectedModule>
  -> yawgpu_tint::Program
```

Every C handle is released correctly and the FFI handle itself drops; at
`wgpuShaderModuleRelease` the core object still shows `strong = 2`, the
second reference being the pipeline held by the ended pass encoder. The
same four API calls in a standalone program — no pass encoder — destroy
the Tint program every time. Full record:
`tracking/pass-encoder-retention.md`.

### Decisions

**D1 — a successful `end()` drops the pass's encoding-time references.**
After `end()` returns on any path that sets `ended = true`,
`PassEncoderState` must hold no strong reference to a resource that
existed only to validate or record commands. Clear:

`render_pipeline`, `compute_pipeline`, `bind_groups`, `vertex_buffers`,
`index_buffer`, `attachment_textures`, `attachment_texture_uses`,
`render_color_attachments`, `render_depth_stencil_attachment`,
`occlusion_query_set`, `command_referenced_buffers`, `scope_buffer_uses`,
`scope_texture_uses`, **`scope_usage_index`**, **`render_commands`**.

**`scope_usage_index` added 2026-09-21**, after C1/C2 measured that the
original fourteen do not deliver acceptance criterion 1's "attachment
textures" for a render pass: `LenientUsageScopeIndex::texture_uses_by_identity`
stores *clones* of `TextureScopeUse`, and `TextureScopeUse.texture` is an
`Arc` handle, so a leaked ended render-pass encoder still pinned its
attachment textures through the private index. Clearing it is safe by the
same argument as the rest — the index is read only by
`LenientUsageScopeIndex::validate_and_record`, reached only through
`record_resource_usage_scope_uses`, i.e. only from `record_pass_command`
closures. It also *restores* an invariant rather than weakening one: the
index's own doc comment justifies its raw-address keys with "every
successful index insertion has a corresponding owning handle in the scope
history", which clearing `scope_buffer_uses` / `scope_texture_uses` alone
would falsify.

**Corrected by the Phase Review, 2026-09-21 — two of this list's original
entries were wrong:**

- **`render_commands` must be cleared** (M1). This text originally said it
  "is already `mem::take`n and stays that way". It is not: the `mem::take`
  in `end()` sits *inside* the `if !render_color_attachments.is_empty() ||
  render_depth_stencil_attachment.is_some()` branch. A render pass begun
  with zero color attachments and no depth-stencil takes the `else` arm, so
  its recorded commands — which own `Arc<RenderPipeline>`,
  `Arc<BindGroup>`, `Arc<Buffer>`, `Arc<RenderBundle>` — were neither taken
  nor cleared. That path is reachable: `begin_render_pass` records the
  descriptor error and still returns a usable encoder, so the leak shape
  this section exists to fix survived on the error path.
- **`immediate_data` must NOT be cleared** (M2). It was listed as
  "unbounded caller data"; it is a fixed 64-byte scratch
  (`MAX_IMMEDIATE_DATA_BYTES`), allocated once, only ever overwritten
  byte-range-wise, and `PassEncoderState::new`'s own comment states it "is
  never reset for the lifetime of the pass". `record_set_immediates` and
  `overlay_written_immediates` index it **unchecked** under that documented
  precondition. Clearing it recovers 64 bytes and arms a latent panic
  behind an invariant the same change silently broke — a bad trade against
  CLAUDE.md principle 3 and this section's own acceptance criterion 6.

Scalar/bookkeeping fields (`ended`, `debug_group_depth`, `draw_count`, the
occlusion-query index sets, the dirty flag, `limits`) and the 64-byte
`immediate_data` scratch are untouched: they are cheap and there is nothing
to release. (An earlier wording claimed some are "read by later
validation" — they are not; every one is consumed *above* the clear point
or never read again once `ended`.)

**Why this is safe.** The parent owns everything execution needs before
`end()` returns: compute commands are recorded into the `CommandEncoder`
at dispatch time (`compute_pass.rs:94,134`), each carrying its own
`pipeline` `Arc` and a *clone* of `bind_groups`; render commands are
moved into the parent's `RenderPassCommand` inside `end()` itself. The
state's copies are redundant the moment the pass ends.

**Precedent.** `subpass.rs:424-425` already does exactly this shape —
`draw_state.render_pipeline = None; draw_state.bind_groups.clear();` —
when bundle replay invalidates outer render bindings.

**D2 — error behaviour must not change, and that must be verified, not
assumed.** Every operation on an ended pass encoder must still produce a
byte-identical device error. `record_pass_command` checks `ended` before
touching resource state, so clearing should be unobservable — but the
slice must **enumerate every read of each cleared field and show it is
gated on `!ended`** (or on a path unreachable after `end()`). A read that
is not so gated is a finding to report, not something to work around.

**D3 — failure paths leave state untouched.** `end()`'s early returns
(already ended, parent finished, not the active pass) return before
`ended = true` and must not clear anything. Only the first two are
constructible from the public API with bindings still in place — a pass
that is not the active pass has necessarily already been ended — so the
parent-finished path is the one a test should pin (C1 measured this). The "soft" error paths
that run *after* `ended = true` (unbalanced debug groups, open occlusion
query, draw count exceeded) still record their error and still clear.

**D4 — this bounds the damage, it does not fix leaked handles.** A pass
encoder that is leaked **without** being ended still retains everything;
nothing can be done about that, because an unended encoder is
legitimately still encoding. A released encoder — ended or not — frees
everything through `Drop` today and keeps doing so. The library's
obligation is to stop *amplifying* a caller's handle leak, which this
does: the CTS's ended-and-leaked encoders would retain only the
`PassEncoderInner` shell.

### Slices

- **C1 — clear on end.** D1 + D3 in `PassEncoderInner::end()`.
  Inline `#[cfg(test)]` test in `pass.rs`: build a compute pass, set a
  pipeline, dispatch, `end()`, then assert `Arc::strong_count` of the
  pipeline is back to the count the caller holds — i.e. the encoder
  released its reference. A second test asserts the same for a render
  pass with a bound bind group and vertex buffer. No GPU: Noop.
- **C2 — the no-regression proof.** D2. Enumerate the reads; run the full
  Noop suite and the Dawn-ported validation ports, which are where an
  error-message change would surface.
- **C3 — real-GPU confirmation.** Re-run the CTS repro *without* fixing
  the CTS side, so the measurement isolates this change: peak RSS for
  `bitwise_or:*` must drop from 636 MB, and `tint program destroy` must
  become non-zero. Record in `tracking/pass-encoder-retention.md`.

### Acceptance criteria

1. After `end()`, the pass encoder holds no strong reference to its bound
   pipeline, bind groups, buffers, attachment textures or query set —
   pinned by inline unit tests via `Arc::strong_count`. The
   attachment-texture half of this needs `scope_usage_index` cleared (see
   D1) and must have its own assertion; the other resources were not
   enough to prove it.
2. `cargo test --workspace` green on Noop with **no test edited to
   accommodate the change**. A test that has to change is a signal the
   behaviour changed observably; stop and report instead.
3. No device-error message, and no error's presence or absence, changes.
4. With the CTS still leaking pass-encoder handles, its
   `bitwise_or:*` peak RSS drops from **636 MB to ≤ 300 MB** and Tint
   program destroys go from 0 to ≥ 400 of 442.
5. `cargo clippy --workspace --all-targets -- -D warnings` clean, and the
   same with `--features vulkan`.
6. No new panic path; no `unwrap`/`expect` added to non-test code.
7. Phase Review clean (no open CRITICAL/MAJOR).

### Out of scope

- The webgpu-native-cts side (releasing its pass encoders). Independent,
  tracked in that repo; this change must be correct and measurable on its
  own without it.
- Clearing on `CommandEncoder::finish()` or on parent drop. `end()` is
  the point where the pass provably no longer needs its bindings; extra
  clearing points are unneeded complexity.
- Any change to what a `CommandBuffer` retains. A submitted command
  buffer legitimately holds its pipelines until released.
- The `ReflectedModule` memoization caches (block 95). They live and die
  with the module and are not implicated — the module was being kept
  alive, not leaking on its own.

## Open questions

- CommandBuffer model: store the recorded command list + referenced
  resource `Arc`s + a first-error; `Finish` validates the deferred
  rules; `Submit` runs C80–C82.
- Usage-scope granularity: per-buffer-range / per-texture-subresource
  tracking sufficient for the ported tests.
- How much pass/draw validation is deferred-to-Finish vs immediate —
  decide per rule from the Dawn test (record divergences).
