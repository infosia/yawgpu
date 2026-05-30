# CTS coverage ledger — `api/validation`

Live status of porting the WebGPU CTS (`gpuweb/cts`,
`src/webgpu/api/validation/`) onto the yawgpu C FFI. Methodology,
exclusions, harness contract, directory layout, and the `$CTS`
checkout-path convention: `specs/blocks/91-cts-conformance.md`.

This ledger is at **spec-file granularity** (129 rows). The per-`g.test()`
checklist for an area lives in that area's Phase-B task handoff; this
table is the master index of which areas are done vs untouched.

**The CTS port is counted independently of the legacy Dawn tests.** A
spec is `ported` once every non-excluded `g.test()` has its own Rust
`#[test]` under `tests/cts/validation/…`, **regardless of whether a
legacy `yawgpu/tests/*.rs` already exercises the same rule** —
duplication across the two layers is allowed. There is therefore no
"partially covered by legacy" state: a spec is `todo` until its CTS
cases are ported, then `ported`.

## Status legend

| status | meaning |
|---|---|
| `ported` | every non-excluded `g.test()` has a Rust `#[test]` under `tests/cts/validation/…`, green on Noop |
| `ported*` | as `ported`, but some subcases were excluded or `#[ignore]`d (web-only, or deferred behind a core gap with a spec-correct assertion) — reason in-row |
| `todo` | not yet ported to `tests/cts/` (a legacy Dawn test may still overlap — see related-test column) |
| `N/A` | excluded (web/canvas/WebCodecs/IDL/empty); reason in-row |

The **related legacy test** column is *informational only*: the legacy
`yawgpu/tests/*.rs` (Dawn-ported) file that overlaps this CTS spec, kept
as a pointer to prior art a porter may consult for the Rust idiom. It is
never a reason to skip a CTS case.

## Snapshot

- 129 spec files / 704 `g.test()` cases total in `api/validation`.
- Excluded (`N/A`): 7 whole spec files (web/empty/multiDraw + setImmediates/immediate absent).
- `ported`: 112. Complete areas: `buffer/`, texture creation,
  `image_copy/`, `queue/`, shader_module, compute_pipeline, immediates,
  layout_shader_compat, `render_pipeline/`, bind-group family,
  `render_pass/`, `encoding/`, state & lifecycle, `resource_usages/`,
  `capability_checks/limits/` (35). Several `ported*` with subcases
  `#[ignore]`d behind core gaps — assertions are spec-correct.
- `todo`: 10 spec files — only `capability_checks/features/` (7) and
  texture format-capability (`texture/bgra8unorm_storage`,
  `float32_filterable`, `rg11b10ufloat_renderable`).
- Known core gaps surfaced (recommended follow-up): evaluate
  pipeline-overridable constants at createComputePipeline (workgroup-size
  / storage-size limits + override-expression errors); **inter-stage
  vertex-output↔fragment-input matching** (location/type/interpolation/
  max-variables — currently unvalidated, 8/9 inter_stage cases ignored);
  pipeline-layout/shader resource compatibility (createComputePipeline +
  createRenderPipeline); depth-clip-control gating of unclippedDepth;
  storage-texture format/access in render auto-layout.

## Coverage matrix

| spec file | cases | related legacy test (info) | status |
|---|---|---|---|
| **buffer/** | | | |
| `create.spec.ts` | 5 | buffer_creation_validation.rs | `ported` → `cts/validation/buffer/create.rs` |
| `destroy.spec.ts` | 4 | — | `ported` → `cts/validation/buffer/destroy.rs` |
| `mapping.spec.ts` | 33 | buffer_map_validation.rs / buffer_mapped_range_validation.rs | `ported*` → `cts/validation/buffer/mapping.rs` (gc_behavior,* N/A: JS GC; earlyRejection timing adapted) |
| `threading.spec.ts` | 0 | — | `N/A` — web (worker postMessage); 0 cases |
| **capability_checks/** | | | |
| `features/clip_distances.spec.ts` | 2 | — | `todo` |
| `features/query_types.spec.ts` | 2 | — | `todo` |
| `features/subgroup_size_control.spec.ts` | 1 | — | `todo` |
| `features/texture_component_swizzle.spec.ts` | 4 | — | `todo` — compatibility_mode deferred |
| `features/texture_formats.spec.ts` | 13 | features_validation.rs / texture_format_validation.rs | `todo` — canvas_configuration* excluded |
| `features/texture_formats_tier1.spec.ts` | 8 | — | `todo` |
| `features/texture_formats_tier2.spec.ts` | 3 | — | `todo` |
| `limits/maxBindGroups.spec.ts` | 4 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxBindGroupsPlusVertexBuffers.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxBindingsPerBindGroup.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxBufferSize.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxColorAttachmentBytesPerSample.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxColorAttachments.spec.ts` | 5 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxComputeInvocationsPerWorkgroup.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxComputeWorkgroupSizeX.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxComputeWorkgroupSizeY.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxComputeWorkgroupSizeZ.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxComputeWorkgroupStorageSize.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxComputeWorkgroupsPerDimension.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxDynamicStorageBuffersPerPipelineLayout.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxDynamicUniformBuffersPerPipelineLayout.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxInterStageShaderVariables.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxSampledTexturesPerShaderStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxSamplersPerShaderStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageBufferBindingSize.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageBuffersInFragmentStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageBuffersInVertexStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageBuffersPerShaderStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageTexturesInFragmentStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageTexturesInVertexStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxStorageTexturesPerShaderStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxTextureArrayLayers.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxTextureDimension1D.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxTextureDimension2D.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxTextureDimension3D.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxUniformBufferBindingSize.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxUniformBuffersPerShaderStage.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxVertexAttributes.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxVertexBufferArrayStride.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/maxVertexBuffers.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/minStorageBufferOffsetAlignment.spec.ts` | 4 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| `limits/minUniformBufferOffsetAlignment.spec.ts` | 4 | — | `ported*` → `cts/validation/capability_checks/limits/` (active where core enforces; rest `#[ignore]`d with real at/over bodies) |
| **(top-level)/** | | | |
| `compute_pipeline.spec.ts` | 19 | compute_pipeline_validation.rs | `ported*` → `cts/validation/compute_pipeline.rs` (override/storage + resource_compatibility cases `#[ignore]`d: core does not yet evaluate pipeline overrides at createComputePipeline nor reject layout/shader resource mismatches) |
| `createBindGroup.spec.ts` | 27 | bind_group_validation.rs | `ported*` → `cts/validation/create_bind_group.rs` (5 external_texture,* N/A: web; 8 `#[ignore]`d: component-type, destroyed buffer/texture, BGL device-mismatch, storage-texture mip/format, effective-binding-size %4, sampler compare-type core gaps) |
| `createBindGroupLayout.spec.ts` | 11 | bind_group_layout_validation.rs | `ported*` → `cts/validation/create_bind_group_layout.rs` (6 `#[ignore]`d: vertex-stage storage restrictions, multisample sampleType, cross-BGL resource aggregation, storage-texture dimension/format core gaps) |
| `createPipelineLayout.spec.ts` | 7 | pipeline_layout_validation.rs | `ported*` → `cts/validation/create_pipeline_layout.rs` (5 `#[ignore]`d: dynamic-buffer max, 3 null/sparse-BGL slots, immediate_data_size) |
| `createSampler.spec.ts` | 2 | sampler_validation.rs | `ported` → `cts/validation/texture/create_sampler.rs` |
| `createTexture.spec.ts` | 21 | texture_creation_validation.rs | `ported` → `cts/validation/texture/create_texture.rs` |
| `createView.spec.ts` | 10 | texture_view_validation.rs | `ported` → `cts/validation/texture/create_view.rs` |
| `debugMarker.spec.ts` | 2 | debug_marker_validation.rs | `ported` → `cts/validation/debug_marker.rs` |
| `dispatch.spec.ts` | 2 | — | `ported*` → `cts/validation/dispatch.rs` (2 `#[ignore]`d: linear_indexing shader-feature/range unvalidated; indirect variant is operation/readback) |
| `error_scope.spec.ts` | 6 | error_scope_validation.rs | `ported` → `cts/validation/error_scope.rs` |
| `getBindGroupLayout.spec.ts` | 4 | get_bind_group_layout_validation.rs | `ported*` → `cts/validation/get_bind_group_layout.rs` (2 index_range `#[ignore]`d: core rejects index beyond concrete layout count, CTS expects empty layout < maxBindGroups; unique_js_object adapted — JS identity N/A) |
| `gpu_external_texture_expiration.spec.ts` | 6 | — | `N/A` — web (WebCodecs external texture) |
| `layout_shader_compat.spec.ts` | 1 | — | `ported*` → `cts/validation/layout_shader_compat.rs` (the case is `#[ignore]`d: core does not reject layout/shader resource mismatches — the earlier "active mismatch cases" were false-greens, corrected) |
| `non_filterable_texture.spec.ts` | 1 | — | `ported*` → `cts/validation/non_filterable_texture.rs` (`#[ignore]`d: core does not reject filtering sampler + non-filterable texture in shader use) |
| **encoding/** | | | |
| `beginComputePass.spec.ts` | 4 | — | `ported*` → `cts/validation/encoding/begin_compute_pass.rs` (2 active; 2 `#[ignore]`d: timestamp query-set device-mismatch, dup-undefined index) |
| `beginRenderPass.spec.ts` | 4 | — | `ported*` → `cts/validation/encoding/begin_render_pass.rs` (4 `#[ignore]`d: attachment/query-set device-ownership not validated at finish — core gap) |
| `createRenderBundleEncoder.spec.ts` | 6 | render_bundle_validation.rs | `ported*` → `cts/validation/encoding/create_render_bundle_encoder.rs` (4 active; 2 `#[ignore]`d: maxColorAttachmentBytesPerSample not enforced) |
| `encoder_open_state.spec.ts` | 4 | command_encoder_lifecycle_validation.rs | `ported` → `cts/validation/encoding/encoder_open_state.rs` (setImmediates/multiDraw* subcommands N/A: absent in C ABI) |
| `encoder_state.spec.ts` | 6 | command_encoder_lifecycle_validation.rs / pass_state_validation.rs | `ported*` → `cts/validation/encoding/encoder_state.rs` (4 active; 2 `#[ignore]`d: core poisons parent encoder on invalid pass-end, CTS expects finish to still succeed) |
| `programmable/pipeline_bind_group_compat.spec.ts` | 10 | resource_usage_tracking_validation.rs | `ported` → `cts/validation/encoding/programmable/pipeline_bind_group_compat.rs` (all 10 active; core fix: skip empty BGL slots + binding-number-keyed BGL compat) |
| `programmable/pipeline_immediate.spec.ts` | 4 | — | `N/A` — depends on setImmediates (no yawgpu export / core immediate-data command) |
| `queries/begin_end.spec.ts` | 4 | query_validation.rs | `ported*` → `cts/validation/encoding/queries/begin_end.rs` (3 active; nesting `#[ignore]`d: CTS-unimplemented) |
| `queries/general.spec.ts` | 3 | query_validation.rs | `ported` → `cts/validation/encoding/queries/general.rs` |
| `queries/resolveQuerySet.spec.ts` | 6 | query_validation.rs | `ported*` → `cts/validation/encoding/queries/resolve_query_set.rs` (4 active; 2 `#[ignore]`d: destroyed submit-timing, device-mismatch) |
| `render_bundle.spec.ts` | 6 | render_bundle_validation.rs | `ported*` → `cts/validation/encoding/render_bundle.rs` (5 active; 1 `#[ignore]`d: depth/stencil readonly not in attachment signature) |
| **encoding/cmds/** | | | |
| `clearBuffer.spec.ts` | 8 | — | `ported*` → `cts/validation/encoding/cmds/clear_buffer.rs` (6 active; 2 `#[ignore]`d: destroyed-buffer submit-timing, device-mismatch) |
| `compute_pass.spec.ts` | 6 | — | `ported*` → `cts/validation/encoding/cmds/compute_pass.rs` (3 active; 3 `#[ignore]`d: error-pipeline set-time, destroyed indirect submit-timing, indirect device-mismatch) |
| `copyBufferToBuffer.spec.ts` | 8 | command_buffer_copy_validation.rs | `ported*` → `cts/validation/encoding/cmds/copy_buffer_to_buffer.rs` (7 active; 1 `#[ignore]`d: destroyed-buffer submit-timing) |
| `copyTextureToTexture.spec.ts` | 12 | command_texture_copy_validation.rs | `ported*` → `cts/validation/encoding/cmds/copy_texture_to_texture.rs` (8 active; 4 `#[ignore]`d: destroyed-texture submit-timing, device-mismatch, aspect strictness, compressed-format feature) |
| `debug.spec.ts` | 3 | debug_marker_validation.rs | `ported` → `cts/validation/encoding/cmds/debug.rs` |
| `index_access.spec.ts` | 2 | — | `ported` → `cts/validation/encoding/cmds/index_access.rs` |
| `render/draw.spec.ts` | 8 | — | `ported*` → `cts/validation/encoding/cmds/render/draw.rs` (5 active; 3 `#[ignore]`d: vertex-OOB lastStride, maxDrawCount unmodeled, last_buffer_setting CTS-unimplemented) |
| `render/dynamic_state.spec.ts` | 8 | — | `ported*` → `cts/validation/encoding/cmds/render/dynamic_state.rs` (5 active; 3 `#[ignore]`d: viewport/scissor attachment-bounds gaps; scissor negative-arg N/A: C unsigned) |
| `render/indirect_draw.spec.ts` | 5 | — | `ported*` → `cts/validation/encoding/cmds/render/indirect_draw.rs` (3 active; 2 `#[ignore]`d: destroyed-buffer submit-timing, indirect-buffer device-mismatch) |
| `render/indirect_multi_draw.spec.ts` | 6 | — | `N/A` — multiDraw* absent from yawgpu C ABI (no multiDrawIndirect/Indexed symbols) |
| `render/setIndexBuffer.spec.ts` | 5 | — | `ported*` → `cts/validation/encoding/cmds/render/set_index_buffer.rs` (3 active; 2 `#[ignore]`d: destroyed-buffer submit-timing, bundle device-mismatch) |
| `render/setPipeline.spec.ts` | 2 | — | `ported*` → `cts/validation/encoding/cmds/render/set_pipeline.rs` (2 `#[ignore]`d: error-pipeline validated at draw-time not setPipeline; bundle device-mismatch) |
| `render/setVertexBuffer.spec.ts` | 6 | — | `ported*` → `cts/validation/encoding/cmds/render/set_vertex_buffer.rs` (4 active; 2 `#[ignore]`d: destroyed-buffer submit-timing, bundle device-mismatch) |
| `render/state_tracking.spec.ts` | 4 | — | `ported*` → `cts/validation/encoding/cmds/render/state_tracking.rs` (2 active; 2 `#[ignore]`d: CTS-unimplemented all_needed_*) |
| `render_pass.spec.ts` | 0 | — | `N/A` — empty placeholder; 0 cases |
| `setBindGroup.spec.ts` | 6 | — | `ported*` → `cts/validation/encoding/cmds/set_bind_group.rs` (6 `#[ignore]`d: core defers all setBindGroup validation to draw/dispatch — index/offset/state/compat unchecked at call; u32array start/length N/A) |
| `setImmediates.spec.ts` | 3 | — | `N/A` — yawgpu has no `wgpu*SetImmediates` export / core immediate-data command (header declares, not implemented) |
| **image_copy/** | | | |
| `buffer_related.spec.ts` | 4 | — | `ported` → `cts/validation/image_copy/buffer_related.rs` |
| `buffer_texture_copies.spec.ts` | 7 | — | `ported*` → `cts/validation/image_copy/buffer_texture_copies.rs` (depth32float-stencil8 subcases deferred: Noop lacks feature) |
| `layout_related.spec.ts` | 7 | — | `ported*` → `cts/validation/image_copy/layout_related.rs` (compressed-format subcases deferred: Noop lacks feature) |
| `texture_related.spec.ts` | 9 | — | `ported*` → `cts/validation/image_copy/texture_related.rs` (compressed-format subcases deferred: Noop lacks feature) |
| **pipeline/** | | | |
| `immediates.spec.ts` | 1 | — | `ported*` → `cts/validation/pipeline/immediates.rs` (immediateSize limit only; shader-side immediate mismatch N/A — yawgpu has no shader immediate model) |
| **query_set/** | | | |
| `create.spec.ts` | 1 | query_validation.rs | `ported*` → `cts/validation/query_set/create.rs` (`#[ignore]`d: core rejects count=0, CTS allows; only >4096 should fail) |
| `destroy.spec.ts` | 2 | query_validation.rs | `ported` → `cts/validation/query_set/destroy.rs` |
| **queue/** | | | |
| `buffer_mapped.spec.ts` | 5 | — | `ported` → `cts/validation/queue/buffer_mapped.rs` |
| `copyToTexture/CopyExternalImageToTexture.spec.ts` | 12 | — | `N/A` — web (ImageBitmap/canvas source) |
| `destroyed/buffer.spec.ts` | 8 | — | `ported` → `cts/validation/queue/destroyed_buffer.rs` |
| `destroyed/query_set.spec.ts` | 4 | — | `ported` → `cts/validation/queue/destroyed_query_set.rs` |
| `destroyed/texture.spec.ts` | 6 | — | `ported` → `cts/validation/queue/destroyed_texture.rs` |
| `submit.spec.ts` | 4 | queue_submit_validation.rs | `ported` → `cts/validation/queue/submit.rs` |
| `writeBuffer.spec.ts` | 4 | queue_buffer_validation.rs | `ported` → `cts/validation/queue/write_buffer.rs` |
| `writeTexture.spec.ts` | 4 | queue_write_texture_validation.rs | `ported` → `cts/validation/queue/write_texture.rs` |
| **render_pass/** | | | |
| `attachment_compatibility.spec.ts` | 12 | — | `ported*` → `cts/validation/render_pass/attachment_compatibility.rs` (6 active: pass↔bundle compat; 6 `#[ignore]`d: pass↔pipeline attachment compat at setPipeline + depthReadOnly — core gap) |
| `render_pass_descriptor.spec.ts` | 32 | render_pass_descriptor_validation.rs | `ported*` → `cts/validation/render_pass/render_pass_descriptor.rs` (21 active; 11 `#[ignore]`d: depthSlice/3D, bytes-per-sample, attachment mip-level-count, transient load/store, depthReadOnly, resolve-format-support core gaps; bindTextureResource subcases N/A) |
| `resolve.spec.ts` | 1 | — | `ported*` → `cts/validation/render_pass/resolve.rs` (`#[ignore]`d: transient resolve target + mip-level-count core gap) |
| **render_pipeline/** | | | |
| `depth_stencil_state.spec.ts` | 9 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/depth_stencil_state.rs` (6 `#[ignore]`d: core gaps in depth/stencil state rules) |
| `float32_blendable.spec.ts` | 1 | — | `ported` → `cts/validation/render_pipeline/float32_blendable.rs` (no-feature validation active; float32-blendable feature subcase deferred on Noop) |
| `fragment_state.spec.ts` | 13 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/fragment_state.rs` (7 `#[ignore]`d: maxColorAttachments/byte-align/blend/write-mask core gaps; dual-source-blending feature) |
| `inter_stage.spec.ts` | 9 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/inter_stage.rs` (8/9 `#[ignore]`d: core does not validate inter-stage location/type/interpolation/limits; only location_superset active) |
| `misc.spec.ts` | 6 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/misc.rs` (external_texture N/A: web; storage_texture format `#[ignore]`d: core gap) |
| `multisample_state.spec.ts` | 3 | render_pipeline_validation.rs | `ported` → `cts/validation/render_pipeline/multisample_state.rs` |
| `overrides.spec.ts` | 10 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/overrides.rs` (2 f16 cases deferred: shader-f16 not on Noop) |
| `primitive_state.spec.ts` | 2 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/primitive_state.rs` (unclipped_depth `#[ignore]`d: depth-clip-control not enforced) |
| `resource_compatibility.spec.ts` | 1 | render_pipeline_validation.rs | `ported*` → `cts/validation/render_pipeline/resource_compatibility.rs` (`#[ignore]`d: layout/shader resource compat core gap) |
| `shader_module.spec.ts` | 3 | render_pipeline_validation.rs / shader_module_validation.rs | `ported` → `cts/validation/render_pipeline/shader_module.rs` |
| `vertex_state.spec.ts` | 12 | vertex_state_validation.rs | `ported` → `cts/validation/render_pipeline/vertex_state.rs` |
| **resource_usages/** | | | |
| `buffer/in_pass_encoder.spec.ts` | 6 | — | `ported*` → `cts/validation/resource_usages/buffer/in_pass_encoder.rs` (5 active; 1 `#[ignore]`d: compute dispatch accessibility matrix) |
| `buffer/in_pass_misc.spec.ts` | 3 | — | `ported*` → `cts/validation/resource_usages/buffer/in_pass_misc.rs` (2 active; 1 `#[ignore]`d: reset-before-draw matrix) |
| `texture/in_pass_encoder.spec.ts` | 11 | — | `ported*` → `cts/validation/resource_usages/texture/in_pass_encoder.rs` (4 active; 7 `#[ignore]`d: subresource mip/layer/aspect overlap, visibility-independent storage-write, replaced-binding scope, bundle usages, unused-bindings — core tracking coarser than CTS) |
| `texture/in_render_common.spec.ts` | 5 | — | `ported*` → `cts/validation/resource_usages/texture/in_render_common.rs` (2 active; 3 `#[ignore]`d: attachment-aliasing / depth-stencil+bind-group / multi-bind-group matrices) |
| `texture/in_render_misc.spec.ts` | 5 | — | `ported*` → `cts/validation/resource_usages/texture/in_render_misc.rs` (1 active; 4 `#[ignore]`d: same-index replacement, unused bind group, per-view usage override) |
| **shader_module/** | | | |
| `entry_point.spec.ts` | 6 | shader_module_validation.rs | `ported` → `cts/validation/shader_module/entry_point.rs` |
| `overrides.spec.ts` | 2 | shader_module_validation.rs | `ported` → `cts/validation/shader_module/overrides.rs` |
| **state/** | | | |
| `device_lost/destroy.spec.ts` | 32 | device_lost_validation.rs | `ported*` → `cts/validation/state/device_lost/destroy.rs` (24 active; 5 `#[ignore]`d: 3 compressed-format feature, 2 async-pipeline lost-device returns ValidationError; 3 N/A web external-texture) |
| **texture/** | | | |
| `bgra8unorm_storage.spec.ts` | 4 | — | `todo` — canvas subcases excluded |
| `destroy.spec.ts` | 4 | — | `ported` → `cts/validation/texture/destroy.rs` |
| `float32_filterable.spec.ts` | 1 | — | `todo` |
| `rg11b10ufloat_renderable.spec.ts` | 5 | — | `todo` |

**Total: 129 spec files / 704 `g.test()` cases.**

## Regenerating this matrix

The case counts come straight from the CTS checkout. To refresh after a
CTS pull (counts only — the mapping / status / exclusion columns are
hand-maintained), point `CTS` at your local `gpuweb/cts` checkout root:

```sh
python3 - "$CTS" <<'PY'
import re, glob, sys
root=sys.argv[1]+"/src/webgpu/api/validation"
for f in sorted(glob.glob(root+"/**/*.spec.ts", recursive=True)):
    rel=f[len(root)+1:]
    n=len(re.findall(r"g\.test\(\s*[`'\"]([^`'\"]+)", open(f,encoding="utf-8",errors="replace").read()))
    print(f"{n:3d}  {rel}")
PY
```
