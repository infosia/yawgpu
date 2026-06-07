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
- `ported`: 122 (all non-`N/A` specs). **The `api/validation` CTS port
  is COMPLETE** — every non-excluded `g.test()` across all 129 spec files
  has a Rust `#[test]` under `tests/cts/validation/…`. Many `ported*`
  with subcases `#[ignore]`d behind core gaps (real spec-correct bodies)
  or feature-gated on Noop — see rows and the core-gap list below.
- `todo`: 0.
- **Phase E (legacy cleanup) — two sweeps, 21 legacy files deleted.**
  *Sweep 1* removed 8 files fully covered by active CTS (66 redundant
  tests): `buffer_creation_validation`, `buffer_map_validation`,
  `buffer_mapped_range_validation`, `debug_marker_validation`,
  `queue_submit_validation`, `texture_creation_validation`,
  `texture_view_validation`, `vertex_state_validation`.
  *Sweep 2* (after follow-ups #1–#8 + findings F-005..F-011 closed the
  gaps that had forced subcases to `#[ignore]`) removed 13 more (~84
  redundant tests), each re-verified per file against the *actual* CTS
  files (the matrix rows below can lag): `command_buffer_copy_validation`,
  `command_texture_copy_validation`, `queue_buffer_validation`,
  `queue_write_texture_validation`, `command_encoder_lifecycle_validation`,
  `bind_group_validation`, `bind_group_layout_validation`,
  `get_bind_group_layout_validation`, `pipeline_layout_validation`,
  `sampler_validation`, `compute_pipeline_validation`,
  `shader_module_validation`, `resource_usage_tracking_validation`.
  **Still KEPT** (each has ≥1 rule CTS only `#[ignore]`s or doesn't cover):
  `render_pipeline_validation` (inter-stage / bytes-per-sample),
  `render_bundle_validation` (maxColorAttachmentBytesPerSample),
  `render_pass_descriptor_validation` (resolve-format / depthReadOnly /
  transient / bytes-per-sample), `pass_state_validation` (eager
  setBindGroup + viewport/scissor bounds + indirect), `device_lost_validation`
  (lost-callback ordering/single-fire/getLostFuture), `error_scope_validation`
  (first-error-kept / uncaptured-callback / destroyed-pop / WaitAnyOnly),
  `features_validation` (CoreFeaturesAndLimits core-vs-compat + tier
  implications), `limits_validation` (request_device clamping — CTS only
  covers the at/over pipeline path), `query_validation` (count==0 allowed —
  CTS case ignored), `texture_format_validation` (caps-sanity asserts +
  F-009 storage regression lock). These remain the worklist for the next
  core-gap closures. KEEP-forever (no CTS analog): future_modes,
  gles_context_backend_chain, instance_smoke, label_validation,
  multiple_device_validation, object_caching_validation,
  pipeline_async_validation, surface_validation, unsafe_api_validation.
- **Core-gap follow-up #1 (device-ownership) — DONE.** yawgpu now
  validates resource owner-device at the record-time FFI entry points
  (createBindGroup BGL; begin{Render,Compute}Pass attachments/query-sets;
  resolveQuerySet; clearBuffer; copyTextureToTexture; indirect
  dispatch/draw; render-bundle setPipeline/BindGroup/Vertex/IndexBuffer).
  15 device-mismatch CTS tests un-ignored → active & passing. (Per-row
  "device-mismatch ignored" sub-notes below are superseded for these.)
- **Core-gap follow-up #2 (feature-aware format caps) — DONE.** Added
  `Feature` variants (BC/ETC2/ASTC[+sliced-3d] compression,
  depth32float-stencil8, bgra8unorm-storage, float32-filterable) + FFI
  mapping; Noop advertises them; `TextureFormat::caps` is now feature-keyed
  and threaded through all texture/view/BGL/pipeline/pass/bundle/queue/copy
  validation (via `Texture::format_caps()` using stored device features).
  ~15 format-feature CTS tests un-ignored → active & passing
  (`capability_checks/features/texture_formats{,_tier1,_tier2}`,
  `texture/{bgra8unorm_storage,float32_filterable,rg11b10ufloat_renderable}`);
  only canvas/surface fixture subcases remain ignored. Regressions in
  compressed-format tests/e2e updated to request the feature.
- **Core-gap follow-up #3 (Batch B, create-time validation) — DONE.**
  +38 CTS tests un-ignored → active: compute pipeline-override evaluation
  (naga `process_overrides` → workgroup size/storage/arithmetic);
  render-pipeline inter-stage matching (all 8: location/type/interpolation/
  maxInterStageShaderVariables); fragment color/blend (maxColorAttachments,
  bytes-per-sample, blend-factor, writeMask, vec4 source-alpha);
  depth-stencil `depthCompare=always` inert handling (6); device-limit
  relationship validation (~16: invocations vs workgroup,
  maxBindGroupsPlusVertexBuffers, min*Alignment pow2/≥32, binding-size vs
  maxBufferSize, etc.). Still deferred → Batch C: layout/resource
  compatibility matrices, dual-source-blending/shader-f16 (feature adds),
  bytes-per-sample format-selection matrices, draw-time relationship.
- **Core-gap follow-up #4 (Batch C, layout/resource compat + misc) — DONE.**
  +30 CTS un-ignored (152→122 remaining): BGL validation (vertex-stage
  writable-storage rejection, multisample float sampleType, cube storage
  dimension, rw-storage format), pipeline-layout (immediateSize %4,
  cross-BGL dynamic + per-stage aggregation), bind-group (destroyed
  resources, effective-size %4, sampler compare-vs-type, component class,
  storage mip/format), getBindGroupLayout (empty default for unused
  in-range slot), compute/render resource compatibility, non-filterable
  gather, query count==0. Also fixed a real bug: explicit pipeline-layout
  cache keys now use core Arc identity (not transient FFI handle address).
  Deferred → Batch D: render-pass/bundle attachment-misc matrices,
  resource_usages subresource granularity, destroyed-resource timing.
- **Core-gap follow-up #5 (Batch D, encoder/command rules) — DONE.**
  +8 CTS un-ignored (122→114): setBindGroup eager validation (index <
  maxBindGroups, dynamic-offset count/alignment/range, error bind group —
  closes the long-standing setBindGroup-deferred gap), setPipeline rejects
  error pipelines immediately (render + compute), setViewport/setScissorRect
  bounds validation. Fixed dynamic-offset range semantics
  (binding_offset+dynamic_offset+binding_size ≤ buffer.size).
  **Still deferred (each a larger model change, "close all gaps" residue):**
  Cluster 1 render-pass/bundle attachment-misc (needs attachment-signature/
  descriptor model expansion: depthSlice 3D, mip-level-count,
  depthReadOnly/stencilReadOnly, resolve-format, transient, pass↔pipeline
  compat); Cluster 3 resource-usage subresource granularity (fine-grained
  mip/layer/aspect usage-scope tracking); Cluster 4 destroyed-resource
  finish→submit timing (behavior change). Plus feature-adds + native-surface
  + a few C-ABI-N/A (u32array start/length, scissor negative args,
  maxDrawCount, vertex-OOB lastStride).
- **Core-gap follow-up #6 (Cluster 1, render-pass attachment-misc) — DONE.**
  +22 CTS un-ignored (114→92); **`render_pass/` now 0 ignores (fully
  active)**. Expanded attachment model: `RenderPassColorAttachment.depth_slice`,
  `RenderPassDepthStencilAttachment.depth_read_only/stencil_read_only`,
  `AttachmentSignature` readonly state (+ FFI conv). New validation:
  3D-color depthSlice (definedness/bounds/overlap), attachment+resolve
  mip-level-count==1, depthReadOnly/stencilReadOnly loadOp/storeOp match,
  resolve-format-support, transient load/store, render-pass↔pipeline
  attachment compat (color/depth/sample + readonly write-state),
  createRenderBundleEncoder bytes-per-sample, storage_texture format.
  Remaining closeable → Cluster 3 (resource-usage subresource
  granularity) + Cluster 4 (destroyed-resource finish→submit timing).
- **Core-gap follow-up #7 (Cluster 4, destroyed-resource timing) — DONE.**
  +10 CTS un-ignored (114→104): destroyed buffers/textures/query-sets
  referenced by a recorded command now make command-buffer/bundle
  `finish()` succeed and **queue `submit()` fail** (was: rejected at
  finish), matching the spec; error/invalid resources still fail at
  finish; invalid `pass.end()` no longer poisons the parent encoder.
  Legacy `command_buffer_copy_validation` / `command_texture_copy_validation`
  and CTS `image_copy` destroyed sub-cases updated to submit-time.
  (Surfaced a pre-existing release-only test UB — dangling `&[temp]`
  render-pass descriptors — fixed separately.) Remaining closeable →
  Cluster 3 (resource-usage subresource granularity).
- **Core-gap follow-up #8 (Cluster 3, resource-usage subresource
  granularity) — DONE. Closeable validation gaps now COMPLETE.**
  +15 CTS un-ignored (82→67); `resource_usages/` now 0 ignored. Extended
  in-pass usage tracking to mip/layer/aspect subresource ranges, pass-scope
  usage accumulation (replaced bindings contribute only after use; render
  bundles import recorded usages on execute), and per-view usage-override
  enforcement (sampled/storage bind groups + attachments).
- **Remaining CTS ignores: ~50** (was ~67; **P1 closed 11**, **P2-core
  closed 6**). P1 wired `capability_checks/limits/*::create_pipeline_layout_at_over`
  aggregation creators (core already validated; no production change).
  P2-core fixed 5 genuine core-validation gaps + un-ignored their 6 cases:
  **maxDrawCount** now parsed from the `WGPURenderPassMaxDrawCount` chain
  and enforced at render-pass finish; **draw-time
  maxBindGroupsPlusVertexBuffers** now checked on bound state;
  **copyTextureToTexture** no longer requires equal src/dst aspect (each
  validated independently) while **combined** depth-stencil T2T still
  requires `All`; **compute-pass timestampWrites** now validated (≥1 index,
  distinct); **device-lost async pipeline** creation now resolves Success
  (no ValidationError). **NOTE: inter-stage matching + maxInterStageShaderVariables
  were already implemented** — the prior matrix row claiming "8/9 ignored"
  was stale (verified: all 9 `render_pipeline/inter_stage.rs` active).
  The rest split into: **(a) test-wiring still pending** —
  `create_pipeline_at_over` shader-resource matrices (core enforces the
  per-stage limits at BGL/pipeline-layout creation; mostly wiring),
  command-encoder matrices (setBindGroup / setVertexBuffer / dynamic-offset
  — gated on the deferred eager-setBindGroup gap), render-bundle
  maxColorAttachments, required-limit `validate`; **(b) remaining genuine
  core gaps** — vertex-buffer draw OOB lastStride (needs investigation),
  dual-source-blending validation, storage-texture format/access in render
  auto-layout; **(c) optional-feature additions** (shader-f16,
  dual-source-blending, subgroups, clip-distances, setImmediates,
  linear_indexing — implement the feature, not just validate);
  **(d) native-surface** (canvas/configure/
  getCurrentTexture — no Noop fixture); and **(e) C-ABI-N/A** /
  CTS-`.unimplemented()` (permanent).
- **External-CTS createTexture findings (webgpu-native-cts 3-way re-test)
  — RESOLVED.** The external runner (vs real Dawn + wgpu-native) surfaced
  three gaps the Noop port did not: **F-005b** `Depth24PlusStencil8`
  aborted on Apple-Silicon Metal — the HAL mapped it to
  `Depth24Unorm_Stencil8` (Intel-only); now maps to
  `Depth32Float_Stencil8` (Metal) / `D32_SFLOAT_S8_UINT` (Vulkan, avoids
  the optional `D24_UNORM_S8_UINT`), verified by a real-M2 Metal e2e
  (`e2e_metal_texture::metal_depth24_plus_stencil8_texture_creation_has_no_device_error`).
  **F-009** `RGBA8Snorm` is storage-capable under `texture-formats-tier1`
  (reversed a prior over-correction that trusted a stale in-tree sanity
  assertion over the CTS). **F-010** compressed `createTexture` now
  rejects non-block-aligned width/height. Each has an active in-tree
  assertion (`cts/validation/texture/create_texture.rs`, `format.rs`
  caps tests, `texture_format_validation.rs`). External
  `expectations/yawgpu.txt` rebaseline is the CTS project's job.
- **External-CTS createView finding F-011 — RESOLVED.** The external
  `createView` port (Texture T9) surfaced three view-dimension gaps (12
  failing cases) the in-tree port missed because its `assert_view_ok`
  only checked the handle was non-null — yawgpu returns a non-null
  *error-view* on validation failure, so positive cases passed
  vacuously. Fixed: (1) `resolve_view_descriptor` now resolves view
  `dimension` before defaulting `arrayLayerCount` *from the resolved
  dimension* (D1/D2/D3→1, Cube→6, D2Array/CubeArray→layers−base), so a
  `2d` view of a multi-layer texture no longer over-defaults its layer
  count; (2) a valid 6-layer `Cube` view is accepted (a missing match
  arm had dropped it to "unsupported"); (3) `Cube`/`CubeArray` views now
  require square faces (`width == height`). `assert_view_ok` was hardened
  to assert an empty error sink, de-vacuuming the whole `create_view.rs`
  suite; active in-tree assertions in
  `cts/validation/texture/create_view.rs` + core unit tests. All
  Noop-verifiable (no real-GPU component).
- **External-CTS createView finding F-014 — RESOLVED.** A later
  `createView:array_layers` slice (Texture T10) found yawgpu under-validated
  **3D-texture** view array-layer ranges: `validate_texture_view_descriptor`
  skipped the layer-range check entirely for `D3` (the `!= D3` guard), so a
  3D view with `arrayLayerCount != 1` or `baseArrayLayer > 0` was wrongly
  accepted. A 3D texture is one non-arrayed image (`depth_or_array_layers`
  is depth, not layers), so the fix validates `base_array_layer +
  array_layer_count <= 1` for `D3` while leaving the 1D/2D `> texture_layers`
  check unchanged. Active in-tree 3D cases in
  `cts/validation/texture/create_view.rs::array_layers` + a core unit test.
  Noop-verifiable.
- **External-CTS createBindGroupLayout finding F-016 — RESOLVED.** The BGL
  storage-texture slice (T13) found yawgpu rejected `read-write` storage
  access on `r32uint`/`r32sint`/`r32float`: `read_write_storage_capable`
  was set only inside the `TextureFormatsTier2` block, but per spec those
  three formats support read-write storage with **no feature gate**. Fixed
  by marking the r32 trio read-write-capable in their base `FormatCaps`
  (new `read_write_storage()` builder; the redundant tier2 `R32_FLOAT`
  entry dropped). Active in-tree r32 read-write cases in
  `cts/validation/create_bind_group_layout.rs::storage_texture_formats` +
  a core unit test. Noop-verifiable.
- **External-CTS createBindGroupLayout finding F-018 — RESOLVED.** The BGL
  storage-texture slice (T14) found two over-restrictions: (1) yawgpu
  rejected a **1D** storage-texture view dimension — `1d` is a valid
  storage view (only cube/cube-array are disallowed), removed the
  `bind_group_layout.rs` D1 rejection; (2) yawgpu gated **rgba8snorm
  storage** behind `texture-formats-tier1`, but per the CTS
  `kStorageTextureFormats` table rgba8snorm is a **base** (no-feature)
  storage format — **this reverses part of F-009**, which over-narrowed it
  (the F-009 createTexture test ran with a tier1-on fixture and couldn't
  tell base from tier1). Fixed by adding `.storage()` to the base
  `RGBA8_SNORM` caps and removing it (and the redundant `RGBA16_FLOAT`)
  from the tier1 storage block; rgba8snorm's renderable/multisample stays
  tier1 (F-006, unchanged). Corrected the three F-009 tests that asserted
  "rgba8snorm storage requires tier1" + added 1D / rgba8snorm-storage CTS
  cases and a base-storage unit test. Noop-verifiable.
- **External-CTS createPipelineLayout finding F-020 — RESOLVED.** The
  createPipelineLayout slice (T18) found yawgpu rejected **null
  bind-group-layout slots** — a null slot is a valid unused bind-group
  index (Dawn models it as an empty BGL). `conv/bind.rs::map_pipeline_layout_descriptor`
  both errored on a null element AND `filter_map`-dropped it (which would
  shift later groups' indices). Fixed by `map`ping a null element to a new
  `BindGroupLayout::empty_unused()` (empty, non-default, non-error) so the
  slot is preserved in place; removed the per-element null error (the
  whole-array-null error stays). Flipped the three in-tree null-BGL tests
  to expect success + added a `[bgl0, null, bgl2]` / `@group(2)`
  slot-preservation case and a conv unit test. Noop-verifiable.
- **External-CTS createPipelineLayout finding F-022 — RESOLVED.** The
  completed `createPipelineLayout` slice (T21) found yawgpu rejected a BGL
  buffer entry with `minBindingSize = 0` at pipeline creation. `0` means
  *unspecified* — the size check defers to bind time — so pipeline creation
  must not reject it (Dawn defers). `compute_pipeline.rs::validate_shader_binding_compat`
  compared the layout's `minBindingSize` (0) against the shader's required
  size and errored; fixed by guarding the check with `min_binding_size != 0`
  (non-zero-but-too-small still errors), mirroring the existing bind-time
  rule in `bind_group.rs`. render-pipeline has no analogous check (no change
  needed). Reverted the F-020 test workaround (those null-BGL tests had used
  `minBindingSize = 16` to dodge this bug; now back to the default 0, so
  they exercise the deferral) + a core unit test. Noop-verifiable.
- **External-CTS api/operation finding F-023 — RESOLVED.** The first
  `api/operation` slice (T22, command_buffer) found that a **0-byte**
  `copyBufferToBuffer`/`clearBuffer` (a valid no-op) aborted the **Metal**
  validation layer ("Command encoder released without endEncoding"): the
  Metal blit encoder issued a 0-length `copyFromBuffer` and was torn down
  un-ended. (Both failing tests reduce to a 0-size copy — `copyBufferToBuffer`
  directly, `clearBuffer` via its readback copy.) **First operation-area
  finding, and a real-Metal-only defect (Noop cannot catch it).** Fixed in
  `yawgpu-core/src/queue.rs::hal_command_execution`: a 0-size buffer copy
  (and a 0-extent texture copy) now translates to no HAL command (a
  validated no-op — backend-agnostic, also avoids Vulkan 0-size VUIDs);
  plus `yawgpu-hal/src/metal/queue.rs` now always calls `endEncoding()`
  even when an `encode_*` helper errors (defensive against the un-ended
  encoder class). **Second part (CTS re-test):** removing the abort
  unmasked a deeper bug — yawgpu's `clearBuffer` was validation-only and
  never zeroed the buffer (it recorded no `CommandExecution`, and no HAL
  backend had a fill primitive). Implemented `clearBuffer` execution
  end-to-end: a `CommandExecution::BufferClear` → `HalCopy::BufferClear`
  with a backend fill (Metal `fillBuffer:range:value:0`, Vulkan
  `vkCmdFillBuffer(…,0)`, GLES chunked zero `bufferSubData`, Noop no-op —
  Noop has no byte storage); the 0-size clear stays a no-op. Noop unit
  tests + a **data-readback** Metal e2e
  (`e2e_metal_buffer::metal_clear_buffer_zeroes_full_and_partial_ranges`,
  plus the 0-size no-op case) that **Claude ran on the M2** (5/5 green;
  the cleared range reads back all-zero and partial-range bytes outside
  are unchanged). **Third part (CTS re-test):** the data-readback case
  passed on the M2 but the real CTS still failed 10/50 — the
  `size = undefined` subcases — because the clearBuffer C FFI did not
  resolve `WGPU_WHOLE_SIZE` (it passed `u64::MAX` to core, which rejected
  the range). This was a pre-existing FFI gap, unmasked once clearBuffer
  executed (and missed by the hand-written e2e, which only used explicit
  sizes — lesson: the real gate is the CTS sequence, not a bespoke e2e).
  Fixed in `yawgpu/src/ffi/encoder.rs::wgpuCommandEncoderClearBuffer`
  (`size == WGPU_WHOLE_SIZE` → `buffer.size − offset`). **Verified by
  running the real webgpu-native-cts binary directly** (rebuilt the
  `--features metal` staticlib + relinked cts):
  `api,operation,command_buffer,clearBuffer:clear` now **pass=50 / fail=0**
  (was 40/10), `copyBufferToBuffer` + `createPipelineLayout` slices clean.
  **With F-023 closed, every yawgpu finding this suite had surfaced
  through T22 is resolved.**
- **External-CTS api/operation finding F-024 — RESOLVED.** The first
  buffer↔texture operation slice (T23, `command_buffer,basic:{b2t2b,b2t2t2b}`)
  found a `rgba8uint` `copyBufferToTexture`→`copyTextureToBuffer` roundtrip
  read back all zeros. Root cause: the HAL `HalTextureFormat` enum is a
  minimal ~10-format subset, and core's `hal_texture_format` mapped
  `RGBA8_UINT` via the `_ => Unsupported` catch-all → the rgba8uint texture
  had no real GPU backing → `hal_texture_copy_execution`'s `texture.hal()?`
  short-circuited and the copy was silently dropped (Noop validation passes
  rgba8uint, so it only surfaced on a real backend in an operation test).
  Fixed by adding `HalTextureFormat::Rgba8Uint` + the core mapping +
  per-backend formats (Metal `RGBA8Uint`, Vulkan `R8G8B8A8_UINT`, GLES
  `RGBA8UI`/`RGBA_INTEGER`). **Verified by running the real CTS**
  (`command_buffer,basic` `b2t2b`/`b2t2t2b` now pass) + a data-readback
  Metal e2e on the M2. **Follow-up — HAL format-table expansion (DONE).**
  Rather than chase formats one finding at a time, `HalTextureFormat` was
  expanded to cover **all uncompressed color formats** (~40: r8/rg8/rgba8,
  r16/rg16/rgba16, r32/rg32/rgba32 × unorm/snorm/uint/sint/float, bgra8±srgb,
  rgba8-srgb, rgb10a2 uint/unorm, rg11b10ufloat, rgb9e5ufloat) across core +
  Metal/Vulkan/GLES (mappings cross-checked against wgpu-hal). Verified: CTS
  `createTexture` validation (48343 pass) + a parameterized real-M2 e2e that
  byte-roundtrips every integer format and asserts non-zero readback for the
  float/packed ones (`e2e_metal_texture::metal_added_uncompressed_color_texture_copy_round_trips_data`).
  Compressed formats (BC/ETC2/EAC/ASTC) remain `Unsupported` — deferred
  (feature-gated + block-size handling).
  **With F-024 closed, every yawgpu finding this suite has surfaced
  (F-005/006/008/009/010/011/014/016/018/020/022/023/024) is resolved**; all
  other open findings (F-001–F-004, F-007, F-012, F-013, F-015, F-017,
  F-019, F-021) are wgpu-native defects.
- **External-CTS api/operation findings F-025 + F-026 — RESOLVED.**
  The `image_copy` slice (T24b) surfaced two yawgpu findings on real-GPU Metal.
  Claude reproduced both on the M2 and **root-caused them with instrumented
  `submit_copies`** (the findings doc's "over-strict bytesPerRow" guess is
  wrong). Three distinct defects, two sharing a foundation:
  - **Defect 1 (bulk of F-026): the HAL backends cannot create array / 3D /
    mipmapped textures.** Metal *and* Vulkan `create_texture` reject
    `depth_or_array_layers != 1 || mip_level_count != 1 || sample_count != 1`
    and always build a single-layer single-mip 2D image; `MetalDevice::
    create_texture` swallows the `Err` into a null `MetalTexture {inner:None,
    bytes_per_pixel:0}` with **no device error surfaced**. So `createTexture`
    "succeeds" (validation CTS green) but the texture is a husk; any later copy
    fails in the encode (`bytes_per_pixel == 0`) and aborts the **whole**
    `submit_copies`, so the readback buffer keeps its initial bytes (the
    constant `got 0.705882` = `generateData(0,17)[0]` = 180 proves the buffer
    was never written). image_copy uses `baseTextureSize` 256×16×4 (4-layer
    2D-array), 256×16×8 (3D), and `mipLevelCount>1`; only `undefined_params`
    (8×1×1) passes today. `HalTextureDescriptor` doesn't even carry the texture
    dimension, and core's `hal_texture_descriptor` drops `descriptor.dimension`.
  - **Defect 2 (all of F-025): `wgpuQueueWriteTexture` ignores its `data`
    pointer** (`_data`) — validation-only, uploads nothing → `got 0` even on the
    createable 8×1×1 texture. Same class as the old F-023 "validation-only
    clearBuffer".
  - **Defect 3: the Metal copy encode hard-codes array-slice semantics**
    (`destinationSlice = origin.z`, `z-origin = 0`) — correct for a 2D-array
    single layer, wrong for 3D (z is depth) and for multi-layer 2D-array copies
    (Metal copies one slice/call → needs a slice loop). Blocks 3D once Defect 1
    is fixed.
  Handed off to the coding agent (HANDOFF.md): add `HalTextureDimension` +
  thread it into `HalTextureDescriptor` + every backend texture handle; Metal &
  Vulkan `create_texture` full 1D/2D/2D-array/3D/mip support; Metal copy encode
  branch (3D single-call vs 2D-array slice loop) + `validate_buffer_texture_range`
  multi-image extension; GLES Tier-2 best-effort; `queueWriteTexture` upload via
  staging-buffer→B2T (reuses the fixed copy path, so array/3D/mip come for free).
  The fix landed: `HalTextureDimension::{D1,D2,D3}` threaded through
  `HalTextureDescriptor` + each backend texture handle; Metal & Vulkan
  `create_texture` now map 1D/2D/2D-array/3D + mip/array length; the copy encode
  branches on dimension (Metal: 3D single blit vs 1D/2D per-slice loop advancing
  the buffer by `bytes_per_image`; Vulkan: `baseArrayLayer`/`layerCount` vs 3D
  depth); `validate_buffer_texture_range` accounts for all images;
  `wgpuQueueWriteTexture` stages the real data into a Shared CopySrc buffer and
  submits a `HalCopy::BufferToTexture`. GLES (Tier 2) compiles + allocates the new
  shapes but still returns `HalError` for array/3D copy execution (catalogued in
  `specs/blocks/67-gles-backend.md`). **Verified on real-GPU Metal (sandbox off):
  the full `api,operation,command_buffer,image_copy` suite is `pass=137256 fail=0
  crash=0` (Dawn-equal — up from `pass=21860 fail=115396`), `command_buffer,basic`
  still `pass=3`, and `createTexture`/`createView` validation unchanged
  (`48343`/`26619`, `fail=0`).** In-tree regression:
  `e2e_metal_texture::metal_queue_write_texture_uploads_color_data_round_trips`
  (new) + the existing 6 e2e all pass on the M2; workspace release backstop green.
  3-way confirmed throughout (Dawn + wgpu-native always passed).
- **External-CTS api/operation finding F-031 — RESOLVED.** The
  `copyTextureToTexture:copy_depth_stencil` port (T26) surfaced that yawgpu's
  **depth** aspect failed (every depth format `fail=36/36`; stencil-only
  `Stencil8` passed). **It was NOT the texture-to-texture copy.** The stencil
  path uses `writeTexture` + `copyTextureToBuffer` (no rendering); the depth path
  uses *render passes* (`initializeDepthAspect` renders depth,
  `verifyDepthAspect` re-renders with `depthCompare=equal`). Real-GPU-Metal
  isolation (new `yawgpu/tests/e2e_metal_depth.rs`, plus device-error tracing)
  localised **seven** distinct gaps in the regular (non-`tiled`) real-backend
  render path, fixed in sequence:
  1. **Render-pass depth-stencil attachment dropped.** `HalRenderPass` carried
     only a mandatory `color_target`; the regular `render_pass_descriptor` bound
     no depth attachment. Added `HalRenderDepthStencilAttachment`, made
     `color_target` optional, threaded the (already-parsed/validated)
     depth-stencil attachment core→HAL, and bound it in Metal/Vulkan/GLES.
  2. **No-colour (depth-only) render passes rejected.** `draw` required a colour
     attachment; relaxed to require ≥1 attachment.
  3. **Render pipeline rejected depth-stencil + vertex-only.**
     `create_hal_render_pipeline` bailed on `depth_stencil.is_some()` and
     required exactly one colour target + a fragment. Now accepts colour+depth
     and vertex-only (no-fragment) pipelines; `select_render_shader_source` +
     `HalDevice::create_render_pipeline` carry an optional fragment entry;
     vertex-only MSL/SPIR-V/GLSL generation added; Metal allows empty colour
     formats + nil fragment function.
  4. **Cross-pipeline vertex-position invariance.** Metal render MSL now compiles
     with `MTLCompileOptions.preserveInvariance = true` (Dawn parity).
  5. **Separate vertex & fragment shader modules rejected on Metal.** The CTS
     verify pipeline uses two distinct WGSL modules; Metal combined them into one
     MSL and required a single module. Added `HalShaderSource::MslStages` +
     per-stage MSL generation + a two-library Metal pipeline (Vulkan/GLES already
     did per-stage). Broad fix — separate vs/fs modules are common WebGPU usage.
  6. **Render-attachment mip-level / array-layer dropped.** The core
     `RenderPass{Color,DepthStencil}Execution` captured only the view's texture,
     not its `base_mip_level`/`base_array_layer`, so every attachment rendered to
     mip 0 / layer 0. Threaded the view subresource core→HAL
     (`HalRender{ColorTarget,DepthStencilAttachment}.{mip_level,array_layer}`) and
     Metal sets the attachment `level`/`slice`. **Implemented for Metal; Vulkan &
     GLES still target the base mip/layer for non-default attachment views — a
     follow-up to implement + verify on Windows/Vulkan + Android GLES.**
  7. **Depth/stencil copy validation over-strict.**
     `validate_texture_copy_subresource` (and the `queueWriteTexture` analogue)
     applied the texture-*buffer* "single 2D layer" + "origin-zero full
     subresource" rules to **all** copies, rejecting multi-layer / layer-ranged
     depth-stencil `copyTextureToTexture` (and multi-layer stencil
     write/`copyTextureToBuffer`). Corrected to require only a full-width/height
     2D subresource at a zero x/y origin while allowing a range of array layers —
     matching WebGPU/Dawn for buffer and texture-to-texture copies alike. (This
     un-masked the real multi-layer stencil read; the prior `Stencil8` "pass" was
     a false pass — its readback copy was also being rejected, leaving the compare
     buffer at its expected seed.)
  **Verified on real-GPU Metal (sandbox off):
  `copyTextureToTexture:copy_depth_stencil` is `pass=216 fail=0` (Dawn-equal — up
  from `pass=36 fail=180`); full `copyTextureToTexture` `pass=31126 fail=0`;
  `image_copy` regression `pass=137256 fail=0`; `command_buffer,basic` `pass=3`.**
  In-tree: `e2e_metal_depth.rs` (7 tests — depth render+readback, color+depth,
  depthCompare=Equal+Load, gradient-Equal, separate vs/fs modules, multi-layer
  depth t2t, t2t-preserves-depth) all pass on the M2; workspace release backstop
  `1080 passed / 0 failed`. 3-way confirmed (Dawn + wgpu-native pass all 216).
  Verification + the gap-6/gap-7 fixes were done by Claude directly (per request);
  Rounds 1–4 lib work was the coding agent's.
- **External-CTS finding F-031 on the Vulkan backend — RESOLVED.** The Metal F-031
  fix left `copyTextureToTexture:copy_depth_stencil` at `pass=36 fail=180` on yawgpu's
  Vulkan HAL (verified real-GPU via MoltenVK; `Stencil8` passed, all depth formats
  failed). Claude localised four independent Vulkan-only gaps with `e2e_vulkan_depth.rs`
  isolation probes and handed each to the coding agent:
  1. **Copy aspect hardcoded COLOR.** `image_subresource_layers` /
     `color_subresource_range` forced `VK_IMAGE_ASPECT_COLOR` for every copy, so the
     depth/stencil aspect of `copyTextureToTexture` (and buffer⇄texture copies) copied the
     wrong plane. Derive the aspect from the format (`copy_format_aspect_flags`) / the
     copy's `HalTextureAspect`; use aspect-aware subresources + `transition_image_aspect`.
  2. **`LoadOp::Load` rejected.** The non-tiled render path errored on any load op except
     `Clear`. The CTS `verifyDepthAspect` re-renders with `depthLoadOp=Load`; without it the
     verify never ran. Build the execution render pass from the pass's actual load/store ops
     with a contents-preserving `initial_layout`.
  3. **Attachment views ignored mip/array layer** (the Vulkan half of the Metal-only gap 6
     above). `create_framebuffer` used the texture's default full-image view, so a
     `baseArrayLayer`/`baseMipLevel` attachment rendered to layer 0 / mip 0. Create transient
     2D views scoped to `HalRender*Attachment.{mip_level,array_layer}`, freed via
     `RetireOp::ImageView`.
  4. **Render extent used the base size.** `render_pass_extent_from_targets` returned the
     base texture size, so a depth-only mip-2 staging pass rasterised the gradient over the
     base extent (only the top-left mip-sized corner landed in the mip region). Compute
     `max(1, dim >> mip_level)` for the chosen attachment (`mip_extent`), feeding render-area,
     viewport, and framebuffer. (A constant depth masked this; the CTS uses a depth gradient.)
  **Verified real-GPU Vulkan/MoltenVK:** `copyTextureToTexture:copy_depth_stencil`
  `pass=216 fail=0` (Dawn-equal, from `36/180`); `e2e_vulkan_render` 2/2 + `e2e_vulkan_depth`
  10/10 (no regression); Noop clippy + `--features vulkan` clippy clean; workspace test green.
  In-tree `e2e_vulkan_depth.rs` (Claude-authored) — see the F-032 Vulkan note below for the
  grown probe set.
- **External-CTS finding F-032 on the Vulkan backend — RESOLVED.** The Metal F-032 fix left the
  Vulkan `image_copy` depth/stencil aspect buffer copies at `pass=352 fail=800` (confirmed on
  **native Windows/Vulkan**, NVIDIA, byte-identical to MoltenVK — a genuine yawgpu Vulkan-HAL gap,
  not a MoltenVK artifact). Claude localised two Vulkan-only gaps with `e2e_vulkan_depth.rs` probes
  and handed each to the coding agent:
  1. **Buffer-copy byte size was whole-format.** `texture_bytes_per_pixel` returned
     `texture.bytes_per_pixel` for every copy, so the *aspect's* row stride was wrong (stencil =
     1 byte, packed-depth = 4 not 5) and the copy produced zeros. Made it aspect-aware (mirroring
     the Metal `metal/format.rs` version) and threaded it into `buffer_image_copy`. Fixed packed
     stencil (576) + packed depth (96): `352 → 960`.
  2. **Sampled-texture binding ignored the view subresource.** `descriptor_info`
     (`vulkan/pipeline.rs`) bound the texture's default full-image `.view`, so the CTS depth
     staging (which samples a per-layer `r32float` view) sampled layer 0 — every multi-layer depth
     stage wrote the wrong depth. Bind a transient `vk::ImageView` scoped to `HalBoundTexture`'s
     `{format,dimension,base_mip_level,mip_level_count,base_array_layer,array_layer_count,aspect}`
     (the Vulkan analog of the Metal "a2" view fix), tracked via `RetireOp::ImageView` for both the
     render and compute descriptor paths. Fixed the depth-aspect staging (192): `960 → 1152`.
  **Verified real-GPU Vulkan/MoltenVK:** `image_copy` depth/stencil
  `rowsPerImage…_depth_stencil` `864/0` + `offsets…` `288/0` = **`1152/0`** (Dawn-equal, from
  `352/800`); `e2e_vulkan_{depth 12/12, compute 3/3, texture 4/4, render 2/2}` (no regression);
  Noop + `--features vulkan` clippy clean; workspace test green. `e2e_vulkan_depth.rs` grew to 12
  Claude-authored probes (incl. `vulkan_packed_stencil_buffer_roundtrips`,
  `vulkan_sampled_frag_depth_layer1`). With this, **F-032 is fully resolved on Metal *and* Vulkan.**
- **External-CTS finding F-034 — RESOLVED (treated as a phase, with Clean Review).** The T30
  `rendering/draw:{arguments,default_arguments}` ports surfaced that `drawIndexed`, `drawIndirect`,
  and `drawIndexedIndirect` were **validation-only stubs** in `render_pass.rs` — they validated +
  bumped `draw_count` but never called `record_render_pass`, so (unlike plain `draw`) **no HAL
  command was emitted**: the draws never rasterized and their `@fragment` `read_write` storage write
  never ran (`result==0`). HAL-agnostic — byte-identical on Metal and Vulkan (`pass=340 fail=224`).
  - **Fix (coding agent):** added shared draw-execution variants (`Direct`/`Indexed`/`Indirect`/
    `IndexedIndirect`) through core (`RenderDrawExecution`/`RenderPassCommand` now carry the bound
    index buffer + indirect buffer; `draw_indexed`/`draw_indirect`/`draw_indexed_indirect` record a
    command like `draw`; `base_vertex` wired), the HAL (`HalDraw` enum + `HalIndexFormat` +
    index/indirect buffers on `HalRenderPass`), and execution in Noop/Metal/Vulkan
    (`drawIndexedPrimitives…`/`cmd_draw_indexed`/`…indirect`); GLES maps GLES-3.1 paths and returns
    `HalError` for `baseVertex != 0` and indexed-indirect nonzero index offset (catalogued in
    `specs/blocks/67-gles-backend.md`).
  - **Verified real-GPU (Claude):** `rendering/draw:{arguments,default_arguments}` = `564/0`
    (180 `indirect-first-instance` feature-skips) on **Metal and Vulkan/MoltenVK**, up from `340/224`;
    Noop + metal + vulkan + gles clippy clean; workspace test green. Claude authored
    `yawgpu/tests/e2e_metal_draw.rs` (3 probes — indexed / indirect / indexed-indirect, each asserts
    the fragment storage write `==1` AND green raster); all pass on the M2.
  - **Phase Review (Clean Review, fresh no-context subagent on the cumulative diff):** **0 CRITICAL**,
    **1 MAJOR**, 2 MINOR. The MAJOR — "no in-tree e2e exercises the new variants + reads back" — was
    fixed by the `e2e_metal_draw.rs` probes above (GPU tests are Claude-owned). The 2 MINOR are
    **deferred with rationale**: (i) `render_pass.rs:368/476` keep a defensive `ok_or_else("…requires
    an index buffer")` that is unreachable because `validate_render_draw_state` errors first with a
    different message — harmless (no panic, returns a `Result`); only the dead message string differs.
    (ii) the GLES `Indirect`/`IndexedIndirect` `first_instance→0` mapping
    (`gles/queue.rs`) lacks an explanatory comment. Neither blocks COMPLETE; both logged here for a
    follow-up cleanup. Gate: **no open CRITICAL/MAJOR → F-034 COMPLETE.**
- **External-CTS finding F-035 — RESOLVED (treated as a phase, with Clean Review).** The T31
  `rendering/color_target_state` ports surfaced that yawgpu ignored `GPUColorTargetState`
  **`writeMask`** and **`blend`** (and `setBlendConstant`): core parsed + validated
  `ColorTargetState{format, blend, write_mask}` but `HalRenderPipelineDescriptor` carried only
  `color_formats`, and `set_blend_constant` was a validation-only stub — so the raw clamped fragment
  output was written to every channel. HAL-agnostic (Metal + Vulkan byte-identical, `pass=2 fail=21`).
  - **Fix (coding agent):** `HalRenderPipelineDescriptor.color_formats` → `color_targets:
    Vec<HalColorTargetState{format, blend: Option<HalBlendState>, write_mask}>` with new
    `HalBlendState`/`HalBlendComponent`/`HalBlendOperation`/`HalBlendFactor`; core maps every
    `ColorTargetState`; `set_blend_constant` records the constant into pass state, every draw site
    snapshots it into `RenderPassCommand`, and it threads through `queue.rs` to
    `HalRenderPass.blend_constant`. Backends apply write_mask + blend in the pipeline color attachment
    and the blend constant at draw (Metal `setBlendColor…`; Vulkan dynamic `cmd_set_blend_constants`;
    GLES `glColorMask`/`glBlendFuncSeparate`/`glBlendEquationSeparate`/`glBlendColor`). GLES rejects
    dual-source blend factors with `HalError` (catalogued in `specs/blocks/67-gles-backend.md`).
  - **Verified real-GPU (Claude):** `rendering/color_target_state:*` = `23/0` (3 skips) on **Metal and
    Vulkan/MoltenVK**, up from `2/21`; Noop+metal+vulkan+gles clippy clean; workspace test green.
    Claude authored `yawgpu/tests/e2e_metal_color_target.rs` (2 probes: `writeMask=Red` gates G/B
    → `[255,0,0,255]`; `blend src*constant` with `setBlendConstant 0.5` → `[128,128,128,255]`);
    both pass on the M2.
  - **Phase Review (Clean Review, fresh no-context subagent on the cumulative diff):** **0 CRITICAL**,
    **1 MAJOR**, 2 MINOR. The MAJOR — the GLES dual-source-blend `HalError` was not catalogued in
    `67-gles-backend.md` — is a **spec fix (Claude's)** and was applied (the "Render pipeline state"
    row now lists writeMask + blend + blend constant + the dual-source Tier-2 `HalError`). The
    reviewer separately verified soundness (the `write_mask: u64→u32` `try_from` is unreachable-fail
    because core validation rejects `&!0xF` bits before pipeline creation; no panic), the blend
    factor/op mappings on all three backends (color/alpha not swapped, Constant vs OneMinusConstant
    correct), the per-pass blend-constant plumbing, and the e2e logic. 2 MINOR **deferred with
    rationale**: (i) `subpass.rs` `SubpassRenderPassCommand` has no `blend_constant` field — harmless,
    the subpass encoder exposes no `setBlendConstant`; (ii) the GLES `Src1*` `gles_blend_factor` arms
    are unreachable at runtime (the pipeline rejects dual-source first) but kept for `match`
    exhaustiveness. Neither blocks COMPLETE. Gate: **no open CRITICAL/MAJOR → F-035 COMPLETE.**
- **External-CTS finding F-037 — RESOLVED (treated as a phase, with Clean Review).** The T32
  `rendering/depth` ports flaked non-deterministically on yawgpu's **Metal** HAL (~35-44/130 fail,
  varying run to run; the drawn point read back as the clear value), while Vulkan/MoltenVK + Dawn +
  wgpu-native passed 130/130. Despite the "race" framing, it was **point-primitive-specific**, not a
  sync/depth race.
  - **Diagnosis (Claude, real-GPU experiments):** ruled out — missing render→readback sync (render /
    t2b / buffer-copy are three separate `wgpuQueueSubmit`s, each its own command buffer with
    `waitUntilCompleted`), texture storage mode (Shared→Private stayed flaky), explicit `setViewport`
    (Metal's default viewport is already znear=0/zfar=1; stayed flaky), and depth-stencil-state lifetime
    (retained via the pipeline `Arc`). Found it's flaky even for a SINGLE case alone (~30%). Root cause:
    the depth tests draw **points** (`PointList`), and yawgpu's naga→MSL generation never set
    `allow_and_force_point_size`, so the Metal vertex shader emitted no `[[point_size]]` → Metal point
    size is **undefined** → the point intermittently rasterized at size 0 (not drawn). Confirmed: forcing
    the flag made `rendering/depth` deterministically 130/130.
  - **Fix (coding agent):** thread `force_point_size = (topology == PrimitiveTopology::PointList)` from
    the render pipeline descriptor into the render MSL generators (`render_pipeline.rs`), setting
    `naga::back::msl::PipelineOptions::allow_and_force_point_size` in both the combined and
    separate-vertex paths (`shader_naga.rs`); NOT for non-point topologies (naga: Metal dislikes it
    there), nor compute/fragment-only/Vulkan/GLES.
  - **Verified real-GPU (Claude):** `rendering/depth:*` deterministically **`130/130` across 12
    consecutive Metal runs** (from ~35-44 flaky); `rendering/draw` `540/0` + `color_target_state` `23/0`
    no regression; Vulkan/MoltenVK `rendering/depth` stays `130/130`; Noop + metal + vulkan + gles clippy
    clean; workspace test green. Claude authored `yawgpu/tests/e2e_metal_point.rs` (a point-list draw
    into a colour+depth attachment that asserts the point rasterizes; passes on the M2).
  - **Phase Review (Clean Review, fresh no-context subagent on the cumulative diff):** **0 CRITICAL,
    0 MAJOR, 0 MINOR** — the conditional is exactly `topology == PointList`, threaded to both render MSL
    paths, not applied to compute/fragment/Vulkan/GLES; no panic; Noop unit test
    (`generate_render_msl_forces_point_size_only_when_requested`) + the e2e present. Gate: **no open
    CRITICAL/MAJOR → F-037 COMPLETE.**
- **External-CTS finding F-038 — RESOLVED (treated as a phase, with Clean Review).** The
  `rendering/stencil` ports failed `pass=97 fail=91`, **deterministically and byte-identically on Metal
  and Vulkan/MoltenVK** (so a shared-core bug, not a per-HAL stencil enum mapping), while Dawn +
  wgpu-native passed 188/188. The failing compares showed the "reflexive" pattern (pass for
  equal/LE/GE/always, fail for less/greater/not-equal/never regardless of the requested reference) — the
  hallmark of the stencil **reference** never being applied.
  - **Diagnosis (Claude, source-conclusive):** `wgpuRenderPassEncoderSetStencilReference` was a
    **validation-only stub** — `render_pass.rs` `set_stencil_reference(&self, _reference: u32)` discarded
    the value; there was no `stencil_reference` field on `HalRenderPass` and no
    `setStencilReference`/`cmd_set_stencil_reference`/`glStencilFunc` reference anywhere in the HAL, so
    every backend used a default reference of 0. The stencil pipeline state (compare/failOp/depthFailOp/
    passOp + read/write masks) was already mapped correctly; only the dynamic reference was missing. This
    is the **stencil analog of the F-035 `blend_constant` fix** — and was the deferred-MINOR observation
    the F-035 Clean Review had flagged.
  - **Fix (coding agent):** mirror the `blend_constant` plumbing — `HalRenderPass.stencil_reference: u32`;
    core `set_stencil_reference` records `state.stencil_reference` (default 0); all four render draw sites
    (`draw`/`draw_indexed`/`draw_indirect`/`draw_indexed_indirect`) plus the clear-only-pass path snapshot
    it; `queue.rs` threads it into `HalRenderPass`. Backends: Metal `setStencilReferenceValue`; Vulkan
    `VK_DYNAMIC_STATE_STENCIL_REFERENCE` in the pipeline dynamic-state list + `cmd_set_stencil_reference
    (FRONT_AND_BACK, …)`; GLES (Tier 2) per-draw `glStencilFuncSeparate/OpSeparate/MaskSeparate` from the
    pipeline depth-stencil state + dynamic reference (a reference `> i32::MAX` returns a catalogued Tier-2
    `HalError`); Noop records. +Noop unit test
    (`render_pass_encoder_set_stencil_reference_records_draw_reference`).
  - **Verified real-GPU (Claude):** `rendering/stencil:*` reaches **`188/0` on Metal and `188/0` (skip=1)
    on Vulkan/MoltenVK** (from `97/91`); `rendering/depth` `130/0` + `color_target_state` `23/0` no
    regression on both backends. Noop + metal + vulkan + gles clippy clean; workspace test green (67
    groups, 0 fail). Claude authored `yawgpu/tests/e2e_metal_stencil.rs` — clears stencil to 1 via
    `stencilClearValue` (independent of the reference), draws with `compare=Equal` + `setStencilReference
    (1)`; green only if the reference reached the GPU (a stuck 0 → `Equal(0,1)` → black → fail, no
    reflexive escape). Passes on the M2.
  - **Phase Review (Clean Review, fresh no-context subagent on the cumulative diff):** **0 CRITICAL, 1
    MAJOR (resolved), 1 MINOR (deferred).** MAJOR — the GLES `> i32::MAX` `HalError` was shipping
    uncatalogued; resolved by extending the `67-gles-backend.md` mapping matrix (render-pass row) with the
    F-038 stencil-test application + the catalogued `HalError`. MINOR — the GLES error message
    `"stencil reference value exceeds GLES limit"` wording; deferred — it matches the existing in-tree
    convention (`"draw firstVertex exceeds GLES limit"` etc.) and is defensible. Subagent confirmed: all
    four draw sites thread the reference, default 0, Vulkan dynamic-state added unconditionally beside
    `BLEND_CONSTANTS`, Metal once-per-pass, GLES no-panic `?`-based, and the e2e is a sound guard. Gate:
    **no open CRITICAL/MAJOR → F-038 COMPLETE.**
- **External-CTS finding F-039 — RESOLVED (treated as a phase, with Clean Review).** The T35 (V7)
  `memory_sync/buffer/single_buffer:two_dispatches_in_the_same_compute_pass` port: two compute dispatches
  in ONE pass write `1` then `2` to one storage buffer (spec-ordered ⇒ expect `2`); Dawn + wgpu-native
  pass, **yawgpu read back `0`** (the initial value — neither write visible), **deterministic and
  byte-identical on Metal and Vulkan/MoltenVK** → a shared-core bug. (Reported batch-only, but reproduced
  standalone on `40f5d7f`.)
  - **Diagnosis (Claude, source-conclusive + real-GPU confirmed):** `dispatch_workgroups`
    (`compute_pass.rs`) called `record_pipeline_usage_scope`, which accumulates a **pass-wide** resource
    usage scope into `PassEncoderState.scope_buffer_uses`/`scope_texture_uses` and re-validates the running
    union. Per WebGPU **each compute dispatch is its own usage scope** (a render pass, by contrast, is one
    scope across all draws). So dispatch 2's storage write collided with dispatch 1's in the accumulator →
    `validate_buffer_usage_scope` returned `Err("usage scope cannot … write the same buffer range twice")`
    → the `?` aborted before `record_compute_pass`, and `record_first_error` poisoned the encoder →
    `finish()` yielded an error command buffer → `submit` rejected it wholesale → **neither dispatch
    executed** → buffer stayed `0`. Confirmed by HAL instrumentation (the compute submit produced **zero**
    `HalCopy`) and a throwaway revert experiment (removing the two lines → 2 `ComputePass` reach the HAL →
    readback `2`). Corroboration: `dispatch_workgroups_indirect` already omitted the accumulation — only
    direct dispatch called it, erroneously.
  - **Fix (coding agent):** remove the two `record_pipeline_usage_scope` lines from `dispatch_workgroups`;
    each dispatch is now validated as its own usage scope by the existing `validate_compute_dispatch_state`
    (→ `validate_usage_scope` over the current bind groups). Render-pass / render-bundle accumulation
    untouched (correct there). +Noop unit test
    (`compute_pass_direct_dispatches_have_separate_usage_scopes`): two distinct pipelines writing the same
    storage buffer in one pass ⇒ no error + two recorded `ComputePass` ops. Pure `yawgpu-core` fix; no HAL
    change (the bug never reached a backend).
  - **Verified real-GPU (Claude):** `single_buffer:*` reaches **`pass=25 fail=0` on Metal and
    Vulkan/MoltenVK** (from `pass=24 fail=1`); no memory_sync/compute regression; Noop + metal + vulkan +
    gles clippy clean; workspace test green (67 groups, 0 fail). Claude authored the Metal e2e
    `metal_two_dispatches_in_one_pass_second_write_wins` (`e2e_metal_compute.rs`) — clears a storage buffer
    to 0, two dispatches write `1` then `2` in one pass through distinct pipelines, separate readback
    submit asserts `2` (a stuck pre-fix path reads `0`). Passes on the M2.
  - **Phase Review (Clean Review, fresh no-context subagent on the cumulative diff):** **0 CRITICAL,
    0 MAJOR, 1 MINOR (deferred).** MINOR — the Noop unit test's two pipelines are functionally identical
    (same WGSL); deferred — the test is still a sound guard (the subagent empirically reintroduced the
    pre-fix lines and confirmed it FAILS, 1 op + poisoned encoder), and the GPU e2e uses genuinely distinct
    `1`/`2` shaders with readback. Subagent independently confirmed: the per-dispatch within-dispatch alias
    check is preserved by `validate_compute_dispatch_state`; `scope_*` fields are read only by render
    paths, so removal is clean (no latent submit-sync bug); direct/indirect dispatch now consistent; no
    panics; core rule tightened, not relaxed. Gate: **no open CRITICAL/MAJOR → F-039 COMPLETE.**
- **External-CTS finding F-040 — RESOLVED (3-slice feature; slices 1 & 2 done, slice 3 subsumed).**
  F-040 (`render_pass,resolve` T36, V8): yawgpu's multisample resolve never writes the
  `resolveTarget` — `pass=0 fail=12` on Metal and Vulkan/MoltenVK ("expected 1, got 0"), Dawn/wgpu-native
  pass. Root cause is a **feature gap**, not a bug: the regular render path supported only one
  single-sample color attachment with no resolve, and two intentional gates blocked it
  (`render_pipeline.rs:783` multisample > 1, `:789` at-most-one-color-target). User approved a **3-slice**
  implementation (each a phase): **(1) multiple color attachments**, (2) MSAA pipeline + attachment, (3)
  per-attachment resolve → CTS green.
  - **Slice 1 — multiple color attachments (non-MSAA), COMPLETE.** Relaxed the `target_count > 1` gate;
    `HalRenderPass.color_target: Option<…>` → `color_targets: Vec<HalRenderColorTarget>`; threaded N color
    attachments in slot order through core pass state / command recording / queue submission; Metal sets
    `colorAttachments[i]` per target; Vulkan emits N `VkAttachmentDescription`/references + framebuffer
    views (+ a partial-view cleanup-on-error fix); GLES (Tier 2) returns a catalogued `HalError` for `> 1`
    color attachment (single still works) — catalogued in `specs/blocks/67-gles-backend.md`. +2 Noop unit
    tests (records two color attachments; rejects pipeline/pass count mismatch via the existing
    `AttachmentSignature` compatibility check).
  - **Verified real-GPU (Claude):** Metal `metal_two_color_attachments_write_distinct_targets` and Vulkan
    `vulkan_render_two_color_attachments_write_distinct_targets` (e2e probes — attachment 0 reads red,
    attachment 1 reads green) pass on the M2; no regression: `rendering/color_target_state` 23/0,
    `rendering/draw` 564/0, `rendering/depth` 130/0 on Metal; Noop workspace test green (67 groups); all
    four clippy gates clean. `render_pass,resolve` still `fail=12` (expected — resolve is slice 3).
  - **Phase Review (Clean Review, fresh no-context subagent):** **0 CRITICAL, 0 MAJOR, 2 MINOR
    (deferred).** Subagent independently ran the Noop tests + clippy + compiled both probes. MINOR-1: a
    sparse "hole" color array (`[Some, None, Some]`) would compact in the execution `Vec` (`.flatten()`)
    but not in the `AttachmentSignature`, a latent slot-misalignment — **currently unreachable** (an
    undefined-format pipeline target maps to `Unsupported` and fails pipeline creation in both backends).
    **Dense-only assumption recorded: slices 2/3 must not build on sparse color arrays without carrying
    slot indices or rejecting `None`-gap arrays in core.** MINOR-2: a pre-existing garbled doc comment on
    `HalRenderPass` (not introduced here). Both deferred. Gate: **no open CRITICAL/MAJOR → F-040 slice 1
    COMPLETE.**
  - **Slice 2 — MSAA pipeline + multisample resolve, COMPLETE (and completed F-040).** Removed the
    `multisample.count > 1` gate; added `sample_count` to `HalRenderPipelineDescriptor` (from
    `multisample.count`); added per-color `resolve_target` (+ resolve mip/layer) to
    `RenderPassColorExecution` and `HalRenderColorTarget`. Metal: pipeline `setRasterSampleCount`;
    attachment `setResolveTexture`/level/slice + `StoreAndMultisampleResolve`/`MultisampleResolve`; regular
    `create_texture` now allocates `sampleCount=4` single-layer 2D textures as `Type2DMultisample` with
    `Private` storage. Vulkan: pipeline `rasterizationSamples` + render-pass color samples from the
    texture's count; per-target resolve `VkAttachmentDescription` + subpass `p_resolve_attachments`
    (`VK_ATTACHMENT_UNUSED` for non-resolve) + resolve framebuffer views; regular `create_texture` removed
    the `sample_count != 1` rejection. GLES (Tier 2): MSAA pipelines + `resolveTarget` return a catalogued
    `HalError` (`67-gles-backend.md`). Resolve was implemented **generically per color target**
    (subset-safe), so the CTS's two-attachment resolve-subset shape works — **slice 3 is subsumed**, no
    separate code needed. +3 Noop unit tests (MSAA pipeline `sample_count` threading; resolve target
    recorded; Noop HAL accepts a `sample_count=4` descriptor).
  - **Diagnosis note (Claude, real-GPU):** the agent's first slice-2 pass threaded sample count + resolve
    but the Metal e2e read back `[0,0,0,0]` — the regular `create_texture` in BOTH backends still rejected
    `sample_count != 1` (MSAA texture allocation existed only in the `tiled` transient path). The HANDOFF
    had wrongly said "MSAA textures already work". Claude caught it on real-GPU (Noop+clippy could not),
    amended the handoff, and the agent added MSAA texture creation. Reinforces [[feedback-claude-owns-gpu-tests]].
  - **Verified real-GPU (Claude):** **`render_pass,resolve:* = 12/0` on Metal AND Vulkan/MoltenVK** (from
    `0/12`). e2e probes `metal_msaa_resolve_writes_resolve_target` + `vulkan_msaa_resolve_writes_resolve_target`
    (single `sampleCount=4` attachment + single-sample resolve target; the resolved pixel reads the drawn
    colour — a stuck pre-fix path read `0`) pass on the M2. No regression: `color_target_state` 23/0,
    `draw` 564/0, `depth` 130/0; Noop workspace green (67 groups); all four clippy gates clean.
  - **Phase Review (Clean Review, fresh no-context subagent):** **0 CRITICAL, 0 MAJOR, 2 MINOR
    (deferred).** Subagent built default/metal/vulkan/gles, ran clippy + Noop tests, and traced the
    subset-resolve attachment/framebuffer/clear-value ordering (consistent; `p_resolve_attachments` one
    entry per color target). MINOR-1: rustfmt churn on 3 pre-existing call sites in the Metal e2e file.
    MINOR-2: a redundant `|| target.resolve_target.is_some()` in Vulkan `vk_resolve_attachment_description`
    (always true in context → always `STORE`, correct but misleading). Both deferred. Gate: **no open
    CRITICAL/MAJOR → F-040 slice 2 COMPLETE → F-040 RESOLVED** (CTS resolve green on both Tier-1 backends).
- **External-CTS finding F-041 — RESOLVED (treated as a phase, with Clean Review).** The T37 (V9)
  `storage_texture/read_only` port: `textureLoad` on a `texture_storage_2d<format, read>` read back `0`
  (`pass=0 fail=3`, byte-identical on Metal and Vulkan/MoltenVK), Dawn/wgpu-native pass. **Two root causes**
  (Claude, source-conclusive + real-GPU + wgpu cross-check):
  - **(1) Storage-texture bindings were dropped from the pipeline binding map.** `compute_pipeline.rs
    metal_buffer_binding_map` (shared by compute AND render) skipped `BindingLayoutKind::StorageTexture` via
    `_ => continue` → the texture was never bound → `textureLoad` read an unbound texture → 0. (First
    storage-texture *operation* coverage; the binding path was never exercised.)
  - **(2) Metal: runtime-sized output array needed naga's MSL buffer-sizes buffer.** The shader's output is
    `array<u32>` (runtime-sized); naga MSL then needs a `_mslBufferSizes` buffer, but
    `EntryPointResources.sizes_buffer` was `None` → naga returned `Internal: "mapping for sizes buffer is
    missing"` → the compute pipeline became an error pipeline → nothing ran → 0. **Not a naga bug** —
    Claude confirmed **wgpu-native passes this test 3/3 on Metal** (same naga→MSL); wgpu provides the sizes
    buffer, yawgpu did not. SPIR-V has native `OpArrayLength`, so Vulkan was unaffected by (2) — once (1)
    landed, Vulkan already passed 3/3.
  - **Fix (coding agent):** (1) `MetalBindingKind::StorageTexture { access }` + `HalDescriptorBindingKind::
    StorageTexture { access }` (+ `HalStorageTextureAccess`) + `HalBoundTexture.storage_access`, threaded to
    the HAL; Metal `map_texture_usage` adds `ShaderRead` for `storage_binding`; Vulkan binds `STORAGE_IMAGE`
    in `GENERAL` layout (descriptor type + pool + pre-dispatch transition). (2) `shader_naga.rs` reflects
    runtime-sized storage globals, reserves a non-colliding `_mslBufferSizes` slot, sets
    `bounds_check_policies = Restrict`, threads slot+bindings via `HalShaderSource::{MslWithBufferSizes,
    MslStagesWithBufferSizes}`; the Metal HAL fills a `uint` byte-length array and binds it via
    `setBytes`/`setVertex/FragmentBytes`. GLES (Tier 2): `submit_compute_pass` returns a catalogued
    `HalError` for any texture binding (was silently ignoring `bind_textures`) — `67-gles-backend.md`.
  - **Verified real-GPU (Claude):** `storage_texture,read_only:* = 3/3` on Metal AND Vulkan/MoltenVK (from
    `0/3`); no regression (compute/basic 1/0, draw 564/0, color_target 23/0, single_buffer 25/0); Noop
    workspace green (67 groups); all four clippy gates clean. e2e `metal_read_only_storage_texture_reads_texel`
    + `vulkan_read_only_storage_texture_reads_texel` (upload texel 7 → `textureLoad` → runtime-sized output →
    read 7; pre-fix read 0) pass on the M2.
  - **Phase Review (Clean Review, fresh no-context subagent, read naga 29.0.3 `back/msl/writer.rs` + ran the
    Metal probe):** **0 CRITICAL, 2 MAJOR (both fixed + re-verified), 3 MINOR (1 fixed, 2 deferred).**
    MAJOR-1 — `_mslBufferSizes` was filled from the per-entry-point subset, but naga lays the struct over
    **all** module runtime-array globals (handle order, positional offsets); a multi-entry-point module
    would misalign → garbage (single-entry, the tested case, coincided). Fixed: reflect all module globals
    in `global_variables` order; the Metal fill writes `0` for unbound entries. MAJOR-2 — the reserved sizes
    slot was `max(buffer-resource idx)+1`, colliding with vertex-buffer `[[buffer(n)]]` slots on the render
    path. Fixed: reserve above resource + vertex-buffer indices. Both got Noop guard tests
    (`msl_buffer_sizes_cover_all_runtime_arrays_in_module_order`,
    `render_msl_buffer_sizes_slot_avoids_vertex_buffer_slots`). MINOR-1 (dead `MslWithBufferSizes` render
    arm) removed; MINOR-2 (Vulkan error wording) + MINOR-3 (unconditional transfer→compute barrier) deferred.
    Subagent confirmed storage-texture binding, Vulkan STORAGE_IMAGE/GENERAL, `Restrict` policy (safety
    improvement, no regression), GLES `HalError`, no panics, and sound e2e guards. Gate: **no open
    CRITICAL/MAJOR → F-041 COMPLETE.** Reinforces [[feedback-claude-owns-gpu-tests]] (Noop+clippy passed
    while real-GPU exposed the MSL gap) and [[feedback-gpu-probe-false-signals]].
- **External-CTS finding F-042 — RESOLVED (2-slice; both slices COMPLETE).**
  F-042 (T39/V7b `memory_sync/buffer/single_buffer:two_draws_*`): a fragment-stage storage write from a
  point draw read back `0` (`pass=0 fail=5`, cross-HAL), Dawn/wgpu pass. **Two independent root causes**
  (Claude, real-GPU + experiment); user approved a **2-slice** plan: **(1) usage-scope write+write false
  rejection**, (2) render bundle execution.
  - **Slice 1 — render usage-scope allows write+write across draws, COMPLETE.** The two draws write the
    same storage buffer via separate bind groups; `validate_buffer_usage_scope` errored whenever *either*
    overlapping use was a Write, but WebGPU allows write+write of the same buffer in a render-pass usage
    scope (content-undefined but valid). A throwaway experiment confirmed it: relaxing the rule took
    `two_draws_in_the_same_render_pass` from `0/5` to `3/5` (the non-bundle subcases). **Subtlety:** compute
    *within-dispatch* two-binding write+write must still error (`assert_compute_buffer_alias`), and render
    *within-draw* write+read must still error. Fix (coding agent): `record_pipeline_usage_scope` now does a
    **strict per-draw** check (the draw's own uses incl. attachments — catches within-draw two-binding
    aliasing) + a **lenient cross-draw** accumulated check (`validate_*_usage_scope_lenient`: error only on
    `access != access` = read↔write, allowing write+write/read+read). Compute path
    (`validate_compute_dispatch_state` → `validate_usage_scope`, strict, per-dispatch) unchanged. +3 Noop
    render-pass unit tests.
  - **Verified real-GPU (Claude):** `two_draws_in_the_same_render_pass:*` reaches `pass=3` on Metal and
    Vulkan/MoltenVK (the 3 non-bundle subcases; the both-via-bundle subcase + `two_draws_in_the_same_render_bundle`
    remain for slice 2); no regression (`rw`/`ww` 8/0, `draw` 564/0, `compute` 1/0). Noop workspace green
    (67 groups); all four clippy gates clean. Claude authored e2e
    `metal_two_draws_write_same_storage_buffer` (two point draws, separate bind groups, same storage buffer
    via an explicit shared layout → buffer reads 1 or 2; pre-fix the usage-scope error left it 0).
  - **Phase Review (Clean Review, fresh no-context subagent):** **0 CRITICAL, 0 MAJOR, 0 MINOR.** Subagent
    independently verified all four anchors (A1 render cross-draw write+write OK; A2 compute within-dispatch
    write+write errors; A3 render within-draw write+read errors; A4 render cross-draw write+read errors),
    ran the Noop tests + 30 CTS `resource_usages` tests + clippy + the Metal e2e on the M2. Render
    within-draw write+write is still rejected (matches compute). Gate: **no open CRITICAL/MAJOR → F-042
    slice 1 COMPLETE.**
  - **Slice 2 — render bundle execution, COMPLETE (and completed F-042).** Render bundles were
    validation-only: `RenderBundleEncoder` draw methods validated + recorded usage scope but recorded **no
    draw command**, and `execute_bundles` replayed **none** → bundle draws were GPU no-ops (the bundle
    subcases read 0, unmasked once slice 1 landed — [[feedback-crash-masks-behavior]]). Fix (coding agent,
    core-only — the HAL already does one-draw-per-`RenderPassCommand`): a `RenderBundleDraw` snapshot
    (pipeline + bind_groups + vertex_buffers + index_buffer + indirect_buffer + `RenderDrawExecution`) is
    recorded per bundle draw (all 4 kinds) into `RenderBundleInner.draws`; `execute_bundles` replays each as
    a `RenderPassCommand` combining the bundle draw with the executing pass's attachments + `blend_constant`
    + `stencil_reference`, increments `draw_count`, sets `render_pass_recorded`, and `clear_render_state()`
    after (WebGPU resets pass render state post-ExecuteBundles). Bundle-draw resources are added to the
    bundle's referenced set (destroy-at-submit validation). +3 Noop unit tests.
  - **Verified real-GPU (Claude):** `two_draws_in_the_same_render_pass:* = 4/4` and
    `two_draws_in_the_same_render_bundle:* = 1/1` on Metal AND Vulkan/MoltenVK (F-042 → `two_draws_*` 5/5;
    `single_buffer:*` whole group `30/0`); no regression (`rendering/draw` 564/0). Noop workspace green (67
    groups); all four clippy gates clean. Claude authored e2e
    `metal_render_bundle_two_draws_write_storage_buffer` (two draws recorded in a bundle, executed via
    `executeBundles`, fragment storage write → 1 or 2; pre-fix the bundle no-op left it 0).
  - **Phase Review (Clean Review, fresh no-context subagent):** **0 CRITICAL, 0 MAJOR, 2 MINOR (1 fixed, 1
    deferred).** Subagent built yawgpu-core, ran clippy + the Metal e2e on the M2, and verified replay
    field-sourcing, snapshot isolation, post-ExecuteBundles state reset, validation order, and
    referenced-resource/destroy coverage. MINOR-1 (`render_bundle_draw_snapshot` used `.expect` for the
    pipeline — CLAUDE.md principle 3 no-panics-in-core) **fixed** (the resolved `Arc<RenderPipeline>` is now
    passed in). MINOR-2 (the inline draw's empty-attachment guard not mirrored in the replay loop) deferred
    — unreachable (a pass can't begin with zero attachments and the bundle signature must match). Gate:
    **no open CRITICAL/MAJOR → F-042 slice 2 COMPLETE → F-042 RESOLVED** (`two_draws_*` 5/5 on both Tier-1
    backends).
- **External-CTS finding F-043 — RESOLVED (treated as a phase, with Clean Review).** T43 (V13)
  `rendering/3d_texture_slices:one_color_attachment,mip_levels`: `WGPURenderPassColorAttachment.depthSlice`
  (which z-slice of a 3D render target a draw hits) was ignored — yawgpu always rendered to slice 0
  (`pass=3 fail=3`, byte-identical Metal + Vulkan/MoltenVK; `depthSlice=1` cases got slice-0's pattern).
  Root cause (same shape as F-038/F-041): `validate_color_attachment_depth_slice` validated `depthSlice`
  but `RenderPassColorExecution` had no `depth_slice` field, so it was dropped before the HAL. Fix (coding
  agent): `RenderPassColorExecution.depth_slice` (from `attachment.depth_slice.unwrap_or(0)`) →
  `HalRenderColorTarget.depth_slice` → Metal `setDepthPlane(depth_slice)` + `setSlice(0)` for 3D targets
  (non-3D keep `setSlice(array_layer)`); Vulkan `baseArrayLayer = depth_slice` for a `TYPE_2D` view of the
  3D slice + 3D images created with `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`. GLES already rejects non-2D
  color attachments with `HalError` (catalogued), so no silent mis-render. +Noop unit test.
  - **Diagnosis note (Claude):** the CTS query path was initially mis-typed (group `3d_texture_slices`,
    test `one_color_attachment,mip_levels` are colon-separated); the CTS runner also needed a rebuild to
    compile the new T43 spec.
  - **Verified real-GPU (Claude):** `3d_texture_slices:one_color_attachment,mip_levels:* = 6/6` on Metal
    AND Vulkan/MoltenVK (from `3/3`); no regression (`draw` 564/0, `depth` 130/0, `render_pass,resolve`
    12/12 on Vulkan — re-checked after the addendum). Claude authored e2e
    `metal_render_pass_depth_slice_targets_requested_3d_slice` (init 3D slice0=10/slice1=20, clear-only pass
    `depthSlice=1` clears 255 → slice0 stays 10, slice1=255; pre-fix slice0 was cleared instead).
  - **Phase Review (Clean Review, fresh no-context subagent):** **0 CRITICAL, 0 MAJOR, 0 MINOR.** Subagent
    empirically reverted the fix to confirm the e2e fails pre-fix, and determined GLES errors cleanly on 3D
    color attachments (no silent wrong). It also surfaced a **pre-existing broken Vulkan-feature HAL test**
    (`render_attachment_descriptions_preserve_contents_for_load_ops` — a Noop `dummy_texture` used where
    `vk_color_attachment_description` has required `HalTexture::Vulkan` since F-040 slice 2; latent because
    the gates never ran `cargo test -p yawgpu-hal --features vulkan --lib` — only the default + clippy
    compile). Fixed in this phase (the test now uses a Vulkan-backed dummy; `sample_count` moved from
    `VulkanTextureInner` to the outer `VulkanTexture` so attachment-description tests don't need an
    allocated image). Both feature-gated HAL suites now pass (vulkan 76/0, metal 55/0); added
    [[feedback-run-feature-gated-hal-tests]] so reviews run them. Gate: **no open CRITICAL/MAJOR → F-043
    COMPLETE.**
- **External-CTS finding F-048 — RESOLVED.** T51 (V22) `render_pass/clear_value:stencil_clear_value`: the
  stencil **reference** value was not masked to the stencil aspect's 8-bit width before the `equal` compare
  (`pass=24 fail=6`, Metal == Vulkan/MoltenVK; also affects wgpu-native), so `stencilReference ∈ {258, 65539}`
  with `applyAsReference=true` mismatched the correctly-masked cleared stencil (2 / 3). Fix (coding agent):
  mask `stencil_reference & 0xFF` in core `queue.rs` when building `HalRenderPass` (backend-independent;
  every WebGPU stencil format is 8-bit). + Noop unit test (258→2, 7→7). **Verified real-GPU (Claude):**
  `clear_value:stencil_clear_value = 30/30` on Metal AND Vulkan/MoltenVK (from `24/6`); `rendering,stencil`
  188/0 (no regression). 1-line prescribed fix, fully CTS-verified on both backends → self-reviewed.
  **Re-verified via CTS re-run against current yawgpu: F-046 (culling/winding) `12/12` and F-049
  (render_bundle) `4/4` are already resolved by the threading audit (`de4a99f`) — stale in FINDINGS.** Open
  CTS findings remaining (F-044/F-047/F-045 since RESOLVED below — F-045 on Metal; its Vulkan/MoltenVK case
  is a documented MoltenVK artifact): F-050 (occlusion query).
- **External-CTS finding F-044 — RESOLVED.** T46 (V16) `vertex_state/correctness:
  vertex_format_to_shader_format_conversion`: yawgpu implemented ONLY the 4 `float32` vertex formats; every
  other `GPUVertexFormat` decoded to **zero** (`pass=1 fail=8`, Metal == Vulkan/MoltenVK). Root cause:
  `hal_vertex_format` mapped only `0x1C..0x1F` → `Float32*`, else `Unsupported` (which the backends error on),
  and the naga MSL metadata (`MslVertexFormat`) was likewise float32-only. Fix (coding agent): expand
  `HalVertexFormat` + `MslVertexFormat` to the full set (0x01..=0x29), map every raw value in
  `hal_vertex_format` (core), map each to `vk::Format` / `MTLVertexFormat` / `naga::back::msl::VertexFormat`,
  and GLES attrib metadata (`glVertexAttribIPointer` for int formats, normalized for unorm/snorm,
  `UNSIGNED_INT_2_10_10_10_REV` for packed); `unorm8x4-bgra` is a catalogued Tier-2 `HalError` on GLES
  (`67-gles-backend.md`). The GPU/naga does the conversion — no shader/core-validation change. **Verified
  real-GPU (Claude):** `vertex_state,correctness = 9/9` on Metal AND Vulkan/MoltenVK (from `1/8`);
  `rendering,draw` 564/0 (no regression). **Clean Review: 0 CRITICAL/MAJOR** (verified all 41 formats handled
  consistently across all 4 mappers + raw values match webgpu.h; 1 MINOR = the GLES catalogue entry, added).
- **External-CTS finding F-047 — RESOLVED.** `render_pipeline/overrides:basic`: WGSL `override` constants —
  both their WGSL defaults (`override R = 1.0;`) and pipeline-provided `WGPUConstantEntry` values — were
  **ignored, emitted as 0** (`pass=1 fail=5`, Metal == Vulkan/MoltenVK; also affects wgpu-native). Same
  "validate but don't act" shape: yawgpu PARSED + VALIDATED the constants (`resolve_pipeline_constants` vs
  `module.overrides()`) but `generate_msl`/`generate_glsl`/`generate_spirv` codegen'd from the RAW
  `self.module`/`self.info` and never ran `naga::back::pipeline_constants::process_overrides` (that helper
  existed but was used only to resolve `@workgroup_size`). Fix (coding agent): add a
  `pipeline_constants: &naga::back::PipelineConstants` param to each `generate_*`, run `process_overrides`
  (per-stage, `Some((stage, entry))`) first and codegen from the PROCESSED `(module, info)` — naga applies
  provided values AND fills WGSL defaults; thread the per-stage map (keyed exactly like
  `resolve_pipeline_constant_key`: numeric `@id` string / name) from `render_pipeline.rs` (vertex+fragment) +
  `compute_pipeline.rs`; reflection/buffer-sizes now computed from the processed module. The Metal render
  path was unified to always split vertex/fragment generation (the combined `generate_render_msl` became
  test-only). + Noop unit tests (MSL/GLSL/SPIR-V: empty map → default `1.0`; `{R:0.6}` → `0.6`). **Verified
  real-GPU (Claude):** `render_pipeline,overrides:basic = 6/6` on Metal AND Vulkan/MoltenVK (from `1/5`);
  `compute_pipeline,overrides` 1/1; `rendering,draw` 564/0, `primitive_topology`/`pipeline_output_targets`/
  `culling_tests` clean (no regression from the Metal split). **Clean Review: 0 CRITICAL/MAJOR** (override
  keying matches naga's contract; all reflection uses the processed module; inter-stage IO intact across the
  split; 3 MINOR — chief: the now-test-only `generate_render_msl` is a drift hazard, candidate for removal).
- **External-CTS finding F-045 — RESOLVED on Metal (Vulkan/MoltenVK = MoltenVK artifact).**
  `rendering/depth_clip_clamp:depth_test_input_clamped`: a fragment-shader-written `@builtin(frag_depth)` must
  be **clamped to the viewport `[minDepth,maxDepth]` before the depth test**; yawgpu didn't → the r8unorm
  target got `255` where a correctly-clamped depth keeps it `0` (Metal == Vulkan/MoltenVK; also wgpu-native).
  Metal/D3D (unlike OpenGL/Vulkan, which clamp automatically) do NOT clamp shader-written depth — that's why
  Dawn injects Tint's `ClampFragDepth` for Metal/D3D and why naga has no such transform. Fix is **two-repo**:
  (1) a new naga `back::clamp_frag_depth` transform (`infosia/wgpu` fork, `feature/tiled`
  `3d7d7944d`): injects an `AddressSpace::Immediate` `vec2<f32>` global `[min,max]` and wraps each returned
  depth with `clamp(depth, range.x, range.y)` (handles scalar + struct-member outputs, recurses control flow);
  yawgpu's naga `rev` bumped to it. (2) yawgpu wiring (coding agent): the MSL fragment path runs the transform
  after `process_overrides` when the FS writes frag_depth, reserving an immediate buffer slot ABOVE the
  resource + `_mslBufferSizes` slots (`msl_next_buffer_slot`); the Metal HAL binds `[minDepth,maxDepth]`
  (default `[0,1]`, from the per-draw viewport) at that slot before every render + tiled-subpass draw.
  **Metal-only by design** — the SPIR-V (Vulkan) and GLSL (GLES) paths are untouched because native
  Vulkan/GL clamp automatically per spec. + Noop unit tests (naga: scalar/struct/no-op/MSL-string; yawgpu:
  clamp present + slot `Some` for frag_depth FS, absent + `None` otherwise). **Verified real-GPU (Claude):**
  `rendering,depth_clip_clamp = 1/1` on **Metal** (unclippedDepth subcase skips — no depth-clip-control;
  was `0/1`); no regression (`rendering,depth` 130/0, `rendering,draw` 564/0). **Vulkan/MoltenVK still fails
  `0/1`** — expected: the SPIR-V path is deliberately unchanged; native Vulkan clamps per spec, so this is a
  MoltenVK translation artifact (F-033 class), **unverified on this Mac** (no native Vulkan); confirm on
  native Windows/Vulkan, or optionally extend the transform to the SPIR-V path (idempotent double-clamp) to
  turn MoltenVK green. **Clean Review: 0 CRITICAL/0 MAJOR/0 MINOR.**
- **External-CTS api/operation finding F-032 — RESOLVED.**
  The T27 `image_copy` depth/stencil ports surfaced that yawgpu zeroed the depth/stencil
  aspect of buffer⇄texture copies — un-masked once F-031's gap-7 stopped rejecting them.
  Root-caused on real-GPU Metal into several sub-gaps, fixed in sequence:
  - **(a1)** the regular render pipeline rejected a fragment with **zero colour targets**;
    a frag-depth-only fragment is valid WebGPU (relaxed the validation).
  - **(b)** the Metal buffer⇄texture copy ignored the copy **aspect**: added
    `aspect`/`format` to `HalBufferTextureCopy`, used
    `MTLBlitOption::{depth,stencil}FromDepthStencil` for packed formats, and made the
    HAL byte-size aspect-aware (stencil = 1, depth = 2/4) so tight aspect-sized buffers
    pass `validate_buffer_texture_range`. Fixes packed-stencil fully.
  - **(a2)** sampled-texture/sampler bind-group execution was entirely unimplemented
    (only buffers were bound at draw/dispatch); the depth staging samples an `r32float`
    in a frag-depth shader. Implemented texture/sampler bindings across core + Metal
    (+ Vulkan/GLES), incl. preserving `TextureView` metadata so Metal binds an actual
    single-layer view (`newTextureViewWithPixelFormat:textureType:levels:slices:`)
    instead of the parent array texture. Fixes depth-only (all sizes/mips).
  - **(c)** the **core** copy-layout helper `texel_copy_block_size` returned the
    *whole-format* `texel_block_size` for the **depth aspect of a packed format**
    (`Depth32FloatStencil8` = 5) instead of the 4-byte depth plane, over-reporting the
    required buffer size (a 3×3 copy needs `(3-1)*256 + 3*4 = 524` bytes, not `527`).
    `validate_buffer_texture_copy` then *rejected* the CTS's tightly-sized buffers, so
    no HAL copy was emitted and the zero-initialised output stayed zero. Fixed to return
    the depth-plane size (`texel_block_size − 1`; the stencil plane is always 1 byte) for
    `DepthOnly` of a packed format. This is what kept the **packed depth** sub-case open.
  **Verified real-GPU Metal:** `image_copy` depth/stencil is now `pass=1152 fail=0`
  (Dawn-equal, from `288/864`); colour `image_copy` regression-checked
  (`undefined_params 2064/0`); `copyTextureToTexture` `copy_depth_stencil 216/0` (no
  F-031 regression); workspace test green. In-tree `e2e_metal_depth.rs` grew to 18 probes
  (all pass), incl. `metal_sampled_frag_depth_packed_3x3` with a deliberately **tight**
  readback buffer that reproduces the (c) rejection. The (c) core fix carries a Noop unit
  test in `yawgpu-core/src/copy.rs`. a1/b/c by Claude; a2 + the view-binding fix by the
  coding agent.
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
| `features/clip_distances.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/features/clip_distances.rs` (0 active: Noop lacks clip-distances; real bodies) |
| `features/query_types.spec.ts` | 2 | — | `ported*` → `cts/validation/capability_checks/features/query_types.rs` (2 active: occlusion + timestamp-query + missing-feature rejection) |
| `features/subgroup_size_control.spec.ts` | 1 | — | `ported*` → `cts/validation/capability_checks/features/subgroup_size_control.rs` (0 active: Noop lacks subgroups) |
| `features/texture_component_swizzle.spec.ts` | 4 | — | `ported*` → `cts/validation/capability_checks/features/texture_component_swizzle.rs` (1 active (identity); feature/compat-mode ignored) |
| `features/texture_formats.spec.ts` | 13 | features_validation.rs / texture_format_validation.rs | `ported*` → `cts/validation/capability_checks/features/texture_formats.rs` (1 active (capability-guarantee probe); format matrices ignored — static caps not feature-keyed) |
| `features/texture_formats_tier1.spec.ts` | 8 | — | `ported*` → `cts/validation/capability_checks/features/texture_formats_tier1.rs` (1 active (implication); format effects ignored) |
| `features/texture_formats_tier2.spec.ts` | 3 | — | `ported*` → `cts/validation/capability_checks/features/texture_formats_tier2.rs` (1 active (implication); rw-storage effects ignored) |
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
| `bgra8unorm_storage.spec.ts` | 4 | — | `ported*` → `cts/validation/texture/bgra8unorm_storage.rs` (0 active: Noop lacks bgra8unorm-storage; canvas N/A; real bodies) |
| `destroy.spec.ts` | 4 | — | `ported` → `cts/validation/texture/destroy.rs` |
| `float32_filterable.spec.ts` | 1 | — | `ported*` → `cts/validation/texture/float32_filterable.rs` (0 active: Noop lacks float32-filterable; real body) |
| `rg11b10ufloat_renderable.spec.ts` | 5 | — | `ported*` → `cts/validation/texture/rg11b10ufloat_renderable.rs` (feature advertised but renderability not feature-applied — ignored, real bodies) |

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
