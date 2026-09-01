# Refactor — duplication & inefficiency sweep (2026-09-02)

Goal: remove copy-pasted code and hot-path waste across `yawgpu-hal`,
`yawgpu-core`, and `yawgpu` (FFI) **without changing public API or
observable behaviour**, except where a duplicate has drifted from the
spec (called out explicitly below). Every slice must keep
`cargo test --workspace` + `cargo clippy --workspace --all-targets
-- -D warnings` green, and HAL slices are additionally re-verified on
real Metal + MoltenVK e2e before commit.

Method: three fresh-context audit agents (one per crate) produced
file:line findings; each was verified by reading the code before
being scheduled. Findings that are behaviour-sensitive and cannot be
verified on this machine (GLES draw-time dirty tracking) are deferred,
not silently dropped.

## Slices

| Slice | Crate | Status | Commit |
|---|---|---|---|
| H1 | yawgpu-hal (lib.rs dispatch, Vulkan, Metal, shared format helpers) | DONE | see log |
| H2 | yawgpu-hal (GLES-only dedupe + program cache) | in progress | |
| C1 | yawgpu-core (pipeline creation, pass recording, copy validation) | planned | |
| F1 | yawgpu (FFI: handle borrowing, chain walks, callbacks, macros) | planned | |

## H1 — yawgpu-hal: dispatch + Vulkan + Metal + shared helpers

| # | Finding | Location | Action |
|---|---|---|---|
| H1.1 | 54 hand-written 4-arm `match self` dispatch methods, no macro in crate | `lib.rs:244-1555` | `hal_dispatch!` macro (plain forward + `.map(Variant)` constructor form) |
| H1.2 | `create_compute_descriptor_pool` is an inlined copy of `create_descriptor_pool`; `allocate_*_descriptor_sets` and the `write_specs → Vec<WriteDescriptorSet>` block in `update_*_descriptor_sets` are duplicated; pool counting does six passes | `vulkan/pipeline.rs:1236-1349` vs `2006-2130`; `1352-1362`/`1454-1464`; `1403-1435`/`1524-1556` | delegate like the render path already does; generic allocate; shared `build_descriptor_writes`; one counting pass |
| H1.3 | `validate_buffer_texture_range` + aspect bytes-per-pixel duplicated Metal/Vulkan (3 differing lines) | `metal/format.rs:16-73` vs `vulkan/encode.rs:3495-3558` | move to `format.rs` as `pub(crate)` helpers taking `buffer_size: u64` |
| H1.4 | Metal per-stage slot resolution triplicated in `encode_render_bind_{buffer,texture,sampler}`; compute counterparts repeat the resource-unwrap prologue | `metal/encode.rs:1448-1524, 1701-1736, 585-676` | `per_stage_slots()` + `metal_bound_*()` unwrap helpers |
| H1.5 | Vulkan submit walks the copy list three times; `collect_retained_resources` pushes one `Arc` clone per occurrence with no dedup | `vulkan/encode.rs:43, 67-116, 187, 399-630` | dedup by `Arc::as_ptr` (identity set) |

Deferred from H1 (behaviour-sensitive, needs its own real-GPU pass):
Metal derived-texture-view caching (`metal/encode.rs:1654-1664`),
per-draw scratch-buffer reuse in `encode_metal_stream_draw`, unified
`HalRenderStreamState` interpreter across the three backends,
per-backend bytes-per-block column removal from `map_texture_format`.

## H2 — yawgpu-hal: GLES

| # | Finding | Location | Action |
|---|---|---|---|
| H2.1 | GLSL compute program compiled+linked+deleted on every texture→buffer copy of snorm/16-bit/depth formats | `gles/queue.rs:3723, 3796, 3801-3925` | cache on `GlesDeviceInner` keyed `(target, encoding)` incl. uniform locations |
| H2.2 | Five near-identical framebuffer-attach helpers | `gles/queue.rs:1879-2004, 3066-3102, 3390-3415` | one `attach_texture_to_framebuffer` |
| H2.3 | `GlesClearKind` recomputed via the 478-line format table; `HalTextureFormat::color_clear_kind` already exists | `gles/format.rs:605-638`; `gles/queue.rs:1793, 2957` | use `format.color_clear_kind()` if provably identical |
| H2.4 | Two `row_spans` precompute loops; per-row staging `Vec`; compute bind loop collects two Vecs then iterates | `gles/queue.rs:3304-3372, 3651-3690, 3426, 359-396` | shared `buffer_texture_row_spans`; hoist staging; share bind loop with render path |

Deferred (cannot be verified here — GLES real-GPU runs are Windows
ANGLE only): draw-time dirty tracking in `run_render_draw`
(`gles/queue.rs:2140-2223`), precomputed binding index maps
(`gles/queue.rs:463-505, 535-559, 800-820, 1042-1055`).

## C1 — yawgpu-core

| # | Finding | Location | Action |
|---|---|---|---|
| C1.1 | Full pipeline resolve (reflection + auto-layout BGL allocation) runs twice per `create_*_pipeline`: once via `validate_*_pipeline_descriptor` (with `pipeline_id = 0`), again in `*Pipeline::new` | `device.rs:528-566, 599-635`; `render_pipeline.rs:866-872, 946-963`; `compute_pipeline.rs:158-169, 904-910` | resolve once in `Device`, pass the `Result` into `new` |
| C1.2 | `create_*_pipeline` vs `_without_error_dispatch` are four near-identical bodies (+ subpass fifth) | `device.rs:527-737` | single inner fn with `dispatch: bool` |
| C1.3 | `record_pipeline_usage_scope` allocates two Vecs per draw and discards them; dead `_attachment_uses` param | `pass.rs:959-985` | non-allocating overflow check |
| C1.4 | 8 draw fns copy-pasted between render pass and render bundle; the "requires attachment / open occlusion query" block repeated 4× | `render_pass.rs:278-487` vs `render_bundle.rs:379-538` | shared `record_draw` in `pass.rs` |
| C1.5 | `validate_queue_write_texture` duplicates `validate_texture_copy_subresource`. **Behaviour drift:** write path bounds against logical mip size, copy path against physical (block-rounded) size; spec + Dawn use physical | `texture.rs:676-795` vs `command_encoder.rs:1798-1882, 1969-2050, 2149-2155` | shared range validator using physical size; unit test for compressed odd mip via `write_texture` |
| C1.6 | `Texture::format_caps()` re-walks the 330-line caps match + 6 `BTreeSet` probes on every call; inputs are immutable | `texture.rs:216-226`; `format.rs:713-802` | cache in `TextureInner`; merge the two `TextureFormatsTier1` blocks |
| C1.7 | bind-groups-plus-vertex-buffers limit check written three times | `pass.rs:584-613`; `subpass.rs:778-808`; `render_pipeline.rs:2533-2552` | one `pub(crate)` helper |
| C1.8 | Render pipeline collects stage resource bindings twice (validate + effective BGLs); compute does it once | `render_pipeline.rs:3423-3449, 3527-3559` | hoist into `resolve_render_pipeline_descriptor` |
| C1.9 | `hal_set_bind_group` allocates Vec + one-entry BTreeMap + deep clone per SetBindGroup at submit; texture/storage-texture arms duplicated | `queue.rs:1702-1730, 2032-2172` | single-group variant; `hal_bound_texture` helper |
| C1.10 | `record_pass_command` takes the pass lock twice per command | `pass.rs:439-467` | single guard |
| C1.11 | `execute_bundles` `to_vec()`s four slices per bundle; `bind_group_{buffer,texture}_resources` allocate per `set_bind_group` | `render_pass.rs:621-632`; `pass.rs:1369-1390` | `impl IntoIterator` params / iterator returns |
| C1.12 | Indirect dispatch collects + O(n²)-validates the usage scope twice | `compute_pass.rs:113-155, 183-199` | thread the indirect buffer into the single pass |
| C1.13 | `reflected_storage_texture_format` uses raw hex instead of `TextureFormat::*` constants | `compute_pipeline.rs:1727-1780` | named constants |
| C1.14 | Three buffer write/clear validators differ only in prefix and options | `command_encoder.rs:1042-1090`; `buffer.rs:722-748` | one parameterised validator |
| C1.15 | `resolve_*_pipeline_immediate_size` twins; passthrough prologue duplicated; test helpers duplicated across modules | `render_pipeline.rs:3498-3512`; `compute_pipeline.rs:953-1026, 1262-1280`; test mods | fold; move helpers to `test_helpers.rs` |

Deferred: subpass attachment validation dedupe (`subpass.rs:1009-1163`,
tiled vendor ext, low traffic), single-table format caps merge.

## F1 — yawgpu (FFI)

| # | Finding | Location | Action |
|---|---|---|---|
| F1.1 | `clone_handle` (2 atomic RMWs) at ~70 sites where the `Arc` is only read and dropped | `render_pass.rs`, `bundle.rs`, `compute_pass.rs`, `tiled.rs`, `encoder.rs`, `conv/*.rs`, `ffi/mod.rs` | `borrow_handle` at non-retaining sites |
| F1.2 | 15 hand-rolled `WGPUChainedStruct` walks | `ffi/mod.rs`, `ffi/device.rs`, `conv/{descriptors,bind,limits,shader}.rs` | generic `find_in_chain::<T>(chain, s_type)` |
| F1.3 | `PendingCallback::fire` pipeline arms are twins and each has a dead duplicated branch; `create_*_pipeline_handle` twins; cache-insert tail copied 4× | `ffi/mod.rs:1842-1911, 1976-2068`; `ffi/device.rs:316-324, 426-434` | collapse; `cache_if_valid` helper |
| F1.4 | `cache_handle` prunes the whole map on every miss (O(n²)) | `ffi/mod.rs:848-867` | amortised prune |
| F1.5 | `label_from_string_view` re-implements `string_view_to_str`; three `SetLabel` paths allocate a `String` to pass `&str` | `conv/strings.rs:21-58`; `ffi/{device,queue,query}.rs` | reuse decoder; borrow |
| F1.6 | `map_shader_module_descriptor` three sequential `if`s + 4× error literal | `conv/shader.rs:20-73` | `match node.sType` |
| F1.7 | `wgpuCreateInstance` backend block triplicated | `ffi/instance.rs:42-144` | shared finish helper |
| F1.8 | `validate_render_pass_descriptor_devices` duplicated for subpasses | `ffi/encoder.rs:568-637` vs `ffi/tiled.rs:461-509` | shared view-device check |
| F1.9 | `wgpuInstanceWaitAny` spin loop allocates 3 Vecs per iteration + O(n²) fill | `ffi/instance.rs:532-560`; `ffi/mod.rs:699-721` | hoist buffers; sort/set |
| F1.10 | Surface present clones whole config state incl. Vec twice per frame | `ffi/surface.rs:179-183, 252-256` | borrow under guard |
| F1.11 | 25×AddRef + 25×Release + 17×SetLabel hand-written; 18 `Impl` structs share the same shape | every `ffi/*.rs`; `ffi/mod.rs:132-545` | `declare_wgpu_handle!` macro emitting struct + exports + test accessor |
| F1.12 | Render-pass / bundle / subpass recording families are three near-verbatim copies (only the error noun differs) | `ffi/render_pass.rs:110-515`, `ffi/bundle.rs:43-385`, `ffi/tiled.rs:190-433` | recording macro / shared generic fns |

Deferred: double `Arc<core::X>` where `core::X` is already an `Arc`
newtype (cross-crate signature change, ~10 core APIs); `ExecuteBundles`
double Vec; `adapter_info_from_core` constant allocation.

## Log

- 2026-09-02 — audits complete, baseline `cargo test --workspace` +
  clippy green at `feed066`. H1 dispatched.
- 2026-09-02 — H1 reviewed + landed: 6 files, +564/−818. Gates: workspace
  test 997/0, clippy clean, `yawgpu-hal --features metal,vulkan --lib`
  207/0, Metal e2e (basic/buffer/texture/render/compute/draw/depth/
  stencil/smoke) 48/0, MoltenVK e2e (basic/buffer/texture/render/
  compute/depth) 31/0. `find_surface_pending` fold left as-is (touches
  present control flow). H2 dispatched.
