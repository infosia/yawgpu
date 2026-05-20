# Phase 10.3a - yawgpu-core Public Function Audit

Inventory source: `git show HEAD:yawgpu-core/src/lib.rs` before P10.3a narrowing. Decisions were checked against the edited `yawgpu-core/src/lib.rs`.

## Summary

- Total `pub fn` before audit: 208
- Kept `pub`: 183
- Narrowed to `pub(crate)`: 24
- Deleted as dead code: 1
- Remaining function rows after audit: 207

## Kept Public Distribution

- Instance / Adapter / Device: 51
- Buffer / Texture / Sampler: 40
- Pipeline / Shader: 18
- Encoder / Pass / Bundle: 59
- Query / Error / Future: 14
- Surface / utilities: 1

## Inventory

| Line | Function | Cross-crate use | Decision | Note |
|---:|---|---|---|---|
| 43 | `Instance::new_noop` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 48 | `Instance::from_hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 58 | `Instance::enumerate_adapters` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 63 | `Instance::enumerate_adapters_with_feature_level` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 76 | `Instance::future_registry` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 81 | `Instance::hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 88 | `Instance::create_surface_from_metal_layer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 114 | `Adapter::from_hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 119 | `Adapter::from_hal_with_feature_level` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 126 | `Adapter::name` | `yawgpu/src`, `yawgpu-test/src` | keep pub | Observed downstream use; kept public. |
| 131 | `Adapter::backend` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 136 | `Adapter::limits` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 143 | `Adapter::feature_level` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 148 | `Adapter::features` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 153 | `Adapter::has_feature` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 157 | `Adapter::create_device` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 213 | `Device::from_hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 235 | `Device::queue` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 240 | `Device::allocation_count` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 245 | `Device::hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 250 | `Device::limits` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 255 | `Device::features` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 260 | `Device::has_feature` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 265 | `Device::create_query_set` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 278 | `Device::same` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 282 | `Device::set_label` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 287 | `Device::label` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 291 | `Device::destroy` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 295 | `Device::lose` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 305 | `Device::is_lost` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 310 | `Device::lost_reason` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 314 | `Device::set_uncaptured_error_callback` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 321 | `Device::clear_uncaptured_error_callback` | none | delete (dead code) | No workspace caller found; removed as unambiguous dead code. |
| 325 | `Device::push_error_scope` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 332 | `Device::pop_error_scope` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 342 | `Device::dispatch_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 363 | `Device::create_buffer` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 383 | `Device::create_texture` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 407 | `Device::create_sampler` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 432 | `Device::create_shader_module` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 460 | `Device::create_bind_group_layout` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 479 | `Device::create_bind_group` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 496 | `Device::create_pipeline_layout` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 523 | `Device::create_command_encoder` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 532 | `Device::create_compute_pipeline` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 556 | `Device::create_compute_pipeline_without_error_dispatch` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 577 | `Device::create_render_pipeline` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 598 | `Device::create_render_pipeline_without_error_dispatch` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 634 | `MapMode::from_bits` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 684 | `BufferUsage::from_bits_retain` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 689 | `BufferUsage::bits` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 694 | `BufferUsage::contains` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 720 | `TextureUsage::from_bits_retain` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 725 | `TextureUsage::bits` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 730 | `TextureUsage::contains` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 799 | `TextureFormat::from_raw` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 804 | `TextureFormat::raw` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 809 | `TextureFormat::is_undefined` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 814 | `TextureFormat::caps` | `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 933 | `TextureFormat::srgb_pair` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1293 | `Texture::from_hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1298 | `Texture::usage` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1303 | `Texture::dimension` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1308 | `Texture::size` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1313 | `Texture::format` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1318 | `Texture::mip_level_count` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1323 | `Texture::sample_count` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1328 | `Texture::view_formats` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1337 | `Texture::is_view_format_compatible` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1342 | `Texture::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1347 | `Texture::is_destroyed` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1352 | `Texture::same` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1360 | `Texture::destroy` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1365 | `Texture::create_view` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 1419 | `Texture::validate_queue_write` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1515 | `TextureView::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1520 | `TextureView::texture` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1525 | `TextureView::format` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1530 | `TextureView::dimension` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1535 | `TextureView::base_mip_level` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1540 | `TextureView::mip_level_count` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1545 | `TextureView::base_array_layer` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 1550 | `TextureView::array_layer_count` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1555 | `TextureView::aspect` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 1560 | `TextureView::render_extent` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1623 | `ShaderModule::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1628 | `ShaderModule::diagnostic` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1913 | `BindGroupLayout::error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1918 | `BindGroupLayout::entries` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1923 | `BindGroupLayout::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1928 | `BindGroupLayout::is_default` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1933 | `BindGroupLayout::same` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1988 | `BindGroup::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 1993 | `BindGroup::layout` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 1998 | `BindGroup::entries` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 2299 | `PipelineLayout::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 2304 | `PipelineLayout::bind_group_layouts` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 2434 | `ComputePipeline::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 2439 | `ComputePipeline::entry_name` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 2444 | `ComputePipeline::bind_group_layouts` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 3286 | `VertexFormat::from_raw` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 3531 | `RenderPipeline::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 3536 | `RenderPipeline::vertex_entry_name` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 3541 | `RenderPipeline::fragment_entry_name` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 3546 | `RenderPipeline::bind_group_layouts` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 3563 | `RenderPipeline::required_vertex_buffer_count` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 4564 | `Sampler::descriptor` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 4569 | `Sampler::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4730 | `Buffer::size` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4735 | `Buffer::usage` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4740 | `Buffer::map_state` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4745 | `Buffer::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4750 | `Buffer::is_destroyed` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 4755 | `Buffer::same` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4764 | `Buffer::destroy` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4779 | `Buffer::unmap` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4808 | `Buffer::begin_map` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4858 | `Buffer::resolve_pending_map` | public core API / constructor path | keep pub | Kept for downstream API surface; full workspace compile confirms reachability requirements. |
| 4902 | `Buffer::abort_pending_map` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4912 | `Buffer::mapped_range` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4943 | `Buffer::write_from_queue` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 4961 | `Buffer::hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 4965 | `Buffer::validate_queue_write` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 5835 | `supported_features` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 5908 | `QuerySet::kind` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 5913 | `QuerySet::count` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 5917 | `QuerySet::set_label` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 5922 | `QuerySet::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 5927 | `QuerySet::is_destroyed` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 5932 | `QuerySet::same` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 5936 | `QuerySet::destroy` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6381 | `CommandEncoder::begin_render_pass` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6408 | `CommandEncoder::begin_compute_pass` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6452 | `CommandEncoder::insert_debug_marker` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6456 | `CommandEncoder::record_validation_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6460 | `CommandEncoder::copy_buffer_to_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6491 | `CommandEncoder::clear_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6497 | `CommandEncoder::write_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6503 | `CommandEncoder::write_timestamp` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6510 | `CommandEncoder::resolve_query_set` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6529 | `CommandEncoder::copy_buffer_to_texture` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 6552 | `CommandEncoder::copy_texture_to_buffer` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 6580 | `CommandEncoder::copy_texture_to_texture` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 6596 | `CommandEncoder::push_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6612 | `CommandEncoder::pop_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 6632 | `CommandEncoder::record_command_guard` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 6699 | `CommandEncoder::finish` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7498 | `CommandBuffer::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7659 | `RenderPassEncoder::end` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7663 | `RenderPassEncoder::insert_debug_marker` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7667 | `RenderPassEncoder::push_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7671 | `RenderPassEncoder::pop_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7675 | `RenderPassEncoder::set_pipeline` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7682 | `RenderPassEncoder::record_validation_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7686 | `RenderPassEncoder::set_bind_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7711 | `RenderPassEncoder::set_vertex_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7742 | `RenderPassEncoder::set_index_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7765 | `RenderPassEncoder::draw` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7809 | `RenderPassEncoder::draw_indexed` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7832 | `RenderPassEncoder::draw_indirect` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7846 | `RenderPassEncoder::draw_indexed_indirect` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7865 | `RenderPassEncoder::set_viewport` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7878 | `RenderPassEncoder::set_scissor_rect` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7888 | `RenderPassEncoder::set_blend_constant` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7901 | `RenderPassEncoder::set_stencil_reference` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7905 | `RenderPassEncoder::execute_bundles` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7927 | `RenderPassEncoder::begin_occlusion_query` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7946 | `RenderPassEncoder::end_occlusion_query` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7957 | `ComputePassEncoder::end` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7961 | `ComputePassEncoder::insert_debug_marker` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7965 | `ComputePassEncoder::push_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7969 | `ComputePassEncoder::pop_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7973 | `ComputePassEncoder::set_pipeline` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7980 | `ComputePassEncoder::record_validation_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 7984 | `ComputePassEncoder::set_bind_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8009 | `ComputePassEncoder::dispatch_workgroups` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8031 | `ComputePassEncoder::dispatch_workgroups_indirect` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8053 | `RenderBundleEncoder::new` | `yawgpu/src`, `yawgpu/tests`, `yawgpu-test/src` | keep pub | Observed downstream use; kept public. |
| 8083 | `RenderBundleEncoder::finish` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8118 | `RenderBundleEncoder::insert_debug_marker` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8122 | `RenderBundleEncoder::push_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8129 | `RenderBundleEncoder::pop_debug_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8140 | `RenderBundleEncoder::set_pipeline` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8148 | `RenderBundleEncoder::set_bind_group` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8170 | `RenderBundleEncoder::set_vertex_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8198 | `RenderBundleEncoder::set_index_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8218 | `RenderBundleEncoder::draw` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8240 | `RenderBundleEncoder::draw_indexed` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8263 | `RenderBundleEncoder::draw_indirect` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8275 | `RenderBundleEncoder::draw_indexed_indirect` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 8322 | `RenderBundle::is_error` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9060 | `Queue::from_hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9070 | `Queue::hal` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9074 | `Queue::set_label` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9079 | `Queue::label` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9083 | `Queue::write_buffer` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9087 | `Queue::submit` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 9164 | `ErrorFilter::matches` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 9189 | `DeviceError::validation` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 9197 | `DeviceError::internal` | none | narrow to pub(crate) | No cross-crate caller found; compile-checked after narrowing. |
| 9207 | `DeviceError::new` | `yawgpu/src`, `yawgpu/tests`, `yawgpu-test/src` | keep pub | Observed downstream use; kept public. |
| 9254 | `FutureId::get` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9259 | `FutureId::from_raw` | `yawgpu/src`, `yawgpu/tests` | keep pub | Observed downstream use; kept public. |
| 9323 | `FutureRegistry::new` | `yawgpu/src`, `yawgpu/tests`, `yawgpu-test/src` | keep pub | Observed downstream use; kept public. |
| 9328 | `FutureRegistry::register` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9343 | `FutureRegistry::complete` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9350 | `FutureRegistry::process_events` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
| 9374 | `FutureRegistry::wait_any` | `yawgpu/src` | keep pub | Observed downstream use; kept public. |
