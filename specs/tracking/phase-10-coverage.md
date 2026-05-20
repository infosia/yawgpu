# Phase 10 — Public API Unit Test Coverage

P10.3a audit: see `specs/tracking/phase-10-audit.md`.

## yawgpu-core/src/lib.rs - Instance / Adapter / Device / Queue (51 pub fn)

| pub fn | test name(s) |
|---|---|
| `Instance::new_noop` | `creates_noop_device_and_queue` (existing), `noop_device` helper users |
| `Instance::from_hal` | `instance_from_hal_wraps_noop_hal` |
| `Instance::enumerate_adapters` | `creates_noop_device_and_queue` (existing), `noop_adapter` / `noop_device` helper users |
| `Instance::enumerate_adapters_with_feature_level` | `instance_enumerate_adapters_with_feature_level_sets_adapter_feature_level` |
| `Instance::future_registry` | `instance_future_registry_process_events_is_empty_without_futures` |
| `Instance::hal` | `instance_hal_returns_noop_hal_instance` |
| `Instance::create_surface_from_metal_layer` | `instance_create_surface_from_metal_layer_noop_returns_noop_surface` |
| `Adapter::from_hal` | `adapter_from_hal_wraps_noop_hal_adapter` |
| `Adapter::name` | `adapter_from_hal_wraps_noop_hal_adapter`, `adapter_name_backend_limits_and_features_match_noop_contract` |
| `Adapter::backend` | `adapter_name_backend_limits_and_features_match_noop_contract` |
| `Adapter::limits` | `adapter_name_backend_limits_and_features_match_noop_contract` |
| `Adapter::features` | `adapter_name_backend_limits_and_features_match_noop_contract` |
| `Adapter::has_feature` | `adapter_name_backend_limits_and_features_match_noop_contract` |
| `Adapter::create_device` | `creates_noop_device_and_queue` (existing), `adapter_create_device_rejects_unsupported_required_feature`, `adapter_create_device_applies_labels_and_core_feature` |
| `Device::from_hal` | `device_from_hal_wraps_noop_hal_device` |
| `Device::queue` | `creates_noop_device_and_queue` (existing), `adapter_create_device_applies_labels_and_core_feature`, `queue_write_buffer_and_submit_empty_succeed` |
| `Device::allocation_count` | `creates_noop_device_and_queue` (existing), `device_create_buffer_increments_allocation_count`, `device_create_texture_happy_path_and_invalid_size_scope_error` |
| `Device::hal` | `device_from_hal_wraps_noop_hal_device`, `device_hal_limits_and_features_match_noop_contract` |
| `Device::limits` | `device_hal_limits_and_features_match_noop_contract` |
| `Device::features` | `device_hal_limits_and_features_match_noop_contract` |
| `Device::has_feature` | `adapter_create_device_applies_labels_and_core_feature`, `device_hal_limits_and_features_match_noop_contract` |
| `Device::create_query_set` | `device_create_query_set_validates_count_and_creates_happy_path` |
| `Device::same` | `device_same_distinguishes_clone_from_distinct_device` |
| `Device::set_label` | `device_label_defaults_empty_and_set_label_updates_it` |
| `Device::label` | `adapter_create_device_applies_labels_and_core_feature`, `device_label_defaults_empty_and_set_label_updates_it` |
| `Device::destroy` | `device_destroy_lose_is_lost_and_lost_reason_are_idempotent` |
| `Device::lose` | `device_destroy_lose_is_lost_and_lost_reason_are_idempotent` |
| `Device::is_lost` | `device_destroy_lose_is_lost_and_lost_reason_are_idempotent` |
| `Device::lost_reason` | `device_destroy_lose_is_lost_and_lost_reason_are_idempotent` |
| `Device::set_uncaptured_error_callback` | `scoped_error_captures_without_uncaptured_callback` (existing), `uncaptured_error_routes_to_callback_without_scope` (existing) |
| `Device::push_error_scope` | `scoped_error_captures_without_uncaptured_callback` (existing), `device_create_texture_happy_path_and_invalid_size_scope_error`, `device_create_compute_pipeline_happy_path_and_error_scope`, `device_create_render_pipeline_happy_path_and_error_scope` |
| `Device::pop_error_scope` | `scoped_error_captures_without_uncaptured_callback` (existing), `device_create_texture_happy_path_and_invalid_size_scope_error`, `device_create_compute_pipeline_happy_path_and_error_scope`, `device_create_render_pipeline_happy_path_and_error_scope` |
| `Device::dispatch_error` | `scoped_error_captures_without_uncaptured_callback` (existing), `uncaptured_error_routes_to_callback_without_scope` (existing) |
| `Device::create_buffer` | `device_create_buffer_increments_allocation_count`, `queue_write_buffer_and_submit_empty_succeed` |
| `Device::create_texture` | `device_create_texture_happy_path_and_invalid_size_scope_error` |
| `Device::create_sampler` | `device_create_sampler_uses_default_descriptor` |
| `Device::create_shader_module` | `device_create_shader_module_accepts_minimal_compute_wgsl`, `device_create_compute_pipeline_happy_path_and_error_scope`, `device_create_render_pipeline_happy_path_and_error_scope` |
| `Device::create_bind_group_layout` | `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `Device::create_bind_group` | `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `Device::create_pipeline_layout` | `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `Device::create_command_encoder` | `device_create_command_encoder_finishes_empty_encoder` |
| `Device::create_compute_pipeline` | `device_create_compute_pipeline_happy_path_and_error_scope` |
| `Device::create_compute_pipeline_without_error_dispatch` | `device_create_compute_pipeline_without_error_dispatch_keeps_scope_empty` |
| `Device::create_render_pipeline` | `device_create_render_pipeline_happy_path_and_error_scope` |
| `Device::create_render_pipeline_without_error_dispatch` | `device_create_render_pipeline_without_error_dispatch_keeps_scope_empty` |
| `Queue::from_hal` | `queue_from_hal_hal_label_and_set_label_round_trip` |
| `Queue::hal` | `queue_from_hal_hal_label_and_set_label_round_trip` |
| `Queue::set_label` | `queue_from_hal_hal_label_and_set_label_round_trip` |
| `Queue::label` | `adapter_create_device_applies_labels_and_core_feature`, `queue_from_hal_hal_label_and_set_label_round_trip` |
| `Queue::write_buffer` | `queue_write_buffer_and_submit_empty_succeed` |
| `Queue::submit` | `queue_write_buffer_and_submit_empty_succeed` |

## yawgpu-core/src/lib.rs - Buffer / Texture / Sampler (40 pub fn)

| pub fn | test name(s) |
|---|---|
| `MapMode::from_bits` | `map_mode_from_bits_rejects_none_both_and_unsupported_bits` (existing) |
| `BufferUsage::from_bits_retain` | `buffer_usage_from_bits_retain_round_trips_known_and_unknown_bits` |
| `BufferUsage::bits` | `buffer_usage_from_bits_retain_round_trips_known_and_unknown_bits` |
| `TextureUsage::from_bits_retain` | `texture_usage_from_bits_retain_round_trips_known_and_unknown_bits` |
| `TextureUsage::bits` | `texture_usage_from_bits_retain_round_trips_known_and_unknown_bits` |
| `TextureFormat::from_raw` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `TextureFormat::raw` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `From<u32> for TextureFormat` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `From<i32> for TextureFormat` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `From<TextureFormat> for u32` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `From<TextureFormat> for i32` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `TextureFormat::caps` | `texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined` |
| `Texture::from_hal` | `texture_from_hal_and_descriptor_accessors_round_trip` |
| `Texture::usage` | `texture_from_hal_and_descriptor_accessors_round_trip` |
| `Texture::dimension` | `texture_from_hal_and_descriptor_accessors_round_trip` |
| `Texture::size` | `texture_from_hal_and_descriptor_accessors_round_trip`, `device_create_texture_happy_path_and_invalid_size_scope_error` |
| `Texture::format` | `texture_from_hal_and_descriptor_accessors_round_trip`, `texture_is_error_same_destroy_create_view_and_validate_queue_write` |
| `Texture::mip_level_count` | `texture_from_hal_and_descriptor_accessors_round_trip` |
| `Texture::sample_count` | `texture_from_hal_and_descriptor_accessors_round_trip` |
| `Texture::is_error` | `texture_from_hal_and_descriptor_accessors_round_trip`, `texture_error_texture_reports_is_error_and_error_view` |
| `Texture::same` | `texture_is_error_same_destroy_create_view_and_validate_queue_write` |
| `Texture::destroy` | `texture_is_error_same_destroy_create_view_and_validate_queue_write` |
| `Texture::create_view` | `texture_is_error_same_destroy_create_view_and_validate_queue_write`, `texture_error_texture_reports_is_error_and_error_view`, `texture_view_descriptor_fields_round_trip` |
| `Texture::validate_queue_write` | `texture_is_error_same_destroy_create_view_and_validate_queue_write` |
| `TextureView::is_error` | `texture_is_error_same_destroy_create_view_and_validate_queue_write`, `texture_error_texture_reports_is_error_and_error_view`, `texture_view_descriptor_fields_round_trip` |
| `TextureView::format` | `texture_is_error_same_destroy_create_view_and_validate_queue_write`, `texture_view_descriptor_fields_round_trip` |
| `TextureView::dimension` | `texture_view_descriptor_fields_round_trip` |
| `TextureView::mip_level_count` | `texture_view_descriptor_fields_round_trip` |
| `TextureView::base_array_layer` | `texture_view_descriptor_fields_round_trip` |
| `TextureView::aspect` | `texture_view_descriptor_fields_round_trip` |
| `Sampler::descriptor` | `device_create_sampler_uses_default_descriptor`, `sampler_descriptor_and_is_error_pin_valid_and_invalid_descriptors` |
| `Sampler::is_error` | `device_create_sampler_uses_default_descriptor`, `sampler_descriptor_and_is_error_pin_valid_and_invalid_descriptors` |
| `Buffer::size` | `device_create_buffer_increments_allocation_count`, `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |
| `Buffer::usage` | `device_create_buffer_increments_allocation_count`, `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |
| `Buffer::map_state` | `buffer_accessors_error_same_destroy_hal_and_validate_queue_write`, `buffer_map_state_machine_transitions_and_mapped_range_bounds`, `buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted` |
| `Buffer::is_error` | `device_create_buffer_increments_allocation_count`, `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |
| `Buffer::same` | `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |
| `Buffer::destroy` | `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |
| `Buffer::unmap` | `buffer_map_state_machine_transitions_and_mapped_range_bounds` |
| `Buffer::begin_map` | `buffer_map_state_machine_transitions_and_mapped_range_bounds`, `buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted` |
| `Buffer::resolve_pending_map` | `buffer_map_state_machine_transitions_and_mapped_range_bounds`, `buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted` |
| `Buffer::abort_pending_map` | `buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted` |
| `Buffer::mapped_range` | `buffer_map_state_machine_transitions_and_mapped_range_bounds` |
| `Buffer::hal` | `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |
| `Buffer::validate_queue_write` | `buffer_accessors_error_same_destroy_hal_and_validate_queue_write` |

## yawgpu-core/src/lib.rs - Encoder / Pass / Bundle (59 pub fn)

| pub fn | test name(s) |
|---|---|
| `CommandBuffer::is_error` | `command_encoder_create_finish_idempotent_and_command_buffer_is_error_false`, `command_encoder_debug_markers_and_validation_error` |
| `CommandEncoder::begin_render_pass` | `render_pass_encoder_lifecycle_and_debug_markers`, `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw`, `render_pass_encoder_indexed_and_indirect_draws`, `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `CommandEncoder::begin_compute_pass` | `compute_pass_encoder_lifecycle_and_debug_markers`, `compute_pass_encoder_pipeline_bind_group_and_dispatch` |
| `CommandEncoder::insert_debug_marker` | `command_encoder_debug_markers_and_validation_error` |
| `CommandEncoder::record_validation_error` | `command_encoder_debug_markers_and_validation_error` |
| `CommandEncoder::copy_buffer_to_buffer` | `command_encoder_buffer_copies_clear_and_write_validate_offsets` |
| `CommandEncoder::clear_buffer` | `command_encoder_buffer_copies_clear_and_write_validate_offsets` |
| `CommandEncoder::write_buffer` | `command_encoder_buffer_copies_clear_and_write_validate_offsets` |
| `CommandEncoder::write_timestamp` | `command_encoder_query_and_timestamps_pin_validation_and_resolve` |
| `CommandEncoder::resolve_query_set` | `command_encoder_query_and_timestamps_pin_validation_and_resolve` |
| `CommandEncoder::copy_buffer_to_texture` | `command_encoder_texture_copies_record_copy_commands` |
| `CommandEncoder::copy_texture_to_buffer` | `command_encoder_texture_copies_record_copy_commands` |
| `CommandEncoder::copy_texture_to_texture` | `command_encoder_texture_copies_record_copy_commands` |
| `CommandEncoder::push_debug_group` | `command_encoder_debug_markers_and_validation_error` |
| `CommandEncoder::pop_debug_group` | `command_encoder_debug_markers_and_validation_error` |
| `CommandEncoder::finish` | `command_encoder_create_finish_idempotent_and_command_buffer_is_error_false`, `command_encoder_debug_markers_and_validation_error`, `command_encoder_buffer_copies_clear_and_write_validate_offsets`, `command_encoder_texture_copies_record_copy_commands`, `command_encoder_query_and_timestamps_pin_validation_and_resolve` |
| `RenderPassEncoder::end` | `render_pass_encoder_lifecycle_and_debug_markers`, `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw`, `render_pass_encoder_indexed_and_indirect_draws`, `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::insert_debug_marker` | `render_pass_encoder_lifecycle_and_debug_markers` |
| `RenderPassEncoder::push_debug_group` | `render_pass_encoder_lifecycle_and_debug_markers` |
| `RenderPassEncoder::pop_debug_group` | `render_pass_encoder_lifecycle_and_debug_markers` |
| `RenderPassEncoder::set_pipeline` | `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw`, `render_pass_encoder_indexed_and_indirect_draws` |
| `RenderPassEncoder::record_validation_error` | `render_pass_encoder_lifecycle_and_debug_markers` |
| `RenderPassEncoder::set_bind_group` | `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw` |
| `RenderPassEncoder::set_vertex_buffer` | `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw` |
| `RenderPassEncoder::set_index_buffer` | `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw`, `render_pass_encoder_indexed_and_indirect_draws` |
| `RenderPassEncoder::draw` | `render_pass_encoder_set_pipeline_bind_group_buffers_and_draw` |
| `RenderPassEncoder::draw_indexed` | `render_pass_encoder_indexed_and_indirect_draws` |
| `RenderPassEncoder::draw_indirect` | `render_pass_encoder_indexed_and_indirect_draws` |
| `RenderPassEncoder::draw_indexed_indirect` | `render_pass_encoder_indexed_and_indirect_draws` |
| `RenderPassEncoder::set_viewport` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::set_scissor_rect` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::set_blend_constant` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::set_stencil_reference` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::execute_bundles` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::begin_occlusion_query` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `RenderPassEncoder::end_occlusion_query` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles` |
| `ComputePassEncoder::end` | `compute_pass_encoder_lifecycle_and_debug_markers`, `compute_pass_encoder_pipeline_bind_group_and_dispatch` |
| `ComputePassEncoder::insert_debug_marker` | `compute_pass_encoder_lifecycle_and_debug_markers` |
| `ComputePassEncoder::push_debug_group` | `compute_pass_encoder_lifecycle_and_debug_markers` |
| `ComputePassEncoder::pop_debug_group` | `compute_pass_encoder_lifecycle_and_debug_markers` |
| `ComputePassEncoder::set_pipeline` | `compute_pass_encoder_pipeline_bind_group_and_dispatch` |
| `ComputePassEncoder::record_validation_error` | `compute_pass_encoder_lifecycle_and_debug_markers` |
| `ComputePassEncoder::set_bind_group` | `compute_pass_encoder_pipeline_bind_group_and_dispatch` |
| `ComputePassEncoder::dispatch_workgroups` | `compute_pass_encoder_pipeline_bind_group_and_dispatch` |
| `ComputePassEncoder::dispatch_workgroups_indirect` | `compute_pass_encoder_pipeline_bind_group_and_dispatch` |
| `RenderBundleEncoder::new` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles`, `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws`, `render_bundle_encoder_indirect_draws` |
| `RenderBundleEncoder::finish` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles`, `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws`, `render_bundle_encoder_indirect_draws` |
| `RenderBundleEncoder::insert_debug_marker` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::push_debug_group` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::pop_debug_group` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::set_pipeline` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws`, `render_bundle_encoder_indirect_draws` |
| `RenderBundleEncoder::set_bind_group` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::set_vertex_buffer` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::set_index_buffer` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws`, `render_bundle_encoder_indirect_draws` |
| `RenderBundleEncoder::draw` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::draw_indexed` | `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws` |
| `RenderBundleEncoder::draw_indirect` | `render_bundle_encoder_indirect_draws` |
| `RenderBundleEncoder::draw_indexed_indirect` | `render_bundle_encoder_indirect_draws` |
| `RenderBundle::is_error` | `render_pass_encoder_state_setters_occlusion_query_and_execute_bundles`, `render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws`, `render_bundle_encoder_indirect_draws` |

## yawgpu-core/src/lib.rs - Pipeline / Shader (19 pub fn)

| pub fn | test name(s) |
|---|---|
| `ShaderModule::is_error` | `shader_module_accessors_pin_is_error_and_diagnostic`, `device_create_shader_module_accepts_minimal_compute_wgsl` |
| `ShaderModule::diagnostic` | `shader_module_accessors_pin_is_error_and_diagnostic`, `device_create_shader_module_accepts_minimal_compute_wgsl` |
| `BindGroupLayout::error` | `bind_group_layout_accessors_pin_entries_error_and_same` |
| `BindGroupLayout::entries` | `bind_group_layout_accessors_pin_entries_error_and_same`, `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `BindGroupLayout::is_error` | `bind_group_layout_accessors_pin_entries_error_and_same`, `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `BindGroupLayout::same` | `bind_group_layout_accessors_pin_entries_error_and_same` |
| `BindGroup::is_error` | `bind_group_accessors_pin_entries_and_is_error`, `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `BindGroup::entries` | `bind_group_accessors_pin_entries_and_is_error`, `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `PipelineLayout::is_error` | `pipeline_layout_accessors_pin_bind_group_layouts_and_is_error`, `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `PipelineLayout::bind_group_layouts` | `pipeline_layout_accessors_pin_bind_group_layouts_and_is_error`, `device_create_bind_group_layout_bind_group_and_pipeline_layout_empty` |
| `ComputePipeline::is_error` | `compute_pipeline_accessors_and_render_pipeline_accessors`, `device_create_compute_pipeline_happy_path_and_error_scope`, `device_create_compute_pipeline_without_error_dispatch_keeps_scope_empty` |
| `ComputePipeline::entry_name` | `compute_pipeline_accessors_and_render_pipeline_accessors`, `device_create_compute_pipeline_happy_path_and_error_scope` |
| `ComputePipeline::bind_group_layouts` | `compute_pipeline_accessors_and_render_pipeline_accessors` |
| `VertexFormat::from_raw` | `vertex_format_from_raw_pins_known_zero_and_unknown_values` |
| `VertexFormat::raw` | `vertex_format_from_raw_pins_known_zero_and_unknown_values` |
| `From<u32> for VertexFormat` | `vertex_format_from_raw_pins_known_zero_and_unknown_values` |
| `From<i32> for VertexFormat` | `vertex_format_from_raw_pins_known_zero_and_unknown_values` |
| `From<VertexFormat> for u32` | `vertex_format_from_raw_pins_known_zero_and_unknown_values` |
| `From<VertexFormat> for i32` | `vertex_format_from_raw_pins_known_zero_and_unknown_values` |
| `RenderPipeline::is_error` | `compute_pipeline_accessors_and_render_pipeline_accessors`, `device_create_render_pipeline_happy_path_and_error_scope`, `device_create_render_pipeline_without_error_dispatch_keeps_scope_empty` |
| `RenderPipeline::vertex_entry_name` | `compute_pipeline_accessors_and_render_pipeline_accessors`, `device_create_render_pipeline_happy_path_and_error_scope` |
| `RenderPipeline::fragment_entry_name` | `compute_pipeline_accessors_and_render_pipeline_accessors`, `device_create_render_pipeline_happy_path_and_error_scope` |
| `RenderPipeline::bind_group_layouts` | `compute_pipeline_accessors_and_render_pipeline_accessors` |

## yawgpu-core/src/lib.rs - Query / Error / Future (14 pub fn)

| pub fn | test name(s) |
|---|---|
| `QuerySet::kind` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `QuerySet::count` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy`, `device_create_query_set_validates_count_and_creates_happy_path` |
| `QuerySet::set_label` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `QuerySet::is_error` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy`, `device_create_query_set_validates_count_and_creates_happy_path` |
| `QuerySet::same` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `QuerySet::destroy` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `From<u32> for QueryType` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `From<i32> for QueryType` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `From<QueryType> for u32` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `From<QueryType> for i32` | `query_set_accessors_pin_kind_count_label_is_error_same_destroy` |
| `DeviceError::new` | `device_error_new_constructs_with_kind_and_message` |
| `FutureId::get` | `future_id_get_and_from_raw_round_trip`, `future_registry_process_events_respects_callback_mode` |
| `FutureId::from_raw` | `future_id_get_and_from_raw_round_trip` |
| `FutureRegistry::new` | `future_registry_process_events_respects_callback_mode` |
| `FutureRegistry::register` | `future_registry_process_events_respects_callback_mode` |
| `FutureRegistry::complete` | `future_registry_process_events_respects_callback_mode` |
| `FutureRegistry::process_events` | `future_registry_process_events_respects_callback_mode` |
| `FutureRegistry::wait_any` | `future_registry_process_events_respects_callback_mode` |

## yawgpu-hal/src/noop/mod.rs (14 pub fn)

| pub fn | test name(s) |
|---|---|
| `NoopInstance::new` | `noop_instance_new_constructs` |
| `NoopInstance::enumerate_adapters` | `noop_instance_enumerate_adapters_returns_synthetic_adapter` |
| `NoopAdapter::synthetic` | `noop_adapter_synthetic_exposes_documented_name` |
| `NoopAdapter::name` | `noop_adapter_name_returns_fixed_string` |
| `NoopAdapter::create_device` | `noop_adapter_create_device_returns_zero_allocation_device` |
| `NoopDevice::new` | `noop_device_new_starts_with_zero_allocations` |
| `NoopDevice::allocation_count` | `noop_device_allocation_count_tracks_created_resources` |
| `NoopDevice::queue` | `noop_device_queue_returns_same_reference` |
| `NoopDevice::create_buffer` | `noop_device_create_buffer_records_size_and_increments_allocation_count` |
| `NoopDevice::create_texture` | `noop_device_create_texture_increments_allocation_count` |
| `NoopDevice::create_sampler` | `noop_device_create_sampler_increments_allocation_count` |
| `NoopQueue::new` | `noop_queue_new_matches_default_smoke` |
| `NoopBuffer::size` | `noop_buffer_size_returns_created_size` |
| `NoopBuffer::mapped_ptr` | `noop_buffer_mapped_ptr_returns_none` |

## yawgpu-hal/src/lib.rs (25 pub fn)

Public API re-exported from `lib.rs` (paths unchanged); definitions live
in sibling modules -- see file map in HANDOFF P11.1 commit.

| pub fn | test name(s) |
|---|---|
| `HalInstance::new_noop` | `noop_creates_device_with_zero_allocations` (existing) |
| `HalInstance::enumerate_adapters` | `noop_creates_device_with_zero_allocations` (existing) |
| `HalInstance::create_surface_from_metal_layer` | `create_surface_from_metal_layer_noop_ignores_layer_pointer` |
| `HalAdapter::name` | `hal_adapter_name_noop_returns_fixed_string` |
| `HalAdapter::backend` | `hal_adapter_backend_noop_returns_noop` |
| `HalAdapter::create_device` | `noop_creates_device_with_zero_allocations` (existing) |
| `HalDevice::backend` | `hal_device_backend_noop_returns_noop` |
| `HalDevice::allocation_count` | `noop_creates_device_with_zero_allocations` (existing) |
| `HalDevice::queue` | `hal_device_queue_noop_returns_queue_that_submits_empty` |
| `HalDevice::create_buffer` | `hal_device_create_buffer_noop_records_requested_size` |
| `HalDevice::create_texture` | `hal_device_create_texture_noop_returns_texture_and_increments_allocations` |
| `HalDevice::create_sampler` | `hal_device_create_sampler_noop_returns_sampler_and_increments_allocations` |
| `HalDevice::create_compute_pipeline` | `hal_device_create_compute_pipeline_noop_accepts_empty_shader` |
| `HalDevice::create_render_pipeline` | `hal_device_create_render_pipeline_noop_accepts_empty_shader` |
| `HalSurfaceConfiguration::new` | `hal_surface_configuration_new_round_trips_fields` |
| `HalSurface::configure` | `hal_surface_configure_noop_returns_ok` |
| `HalSurface::unconfigure` | `hal_surface_unconfigure_noop_is_idempotent` |
| `HalSurface::acquire_next_texture` | `hal_surface_acquire_next_texture_noop_returns_acquire_failed` |
| `HalSurface::present` | `hal_surface_present_noop_returns_ok_without_acquire` |
| `HalQueue::submit_empty` | `hal_queue_submit_empty_noop_returns_ok` |
| `HalQueue::submit_copies` | `hal_queue_submit_copies_noop_accepts_empty_and_buffer_copy` |
| `HalBuffer::size` | `hal_buffer_size_noop_matches_creation_size` |
| `HalBuffer::write` | `hal_buffer_write_noop_accepts_empty_and_non_empty_data` |
| `HalBuffer::read` | `hal_buffer_read_noop_returns_zeroed_vector` |
| `HalBuffer::mapped_ptr` | `hal_buffer_mapped_ptr_noop_returns_none` |

## yawgpu-hal/src/metal/mod.rs (25 pub fn)

Public API re-exported from `metal/mod.rs` (paths unchanged); definitions
live in sibling modules -- see file map in HANDOFF P11.1 commit.

All tests are ignored real-backend tests gated by `#[cfg(feature = "metal")]`.

| pub fn | test name(s) |
|---|---|
| `MetalInstance::new` | `metal_instance_new_constructs` |
| `MetalInstance::enumerate_adapters` | `metal_instance_enumerate_adapters_returns_devices` |
| `MetalAdapter::new` | `metal_adapter_new_captures_device_name` |
| `MetalAdapter::name` | `metal_adapter_name_returns_non_empty_name` |
| `MetalAdapter::create_device` | `metal_adapter_create_device_returns_zero_allocation_device` |
| `MetalDevice::new` | `metal_device_new_starts_with_zero_allocations` |
| `MetalDevice::allocation_count` | `metal_device_allocation_count_tracks_created_resources` |
| `MetalDevice::queue` | `metal_device_queue_returns_same_reference` |
| `MetalDevice::create_buffer` | `metal_device_create_buffer_records_size_and_maps_memory` |
| `MetalDevice::create_texture` | `metal_device_create_texture_records_descriptor_shape` |
| `MetalDevice::create_sampler` | `metal_device_create_sampler_returns_sampler` |
| `MetalDevice::create_compute_pipeline` | `metal_device_create_compute_pipeline_accepts_msl` |
| `MetalDevice::create_render_pipeline` | `metal_device_create_render_pipeline_accepts_msl` |
| `MetalSurface::from_layer` | `metal_surface_from_layer_rejects_null_layer`, `metal_surface_from_layer_wraps_cametal_layer` |
| `MetalSurface::configure` | `metal_surface_configure_stores_configuration` |
| `MetalSurface::unconfigure` | `metal_surface_unconfigure_clears_configuration` |
| `MetalSurface::acquire_next_texture` | `metal_surface_acquire_next_texture_errors_when_unconfigured` |
| `MetalSurface::present` | `metal_surface_present_errors_without_acquired_drawable` |
| `MetalQueue::new` | `metal_queue_new_constructs_queue` |
| `MetalQueue::submit_empty` | `metal_queue_submit_empty_completes` |
| `MetalQueue::submit_copies` | `metal_queue_submit_copies_accepts_buffer_copy` |
| `MetalBuffer::size` | `metal_buffer_size_returns_created_size` |
| `MetalBuffer::write` | `metal_buffer_write_updates_mapped_memory` |
| `MetalBuffer::read` | `metal_buffer_read_returns_written_bytes` |
| `MetalBuffer::mapped_ptr` | `metal_buffer_mapped_ptr_returns_non_null_pointer` |

## yawgpu-hal/src/vulkan/mod.rs (22 pub fn)

Public API re-exported from `vulkan/mod.rs` (paths unchanged); definitions
live in sibling modules -- see file map in HANDOFF P11.1 commit.

All tests are ignored real-backend tests gated by `#[cfg(feature = "vulkan")]`.
Surface tests use null-surface/error-path coverage rather than adding a
CAMetalLayer dev-dependency; the valid-surface happy path remains covered by
Phase-9 e2e (`examples/surface_smoke`, `examples/triangle`,
`examples/hello_triangle` run with `YAWGPU_BACKEND=vulkan`).

| pub fn | test name(s) |
|---|---|
| `VulkanInstance::new` | `vulkan_instance_new_constructs` |
| `VulkanInstance::enumerate_adapters` | `vulkan_instance_enumerate_adapters_returns_devices` |
| `VulkanInstance::create_surface_from_metal_layer` | `vulkan_instance_create_surface_from_metal_layer_rejects_null_layer` |
| `VulkanAdapter::name` | `vulkan_adapter_name_returns_non_empty_name` |
| `VulkanAdapter::create_device` | `vulkan_adapter_create_device_returns_zero_allocation_device` |
| `VulkanDevice::allocation_count` | `vulkan_device_allocation_count_tracks_created_resources` |
| `VulkanDevice::queue` | `vulkan_device_queue_returns_same_reference` |
| `VulkanDevice::create_buffer` | `vulkan_device_create_buffer_records_size_and_maps_memory` |
| `VulkanDevice::create_texture` | `vulkan_device_create_texture_records_descriptor_shape` |
| `VulkanDevice::create_sampler` | `vulkan_device_create_sampler_returns_sampler` |
| `VulkanDevice::create_compute_pipeline` | `vulkan_device_create_compute_pipeline_accepts_spirv` |
| `VulkanDevice::create_render_pipeline` | `vulkan_device_create_render_pipeline_accepts_spirv_stages` |
| `VulkanSurface::configure` | `vulkan_surface_configure_errors_for_null_surface` |
| `VulkanSurface::unconfigure` | `vulkan_surface_unconfigure_is_idempotent` |
| `VulkanSurface::acquire_next_texture` | `vulkan_surface_acquire_next_texture_errors_when_unconfigured` |
| `VulkanSurface::present` | `vulkan_surface_present_errors_without_acquired_image` |
| `VulkanQueue::submit_empty` | `vulkan_queue_submit_empty_completes` |
| `VulkanQueue::submit_copies` | `vulkan_queue_submit_copies_accepts_buffer_copy` |
| `VulkanBuffer::size` | `vulkan_buffer_size_returns_created_size` |
| `VulkanBuffer::write` | `vulkan_buffer_write_updates_mapped_memory` |
| `VulkanBuffer::read` | `vulkan_buffer_read_returns_written_bytes` |
| `VulkanBuffer::mapped_ptr` | `vulkan_buffer_mapped_ptr_returns_non_null_pointer` |

## yawgpu/src/conv.rs (67 pub fn + 6 conversion impls)

| pub fn | test name(s) |
|---|---|
| `arc_to_handle` | `arc_to_handle_round_trips_with_clone_handle_refcount_math` |
| `release_handle` | `release_handle_drops_owned_reference_once`, `release_handle_null_panics_with_contract_message` |
| `add_ref_handle` | `add_ref_handle_increments_refcount_for_later_release`, `add_ref_handle_null_panics_with_contract_message` |
| `clone_handle` | `arc_to_handle_round_trips_with_clone_handle_refcount_math`, `clone_handle_leaves_original_handle_valid`, `clone_handle_null_panics_with_contract_message` |
| `borrow_handle` | `borrow_handle_returns_reference_without_consuming_arc`, `borrow_handle_null_panics_with_contract_message` |
| `string_view` | `string_view_round_trips_data_and_empty_slice` |
| `string_view_to_str` | `string_view_round_trips_data_and_empty_slice`, `string_view_to_str_handles_explicit_strlen_and_null_data` |
| `label_from_string_view` | `label_from_string_view_returns_owned_label_or_none` |
| `map_shader_module_descriptor` | `map_shader_module_descriptor_decodes_wgsl_source_and_missing_source_error` |
| `map_bind_group_layout_descriptor` | `map_bind_group_layout_descriptor_decodes_buffer_entry_and_null_entries_error` |
| `map_bind_group_entries` | `map_bind_group_entries_decodes_buffer_entry_and_null_entries_error` |
| `map_pipeline_layout_descriptor` | `map_pipeline_layout_descriptor_decodes_layouts_and_null_array_error` |
| `map_compute_pipeline_descriptor` | `map_compute_pipeline_descriptor_decodes_module_entry_layout_and_constants`, `map_compute_pipeline_descriptor_null_module_panics` |
| `map_render_pipeline_descriptor` | `map_render_pipeline_descriptor_decodes_vertex_fragment_and_error_path` |
| `map_feature` | `map_feature_round_trips_defined_and_other_variants` |
| `map_feature_to_native` | `map_feature_round_trips_defined_and_other_variants` |
| `map_query_set_descriptor` | `map_query_set_descriptor_decodes_type_count_label` |
| `native::WGPUQueryType -> core::QueryType` | `from_native_query_type_round_trips_known_and_unknown_variants` |
| `core::QueryType -> native::WGPUQueryType` | `from_native_query_type_round_trips_known_and_unknown_variants` |
| `map_query_type` | `map_query_type_round_trips_defined_and_unknown_variants` |
| `map_query_type_to_native` | `map_query_type_round_trips_defined_and_unknown_variants` |
| `map_feature_level` | `map_feature_level_maps_compatibility_and_default_core` |
| `DeviceLostCallbackInfo::has_callback` | `has_callback_detects_present_and_absent_device_lost_callbacks` |
| `map_device_lost_callback_info` | `map_device_lost_callback_info_round_trips_present_and_absent_callback` |
| `map_device_lost_reason` | `map_device_lost_reason_maps_every_core_variant` |
| `map_error_filter` | `map_error_filter_maps_known_values_and_rejects_unknown` |
| `map_error_type` | `map_error_type_maps_every_core_variant` |
| `map_pop_error_scope_status_error` | `map_pop_error_scope_status_error_returns_error` |
| `map_pop_error_scope_status_success` | `map_pop_error_scope_status_success_returns_success` |
| `map_buffer_usage` | `map_buffer_usage_round_trips_bitmask` |
| `map_buffer_usage_to_native` | `map_buffer_usage_round_trips_bitmask` |
| `map_buffer_map_state` | `map_buffer_map_state_maps_every_core_variant` |
| `map_map_async_status` | `map_map_async_status_maps_every_core_variant` |
| `map_queue_work_done_status` | `map_queue_work_done_status_maps_every_core_variant` |
| `map_compilation_info_request_status_success` | `map_compilation_info_request_status_success_returns_success` |
| `map_compilation_message_type_error` | `map_compilation_message_type_error_returns_error` |
| `map_map_mode` | `map_map_mode_accepts_single_modes_and_rejects_invalid_combinations` |
| `map_buffer_descriptor` | `map_buffer_descriptor_round_trips_fields` |
| `map_address_mode` | `map_address_mode_maps_known_values_and_rejects_unknown` |
| `map_filter_mode` | `map_filter_mode_maps_known_values_and_rejects_unknown` |
| `map_mipmap_filter_mode` | `map_mipmap_filter_mode_maps_known_values_and_rejects_unknown` |
| `map_compare_function` | `map_compare_function_maps_known_values_and_rejects_undefined` |
| `map_sampler_descriptor` | `map_sampler_descriptor_round_trips_fields_with_undefined_compare` |
| `map_texture_usage` | `map_texture_usage_round_trips_bitmask` |
| `map_texture_usage_to_native` | `map_texture_usage_round_trips_bitmask` |
| `map_texture_dimension` | `map_texture_dimension_round_trips_defined_variants` |
| `map_texture_dimension_to_native` | `map_texture_dimension_round_trips_defined_variants` |
| `native::WGPUTextureFormat -> core::TextureFormat` | `from_native_texture_format_round_trips_known_and_unknown_variants` |
| `core::TextureFormat -> native::WGPUTextureFormat` | `from_native_texture_format_round_trips_known_and_unknown_variants` |
| `map_texture_format` | `map_texture_format_round_trips_defined_and_unknown_raw_values` |
| `map_texture_format_to_native` | `map_texture_format_round_trips_defined_and_unknown_raw_values` |
| `native::WGPUVertexFormat -> core::VertexFormat` | `from_native_vertex_format_round_trips_known_and_unknown_variants` |
| `core::VertexFormat -> native::WGPUVertexFormat` | `from_native_vertex_format_round_trips_known_and_unknown_variants` |
| `map_vertex_format` | `from_native_vertex_format_round_trips_known_and_unknown_variants` |
| `map_vertex_format_to_native` | `from_native_vertex_format_round_trips_known_and_unknown_variants` |
| `map_extent_3d` | `map_extent_3d_round_trips_fields` |
| `map_origin_3d` | `map_origin_3d_round_trips_fields` |
| `map_texel_copy_buffer_layout` | `map_texel_copy_buffer_layout_round_trips_fields_and_undefined_strides` |
| `map_texel_copy_texture_info_parts` | `map_texel_copy_texture_info_parts_round_trips_fields` |
| `map_render_pass_descriptor` | `map_render_pass_descriptor_decodes_color_attachment_and_sparse_null_view` |
| `map_render_bundle_encoder_descriptor` | `map_render_bundle_encoder_descriptor_decodes_formats_and_null_format_array` |
| `map_query_index` | `map_query_index_maps_defined_values_and_undefined_to_none` |
| `map_load_op` | `map_load_op_maps_defined_values_and_undefined_fallback` |
| `map_store_op` | `map_store_op_maps_defined_values_and_undefined_fallback` |
| `map_color` | `map_color_round_trips_float_bits_including_nan` |
| `map_texture_descriptor` | `map_texture_descriptor_decodes_usage_format_dimension_size_and_view_formats` |
| `map_texture_view_dimension` | `map_texture_view_dimension_maps_known_values_and_rejects_unknown` |
| `map_texture_aspect` | `map_texture_aspect_maps_known_values_and_rejects_undefined` |
| `map_texture_view_descriptor` | `map_texture_view_descriptor_decodes_fields_and_none_defaults` |
| `map_features_to_native` | `map_features_to_native_allocates_feature_array_and_free_supported_features_releases_it` |
| `free_supported_features` | `map_features_to_native_allocates_feature_array_and_free_supported_features_releases_it`, `free_supported_features_accepts_null_feature_array` |
| `map_limits_to_native` | `map_limits_to_native_round_trips_through_map_limits` |
| `map_limits` | `map_limits_round_trips_every_field_from_native`, `map_limits_to_native_round_trips_through_map_limits` |

## yawgpu/src/lib.rs - Instance + Adapter (16 pub fn)

| pub fn | test name(s) |
|---|---|
| `wgpuCreateInstance` | `wgpuCreateInstance_noop_backend_and_null_descriptor_return_instances`, `request_noop_device_helper_returns_live_device` |
| `wgpuInstanceRelease` | `wgpuInstanceAddRef_and_wgpuInstanceRelease_balance_owned_refs`, `wgpuCreateInstance_noop_backend_and_null_descriptor_return_instances` |
| `wgpuInstanceAddRef` | `wgpuInstanceAddRef_and_wgpuInstanceRelease_balance_owned_refs` |
| `wgpuInstanceCreateSurface` | `wgpuInstanceCreateSurface_accepts_noop_metal_layer_source` |
| `wgpuInstanceRequestAdapter` | `wgpuInstanceRequestAdapter_process_events_returns_success_adapter`, `wgpuInstanceWaitAny_wait_any_only_request_adapter_fires_callback`, `request_noop_device_helper_returns_live_device` |
| `wgpuInstanceProcessEvents` | `wgpuInstanceRequestAdapter_process_events_returns_success_adapter`, `wgpuInstanceProcessEvents_without_registered_futures_is_noop`, `wgpuInstanceWaitAny_wait_any_only_request_adapter_fires_callback`, `wgpuAdapterRequestDevice_process_events_returns_success_device`, `request_noop_device_helper_returns_live_device` |
| `wgpuInstanceWaitAny` | `wgpuInstanceWaitAny_empty_list_returns_timed_out_and_null_list_errors`, `wgpuInstanceWaitAny_wait_any_only_request_adapter_fires_callback` |
| `wgpuAdapterRelease` | `wgpuAdapterAddRef_and_wgpuAdapterRelease_balance_owned_refs`, `wgpuInstanceRequestAdapter_process_events_returns_success_adapter` |
| `wgpuAdapterAddRef` | `wgpuAdapterAddRef_and_wgpuAdapterRelease_balance_owned_refs` |
| `wgpuAdapterGetLimits` | `wgpuAdapterGetLimits_populates_noop_defaults_and_rejects_null_out` |
| `wgpuAdapterGetFeatures` | `wgpuAdapterGetFeatures_populates_supported_features_and_free_members` |
| `wgpuAdapterHasFeature` | `wgpuAdapterHasFeature_reports_supported_and_unknown_features` |
| `wgpuAdapterGetInfo` | `wgpuAdapterGetInfo_populates_noop_info_and_free_members` |
| `wgpuAdapterInfoFreeMembers` | `wgpuAdapterGetInfo_populates_noop_info_and_free_members`, `wgpuAdapterInfoFreeMembers_accepts_empty_members` |
| `wgpuAdapterRequestDevice` | `wgpuAdapterRequestDevice_process_events_returns_success_device`, `request_noop_device_helper_returns_live_device` |
| `wgpuSupportedFeaturesFreeMembers` | `wgpuAdapterGetFeatures_populates_supported_features_and_free_members`, `wgpuSupportedFeaturesFreeMembers_accepts_empty_features` |

## yawgpu/src/lib.rs - Device + Queue (32 C FFI fn + 4 public helper fn)

| pub fn | test name(s) |
|---|---|
| `wgpuDeviceRelease` | `wgpuDeviceAddRef_and_wgpuDeviceRelease_balance_owned_refs`, `wgpuDeviceDestroy_and_wgpuDeviceGetLostFuture_complete_loss` |
| `wgpuDeviceAddRef` | `wgpuDeviceAddRef_and_wgpuDeviceRelease_balance_owned_refs` |
| `wgpuDeviceDestroy` | `wgpuDeviceDestroy_and_wgpuDeviceGetLostFuture_complete_loss` |
| `wgpuDeviceGetLostFuture` | `wgpuDeviceDestroy_and_wgpuDeviceGetLostFuture_complete_loss` |
| `WGPUDeviceImpl::set_uncaptured_error_callback` | `WGPUDeviceImpl_set_uncaptured_error_callback_records_callback_for_dispatch` |
| `WGPUDeviceImpl::dispatch_error` | `WGPUDeviceImpl_dispatch_error_routes_to_uncaptured_callback` |
| `testing_set_uncaptured_error_callback` | `testing_set_uncaptured_error_callback_installs_callback_for_dispatch` |
| `testing_dispatch_device_error` | `testing_dispatch_device_error_routes_to_uncaptured_callback` |
| `wgpuDevicePushErrorScope` | `wgpuDevicePushErrorScope_and_wgpuDevicePopErrorScope_capture_and_empty_stack`, `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuDevicePopErrorScope` | `wgpuDevicePushErrorScope_and_wgpuDevicePopErrorScope_capture_and_empty_stack`, `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuDeviceSetLabel` | `wgpuDeviceSetLabel_limits_features_and_has_feature_pin_noop_device` |
| `wgpuDeviceGetLimits` | `wgpuDeviceSetLabel_limits_features_and_has_feature_pin_noop_device` |
| `wgpuDeviceGetFeatures` | `wgpuDeviceSetLabel_limits_features_and_has_feature_pin_noop_device` |
| `wgpuDeviceHasFeature` | `wgpuDeviceSetLabel_limits_features_and_has_feature_pin_noop_device` |
| `wgpuDeviceCreateBuffer` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuDeviceCreateTexture` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuDeviceCreateSampler` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreateQuerySet` | `wgpuDevicePushErrorScope_and_wgpuDevicePopErrorScope_capture_and_empty_stack`, `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreateShaderModule` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuDeviceCreateComputePipelineAsync_and_render_async_fire_success_callbacks` |
| `wgpuDeviceCreateBindGroupLayout` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreateBindGroup` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreatePipelineLayout` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuDeviceCreateComputePipelineAsync_and_render_async_fire_success_callbacks` |
| `wgpuDeviceCreateComputePipeline` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreateRenderPipeline` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreateCommandEncoder` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors`, `wgpuQueueOnSubmittedWorkDone_and_wgpuQueueSubmit_cover_empty_and_command_buffer` |
| `wgpuDeviceCreateRenderBundleEncoder` | `wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors` |
| `wgpuDeviceCreateComputePipelineAsync` | `wgpuDeviceCreateComputePipelineAsync_and_render_async_fire_success_callbacks` |
| `wgpuDeviceCreateRenderPipelineAsync` | `wgpuDeviceCreateComputePipelineAsync_and_render_async_fire_success_callbacks` |
| `wgpuDeviceGetQueue` | `wgpuDeviceAddRef_and_wgpuDeviceRelease_balance_owned_refs`, `wgpuAdapterRequestDevice_process_events_returns_success_device`, `wgpuDeviceGetQueue_queue_add_ref_release_and_set_label_pin_identity`, `wgpuQueueOnSubmittedWorkDone_and_wgpuQueueSubmit_cover_empty_and_command_buffer`, `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuQueueRelease` | `wgpuDeviceGetQueue_queue_add_ref_release_and_set_label_pin_identity`, `wgpuQueueOnSubmittedWorkDone_and_wgpuQueueSubmit_cover_empty_and_command_buffer`, `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuQueueAddRef` | `wgpuDeviceGetQueue_queue_add_ref_release_and_set_label_pin_identity` |
| `wgpuQueueSetLabel` | `wgpuDeviceGetQueue_queue_add_ref_release_and_set_label_pin_identity` |
| `wgpuQueueOnSubmittedWorkDone` | `wgpuQueueOnSubmittedWorkDone_and_wgpuQueueSubmit_cover_empty_and_command_buffer` |
| `wgpuQueueSubmit` | `wgpuQueueOnSubmittedWorkDone_and_wgpuQueueSubmit_cover_empty_and_command_buffer` |
| `wgpuQueueWriteBuffer` | `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |
| `wgpuQueueWriteTexture` | `wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths` |

## yawgpu/src/lib.rs - Buffer / Texture / TextureView / Sampler (26 pub fn)

| pub fn | test name(s) |
|---|---|
| `wgpuBufferDestroy` | `wgpuBuffer_destroy_unmap_release_addref_lifecycle` |
| `wgpuBufferUnmap` | `wgpuBuffer_destroy_unmap_release_addref_lifecycle`, `wgpuBuffer_map_async_and_mapped_range_walk_state_machine` |
| `wgpuBufferMapAsync` | `wgpuBuffer_map_async_and_mapped_range_walk_state_machine` |
| `wgpuBufferGetMappedRange` | `wgpuBuffer_map_async_and_mapped_range_walk_state_machine` |
| `wgpuBufferGetConstMappedRange` | `wgpuBuffer_map_async_and_mapped_range_walk_state_machine` |
| `wgpuBufferGetMapState` | `wgpuBuffer_destroy_unmap_release_addref_lifecycle`, `wgpuBuffer_map_async_and_mapped_range_walk_state_machine` |
| `wgpuBufferGetSize` | `wgpuBuffer_size_and_usage_accessors_match_descriptor` |
| `wgpuBufferGetUsage` | `wgpuBuffer_size_and_usage_accessors_match_descriptor` |
| `wgpuBufferRelease` | `wgpuBuffer_destroy_unmap_release_addref_lifecycle`, `wgpuBuffer_map_async_and_mapped_range_walk_state_machine`, `wgpuBuffer_size_and_usage_accessors_match_descriptor` |
| `wgpuBufferAddRef` | `wgpuBuffer_destroy_unmap_release_addref_lifecycle` |
| `wgpuTextureDestroy` | `wgpuTexture_create_view_and_destroy_release_addref` |
| `wgpuTextureCreateView` | `wgpuTexture_create_view_and_destroy_release_addref` |
| `wgpuTextureGetFormat` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetDimension` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetWidth` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetHeight` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetDepthOrArrayLayers` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetMipLevelCount` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetSampleCount` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureGetUsage` | `wgpuTexture_accessors_match_descriptor` |
| `wgpuTextureRelease` | `wgpuTexture_accessors_match_descriptor`, `wgpuTexture_create_view_and_destroy_release_addref` |
| `wgpuTextureAddRef` | `wgpuTexture_create_view_and_destroy_release_addref` |
| `wgpuTextureViewRelease` | `wgpuTexture_create_view_and_destroy_release_addref` |
| `wgpuTextureViewAddRef` | `wgpuTexture_create_view_and_destroy_release_addref` |
| `wgpuSamplerRelease` | `wgpuSampler_release_and_addref_lifecycle` |
| `wgpuSamplerAddRef` | `wgpuSampler_release_and_addref_lifecycle` |

## yawgpu/src/lib.rs - Encoder + Pass (49 pub fn)

| pub fn | test name(s) |
|---|---|
| `wgpuCommandEncoderBeginRenderPass` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuCommandEncoderBeginComputePass` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuCommandEncoderFinish` | `wgpuCommandEncoder_lifecycle_release_addref_finish`, `wgpuCommandEncoder_debug_markers_insert_push_pop`, `wgpuCommandEncoder_buffer_copies_and_clear_and_write`, `wgpuCommandEncoder_texture_copies_walk`, `wgpuCommandEncoder_query_and_timestamps`, `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles`, `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuCommandEncoderInsertDebugMarker` | `wgpuCommandEncoder_debug_markers_insert_push_pop` |
| `wgpuCommandEncoderPushDebugGroup` | `wgpuCommandEncoder_debug_markers_insert_push_pop` |
| `wgpuCommandEncoderPopDebugGroup` | `wgpuCommandEncoder_debug_markers_insert_push_pop` |
| `wgpuCommandEncoderCopyBufferToBuffer` | `wgpuCommandEncoder_buffer_copies_and_clear_and_write` |
| `wgpuCommandEncoderClearBuffer` | `wgpuCommandEncoder_buffer_copies_and_clear_and_write` |
| `wgpuCommandEncoderWriteBuffer` | `wgpuCommandEncoder_buffer_copies_and_clear_and_write` |
| `wgpuCommandEncoderWriteTimestamp` | `wgpuCommandEncoder_query_and_timestamps` |
| `wgpuCommandEncoderResolveQuerySet` | `wgpuCommandEncoder_query_and_timestamps` |
| `wgpuCommandEncoderCopyBufferToTexture` | `wgpuCommandEncoder_texture_copies_walk` |
| `wgpuCommandEncoderCopyTextureToBuffer` | `wgpuCommandEncoder_texture_copies_walk` |
| `wgpuCommandEncoderCopyTextureToTexture` | `wgpuCommandEncoder_texture_copies_walk` |
| `wgpuCommandEncoderRelease` | `wgpuCommandEncoder_lifecycle_release_addref_finish`, `wgpuCommandEncoder_debug_markers_insert_push_pop`, `wgpuCommandEncoder_buffer_copies_and_clear_and_write`, `wgpuCommandEncoder_texture_copies_walk`, `wgpuCommandEncoder_query_and_timestamps`, `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles`, `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuCommandEncoderAddRef` | `wgpuCommandEncoder_lifecycle_release_addref_finish` |
| `wgpuCommandBufferRelease` | `wgpuCommandEncoder_lifecycle_release_addref_finish`, `wgpuCommandEncoder_debug_markers_insert_push_pop`, `wgpuCommandEncoder_buffer_copies_and_clear_and_write`, `wgpuCommandEncoder_texture_copies_walk`, `wgpuCommandEncoder_query_and_timestamps`, `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles`, `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuCommandBufferAddRef` | `wgpuCommandEncoder_lifecycle_release_addref_finish` |
| `wgpuRenderPassEncoderEnd` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderBeginOcclusionQuery` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderEndOcclusionQuery` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderInsertDebugMarker` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuRenderPassEncoderPushDebugGroup` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuRenderPassEncoderPopDebugGroup` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuRenderPassEncoderSetPipeline` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderSetBindGroup` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderSetVertexBuffer` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderSetIndexBuffer` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderDraw` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderDrawIndexed` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderDrawIndirect` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderDrawIndexedIndirect` | `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPassEncoderSetViewport` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderSetScissorRect` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderSetBlendConstant` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderSetStencilReference` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderExecuteBundles` | `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderRelease` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuRenderPassEncoderAddRef` | `wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuComputePassEncoderEnd` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePassEncoderInsertDebugMarker` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuComputePassEncoderPushDebugGroup` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuComputePassEncoderPopDebugGroup` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers` |
| `wgpuComputePassEncoderSetPipeline` | `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePassEncoderSetBindGroup` | `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePassEncoderDispatchWorkgroups` | `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePassEncoderDispatchWorkgroupsIndirect` | `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePassEncoderRelease` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePassEncoderAddRef` | `wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers` |

## yawgpu/src/lib.rs - Bundle + Pipeline + Shader + BindGroup + Query + Surface (46 C FFI fn + 1 public helper fn)

| pub fn | test name(s) |
|---|---|
| `wgpuRenderBundleEncoderSetPipeline` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderSetBindGroup` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderSetVertexBuffer` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderSetIndexBuffer` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderDraw` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderDrawIndexed` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderDrawIndirect` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderDrawIndexedIndirect` | `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderInsertDebugMarker` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref` |
| `wgpuRenderBundleEncoderPushDebugGroup` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref` |
| `wgpuRenderBundleEncoderPopDebugGroup` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref` |
| `wgpuRenderBundleEncoderFinish` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref`, `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderRelease` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref`, `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleEncoderAddRef` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref` |
| `wgpuRenderBundleRelease` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref`, `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderBundleAddRef` | `wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref` |
| `wgpuComputePipelineGetBindGroupLayout` | `wgpuComputePipeline_get_bind_group_layout_release_addref` |
| `wgpuComputePipelineRelease` | `wgpuComputePipeline_get_bind_group_layout_release_addref`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuComputePipelineAddRef` | `wgpuComputePipeline_get_bind_group_layout_release_addref` |
| `wgpuRenderPipelineGetBindGroupLayout` | `wgpuRenderPipeline_get_bind_group_layout_release_addref` |
| `wgpuRenderPipelineRelease` | `wgpuRenderPipeline_get_bind_group_layout_release_addref`, `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws` |
| `wgpuRenderPipelineAddRef` | `wgpuRenderPipeline_get_bind_group_layout_release_addref` |
| `wgpuBindGroupLayoutRelease` | `wgpuComputePipeline_get_bind_group_layout_release_addref`, `wgpuRenderPipeline_get_bind_group_layout_release_addref`, `wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref`, `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuBindGroupLayoutAddRef` | `wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref` |
| `testing_bind_group_layout_entry_visibility` | `testing_bind_group_layout_entry_visibility_returns_entry_visibility_and_none` |
| `wgpuBindGroupRelease` | `wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref`, `wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws`, `wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch` |
| `wgpuBindGroupAddRef` | `wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref` |
| `wgpuPipelineLayoutRelease` | `wgpuComputePipeline_get_bind_group_layout_release_addref`, `wgpuRenderPipeline_get_bind_group_layout_release_addref`, `wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref` |
| `wgpuPipelineLayoutAddRef` | `wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref` |
| `wgpuShaderModuleGetCompilationInfo` | `wgpuShaderModule_get_compilation_info_and_release_addref` |
| `wgpuShaderModuleRelease` | `wgpuShaderModule_get_compilation_info_and_release_addref` |
| `wgpuShaderModuleAddRef` | `wgpuShaderModule_get_compilation_info_and_release_addref` |
| `wgpuQuerySetDestroy` | `wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref` |
| `wgpuQuerySetGetType` | `wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref` |
| `wgpuQuerySetGetCount` | `wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref` |
| `wgpuQuerySetSetLabel` | `wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref` |
| `wgpuQuerySetRelease` | `wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref`, `wgpuCommandEncoder_query_and_timestamps`, `wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles` |
| `wgpuQuerySetAddRef` | `wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref` |
| `wgpuSurfaceGetCapabilities` | `wgpuSurface_get_capabilities_capabilities_free_members_and_lifecycle` |
| `wgpuSurfaceCapabilitiesFreeMembers` | `wgpuSurface_get_capabilities_capabilities_free_members_and_lifecycle` |
| `wgpuSurfaceConfigure` | `wgpuSurface_configure_unconfigure_get_current_texture_present_noop_contract` |
| `wgpuSurfaceUnconfigure` | `wgpuSurface_configure_unconfigure_get_current_texture_present_noop_contract` |
| `wgpuSurfaceGetCurrentTexture` | `wgpuSurface_configure_unconfigure_get_current_texture_present_noop_contract` |
| `wgpuSurfacePresent` | `wgpuSurface_configure_unconfigure_get_current_texture_present_noop_contract` |
| `wgpuSurfaceSetLabel` | `wgpuSurface_get_capabilities_capabilities_free_members_and_lifecycle` |
| `wgpuSurfaceRelease` | `wgpuSurface_get_capabilities_capabilities_free_members_and_lifecycle`, `wgpuSurface_configure_unconfigure_get_current_texture_present_noop_contract` |
| `wgpuSurfaceAddRef` | `wgpuSurface_get_capabilities_capabilities_free_members_and_lifecycle` |

## Phase 10 yawgpu-core coverage summary

Total kept pub fn (post-P10.3a audit plus Windows bindgen fix): 184
- Instance / Adapter / Device / Queue (51): ☑ P10.3b
- Buffer / Texture / Sampler (40):           ☑ P10.3c
- Encoder / Pass / Bundle (59):              ☑ P10.3d
- Pipeline / Shader (19):                    ☑ P10.3e + Windows bindgen fix
- Query / Error / Future (14):               ☑ P10.3f
- Surface / utilities (0 — absorbed):        ☑ via P10.3b

P10.3 complete: all 184 yawgpu-core kept-pub fn have at least one direct
inline `#[cfg(test)]` unit test.

## yawgpu (C FFI) - public constants

Literal-value public constants are covered by their consuming tests and helpers:

| pub const | consuming test / helper |
|---|---|
| `conv::WGPU_STRLEN` | `string_view_to_str_handles_explicit_strlen_and_null_data` |
| `WGPU_YAWGPU_INSTANCE_BACKEND_NOOP` | `make_noop_instance` helper used by Noop C FFI unit tests |
| `WGPU_YAWGPU_INSTANCE_BACKEND_METAL` | `e2e_metal_basic` backend-selection descriptor |
| `WGPU_YAWGPU_INSTANCE_BACKEND_VULKAN` | `e2e_vulkan_basic` backend-selection descriptor |
| `WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT` | `make_noop_instance` helper used by Noop C FFI unit tests |

## Phase 10 yawgpu (C FFI) coverage summary

Total pub unsafe extern "C" fn: 169
- Instance + Adapter (16):                         ☑ P10.4a
- Device + Queue (32):                             ☑ P10.4b
- Buffer / Texture / TextureView / Sampler (26):   ☑ P10.4c
- Encoder + Pass (49):                             ☑ P10.4d
- Bundle + Pipeline + Shader + BindGroup +
  Query + Surface (46):                            ☑ P10.4e

P10.4 complete: all 169 yawgpu C FFI fns have at least one
direct inline #[cfg(test)] unit test.
