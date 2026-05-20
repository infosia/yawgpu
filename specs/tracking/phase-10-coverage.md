# Phase 10 — Public API Unit Test Coverage

P10.3a audit: see `specs/tracking/phase-10-audit.md`.

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

All tests are ignored real-backend tests gated by `#[cfg(feature = "vulkan")]`.
Surface tests use null-surface/error-path coverage rather than adding a CAMetalLayer
dev-dependency. **Follow-up:** `VulkanSurface::configure`'s happy path requires a
valid `vk::SurfaceKHR`; a null-surface unit test crashed in
`vkGetPhysicalDeviceSurfaceCapabilitiesKHR` (MoltenVK does not gracefully reject
`VK_NULL_HANDLE`), so direct unit coverage is deferred — the happy path is
covered by Phase-9 e2e (`examples/surface_smoke`, `examples/triangle`,
`examples/hello_triangle` run with `YAWGPU_BACKEND=vulkan`). A defensive
null-handle pre-check in `VulkanSurface::configure` would close this gap; tracked
as a Phase-10 follow-up.

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
| `VulkanSurface::configure` | (deferred — e2e-covered; see follow-up note above) |
| `VulkanSurface::unconfigure` | `vulkan_surface_unconfigure_is_idempotent` |
| `VulkanSurface::acquire_next_texture` | `vulkan_surface_acquire_next_texture_errors_when_unconfigured` |
| `VulkanSurface::present` | `vulkan_surface_present_errors_without_acquired_image` |
| `VulkanQueue::submit_empty` | `vulkan_queue_submit_empty_completes` |
| `VulkanQueue::submit_copies` | `vulkan_queue_submit_copies_accepts_buffer_copy` |
| `VulkanBuffer::size` | `vulkan_buffer_size_returns_created_size` |
| `VulkanBuffer::write` | `vulkan_buffer_write_updates_mapped_memory` |
| `VulkanBuffer::read` | `vulkan_buffer_read_returns_written_bytes` |
| `VulkanBuffer::mapped_ptr` | `vulkan_buffer_mapped_ptr_returns_non_null_pointer` |

## yawgpu/src/conv.rs (66 pub fn)

| pub fn | test name(s) |
|---|---|
| `arc_to_handle` | `arc_to_handle_round_trips_with_clone_handle_refcount_math` |
| `release_handle` | `release_handle_drops_owned_reference_once` |
| `add_ref_handle` | `add_ref_handle_increments_refcount_for_later_release` |
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
| `map_texture_format` | `map_texture_format_round_trips_defined_and_unknown_raw_values` |
| `map_texture_format_to_native` | `map_texture_format_round_trips_defined_and_unknown_raw_values` |
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
