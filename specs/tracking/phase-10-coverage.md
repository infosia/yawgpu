# Phase 10 — Public API Unit Test Coverage

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
