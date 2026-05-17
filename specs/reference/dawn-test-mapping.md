# Dawn validation test → yawgpu port mapping

Source: `dawn/src/dawn/tests/unittests/validation/` (55 files).
Each Dawn `XxxValidationTests.cpp` is ported to
`yawgpu/tests/<area>_validation.rs`. `ValidationTest.{h,cpp}` becomes the
`yawgpu-test` crate (`ValidationTest` fixture + `assert_device_error!`).

Status legend: ☐ not started · ◐ partial · ☑ ported & green

| Dawn file | Phase | yawgpu test file | Status |
|---|---|---|---|
| `ValidationTest` (base) | 0 | `yawgpu-test` crate | ☑ |
| `DeviceValidationTests` | 1 | `device_validation.rs` | ☐ |
| `UnsafeAPIValidationTests` | 1 | `unsafe_api_validation.rs` | ☐ |
| `MultipleDeviceTests` | 1 | `multiple_device_validation.rs` | ☐ |
| `LabelTests` | 1 | `label_validation.rs` | ☐ |
| `BufferValidationTests` | 2 | `buffer_validation.rs` | ☐ |
| `MinimumBufferSizeValidationTests` | 2 | `minimum_buffer_size_validation.rs` | ☐ |
| `QueueSubmitValidationTests` | 2 | `queue_submit_validation.rs` | ☐ |
| `QueueWriteBufferValidationTests` | 2 | `queue_write_buffer_validation.rs` | ☐ |
| `QueueOnSubmittedWorkDoneValidationTests` | 2 | `queue_on_submitted_work_done_validation.rs` | ☐ |
| `WriteBufferTests` | 2 | `write_buffer_validation.rs` | ☐ |
| `TextureValidationTests` | 3 | `texture_validation.rs` | ☐ |
| `TextureViewValidationTests` | 3 | `texture_view_validation.rs` | ☐ |
| `TextureSubresourceTests` | 3 | `texture_subresource_validation.rs` | ☐ |
| `SamplerValidationTests` | 3 | `sampler_validation.rs` | ☐ |
| `QueueWriteTextureValidationTests` | 3 | `queue_write_texture_validation.rs` | ☐ |
| `StorageTextureValidationTests` | 3 | `storage_texture_validation.rs` | ☐ |
| `ShaderModuleValidationTests` | 4 | `shader_module_validation.rs` | ☐ |
| `WGSLFeatureValidationTests` | 4 | `wgsl_feature_validation.rs` | ☐ |
| `BindGroupValidationTests` | 4 | `bind_group_validation.rs` | ☐ |
| `GetBindGroupLayoutValidationTests` | 4 | `get_bind_group_layout_validation.rs` | ☐ |
| `MinimumBufferSizeValidationTests` (bind) | 4 | (shared with P2) | ☐ |
| `OverridableConstantsValidationTests` | 4 | `overridable_constants_validation.rs` | ☐ |
| `ImmediateDataTests` | 4 | `immediate_data_validation.rs` | ☐ |
| `RenderPipelineValidationTests` | 5 | `render_pipeline_validation.rs` | ☐ |
| `ComputeValidationTests` | 5 | `compute_validation.rs` | ☐ |
| `PipelineAndPassCompatibilityTests` | 5 | `pipeline_pass_compat_validation.rs` | ☐ |
| `VertexStateValidationTests` | 5 | `vertex_state_validation.rs` | ☐ |
| `ObjectCachingTests` | 5 | `object_caching_validation.rs` | ☐ |
| `CommandBufferValidationTests` | 6 | `command_buffer_validation.rs` | ☐ |
| `CopyCommandsValidationTests` | 6 | `copy_commands_validation.rs` | ☐ |
| `RenderPassDescriptorValidationTests` | 6 | `render_pass_descriptor_validation.rs` | ☐ |
| `RenderBundleValidationTests` | 6 | `render_bundle_validation.rs` | ☐ |
| `DynamicStateCommandValidationTests` | 6 | `dynamic_state_command_validation.rs` | ☐ |
| `IndexBufferValidationTests` | 6 | `index_buffer_validation.rs` | ☐ |
| `VertexBufferValidationTests` | 6 | `vertex_buffer_validation.rs` | ☐ |
| `DrawIndirectValidationTests` | 6 | `draw_indirect_validation.rs` | ☐ |
| `DrawVertexAndIndexBufferOOBValidationTests` | 6 | `draw_oob_validation.rs` | ☐ |
| `ComputeIndirectValidationTests` | 6 | `compute_indirect_validation.rs` | ☐ |
| `MultiDrawIndirectValidationTests` | 6 | `multi_draw_indirect_validation.rs` | ☐ |
| `DebugMarkerValidationTests` | 6 | `debug_marker_validation.rs` | ☐ |
| `ResourceUsageTrackingTests` | 6 | `resource_usage_tracking_validation.rs` | ☐ |
| `WritableBufferBindingAliasingValidationTests` | 6 | `writable_buffer_aliasing_validation.rs` | ☐ |
| `WritableTextureBindingAliasingValidationTests` | 6 | `writable_texture_aliasing_validation.rs` | ☐ |
| `QuerySetValidationTests` / `QueryValidationTests` | 8 | `query_validation.rs` | ☐ |
| `ErrorScopeValidationTests` | 8 | `error_scope_validation.rs` | ☐ |
| `DeviceLostValidationTests` (in Device) | 8 | `device_lost_validation.rs` | ☐ |
| `ToggleValidationTests` | 8 | `toggle_validation.rs` | ☐ |

## Deferred / extension surface (revisit after Phase 8)

`CompatValidationTests`, `ExternalTextureTests`, `PixelLocalStorageTests`,
`ResourceTableValidationTests`, `TexelBufferValidationTests`,
`YCbCrValidationTests`, `InternalUsageValidationTests`,
`CopyTextureForBrowserTests` — extension/advanced features; port once core
WebGPU is conformant.

## Porting notes

- `TEST_F(XxxValidationTest, Case)` → `#[test] fn case()` using the
  `ValidationTest` fixture from `yawgpu-test`.
- `ASSERT_DEVICE_ERROR(stmt)` → `assert_device_error!(stmt)`.
- `device.CreateBuffer(&desc)` (C++ webgpu_cpp wrapper) → direct
  `wgpuDeviceCreateBuffer(device, &desc)` C calls (the boundary under test).
- Async cases use `yawgpu-test` future/poll helpers; Noop completes on poll.
- Skip backend-parametrized macros (`DAWN_INSTANTIATE_TEST`); yawgpu runs
  Noop only for validation.
