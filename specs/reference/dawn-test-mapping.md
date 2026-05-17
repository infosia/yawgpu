# Dawn validation test → yawgpu port mapping

Source: `dawn/src/dawn/tests/unittests/validation/` (55 files).
Each Dawn `XxxValidationTests.cpp` is ported to
`yawgpu/tests/<area>_validation.rs`. `ValidationTest.{h,cpp}` becomes the
`yawgpu-test` crate (`ValidationTest` fixture + `assert_device_error!`).

Status legend: ☐ not started · ◐ partial · ☑ ported & green

| Dawn file | Phase | yawgpu test file | Status |
|---|---|---|---|
| `ValidationTest` (base) | 0 | `yawgpu-test` crate | ☑ |
| `DeviceValidationTests` | 1 | `limits_validation.rs` + `features_validation.rs` + `device_lost_validation.rs` | ☑ (R1–R8,R10–R14; R9 N/A; R15/R16 are MultipleDevice) |
| `UnsafeAPIValidationTests` | 1→4/8 | `unsafe_api_validation.rs` | ☐ Defer (R18–R20→P4, R21→P8) |
| `MultipleDeviceTests` | 1→5/6 | `multiple_device_validation.rs` | ☐ Defer (R16→P5, R15→P6) |
| `LabelTests` | 1 | `label_validation.rs` | ◐ (R17a Device/Queue done; R17b→per-object phase) |
| `BufferValidationTests` | 2 | `buffer_creation_validation.rs` + `buffer_map_validation.rs` + `buffer_mapped_range_validation.rs` | ☑ (B1–B38; B39–B41 submit→P6) |
| `MinimumBufferSizeValidationTests` | 2→4 | `minimum_buffer_size_validation.rs` | ☐ Defer→P4 (B58, needs BindGroupLayout) |
| `QueueSubmitValidationTests` | 2→6 | `queue_buffer_validation.rs` | ◐ (arg-only; command validation→P6) |
| `QueueWriteBufferValidationTests` | 2 | `queue_buffer_validation.rs` | ☑ (B42–B49) |
| `QueueOnSubmittedWorkDoneValidationTests` | 2 | `queue_buffer_validation.rs` | ☑ (B50–B52) |
| `WriteBufferTests` | 2→6 | `write_buffer_validation.rs` | ☐ Defer→P6 (B53–B57, CommandEncoder.WriteBuffer) |
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
