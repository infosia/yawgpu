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
| `UnsafeAPIValidationTests` | 1→4/8 | `unsafe_api_validation.rs` | ☑ (R18–R21 rejected-direction P8.4; AllowUnsafeAPIs non-canonical N/A) |
| `MultipleDeviceTests` | 1→8 | `multiple_device_validation.rs` | ☑ (MD1/MD2 R15/R16 P8.5) |
| `LabelTests` | 1 | `label_validation.rs` | ◐ (R17a Device/Queue done; R17b→per-object phase) |
| `BufferValidationTests` | 2 | `buffer_creation_validation.rs` + `buffer_map_validation.rs` + `buffer_mapped_range_validation.rs` | ☑ (B1–B38; B39–B41 submit→P6) |
| `MinimumBufferSizeValidationTests` | 2→4 | `minimum_buffer_size_validation.rs` | ☐ Defer→P4 (B58, needs BindGroupLayout) |
| `QueueSubmitValidationTests` | 2→6 | `queue_buffer_validation.rs` + `queue_submit_validation.rs` | ☑ (arg P2; C80–C82 submit P6.9) |
| `QueueWriteBufferValidationTests` | 2 | `queue_buffer_validation.rs` | ☑ (B42–B49) |
| `QueueOnSubmittedWorkDoneValidationTests` | 2 | `queue_buffer_validation.rs` | ☑ (B50–B52) |
| `WriteBufferTests` | 2→6 | `command_buffer_copy_validation.rs` | ☑ (B53–B57 CommandEncoder.WriteBuffer, P6.2) |
| `TextureValidationTests` | 3 | `texture_creation_validation.rs` + `texture_format_validation.rs` | ☑ (T1–T25,T57–T65) |
| `TextureViewValidationTests` | 3 | `texture_view_validation.rs` | ☑ (T26–T33) |
| `TextureSubresourceTests` | 3→6 | `resource_usage_tracking_validation.rs` | ☑ (T54–T56 attachment+sampled, C78 P6.9) |
| `SamplerValidationTests` | 3 | `sampler_validation.rs` | ☑ (T34–T39) |
| `QueueWriteTextureValidationTests` | 3 | `queue_write_texture_validation.rs` | ☑ (T40–T51) |
| `StorageTextureValidationTests` | 3→5 | `texture_format_validation.rs` | ◐ (T52/T53 creation rules; shader-driven storage access→P5) |
| `ShaderModuleValidationTests` | 4 | `shader_module_validation.rs` | ☑ (S1–S7,S9,S11; S8→P5, S10 N/A) |
| `WGSLFeatureValidationTests` | 4 | `shader_module_validation.rs` | ◐ (S11 rejected-direction; AllowUnsafeAPIs non-canonical divergence) |
| `BindGroupValidationTests` | 4 | `bind_group_layout_validation.rs` + `bind_group_validation.rs` | ☑ (S12–S33,S35-BG; S34 N/A) |
| `GetBindGroupLayoutValidationTests` | 4→5 | `get_bind_group_layout_validation.rs` | ☑ (P38–P42; draw-time P41 cross-pipeline→P6) |
| `MinimumBufferSizeValidationTests` (bind) | 4→5 | `bind_group_validation.rs` + pipeline S35 | ☑ (S35 BG-part + pipeline layout/shader compat done) |
| `OverridableConstantsValidationTests` | 4→5 | `compute_pipeline_validation.rs` + render | ☑ (P5/P6/P37 via pipeline constants) |
| `ImmediateDataTests` | 4 | `pipeline_layout_validation.rs` | ◐ (S37 PipelineLayout immediateSize; pipeline use→P5) |
| `RenderPipelineValidationTests` | 5 | `render_pipeline_validation.rs` | ☑ (P7–P37) |
| `ComputeValidationTests` | 5 | `compute_pipeline_validation.rs` | ☑ (P1–P6) |
| `PipelineAndPassCompatibilityTests` | 5→6 | `pipeline_pass_compat_validation.rs` + `pass_state_validation.rs` | ☑ (pipeline-create P5; render-pass compat P6.5) |
| `VertexStateValidationTests` | 5 | `vertex_state_validation.rs` | ☑ (P10–P17) |
| `ObjectCachingTests` | 5 | `object_caching_validation.rs` | ☑ (P43–P50) |
| `CommandBufferValidationTests` | 6 | `command_encoder_lifecycle_validation.rs` | ☑ (C1–C5,C36,C63,C85,C86 P6.1) |
| `CopyCommandsValidationTests` | 6 | `command_buffer_copy_validation.rs` + `command_texture_copy_validation.rs` | ☑ (C6–C22,C79,C83,C84 P6.2/P6.3) |
| `RenderPassDescriptorValidationTests` | 6 | `render_pass_descriptor_validation.rs` | ☑ (C23–C33 P6.4; C34/C35 P8.2) |
| `RenderBundleValidationTests` | 6 | `render_bundle_validation.rs` | ☑ (C65–C74 P6.7) |
| `DynamicStateCommandValidationTests` | 6 | `pass_state_validation.rs` | ☑ (C56–C59 P6.5; consolidated) |
| `IndexBufferValidationTests` | 6 | `pass_state_validation.rs` | ☑ (C43–C46 P6.6; consolidated) |
| `VertexBufferValidationTests` | 6 | `pass_state_validation.rs` | ☑ (C47–C49 P6.6; consolidated) |
| `DrawIndirectValidationTests` | 6 | `pass_state_validation.rs` | ☑ (C53–C55 P6.6; consolidated) |
| `DrawVertexAndIndexBufferOOBValidationTests` | 6 | `pass_state_validation.rs` | ☑ (C50–C52 P6.6; consolidated) |
| `ComputeIndirectValidationTests` | 6 | `pass_state_validation.rs` | ☑ (C55 dispatch-indirect P6.6) |
| `MultiDrawIndirectValidationTests` | 6 | — | ✗ N/A (extension; out of scope) |
| `DebugMarkerValidationTests` | 6 | `debug_marker_validation.rs` | ☑ (C60–C64 P6.8) |
| `ResourceUsageTrackingTests` | 6 | `resource_usage_tracking_validation.rs` | ☑ (C76/C78 P6.9; bundle-usage Defer) |
| `WritableBufferBindingAliasingValidationTests` | 6 | `resource_usage_tracking_validation.rs` | ☑ (C75 P6.9; consolidated) |
| `WritableTextureBindingAliasingValidationTests` | 6 | `resource_usage_tracking_validation.rs` | ☑ (C77 P6.9; consolidated) |
| `QuerySetValidationTests` / `QueryValidationTests` | 8 | `query_validation.rs` | ☑ (QS1–QS4 P8.1; QC1–QC5 query-in-commands P8.2) |
| `ErrorScopeValidationTests` | 8 | `error_scope_validation.rs` | ☑ (ES1–ES5 P8.0) |
| `DeviceLostValidationTests` (in Device) | 8 | `device_lost_validation.rs` | ☑ (DL1–DL4 P8.3; Phase-1 stub completed) |
| `ToggleValidationTests` | 8 | — | ✗ N/A (Dawn toggle API absent from webgpu-headers; P8.4) |

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
