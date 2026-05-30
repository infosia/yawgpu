//! Shared Noop adaptation for CTS `capability_checks/limits`.
//!
//! CTS runs each limit through a browser adapter matrix where default and adapter
//! maxima can differ. yawgpu's Noop adapter exposes one effective limit set, and
//! current device creation may not lower all effective limits to requested
//! values. These tests therefore query `wgpuDeviceGetLimits` and run each
//! creator at the effective limit and just over it.

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

#[derive(Default)]
struct RequestDeviceResult {
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: String,
}

#[test]
fn over_maximum_limit_device_request_is_rejected() {
    let test = ValidationTest::new();
    unsafe {
        let supported = get_adapter_limits(test.adapter());
        let mut required = undefined_limits();
        required.maxBindGroups = supported.maxBindGroups + 1;

        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Error);
        assert!(result.device.is_null());
        assert!(!result.message.is_empty());
    }
}

#[test]
fn under_minimum_alignment_device_request_is_rejected() {
    let test = ValidationTest::new();
    unsafe {
        let supported = get_adapter_limits(test.adapter());
        let mut required = undefined_limits();
        required.minUniformBufferOffsetAlignment = supported.minUniformBufferOffsetAlignment / 2;

        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Error);
        assert!(result.device.is_null());
        assert!(!result.message.is_empty());
    }
}

#[test]
fn requested_lower_limits_do_not_change_effective_limits() {
    unsafe { assert_required_limits_are_not_lowered_to_requested_values() };
}

pub unsafe fn assert_max_bind_groups_create_pipeline_layout_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxBindGroups;
    unsafe { assert_pipeline_layout_bind_group_count(&test, limit, limit + 1) };
}

pub unsafe fn assert_max_bindings_per_bind_group_create_bind_group_layout_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxBindingsPerBindGroup;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit)
                .map(no_visibility_uniform_layout)
                .collect::<Vec<_>>(),
            &(0..=limit)
                .map(no_visibility_uniform_layout)
                .collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_buffer_size_create_buffer_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxBufferSize;
    unsafe {
        assert_buffer_size(&test, limit, limit + 1);
    }
}

pub unsafe fn assert_max_color_attachments_render_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxColorAttachments;
    unsafe { assert_render_pipeline_color_targets(&test, limit, limit + 1) };
}

pub unsafe fn assert_max_color_attachments_render_pass_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxColorAttachments;
    unsafe { assert_render_pass_color_attachments(&test, limit, limit + 1) };
}

pub unsafe fn assert_max_compute_workgroup_size_x_create_compute_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxComputeWorkgroupSizeX;
    unsafe {
        assert_compute_pipeline(
            &test,
            &format!("@compute @workgroup_size({limit}, 1, 1) fn main() {{}}"),
            &format!(
                "@compute @workgroup_size({}, 1, 1) fn main() {{}}",
                limit + 1
            ),
        );
    }
}

pub unsafe fn assert_max_compute_workgroup_size_y_create_compute_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxComputeWorkgroupSizeY;
    unsafe {
        assert_compute_pipeline(
            &test,
            &format!("@compute @workgroup_size(1, {limit}, 1) fn main() {{}}"),
            &format!(
                "@compute @workgroup_size(1, {}, 1) fn main() {{}}",
                limit + 1
            ),
        );
    }
}

pub unsafe fn assert_max_compute_workgroup_size_z_create_compute_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxComputeWorkgroupSizeZ;
    unsafe {
        assert_compute_pipeline(
            &test,
            &format!("@compute @workgroup_size(1, 1, {limit}) fn main() {{}}"),
            &format!(
                "@compute @workgroup_size(1, 1, {}) fn main() {{}}",
                limit + 1
            ),
        );
    }
}

pub unsafe fn assert_max_compute_invocations_create_compute_pipeline_at_over() {
    let test = ValidationTest::new();
    let limits = device_limits(test.device());
    let at = limits.maxComputeInvocationsPerWorkgroup;
    unsafe {
        assert_compute_pipeline(
            &test,
            &format!("@compute @workgroup_size({at}, 1, 1) fn main() {{}}"),
            &format!("@compute @workgroup_size({}, 1, 1) fn main() {{}}", at + 1),
        );
    }
}

pub unsafe fn assert_max_compute_workgroup_storage_create_compute_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxComputeWorkgroupStorageSize / 4;
    unsafe {
        assert_compute_pipeline(
            &test,
            &format!(
                "var<workgroup> scratch: array<u32, {limit}>;
                 @compute @workgroup_size(1) fn main() {{ scratch[0] = 1u; }}"
            ),
            &format!(
                "var<workgroup> scratch: array<u32, {}>;
                 @compute @workgroup_size(1) fn main() {{ scratch[0] = 1u; }}",
                limit + 1
            ),
        );
    }
}

pub unsafe fn assert_max_compute_workgroups_dispatch_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxComputeWorkgroupsPerDimension;
    unsafe {
        assert_compute_dispatch(&test, [limit, 1, 1], [limit + 1, 1, 1]);
    }
}

pub unsafe fn assert_max_dynamic_uniform_bgl_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxDynamicUniformBuffersPerPipelineLayout;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit).map(dynamic_uniform_layout).collect::<Vec<_>>(),
            &(0..=limit).map(dynamic_uniform_layout).collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_dynamic_storage_bgl_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxDynamicStorageBuffersPerPipelineLayout;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit).map(dynamic_storage_layout).collect::<Vec<_>>(),
            &(0..=limit).map(dynamic_storage_layout).collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_sampled_textures_bgl_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxSampledTexturesPerShaderStage;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit).map(texture_layout).collect::<Vec<_>>(),
            &(0..=limit).map(texture_layout).collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_samplers_bgl_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxSamplersPerShaderStage;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit).map(sampler_layout).collect::<Vec<_>>(),
            &(0..=limit).map(sampler_layout).collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_uniform_buffers_bgl_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxUniformBuffersPerShaderStage;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit).map(uniform_layout).collect::<Vec<_>>(),
            &(0..=limit).map(uniform_layout).collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_storage_buffers_bgl_at_over(visibility: native::WGPUShaderStage) {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxStorageBuffersPerShaderStage;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit)
                .map(|binding| storage_buffer_layout(binding, visibility))
                .collect::<Vec<_>>(),
            &(0..=limit)
                .map(|binding| storage_buffer_layout(binding, visibility))
                .collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_storage_textures_bgl_at_over(visibility: native::WGPUShaderStage) {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxStorageTexturesPerShaderStage;
    unsafe {
        assert_bind_group_layout_entries(
            &test,
            &(0..limit)
                .map(|binding| storage_texture_layout(binding, visibility))
                .collect::<Vec<_>>(),
            &(0..=limit)
                .map(|binding| storage_texture_layout(binding, visibility))
                .collect::<Vec<_>>(),
        );
    }
}

pub unsafe fn assert_max_texture_dimension_1d_create_texture_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxTextureDimension1D;
    unsafe {
        assert_texture_size(
            &test,
            native::WGPUTextureDimension_1D,
            extent(limit, 1, 1),
            extent(limit + 1, 1, 1),
        );
    }
}

pub unsafe fn assert_max_texture_dimension_2d_create_texture_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxTextureDimension2D;
    unsafe {
        assert_texture_size(
            &test,
            native::WGPUTextureDimension_2D,
            extent(limit, limit, 1),
            extent(limit + 1, 1, 1),
        );
    }
}

pub unsafe fn assert_max_texture_dimension_3d_create_texture_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxTextureDimension3D;
    unsafe {
        assert_texture_size(
            &test,
            native::WGPUTextureDimension_3D,
            extent(limit, limit, limit),
            extent(limit + 1, 1, 1),
        );
    }
}

pub unsafe fn assert_max_texture_array_layers_create_texture_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxTextureArrayLayers;
    unsafe {
        assert_texture_size(
            &test,
            native::WGPUTextureDimension_2D,
            extent(1, 1, limit),
            extent(1, 1, limit + 1),
        );
    }
}

pub unsafe fn assert_max_vertex_buffers_create_render_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxVertexBuffers;
    let at = vec![empty_vertex_buffer(); limit as usize];
    let over = vec![empty_vertex_buffer(); limit as usize + 1];
    unsafe { assert_render_pipeline_vertex_buffers(&test, vertex_no_input(), &at, &over) };
}

pub unsafe fn assert_max_vertex_attributes_create_render_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxVertexAttributes;
    let at_attrs = (0..limit)
        .map(|location| vertex_attribute(native::WGPUVertexFormat_Float32, 0, location))
        .collect::<Vec<_>>();
    let over_attrs = (0..=limit)
        .map(|location| vertex_attribute(native::WGPUVertexFormat_Float32, 0, location))
        .collect::<Vec<_>>();
    let at = [vertex_buffer(4, &at_attrs)];
    let over = [vertex_buffer(4, &over_attrs)];
    unsafe { assert_render_pipeline_vertex_buffers(&test, vertex_no_input(), &at, &over) };
}

pub unsafe fn assert_max_vertex_buffer_array_stride_create_render_pipeline_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxVertexBufferArrayStride;
    let attr = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
    let at = [vertex_buffer(u64::from(limit), &attr)];
    let over = [vertex_buffer(u64::from(limit + 4), &attr)];
    unsafe { assert_render_pipeline_vertex_buffers(&test, vertex_f32(), &at, &over) };
}

pub unsafe fn assert_uniform_buffer_binding_size_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxUniformBufferBindingSize;
    unsafe { assert_buffer_binding_size(&test, native::WGPUBufferBindingType_Uniform, limit) };
}

pub unsafe fn assert_storage_buffer_binding_size_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).maxStorageBufferBindingSize;
    unsafe { assert_buffer_binding_size(&test, native::WGPUBufferBindingType_Storage, limit) };
}

pub unsafe fn assert_min_uniform_buffer_offset_alignment_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).minUniformBufferOffsetAlignment;
    unsafe { assert_buffer_offset_alignment(&test, native::WGPUBufferBindingType_Uniform, limit) };
}

pub unsafe fn assert_min_storage_buffer_offset_alignment_at_over() {
    let test = ValidationTest::new();
    let limit = device_limits(test.device()).minStorageBufferOffsetAlignment;
    unsafe { assert_buffer_offset_alignment(&test, native::WGPUBufferBindingType_Storage, limit) };
}

pub unsafe fn assert_required_limits_are_not_lowered_to_requested_values() {
    let default_test = ValidationTest::new();
    let default_limits = device_limits(default_test.device());
    let mut requested = undefined_limits();
    requested.maxBindGroups = default_limits.maxBindGroups.saturating_sub(1).max(1);
    let lowered_test = ValidationTest::with_limits(requested);
    let effective = device_limits(lowered_test.device());
    assert_eq!(effective.maxBindGroups, default_limits.maxBindGroups);
}

unsafe fn assert_pipeline_layout_bind_group_count(test: &ValidationTest, at: u32, over: u32) {
    let at_layouts = unsafe { create_empty_layouts(test.device(), at) };
    let over_layouts = unsafe { create_empty_layouts(test.device(), over) };
    test.expect_no_validation_error(|| unsafe {
        let layout = create_pipeline_layout(test.device(), &at_layouts);
        assert!(!layout.is_null());
        yawgpu::wgpuPipelineLayoutRelease(layout);
    });
    assert_device_error!({
        let layout = unsafe { create_pipeline_layout(test.device(), &over_layouts) };
        assert!(!layout.is_null());
        unsafe { yawgpu::wgpuPipelineLayoutRelease(layout) };
    });
    unsafe {
        release_layouts(&at_layouts);
        release_layouts(&over_layouts);
    }
}

unsafe fn assert_bind_group_layout_entries(
    test: &ValidationTest,
    at: &[native::WGPUBindGroupLayoutEntry],
    over: &[native::WGPUBindGroupLayoutEntry],
) {
    test.expect_no_validation_error(|| unsafe {
        let layout = create_bind_group_layout(test.device(), at);
        assert!(!layout.is_null());
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    });
    assert_device_error!({
        let layout = unsafe { create_bind_group_layout(test.device(), over) };
        assert!(!layout.is_null());
        unsafe { yawgpu::wgpuBindGroupLayoutRelease(layout) };
    });
}

unsafe fn assert_buffer_size(test: &ValidationTest, at: u64, over: u64) {
    test.expect_no_validation_error(|| unsafe {
        let buffer = create_buffer(test.device(), at, native::WGPUBufferUsage_CopySrc);
        yawgpu::wgpuBufferRelease(buffer);
    });
    assert_device_error!({
        let buffer = unsafe { create_buffer(test.device(), over, native::WGPUBufferUsage_CopySrc) };
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    });
}

unsafe fn assert_render_pipeline_color_targets(test: &ValidationTest, at: u32, over: u32) {
    let at_targets =
        vec![color_target_with_write_mask(native::WGPUColorWriteMask_All); at as usize];
    let over_targets =
        vec![color_target_with_write_mask(native::WGPUColorWriteMask_All); over as usize];
    let at_fragment = fragment_outputs(at);
    let over_fragment = fragment_outputs(over);
    test.expect_no_validation_error(|| unsafe {
        let pipeline = create_render_pipeline_with_fragment(
            test.device(),
            vertex_no_input(),
            &[],
            &at_targets,
            &at_fragment,
        );
        assert!(!pipeline.is_null());
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    });
    assert_device_error!({
        let pipeline = unsafe {
            create_render_pipeline_with_fragment(
                test.device(),
                vertex_no_input(),
                &[],
                &over_targets,
                &over_fragment,
            )
        };
        assert!(!pipeline.is_null());
        unsafe { yawgpu::wgpuRenderPipelineRelease(pipeline) };
    });
}

unsafe fn assert_render_pass_color_attachments(test: &ValidationTest, at: u32, over: u32) {
    let target = unsafe { create_render_target(test.device()) };
    let at_attachments = (0..at)
        .map(|_| color_attachment(target.view))
        .collect::<Vec<_>>();
    let over_attachments = (0..over)
        .map(|_| color_attachment(target.view))
        .collect::<Vec<_>>();
    unsafe {
        expect_render_pass(test, &at_attachments, true);
        expect_render_pass(test, &over_attachments, false);
        release_render_target(target);
    }
}

unsafe fn assert_compute_pipeline(test: &ValidationTest, at_source: &str, over_source: &str) {
    test.expect_no_validation_error(|| unsafe {
        let pipeline = create_compute_pipeline(test.device(), at_source);
        assert!(!pipeline.is_null());
        yawgpu::wgpuComputePipelineRelease(pipeline);
    });
    assert_device_error!({
        let pipeline = unsafe { create_compute_pipeline(test.device(), over_source) };
        assert!(!pipeline.is_null());
        unsafe { yawgpu::wgpuComputePipelineRelease(pipeline) };
    });
}

unsafe fn assert_compute_dispatch(test: &ValidationTest, at: [u32; 3], over: [u32; 3]) {
    unsafe {
        expect_compute_dispatch(test, at, true);
        expect_compute_dispatch(test, over, false);
    }
}

unsafe fn assert_texture_size(
    test: &ValidationTest,
    dimension: native::WGPUTextureDimension,
    at: native::WGPUExtent3D,
    over: native::WGPUExtent3D,
) {
    test.expect_no_validation_error(|| unsafe {
        let texture = create_texture(test.device(), dimension, at);
        yawgpu::wgpuTextureRelease(texture);
    });
    assert_device_error!({
        let texture = unsafe { create_texture(test.device(), dimension, over) };
        unsafe { yawgpu::wgpuTextureRelease(texture) };
    });
}

unsafe fn assert_render_pipeline_vertex_buffers(
    test: &ValidationTest,
    vertex_source: &str,
    at: &[native::WGPUVertexBufferLayout],
    over: &[native::WGPUVertexBufferLayout],
) {
    test.expect_no_validation_error(|| unsafe {
        let pipeline = create_render_pipeline(test.device(), vertex_source, at, &[color_target()]);
        assert!(!pipeline.is_null());
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    });
    assert_device_error!({
        let pipeline = unsafe {
            create_render_pipeline(test.device(), vertex_source, over, &[color_target()])
        };
        assert!(!pipeline.is_null());
        unsafe { yawgpu::wgpuRenderPipelineRelease(pipeline) };
    });
}

unsafe fn assert_buffer_binding_size(
    test: &ValidationTest,
    binding_type: native::WGPUBufferBindingType,
    limit: u64,
) {
    let usage = if binding_type == native::WGPUBufferBindingType_Uniform {
        native::WGPUBufferUsage_Uniform
    } else {
        native::WGPUBufferUsage_Storage
    };
    let layout = buffer_layout(0, binding_type, native::WGPUShaderStage_Compute, false, 0);
    let bgl = unsafe { create_bind_group_layout(test.device(), &[layout]) };
    let at_buffer = unsafe { create_buffer(test.device(), limit, usage) };
    let over_buffer = unsafe { create_buffer(test.device(), limit + 4, usage) };
    test.expect_no_validation_error(|| unsafe {
        let group = create_bind_group(test.device(), bgl, at_buffer, 0, limit);
        yawgpu::wgpuBindGroupRelease(group);
    });
    assert_device_error!({
        let group = unsafe { create_bind_group(test.device(), bgl, over_buffer, 0, limit + 4) };
        unsafe { yawgpu::wgpuBindGroupRelease(group) };
    });
    unsafe {
        yawgpu::wgpuBufferRelease(over_buffer);
        yawgpu::wgpuBufferRelease(at_buffer);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

unsafe fn assert_buffer_offset_alignment(
    test: &ValidationTest,
    binding_type: native::WGPUBufferBindingType,
    alignment: u32,
) {
    let usage = if binding_type == native::WGPUBufferBindingType_Uniform {
        native::WGPUBufferUsage_Uniform
    } else {
        native::WGPUBufferUsage_Storage
    };
    let layout = buffer_layout(0, binding_type, native::WGPUShaderStage_Compute, false, 0);
    let bgl = unsafe { create_bind_group_layout(test.device(), &[layout]) };
    let buffer = unsafe { create_buffer(test.device(), u64::from(alignment) + 256, usage) };
    test.expect_no_validation_error(|| unsafe {
        let group = create_bind_group(test.device(), bgl, buffer, u64::from(alignment), 4);
        yawgpu::wgpuBindGroupRelease(group);
    });
    assert_device_error!({
        let group = unsafe { create_bind_group(test.device(), bgl, buffer, 1, 4) };
        unsafe { yawgpu::wgpuBindGroupRelease(group) };
    });
    unsafe {
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

unsafe fn create_empty_layouts(
    device: native::WGPUDevice,
    count: u32,
) -> Vec<native::WGPUBindGroupLayout> {
    (0..count)
        .map(|_| unsafe { create_bind_group_layout(device, &[]) })
        .collect()
}

unsafe fn release_layouts(layouts: &[native::WGPUBindGroupLayout]) {
    for layout in layouts {
        unsafe { yawgpu::wgpuBindGroupLayoutRelease(*layout) };
    }
}

unsafe fn create_bind_group_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    unsafe { yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor) }
}

unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    layouts: &[native::WGPUBindGroupLayout],
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: layouts.len(),
        bindGroupLayouts: layouts.as_ptr(),
        immediateSize: 0,
    };
    unsafe { yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor) }
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) };
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
        offset,
        size,
        sampler: std::ptr::null_mut(),
        textureView: std::ptr::null_mut(),
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    unsafe { yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor) }
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    dimension: native::WGPUTextureDimension,
    size: native::WGPUExtent3D,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding,
        dimension,
        size,
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    texture
}

unsafe fn create_render_target(device: native::WGPUDevice) -> RenderTarget {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    RenderTarget { texture, view }
}

unsafe fn release_render_target(target: RenderTarget) {
    unsafe {
        yawgpu::wgpuTextureViewRelease(target.view);
        yawgpu::wgpuTextureRelease(target.texture);
    }
}

unsafe fn expect_render_pass(
    test: &ValidationTest,
    attachments: &[native::WGPURenderPassColorAttachment],
    success: bool,
) {
    let encoder =
        unsafe { yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null()) };
    assert!(!encoder.is_null());
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor) };
    assert!(!pass.is_null());
    unsafe { yawgpu::wgpuRenderPassEncoderEnd(pass) };
    if success {
        test.expect_no_validation_error(|| unsafe {
            let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());
            yawgpu::wgpuCommandBufferRelease(command_buffer);
        });
    } else {
        assert_device_error!({
            let command_buffer =
                unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
            assert!(!command_buffer.is_null());
            unsafe { yawgpu::wgpuCommandBufferRelease(command_buffer) };
        });
    }
    unsafe {
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn expect_compute_dispatch(test: &ValidationTest, workgroups: [u32; 3], success: bool) {
    let encoder =
        unsafe { yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null()) };
    assert!(!encoder.is_null());
    let descriptor = native::WGPUComputePassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        timestampWrites: std::ptr::null(),
    };
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &descriptor) };
    assert!(!pass.is_null());
    let pipeline = unsafe {
        create_compute_pipeline(test.device(), "@compute @workgroup_size(1) fn main() {}")
    };
    unsafe {
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(
            pass,
            workgroups[0],
            workgroups[1],
            workgroups[2],
        );
        yawgpu::wgpuComputePassEncoderEnd(pass);
    }
    if success {
        test.expect_no_validation_error(|| unsafe {
            let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());
            yawgpu::wgpuCommandBufferRelease(command_buffer);
        });
    } else {
        assert_device_error!({
            let command_buffer =
                unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
            assert!(!command_buffer.is_null());
            unsafe { yawgpu::wgpuCommandBufferRelease(command_buffer) };
        });
    }
    unsafe {
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    source: &str,
) -> native::WGPUComputePipeline {
    let module = unsafe { create_wgsl_module(device, source) };
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor) };
    unsafe { yawgpu::wgpuShaderModuleRelease(module) };
    pipeline
}

unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    vertex_source: &str,
    buffers: &[native::WGPUVertexBufferLayout],
    targets: &[native::WGPUColorTargetState],
) -> native::WGPURenderPipeline {
    unsafe {
        create_render_pipeline_with_fragment(
            device,
            vertex_source,
            buffers,
            targets,
            fragment_single(),
        )
    }
}

unsafe fn create_render_pipeline_with_fragment(
    device: native::WGPUDevice,
    vertex_source: &str,
    buffers: &[native::WGPUVertexBufferLayout],
    targets: &[native::WGPUColorTargetState],
    fragment_source: &str,
) -> native::WGPURenderPipeline {
    let vertex_module = unsafe { create_wgsl_module(device, vertex_source) };
    let fragment_module = unsafe { create_wgsl_module(device, fragment_source) };
    let fragment_state = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: targets.len(),
        targets: targets.as_ptr(),
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: buffers.len(),
            buffers: buffers.as_ptr(),
        },
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment_state,
    };
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor) };
    unsafe {
        yawgpu::wgpuShaderModuleRelease(fragment_module);
        yawgpu::wgpuShaderModuleRelease(vertex_module);
    }
    pipeline
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    unsafe { yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor) }
}

fn buffer_layout(
    binding: u32,
    binding_type: native::WGPUBufferBindingType,
    visibility: native::WGPUShaderStage,
    has_dynamic_offset: bool,
    min_binding_size: u64,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = empty_layout(binding, visibility);
    entry.buffer.type_ = binding_type;
    entry.buffer.hasDynamicOffset = has_dynamic_offset.into();
    entry.buffer.minBindingSize = min_binding_size;
    entry
}

fn uniform_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    buffer_layout(
        binding,
        native::WGPUBufferBindingType_Uniform,
        native::WGPUShaderStage_Compute,
        false,
        0,
    )
}

fn dynamic_uniform_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    buffer_layout(
        binding,
        native::WGPUBufferBindingType_Uniform,
        native::WGPUShaderStage_Compute,
        true,
        0,
    )
}

fn no_visibility_uniform_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    buffer_layout(
        binding,
        native::WGPUBufferBindingType_Uniform,
        native::WGPUShaderStage_None,
        false,
        0,
    )
}

fn dynamic_storage_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    buffer_layout(
        binding,
        native::WGPUBufferBindingType_Storage,
        native::WGPUShaderStage_Compute,
        true,
        0,
    )
}

fn storage_buffer_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let ty = if visibility & native::WGPUShaderStage_Vertex != 0 {
        native::WGPUBufferBindingType_ReadOnlyStorage
    } else {
        native::WGPUBufferBindingType_Storage
    };
    buffer_layout(binding, ty, visibility, false, 0)
}

fn sampler_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = empty_layout(binding, native::WGPUShaderStage_Fragment);
    entry.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
    entry
}

fn texture_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = empty_layout(binding, native::WGPUShaderStage_Fragment);
    entry.texture.sampleType = native::WGPUTextureSampleType_Float;
    entry.texture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry.texture.multisampled = 0;
    entry
}

fn storage_texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = empty_layout(binding, visibility);
    entry.storageTexture.access = if visibility & native::WGPUShaderStage_Vertex != 0 {
        native::WGPUStorageTextureAccess_ReadOnly
    } else {
        native::WGPUStorageTextureAccess_WriteOnly
    };
    entry.storageTexture.format = native::WGPUTextureFormat_RGBA8Unorm;
    entry.storageTexture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn empty_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: false.into(),
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
        bindingArraySize: 0,
    }
}

fn vertex_buffer(
    array_stride: u64,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Vertex,
        arrayStride: array_stride,
        attributeCount: attributes.len(),
        attributes: attributes.as_ptr(),
    }
}

fn empty_vertex_buffer() -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Undefined,
        arrayStride: 0,
        attributeCount: 0,
        attributes: std::ptr::null(),
    }
}

fn vertex_attribute(
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
) -> native::WGPUVertexAttribute {
    native::WGPUVertexAttribute {
        nextInChain: std::ptr::null_mut(),
        format,
        offset,
        shaderLocation: shader_location,
    }
}

fn color_target() -> native::WGPUColorTargetState {
    color_target_with_write_mask(native::WGPUColorWriteMask_None)
}

fn color_target_with_write_mask(
    write_mask: native::WGPUColorWriteMask,
) -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: write_mask,
    }
}

fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Load,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    }
}

fn vertex_no_input() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn vertex_f32() -> &'static str {
    "@vertex fn vs(@location(0) a: f32) -> @builtin(position) vec4f { return vec4f(a); }"
}

fn fragment_single() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(); }"
}

fn fragment_outputs(count: u32) -> String {
    let fields = (0..count)
        .map(|location| format!("@location({location}) c{location}: vec4f,"))
        .collect::<String>();
    let values = (0..count).map(|_| "vec4f(),").collect::<String>();
    format!(
        "struct Out {{ {fields} }}
         @fragment fn fs() -> Out {{ return Out({values}); }}"
    )
}

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth_or_array_layers,
    }
}

pub fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(device, &mut limits),
            native::WGPUStatus_Success
        );
        limits
    }
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    required_limits: Option<&native::WGPULimits>,
) -> RequestDeviceResult {
    let mut result = RequestDeviceResult::default();
    let mut descriptor: native::WGPUDeviceDescriptor = unsafe { std::mem::zeroed() };
    if let Some(required_limits) = required_limits {
        descriptor.requiredLimits = required_limits;
    }

    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut result as *mut RequestDeviceResult).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let descriptor_ptr = if required_limits.is_some() {
        &descriptor
    } else {
        std::ptr::null()
    };
    let future =
        unsafe { yawgpu::wgpuAdapterRequestDevice(adapter, descriptor_ptr, callback_info) };
    unsafe { wait(instance, future) };
    result
}

unsafe fn get_adapter_limits(adapter: native::WGPUAdapter) -> native::WGPULimits {
    let mut limits = undefined_limits();
    let status = unsafe { yawgpu::wgpuAdapterGetLimits(adapter, &mut limits) };
    assert_eq!(status, native::WGPUStatus_Success);
    limits
}

fn undefined_limits() -> native::WGPULimits {
    native::WGPULimits {
        nextInChain: std::ptr::null_mut(),
        maxTextureDimension1D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureDimension2D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureDimension3D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureArrayLayers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindGroups: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindGroupsPlusVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindingsPerBindGroup: native::WGPU_LIMIT_U32_UNDEFINED,
        maxDynamicUniformBuffersPerPipelineLayout: native::WGPU_LIMIT_U32_UNDEFINED,
        maxDynamicStorageBuffersPerPipelineLayout: native::WGPU_LIMIT_U32_UNDEFINED,
        maxSampledTexturesPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxSamplersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxStorageBuffersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxStorageTexturesPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxUniformBuffersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxUniformBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED,
        maxStorageBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED,
        minUniformBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        minStorageBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBufferSize: native::WGPU_LIMIT_U64_UNDEFINED,
        maxVertexAttributes: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBufferArrayStride: native::WGPU_LIMIT_U32_UNDEFINED,
        maxInterStageShaderVariables: native::WGPU_LIMIT_U32_UNDEFINED,
        maxColorAttachments: native::WGPU_LIMIT_U32_UNDEFINED,
        maxColorAttachmentBytesPerSample: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupStorageSize: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeInvocationsPerWorkgroup: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeX: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeY: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeZ: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupsPerDimension: native::WGPU_LIMIT_U32_UNDEFINED,
        maxImmediateSize: native::WGPU_LIMIT_U32_UNDEFINED,
    }
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let result = unsafe { &mut *(userdata1 as *mut RequestDeviceResult) };
    result.status = status;
    result.device = device;
    result.message = unsafe { string_view_to_string(message) };
}

unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    let bytes = if value.length == native::WGPU_STRLEN {
        unsafe { std::ffi::CStr::from_ptr(value.data).to_bytes() }
    } else {
        unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), value.length) }
    };
    String::from_utf8_lossy(bytes).into_owned()
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

#[derive(Clone, Copy)]
struct RenderTarget {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}
