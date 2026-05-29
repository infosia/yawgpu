use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{create_bind_group_layout, create_pipeline_layout, empty_string_view};

pub unsafe fn expect_bind_group_layout(
    test: &ValidationTest,
    success: bool,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    if success {
        test.clear_errors();
        let layout = unsafe { create_bind_group_layout(test.device(), entries) };
        assert!(!layout.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        layout
    } else {
        let mut layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                layout = unsafe { create_bind_group_layout(test.device(), entries) };
            },
            None,
        );
        assert!(!layout.is_null());
        layout
    }
}

pub unsafe fn expect_pipeline_layout(
    test: &ValidationTest,
    success: bool,
    layouts: &[native::WGPUBindGroupLayout],
    immediate_size: u32,
) -> native::WGPUPipelineLayout {
    if success {
        test.clear_errors();
        let layout = unsafe { create_pipeline_layout(test.device(), layouts, immediate_size) };
        assert!(!layout.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        layout
    } else {
        let mut layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                layout = unsafe { create_pipeline_layout(test.device(), layouts, immediate_size) };
            },
            None,
        );
        assert!(!layout.is_null());
        layout
    }
}

pub unsafe fn create_pipeline_layout_raw(
    device: native::WGPUDevice,
    count: usize,
    layouts: *const native::WGPUBindGroupLayout,
    immediate_size: u32,
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: count,
        bindGroupLayouts: layouts,
        immediateSize: immediate_size,
    };
    unsafe { yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor) }
}

pub fn default_entry(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: 0,
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
    }
}

pub fn buffer_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUBufferBindingType,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding, visibility);
    entry.buffer.type_ = ty;
    entry
}

pub fn dynamic_buffer_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUBufferBindingType,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = buffer_layout(binding, visibility, ty);
    entry.buffer.hasDynamicOffset = 1;
    entry
}

pub fn sampler_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding, visibility);
    entry.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
    entry
}

pub fn texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    sample_type: native::WGPUTextureSampleType,
    view_dimension: native::WGPUTextureViewDimension,
    multisampled: bool,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding, visibility);
    entry.texture.sampleType = sample_type;
    entry.texture.viewDimension = view_dimension;
    entry.texture.multisampled = u32::from(multisampled);
    entry
}

pub fn storage_texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    access: native::WGPUStorageTextureAccess,
    format: native::WGPUTextureFormat,
    view_dimension: native::WGPUTextureViewDimension,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding, visibility);
    entry.storageTexture.access = access;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = view_dimension;
    entry
}

pub fn release_bind_group_layouts(layouts: &[native::WGPUBindGroupLayout]) {
    unsafe {
        for layout in layouts {
            yawgpu::wgpuBindGroupLayoutRelease(*layout);
        }
    }
}

pub fn release_pipeline_layout(layout: native::WGPUPipelineLayout) {
    unsafe {
        yawgpu::wgpuPipelineLayoutRelease(layout);
    }
}
