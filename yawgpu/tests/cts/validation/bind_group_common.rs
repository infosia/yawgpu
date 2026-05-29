use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{create_bind_group_layout, create_pipeline_layout, empty_string_view};

pub unsafe fn expect_bind_group(
    test: &ValidationTest,
    success: bool,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    if success {
        test.clear_errors();
        let bind_group = unsafe { create_bind_group(test.device(), layout, entries) };
        assert!(!bind_group.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        bind_group
    } else {
        let mut bind_group = std::ptr::null();
        test.assert_device_error_after(
            || {
                bind_group = unsafe { create_bind_group(test.device(), layout, entries) };
            },
            None,
        );
        assert!(!bind_group.is_null());
        bind_group
    }
}

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

pub unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    unsafe { yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor) }
}

pub unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) }
}

pub unsafe fn create_sampler(
    device: native::WGPUDevice,
    compare: native::WGPUCompareFunction,
) -> native::WGPUSampler {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_Undefined,
        addressModeV: native::WGPUAddressMode_Undefined,
        addressModeW: native::WGPUAddressMode_Undefined,
        magFilter: native::WGPUFilterMode_Undefined,
        minFilter: native::WGPUFilterMode_Undefined,
        mipmapFilter: native::WGPUMipmapFilterMode_Undefined,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare,
        maxAnisotropy: 1,
    };
    unsafe { yawgpu::wgpuDeviceCreateSampler(device, &descriptor) }
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    dimension: native::WGPUTextureDimension,
    size: native::WGPUExtent3D,
    sample_count: u32,
    mip_level_count: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension,
        size,
        format,
        mipLevelCount: mip_level_count,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) }
}

pub unsafe fn create_texture_2d(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    sample_count: u32,
    layers: u32,
    mip_level_count: u32,
) -> native::WGPUTexture {
    unsafe {
        create_texture(
            device,
            usage,
            format,
            native::WGPUTextureDimension_2D,
            native::WGPUExtent3D {
                width: 16,
                height: 16,
                depthOrArrayLayers: layers,
            },
            sample_count,
            mip_level_count,
        )
    }
}

pub unsafe fn create_texture_view(
    texture: native::WGPUTexture,
    dimension: native::WGPUTextureViewDimension,
    format: native::WGPUTextureFormat,
    base_mip_level: u32,
    mip_level_count: u32,
    base_array_layer: u32,
    array_layer_count: u32,
) -> native::WGPUTextureView {
    let descriptor = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format,
        dimension,
        baseMipLevel: base_mip_level,
        mipLevelCount: mip_level_count,
        baseArrayLayer: base_array_layer,
        arrayLayerCount: array_layer_count,
        aspect: native::WGPUTextureAspect_All,
        usage: native::WGPUTextureUsage_None,
    };
    unsafe { yawgpu::wgpuTextureCreateView(texture, &descriptor) }
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

pub fn sampler_layout_typed(
    binding: u32,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUSamplerBindingType,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding, visibility);
    entry.sampler.type_ = ty;
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

pub fn buffer_binding(
    binding: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

pub fn sampler_binding(binding: u32, sampler: native::WGPUSampler) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: u64::MAX,
        sampler,
        textureView: std::ptr::null(),
    }
}

pub fn texture_binding(
    binding: u32,
    texture_view: native::WGPUTextureView,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: u64::MAX,
        sampler: std::ptr::null(),
        textureView: texture_view,
    }
}

pub fn release_bind_group(bind_group: native::WGPUBindGroup) {
    unsafe {
        yawgpu::wgpuBindGroupRelease(bind_group);
    }
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
