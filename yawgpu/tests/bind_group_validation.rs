use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn entry_count_and_binding_set_must_match_layout() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let layout = create_layout(test.device(), &[buffer_layout(0)]);

        assert_bind_group_error(&test, layout, &[]);
        assert_bind_group_ok(&test, layout, &[buffer_binding(0, buffer, 0, 256)]);

        let layout_two = create_layout(test.device(), &[buffer_layout(0), buffer_layout(1)]);
        assert_bind_group_error(
            &test,
            layout_two,
            &[
                buffer_binding(0, buffer, 0, 256),
                buffer_binding(2, buffer, 0, 256),
            ],
        );
        assert_bind_group_error(
            &test,
            layout_two,
            &[
                buffer_binding(0, buffer, 0, 256),
                buffer_binding(0, buffer, 0, 256),
            ],
        );

        yawgpu::wgpuBindGroupLayoutRelease(layout_two);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn resource_kind_must_match_layout_slot() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let sampler = yawgpu::wgpuDeviceCreateSampler(test.device(), std::ptr::null());
        let texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());

        let sampler_layout = create_layout(test.device(), &[sampler_layout(0)]);
        assert_bind_group_error(&test, sampler_layout, &[buffer_binding(0, buffer, 0, 256)]);

        let texture_layout = create_layout(test.device(), &[texture_layout(0, false)]);
        assert_bind_group_error(&test, texture_layout, &[sampler_binding(0, sampler)]);

        let buffer_layout = create_layout(test.device(), &[buffer_layout(0)]);
        let mut too_many = buffer_binding(0, buffer, 0, 256);
        too_many.sampler = sampler;
        assert_bind_group_error(&test, buffer_layout, &[too_many]);

        assert_bind_group_ok(&test, texture_layout, &[texture_binding(0, view)]);

        yawgpu::wgpuBindGroupLayoutRelease(buffer_layout);
        yawgpu::wgpuBindGroupLayoutRelease(texture_layout);
        yawgpu::wgpuBindGroupLayoutRelease(sampler_layout);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn buffer_offsets_sizes_and_usages_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 20_000);
        let small_uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let storage = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 1024);
        let copy_only = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 1024);

        let uniform_layout = create_layout(test.device(), &[buffer_layout(0)]);
        assert_bind_group_error(&test, uniform_layout, &[buffer_binding(0, uniform, 1, 256)]);
        assert_bind_group_ok(
            &test,
            uniform_layout,
            &[buffer_binding(0, uniform, 256, 256)],
        );
        assert_bind_group_error(
            &test,
            uniform_layout,
            &[buffer_binding(0, uniform, 20_000, u64::MAX)],
        );
        assert_bind_group_error(
            &test,
            uniform_layout,
            &[buffer_binding(0, uniform, 512, 20_000)],
        );
        assert_bind_group_error(
            &test,
            uniform_layout,
            &[buffer_binding(0, uniform, u64::MAX - 255, 512)],
        );
        assert_bind_group_error(
            &test,
            uniform_layout,
            &[buffer_binding(0, uniform, 0, 20_000)],
        );
        assert_bind_group_error(
            &test,
            uniform_layout,
            &[buffer_binding(0, copy_only, 0, 256)],
        );
        assert_bind_group_ok(
            &test,
            uniform_layout,
            &[buffer_binding(0, small_uniform, 0, u64::MAX)],
        );

        let storage_layout = create_layout(test.device(), &[storage_buffer_layout(0, 0)]);
        assert_bind_group_error(&test, storage_layout, &[buffer_binding(0, storage, 1, 256)]);
        assert_bind_group_ok(
            &test,
            storage_layout,
            &[buffer_binding(0, storage, 256, 256)],
        );
        assert_bind_group_error(
            &test,
            storage_layout,
            &[buffer_binding(0, copy_only, 0, 256)],
        );

        let min_layout = create_layout(test.device(), &[uniform_buffer_layout(0, 512)]);
        assert_bind_group_error(&test, min_layout, &[buffer_binding(0, uniform, 0, 256)]);

        yawgpu::wgpuBindGroupLayoutRelease(min_layout);
        yawgpu::wgpuBindGroupLayoutRelease(storage_layout);
        yawgpu::wgpuBindGroupLayoutRelease(uniform_layout);
        yawgpu::wgpuBufferRelease(copy_only);
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuBufferRelease(small_uniform);
        yawgpu::wgpuBufferRelease(uniform);
    }
}

#[test]
fn texture_usage_dimension_multisample_and_storage_layers_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let sampled_layout = create_layout(test.device(), &[texture_layout(0, false)]);
        let sampled_texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
        );
        let sampled_view = yawgpu::wgpuTextureCreateView(sampled_texture, std::ptr::null());
        assert_bind_group_ok(&test, sampled_layout, &[texture_binding(0, sampled_view)]);

        let missing_sampled_usage = create_texture(
            test.device(),
            native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
        );
        let missing_sampled_view =
            yawgpu::wgpuTextureCreateView(missing_sampled_usage, std::ptr::null());
        assert_bind_group_error(
            &test,
            sampled_layout,
            &[texture_binding(0, missing_sampled_view)],
        );

        let d2_array_view = create_view(
            sampled_texture,
            native::WGPUTextureViewDimension_2DArray,
            0,
            1,
        );
        assert_bind_group_error(&test, sampled_layout, &[texture_binding(0, d2_array_view)]);

        let multisampled_texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            4,
            1,
        );
        let multisampled_view =
            yawgpu::wgpuTextureCreateView(multisampled_texture, std::ptr::null());
        assert_bind_group_error(
            &test,
            sampled_layout,
            &[texture_binding(0, multisampled_view)],
        );
        let multisampled_layout = create_layout(test.device(), &[texture_layout(0, true)]);
        assert_bind_group_error(
            &test,
            multisampled_layout,
            &[texture_binding(0, sampled_view)],
        );

        let depth_layout = create_layout(test.device(), &[texture_layout(0, false)]);
        let depth_texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24Plus,
            1,
            1,
        );
        let depth_view = yawgpu::wgpuTextureCreateView(depth_texture, std::ptr::null());
        assert_bind_group_error(&test, depth_layout, &[texture_binding(0, depth_view)]);

        let storage_layout = create_layout(test.device(), &[storage_texture_layout(0)]);
        let storage_texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_StorageBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            2,
        );
        let storage_view = create_view(storage_texture, native::WGPUTextureViewDimension_2D, 0, 1);
        assert_bind_group_ok(&test, storage_layout, &[texture_binding(0, storage_view)]);

        let missing_storage_usage = create_texture(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
        );
        let missing_storage_view =
            yawgpu::wgpuTextureCreateView(missing_storage_usage, std::ptr::null());
        assert_bind_group_error(
            &test,
            storage_layout,
            &[texture_binding(0, missing_storage_view)],
        );

        let storage_array_layout = create_layout(test.device(), &[storage_texture_array_layout(0)]);
        let storage_array_view = create_view(
            storage_texture,
            native::WGPUTextureViewDimension_2DArray,
            0,
            2,
        );
        assert_bind_group_error(
            &test,
            storage_array_layout,
            &[texture_binding(0, storage_array_view)],
        );

        yawgpu::wgpuBindGroupLayoutRelease(storage_array_layout);
        yawgpu::wgpuTextureViewRelease(storage_array_view);
        yawgpu::wgpuTextureViewRelease(missing_storage_view);
        yawgpu::wgpuTextureRelease(missing_storage_usage);
        yawgpu::wgpuTextureViewRelease(storage_view);
        yawgpu::wgpuTextureRelease(storage_texture);
        yawgpu::wgpuBindGroupLayoutRelease(storage_layout);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth_texture);
        yawgpu::wgpuBindGroupLayoutRelease(depth_layout);
        yawgpu::wgpuBindGroupLayoutRelease(multisampled_layout);
        yawgpu::wgpuTextureViewRelease(multisampled_view);
        yawgpu::wgpuTextureRelease(multisampled_texture);
        yawgpu::wgpuTextureViewRelease(d2_array_view);
        yawgpu::wgpuTextureViewRelease(missing_sampled_view);
        yawgpu::wgpuTextureRelease(missing_sampled_usage);
        yawgpu::wgpuTextureViewRelease(sampled_view);
        yawgpu::wgpuTextureRelease(sampled_texture);
        yawgpu::wgpuBindGroupLayoutRelease(sampled_layout);
    }
}

#[test]
fn resources_must_belong_to_the_bind_group_device_and_release_is_safe() {
    let other = ValidationTest::new();
    let test = ValidationTest::new();
    unsafe {
        let local_buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let other_buffer = create_buffer(other.device(), native::WGPUBufferUsage_Uniform, 1024);
        let layout = create_layout(test.device(), &[buffer_layout(0)]);

        let valid = create_bind_group(
            test.device(),
            layout,
            &[buffer_binding(0, local_buffer, 0, 256)],
        );
        yawgpu::wgpuBindGroupAddRef(valid);
        yawgpu::wgpuBindGroupRelease(valid);
        yawgpu::wgpuBindGroupRelease(valid);

        let mut error_group = std::ptr::null();
        test.assert_device_error_after(
            || {
                error_group = create_bind_group(
                    test.device(),
                    layout,
                    &[buffer_binding(0, other_buffer, 0, 256)],
                );
            },
            None,
        );
        assert!(!error_group.is_null());
        yawgpu::wgpuBindGroupRelease(error_group);

        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(other_buffer);
        yawgpu::wgpuBufferRelease(local_buffer);
    }
}

unsafe fn assert_bind_group_ok(
    test: &ValidationTest,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) {
    test.clear_errors();
    let bind_group = create_bind_group(test.device(), layout, entries);
    assert!(!bind_group.is_null());
    assert!(
        test.errors().is_empty(),
        "expected no device error for valid bind group, got {:?}",
        test.errors()
    );
    yawgpu::wgpuBindGroupRelease(bind_group);
}

unsafe fn assert_bind_group_error(
    test: &ValidationTest,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) {
    let mut bind_group = std::ptr::null();
    test.assert_device_error_after(
        || {
            bind_group = create_bind_group(test.device(), layout, entries);
        },
        None,
    );
    assert!(!bind_group.is_null());
    yawgpu::wgpuBindGroupRelease(bind_group);
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor)
}

unsafe fn create_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor)
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        usage,
        size,
        mappedAtCreation: 0,
    };
    yawgpu::wgpuDeviceCreateBuffer(device, &descriptor)
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    sample_count: u32,
    layers: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: layers,
        },
        format,
        mipLevelCount: 1,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    yawgpu::wgpuDeviceCreateTexture(device, &descriptor)
}

unsafe fn create_view(
    texture: native::WGPUTexture,
    dimension: native::WGPUTextureViewDimension,
    base_layer: u32,
    layer_count: u32,
) -> native::WGPUTextureView {
    let descriptor = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        format: native::WGPUTextureFormat_Undefined,
        dimension,
        baseMipLevel: 0,
        mipLevelCount: native::WGPU_MIP_LEVEL_COUNT_UNDEFINED,
        baseArrayLayer: base_layer,
        arrayLayerCount: layer_count,
        aspect: native::WGPUTextureAspect_All,
        usage: native::WGPUTextureUsage_None,
    };
    yawgpu::wgpuTextureCreateView(texture, &descriptor)
}

fn buffer_binding(
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

fn sampler_binding(binding: u32, sampler: native::WGPUSampler) -> native::WGPUBindGroupEntry {
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

fn texture_binding(
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

fn default_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Fragment,
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

fn buffer_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    uniform_buffer_layout(binding, 0)
}

fn uniform_buffer_layout(binding: u32, min_binding_size: u64) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding);
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = min_binding_size;
    entry
}

fn storage_buffer_layout(binding: u32, min_binding_size: u64) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding);
    entry.buffer.type_ = native::WGPUBufferBindingType_Storage;
    entry.buffer.minBindingSize = min_binding_size;
    entry
}

fn sampler_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding);
    entry.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
    entry
}

fn texture_layout(binding: u32, multisampled: bool) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding);
    entry.texture.sampleType = native::WGPUTextureSampleType_Float;
    entry.texture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry.texture.multisampled = native::WGPUBool::from(multisampled);
    entry
}

fn storage_texture_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding);
    entry.storageTexture.access = native::WGPUStorageTextureAccess_WriteOnly;
    entry.storageTexture.format = native::WGPUTextureFormat_RGBA8Unorm;
    entry.storageTexture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn storage_texture_array_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = storage_texture_layout(binding);
    entry.storageTexture.viewDimension = native::WGPUTextureViewDimension_2DArray;
    entry
}
