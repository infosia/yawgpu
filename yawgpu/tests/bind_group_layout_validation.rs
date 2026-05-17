use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn bindings_are_unique_and_below_limit() {
    let test = ValidationTest::new();
    unsafe {
        assert_layout_ok(&test, &[buffer_entry(0)]);
        assert_layout_ok(&test, &[buffer_entry(1), buffer_entry(0)]);
        assert_layout_error(&test, &[buffer_entry(0), buffer_entry(0)]);
        assert_layout_error(&test, &[buffer_entry(1000)]);
    }
}

#[test]
fn exactly_one_sub_layout_must_be_set() {
    let test = ValidationTest::new();
    unsafe {
        assert_layout_ok(&test, &[buffer_entry(0)]);

        let mut none = default_entry(0);
        none.visibility = native::WGPUShaderStage_Fragment;
        assert_layout_error(&test, &[none]);

        let mut two = buffer_entry(0);
        two.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
        assert_layout_error(&test, &[two]);
    }
}

#[test]
fn binding_layout_enums_and_storage_texture_rules_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_layout_ok(
            &test,
            &[buffer_entry_with_type(
                0,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        assert_layout_ok(
            &test,
            &[sampler_entry_with_type(
                0,
                native::WGPUSamplerBindingType_Comparison,
            )],
        );
        assert_layout_ok(&test, &[texture_entry(0)]);
        assert_layout_ok(
            &test,
            &[storage_texture_entry(
                0,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureViewDimension_2D,
            )],
        );

        assert_layout_error(
            &test,
            &[buffer_entry_with_type(
                0,
                native::WGPUBufferBindingType_Force32,
            )],
        );
        assert_layout_error(
            &test,
            &[sampler_entry_with_type(
                0,
                native::WGPUSamplerBindingType_Force32,
            )],
        );

        let mut invalid_texture = texture_entry(0);
        invalid_texture.texture.sampleType = native::WGPUTextureSampleType_Force32;
        assert_layout_error(&test, &[invalid_texture]);

        let mut invalid_multisampled = texture_entry(0);
        invalid_multisampled.texture.multisampled = 1;
        invalid_multisampled.texture.viewDimension = native::WGPUTextureViewDimension_2DArray;
        assert_layout_error(&test, &[invalid_multisampled]);

        assert_layout_error(
            &test,
            &[storage_texture_entry(
                0,
                native::WGPUTextureFormat_BGRA8Unorm,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        assert_layout_error(
            &test,
            &[storage_texture_entry(
                0,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureViewDimension_1D,
            )],
        );
        let mut invalid_storage = storage_texture_entry(
            0,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureViewDimension_2D,
        );
        invalid_storage.storageTexture.access = native::WGPUStorageTextureAccess_Force32;
        assert_layout_error(&test, &[invalid_storage]);
    }
}

#[test]
fn binding_array_size_is_rejected_above_one_and_unknown_visibility_bits_are_allowed() {
    let test = ValidationTest::new();
    unsafe {
        let mut entry = texture_entry(0);
        entry.bindingArraySize = 0;
        assert_layout_ok(&test, &[entry]);

        entry.bindingArraySize = 1;
        assert_layout_ok(&test, &[entry]);

        entry.bindingArraySize = 2;
        assert_layout_error(&test, &[entry]);

        let mut unknown_visibility = buffer_entry(0);
        unknown_visibility.visibility = native::WGPUShaderStage_Vertex | (1 << 12);
        assert_layout_ok(&test, &[unknown_visibility]);
    }
}

#[test]
fn dynamic_buffer_limits_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );

        let entries = (0..limits.maxDynamicUniformBuffersPerPipelineLayout)
            .map(dynamic_uniform_entry)
            .collect::<Vec<_>>();
        assert_layout_ok(&test, &entries);

        let entries = (0..=limits.maxDynamicUniformBuffersPerPipelineLayout)
            .map(dynamic_uniform_entry)
            .collect::<Vec<_>>();
        assert_layout_error(&test, &entries);
    }
}

#[test]
fn per_stage_resource_counts_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );

        let entries = (0..limits.maxSampledTexturesPerShaderStage)
            .map(texture_entry)
            .collect::<Vec<_>>();
        assert_layout_ok(&test, &entries);

        let entries = (0..=limits.maxSampledTexturesPerShaderStage)
            .map(texture_entry)
            .collect::<Vec<_>>();
        assert_layout_error(&test, &entries);

        let mut different_stage = texture_entry(limits.maxSampledTexturesPerShaderStage);
        different_stage.visibility = native::WGPUShaderStage_Vertex;
        let mut entries = (0..limits.maxSampledTexturesPerShaderStage)
            .map(texture_entry)
            .collect::<Vec<_>>();
        entries.push(different_stage);
        assert_layout_ok(&test, &entries);
    }
}

#[test]
fn total_entry_count_is_limited_and_release_is_safe() {
    let test = ValidationTest::new();
    unsafe {
        let entries = (0..1000).map(buffer_entry).collect::<Vec<_>>();
        let layout = create_layout(test.device(), &entries);
        yawgpu::wgpuBindGroupLayoutRelease(layout);

        let entries = (0..1001).map(buffer_entry).collect::<Vec<_>>();
        let mut error_layout = std::ptr::null();
        assert_device_error!({
            error_layout = create_layout(test.device(), &entries);
        });
        assert!(!error_layout.is_null());
        yawgpu::wgpuBindGroupLayoutRelease(error_layout);
    }
}

unsafe fn assert_layout_ok(test: &ValidationTest, entries: &[native::WGPUBindGroupLayoutEntry]) {
    test.clear_errors();
    let layout = create_layout(test.device(), entries);
    assert!(!layout.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn assert_layout_error(test: &ValidationTest, entries: &[native::WGPUBindGroupLayoutEntry]) {
    let mut layout = std::ptr::null();
    assert_device_error!({
        layout = create_layout(test.device(), entries);
    });
    assert!(!layout.is_null());
    yawgpu::wgpuBindGroupLayoutRelease(layout);
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

fn default_entry(binding: u32) -> native::WGPUBindGroupLayoutEntry {
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

fn buffer_entry(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    buffer_entry_with_type(binding, native::WGPUBufferBindingType_Uniform)
}

fn dynamic_uniform_entry(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = buffer_entry(binding);
    entry.buffer.hasDynamicOffset = 1;
    entry
}

fn buffer_entry_with_type(
    binding: u32,
    ty: native::WGPUBufferBindingType,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding);
    entry.buffer.type_ = ty;
    entry
}

fn sampler_entry_with_type(
    binding: u32,
    ty: native::WGPUSamplerBindingType,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding);
    entry.sampler.type_ = ty;
    entry
}

fn texture_entry(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding);
    entry.texture.sampleType = native::WGPUTextureSampleType_Float;
    entry.texture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn storage_texture_entry(
    binding: u32,
    format: native::WGPUTextureFormat,
    view_dimension: native::WGPUTextureViewDimension,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_entry(binding);
    entry.storageTexture.access = native::WGPUStorageTextureAccess_WriteOnly;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = view_dimension;
    entry
}
