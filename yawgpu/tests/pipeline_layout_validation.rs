use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn bind_group_layout_count_is_limited() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let layouts = create_layouts(test.device(), limits.maxBindGroups);
        assert_pipeline_layout_ok(&test, &layouts, 0);

        let too_many = create_layouts(test.device(), limits.maxBindGroups + 1);
        assert_pipeline_layout_error(&test, &too_many, 0);

        release_layouts(&too_many);
        release_layouts(&layouts);
    }
}

#[test]
fn bind_group_layout_array_and_elements_must_be_valid() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_layout_ok(&test, &[], 0);

        let mut null_array_layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                null_array_layout =
                    create_pipeline_layout_raw(test.device(), 1, std::ptr::null(), 0);
            },
            None,
        );
        assert!(!null_array_layout.is_null());
        yawgpu::wgpuPipelineLayoutRelease(null_array_layout);

        let null_element = [std::ptr::null()];
        assert_pipeline_layout_error(&test, &null_element, 0);

        let error_bgl = create_error_bind_group_layout(test.device());
        assert_pipeline_layout_error(&test, &[error_bgl], 0);
        yawgpu::wgpuBindGroupLayoutRelease(error_bgl);
    }
}

#[test]
fn immediate_size_is_limited() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        assert_pipeline_layout_ok(&test, &[], limits.maxImmediateSize);
        assert_pipeline_layout_error(&test, &[], limits.maxImmediateSize + 1);
    }
}

#[test]
fn valid_pipeline_layout_with_bind_group_layouts_and_release_is_safe() {
    let test = ValidationTest::new();
    unsafe {
        let layouts = create_layouts(test.device(), 2);
        let pipeline_layout = create_pipeline_layout(test.device(), &layouts, 0);
        assert!(!pipeline_layout.is_null());
        assert!(test.errors().is_empty());

        yawgpu::wgpuPipelineLayoutAddRef(pipeline_layout);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);

        let mut error_layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                error_layout = create_pipeline_layout(test.device(), &layouts, u32::MAX);
            },
            None,
        );
        assert!(!error_layout.is_null());
        yawgpu::wgpuPipelineLayoutRelease(error_layout);

        release_layouts(&layouts);
    }
}

unsafe fn assert_pipeline_layout_ok(
    test: &ValidationTest,
    layouts: &[native::WGPUBindGroupLayout],
    immediate_size: u32,
) {
    test.clear_errors();
    let pipeline_layout = create_pipeline_layout(test.device(), layouts, immediate_size);
    assert!(!pipeline_layout.is_null());
    assert!(
        test.errors().is_empty(),
        "expected no device error for valid pipeline layout, got {:?}",
        test.errors()
    );
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
}

unsafe fn assert_pipeline_layout_error(
    test: &ValidationTest,
    layouts: &[native::WGPUBindGroupLayout],
    immediate_size: u32,
) {
    let mut pipeline_layout = std::ptr::null();
    test.assert_device_error_after(
        || {
            pipeline_layout = create_pipeline_layout(test.device(), layouts, immediate_size);
        },
        None,
    );
    assert!(!pipeline_layout.is_null());
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
}

unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    layouts: &[native::WGPUBindGroupLayout],
    immediate_size: u32,
) -> native::WGPUPipelineLayout {
    create_pipeline_layout_raw(device, layouts.len(), layouts.as_ptr(), immediate_size)
}

unsafe fn create_pipeline_layout_raw(
    device: native::WGPUDevice,
    bind_group_layout_count: usize,
    bind_group_layouts: *const native::WGPUBindGroupLayout,
    immediate_size: u32,
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        bindGroupLayoutCount: bind_group_layout_count,
        bindGroupLayouts: bind_group_layouts,
        immediateSize: immediate_size,
    };
    yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor)
}

unsafe fn create_layouts(
    device: native::WGPUDevice,
    count: u32,
) -> Vec<native::WGPUBindGroupLayout> {
    (0..count)
        .map(|_| create_bind_group_layout(device, &[]))
        .collect()
}

unsafe fn release_layouts(layouts: &[native::WGPUBindGroupLayout]) {
    for layout in layouts {
        yawgpu::wgpuBindGroupLayoutRelease(*layout);
    }
}

unsafe fn create_error_bind_group_layout(
    device: native::WGPUDevice,
) -> native::WGPUBindGroupLayout {
    let entries = [buffer_layout(0), buffer_layout(0)];
    let mut layout = std::ptr::null();
    yawgpu_test::assert_current_device_error_after(
        || {
            layout = create_bind_group_layout(device, &entries);
        },
        None,
    );
    assert!(!layout.is_null());
    layout
}

unsafe fn create_bind_group_layout(
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

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    let mut limits = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuDeviceGetLimits(device, &mut limits),
        native::WGPUStatus_Success
    );
    limits
}

fn buffer_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Fragment,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Uniform,
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
