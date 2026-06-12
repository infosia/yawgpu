use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn sampler_filtering_compatibility_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let non_filtering_layout =
            create_sampler_layout(test.device(), native::WGPUSamplerBindingType_NonFiltering);
        let filtering_layout =
            create_sampler_layout(test.device(), native::WGPUSamplerBindingType_Filtering);
        let non_filtering_sampler = create_sampler(test.device(), false);
        let filtering_sampler = create_sampler(test.device(), true);

        assert_device_error!(
            {
                let group =
                    create_bind_group(test.device(), non_filtering_layout, filtering_sampler);
                yawgpu::wgpuBindGroupRelease(group);
            },
            "filtering sampler is incompatible with non-filtering sampler binding"
        );

        test.expect_no_validation_error(|| {
            let group =
                create_bind_group(test.device(), non_filtering_layout, non_filtering_sampler);
            yawgpu::wgpuBindGroupRelease(group);
        });
        test.expect_no_validation_error(|| {
            let group = create_bind_group(test.device(), filtering_layout, filtering_sampler);
            yawgpu::wgpuBindGroupRelease(group);
        });

        yawgpu::wgpuSamplerRelease(filtering_sampler);
        yawgpu::wgpuSamplerRelease(non_filtering_sampler);
        yawgpu::wgpuBindGroupLayoutRelease(filtering_layout);
        yawgpu::wgpuBindGroupLayoutRelease(non_filtering_layout);
    }
}

unsafe fn create_sampler_layout(
    device: native::WGPUDevice,
    ty: native::WGPUSamplerBindingType,
) -> native::WGPUBindGroupLayout {
    let entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: native::WGPUShaderStage_Compute,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: 0,
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: ty,
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
    };
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: &entry,
    };
    yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor)
}

unsafe fn create_sampler(device: native::WGPUDevice, filtering: bool) -> native::WGPUSampler {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_Undefined,
        addressModeV: native::WGPUAddressMode_Undefined,
        addressModeW: native::WGPUAddressMode_Undefined,
        magFilter: if filtering {
            native::WGPUFilterMode_Linear
        } else {
            native::WGPUFilterMode_Undefined
        },
        minFilter: native::WGPUFilterMode_Undefined,
        mipmapFilter: native::WGPUMipmapFilterMode_Undefined,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare: native::WGPUCompareFunction_Undefined,
        maxAnisotropy: 1,
    };
    yawgpu::wgpuDeviceCreateSampler(device, &descriptor)
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    sampler: native::WGPUSampler,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler,
        textureView: std::ptr::null(),
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor)
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
