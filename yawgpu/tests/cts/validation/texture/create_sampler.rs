use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn lod_min_and_max_clamp() {
    let test = ValidationTest::new();
    unsafe {
        for (min, max, valid) in [
            (0.0, 32.0, true),
            (1.0, 1.0, true),
            (-1.0, 32.0, false),
            (0.0, -1.0, false),
            (2.0, 1.0, false),
        ] {
            let descriptor = native::WGPUSamplerDescriptor {
                lodMinClamp: min,
                lodMaxClamp: max,
                ..sampler_descriptor()
            };
            if valid {
                test.expect_no_validation_error(|| {
                    release_sampler(yawgpu::wgpuDeviceCreateSampler(test.device(), &descriptor));
                });
            } else {
                assert_sampler_error(&test, descriptor);
            }
        }
    }
}

#[test]
fn max_anisotropy() {
    let test = ValidationTest::new();
    unsafe {
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                maxAnisotropy: 0,
                ..linear_sampler_descriptor()
            },
        );

        for descriptor in [
            native::WGPUSamplerDescriptor {
                maxAnisotropy: 1,
                ..sampler_descriptor()
            },
            linear_sampler_descriptor(),
        ] {
            test.expect_no_validation_error(|| {
                release_sampler(yawgpu::wgpuDeviceCreateSampler(test.device(), &descriptor));
            });
        }

        for descriptor in [
            native::WGPUSamplerDescriptor {
                magFilter: native::WGPUFilterMode_Nearest,
                ..linear_sampler_descriptor()
            },
            native::WGPUSamplerDescriptor {
                minFilter: native::WGPUFilterMode_Nearest,
                ..linear_sampler_descriptor()
            },
            native::WGPUSamplerDescriptor {
                mipmapFilter: native::WGPUMipmapFilterMode_Nearest,
                ..linear_sampler_descriptor()
            },
        ] {
            assert_sampler_error(&test, descriptor);
        }
    }
}

unsafe fn assert_sampler_error(test: &ValidationTest, descriptor: native::WGPUSamplerDescriptor) {
    let mut sampler = std::ptr::null();
    assert_device_error!({
        sampler = yawgpu::wgpuDeviceCreateSampler(test.device(), &descriptor);
    });
    assert!(!sampler.is_null());
    yawgpu::wgpuSamplerRelease(sampler);
}

unsafe fn release_sampler(sampler: native::WGPUSampler) {
    assert!(!sampler.is_null());
    yawgpu::wgpuSamplerRelease(sampler);
}

fn linear_sampler_descriptor() -> native::WGPUSamplerDescriptor {
    native::WGPUSamplerDescriptor {
        maxAnisotropy: 2,
        magFilter: native::WGPUFilterMode_Linear,
        minFilter: native::WGPUFilterMode_Linear,
        mipmapFilter: native::WGPUMipmapFilterMode_Linear,
        ..sampler_descriptor()
    }
}

fn sampler_descriptor() -> native::WGPUSamplerDescriptor {
    native::WGPUSamplerDescriptor {
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
        compare: native::WGPUCompareFunction_Undefined,
        maxAnisotropy: 1,
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
