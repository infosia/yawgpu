use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn lod_clamps_must_be_finite() {
    let test = ValidationTest::new();
    unsafe {
        assert_sampler_ok(&test, default_descriptor());

        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                lodMinClamp: f32::NAN,
                ..default_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                lodMinClamp: f32::INFINITY,
                ..default_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                lodMaxClamp: f32::NAN,
                ..default_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                lodMaxClamp: f32::INFINITY,
                ..default_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                lodMinClamp: -1.0,
                lodMaxClamp: 64.0,
                ..default_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                lodMinClamp: 2.0,
                lodMaxClamp: 1.0,
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn max_anisotropy_must_be_at_least_one() {
    let test = ValidationTest::new();
    unsafe {
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                maxAnisotropy: 0,
                ..linear_anisotropic_descriptor()
            },
        );
        assert_sampler_ok(
            &test,
            native::WGPUSamplerDescriptor {
                maxAnisotropy: 1,
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn anisotropic_sampler_requires_linear_filters() {
    let test = ValidationTest::new();
    unsafe {
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                magFilter: native::WGPUFilterMode_Nearest,
                ..linear_anisotropic_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                minFilter: native::WGPUFilterMode_Nearest,
                ..linear_anisotropic_descriptor()
            },
        );
        assert_sampler_error(
            &test,
            native::WGPUSamplerDescriptor {
                mipmapFilter: native::WGPUMipmapFilterMode_Nearest,
                ..linear_anisotropic_descriptor()
            },
        );
        assert_sampler_ok(&test, linear_anisotropic_descriptor());
    }
}

#[test]
fn default_and_null_descriptors_are_valid() {
    let test = ValidationTest::new();
    unsafe {
        test.clear_errors();
        let sampler = yawgpu::wgpuDeviceCreateSampler(test.device(), std::ptr::null());
        assert!(!sampler.is_null());
        assert!(test.errors().is_empty());
        yawgpu::wgpuSamplerRelease(sampler);

        assert_sampler_ok(&test, default_descriptor());
    }
}

#[test]
fn release_is_safe_for_valid_and_error_samplers() {
    let test = ValidationTest::new();
    unsafe {
        let sampler = yawgpu::wgpuDeviceCreateSampler(test.device(), &default_descriptor());
        assert!(!sampler.is_null());
        yawgpu::wgpuSamplerRelease(sampler);

        let mut error_sampler = std::ptr::null();
        assert_device_error!({
            error_sampler = yawgpu::wgpuDeviceCreateSampler(
                test.device(),
                &native::WGPUSamplerDescriptor {
                    maxAnisotropy: 0,
                    ..default_descriptor()
                },
            );
        });
        assert!(!error_sampler.is_null());
        yawgpu::wgpuSamplerRelease(error_sampler);
    }
}

#[test]
fn compare_function_is_not_an_error_by_itself() {
    let test = ValidationTest::new();
    unsafe {
        assert_sampler_ok(
            &test,
            native::WGPUSamplerDescriptor {
                compare: native::WGPUCompareFunction_LessEqual,
                ..default_descriptor()
            },
        );
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

unsafe fn assert_sampler_ok(test: &ValidationTest, descriptor: native::WGPUSamplerDescriptor) {
    test.clear_errors();
    let sampler = yawgpu::wgpuDeviceCreateSampler(test.device(), &descriptor);
    assert!(!sampler.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuSamplerRelease(sampler);
}

fn linear_anisotropic_descriptor() -> native::WGPUSamplerDescriptor {
    native::WGPUSamplerDescriptor {
        maxAnisotropy: 2,
        magFilter: native::WGPUFilterMode_Linear,
        minFilter: native::WGPUFilterMode_Linear,
        mipmapFilter: native::WGPUMipmapFilterMode_Linear,
        ..default_descriptor()
    }
}

fn default_descriptor() -> native::WGPUSamplerDescriptor {
    native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
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
