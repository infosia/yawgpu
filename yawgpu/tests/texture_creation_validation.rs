use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn usage_must_be_non_zero() {
    let test = ValidationTest::new();
    unsafe {
        let mut texture = std::ptr::null();
        assert_device_error!({
            texture = create_texture(
                test.device(),
                descriptor_with_usage(native::WGPUTextureUsage_None),
            );
        });
        assert!(!texture.is_null());
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn sample_count_accepts_one_and_four_only() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), default_descriptor());
        yawgpu::wgpuTextureRelease(texture);

        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                sampleCount: 4,
                ..default_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);

        let mut texture = std::ptr::null();
        assert_device_error!({
            texture = create_texture(
                test.device(),
                native::WGPUTextureDescriptor {
                    sampleCount: 3,
                    ..default_descriptor()
                },
            );
        });
        assert!(!texture.is_null());
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn multisample_constraints_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                sampleCount: 4,
                mipLevelCount: 2,
                ..default_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 1, 1),
                sampleCount: 4,
                ..default_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                sampleCount: 4,
                size: extent(4, 4, 2),
                ..default_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_StorageBinding,
                sampleCount: 4,
                ..default_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                sampleCount: 4,
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn mip_level_rules_include_non_power_of_two_sizes() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                mipLevelCount: 0,
                ..default_descriptor()
            },
        );

        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(7, 3, 1),
                mipLevelCount: 3,
                ..default_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);

        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(7, 3, 1),
                mipLevelCount: 4,
                ..default_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(2, 1, 1),
                mipLevelCount: 2,
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn array_layers_and_dimension_limits_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );

        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(4, 4, limits.maxTextureArrayLayers + 1),
                ..default_descriptor()
            },
        );

        for descriptor in [
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(0, 1, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(limits.maxTextureDimension1D + 1, 1, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 2, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 1, 2),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                size: extent(0, 4, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                size: extent(4, 0, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                size: extent(4, 4, 0),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                size: extent(limits.maxTextureDimension2D + 1, 4, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                size: extent(4, limits.maxTextureDimension2D + 1, 1),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(0, 4, 4),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 0, 4),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, 0),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(limits.maxTextureDimension3D + 1, 4, 4),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, limits.maxTextureDimension3D + 1, 4),
                ..default_descriptor()
            },
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, limits.maxTextureDimension3D + 1),
                ..default_descriptor()
            },
        ] {
            assert_texture_error(&test, descriptor);
        }
    }
}

#[test]
fn render_attachment_requires_2d_dimension() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, 4),
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn destroy_is_safe_idempotent_and_error_textures_are_destroyable() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), default_descriptor());
        yawgpu::wgpuTextureDestroy(texture);
        yawgpu::wgpuTextureDestroy(texture);
        yawgpu::wgpuTextureRelease(texture);

        let mut error_texture = std::ptr::null();
        assert_device_error!({
            error_texture = create_texture(
                test.device(),
                descriptor_with_usage(native::WGPUTextureUsage_None),
            );
        });
        yawgpu::wgpuTextureDestroy(error_texture);
        yawgpu::wgpuTextureDestroy(error_texture);
        yawgpu::wgpuTextureRelease(error_texture);
    }
}

#[test]
fn getters_reflect_descriptor_for_valid_and_error_textures() {
    let test = ValidationTest::new();
    unsafe {
        let descriptor = native::WGPUTextureDescriptor {
            usage: native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_TextureBinding,
            dimension: native::WGPUTextureDimension_3D,
            size: extent(8, 4, 2),
            format: native::WGPUTextureFormat_R8Unorm,
            mipLevelCount: 2,
            sampleCount: 1,
            ..default_descriptor()
        };
        let texture = create_texture(test.device(), descriptor);
        assert_texture_reflects(texture, descriptor);
        yawgpu::wgpuTextureRelease(texture);

        let error_descriptor = native::WGPUTextureDescriptor {
            usage: native::WGPUTextureUsage_None,
            dimension: native::WGPUTextureDimension_1D,
            size: extent(12, 1, 1),
            format: native::WGPUTextureFormat_Depth24Plus,
            mipLevelCount: 1,
            sampleCount: 1,
            ..default_descriptor()
        };
        let mut error_texture = std::ptr::null();
        assert_device_error!({
            error_texture = create_texture(test.device(), error_descriptor);
        });
        assert!(!error_texture.is_null());
        assert_texture_reflects(error_texture, error_descriptor);
        yawgpu::wgpuTextureRelease(error_texture);
    }
}

unsafe fn assert_texture_error(test: &ValidationTest, descriptor: native::WGPUTextureDescriptor) {
    let mut texture = std::ptr::null();
    assert_device_error!({
        texture = create_texture(test.device(), descriptor);
    });
    assert!(!texture.is_null());
    yawgpu::wgpuTextureRelease(texture);
}

unsafe fn assert_texture_reflects(
    texture: native::WGPUTexture,
    descriptor: native::WGPUTextureDescriptor,
) {
    assert_eq!(yawgpu::wgpuTextureGetFormat(texture), descriptor.format);
    assert_eq!(
        yawgpu::wgpuTextureGetDimension(texture),
        descriptor.dimension
    );
    assert_eq!(yawgpu::wgpuTextureGetWidth(texture), descriptor.size.width);
    assert_eq!(
        yawgpu::wgpuTextureGetHeight(texture),
        descriptor.size.height
    );
    assert_eq!(
        yawgpu::wgpuTextureGetDepthOrArrayLayers(texture),
        descriptor.size.depthOrArrayLayers
    );
    assert_eq!(
        yawgpu::wgpuTextureGetMipLevelCount(texture),
        descriptor.mipLevelCount
    );
    assert_eq!(
        yawgpu::wgpuTextureGetSampleCount(texture),
        descriptor.sampleCount
    );
    assert_eq!(yawgpu::wgpuTextureGetUsage(texture), descriptor.usage);
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    yawgpu::wgpuDeviceCreateTexture(device, &descriptor)
}

fn descriptor_with_usage(usage: native::WGPUTextureUsage) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        usage,
        ..default_descriptor()
    }
}

fn default_descriptor() -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        usage: native::WGPUTextureUsage_TextureBinding,
        dimension: native::WGPUTextureDimension_2D,
        size: extent(4, 4, 1),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth_or_array_layers,
    }
}
