use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn undefined_format_is_invalid() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Undefined,
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn multisample_requires_multisample_capable_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                format: native::WGPUTextureFormat_RGBA8Snorm,
                sampleCount: 4,
                ..default_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                format: native::WGPUTextureFormat_RG11B10Ufloat,
                sampleCount: 4,
                ..default_descriptor()
            },
        );

        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                format: native::WGPUTextureFormat_RGBA8Unorm,
                sampleCount: 4,
                ..default_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn depth_stencil_formats_require_2d() {
    let test = ValidationTest::new();
    unsafe {
        for dimension in [
            native::WGPUTextureDimension_1D,
            native::WGPUTextureDimension_3D,
        ] {
            assert_texture_error(
                &test,
                native::WGPUTextureDescriptor {
                    dimension,
                    size: match dimension {
                        native::WGPUTextureDimension_1D => extent(4, 1, 1),
                        _ => extent(4, 4, 4),
                    },
                    format: native::WGPUTextureFormat_Depth24PlusStencil8,
                    ..default_descriptor()
                },
            );
        }

        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                ..default_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn render_attachment_requires_renderable_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                format: native::WGPUTextureFormat_RGBA8Snorm,
                ..default_descriptor()
            },
        );

        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                format: native::WGPUTextureFormat_RGBA8Unorm,
                ..default_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn storage_binding_requires_storage_capable_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_RG8Unorm,
                ..default_descriptor()
            },
        );

        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_RGBA8Unorm,
                ..default_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn storage_binding_with_multisample_is_invalid() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_RGBA8Unorm,
                sampleCount: 4,
                ..default_descriptor()
            },
        );
    }
}

#[test]
fn populated_format_caps_match_dawn_sanity_checks() {
    let depth_stencil = caps(native::WGPUTextureFormat_Depth24PlusStencil8);
    assert!(depth_stencil.aspects.depth);
    assert!(depth_stencil.aspects.stencil);
    assert!(depth_stencil.renderable);
    assert!(depth_stencil.multisample_capable);

    let snorm = caps(native::WGPUTextureFormat_RGBA8Snorm);
    assert!(snorm.aspects.color);
    assert!(!snorm.renderable);
    assert!(!snorm.multisample_capable);
    // snorm formats are NOT storage-capable (Dawn `Format.cpp`).
    assert!(!snorm.storage_capable);

    let rgba8 = caps(native::WGPUTextureFormat_RGBA8Unorm);
    assert!(rgba8.aspects.color);
    assert!(rgba8.renderable);
    assert!(rgba8.multisample_capable);
    assert!(rgba8.storage_capable);
    assert_eq!(rgba8.texel_block_size, 4);
    assert_eq!(rgba8.block_w, 1);
    assert_eq!(rgba8.block_h, 1);

    let bc1 = caps(native::WGPUTextureFormat_BC1RGBAUnorm);
    assert!(bc1.aspects.color);
    assert!(bc1.is_compressed);
    assert_eq!(bc1.texel_block_size, 8);
    assert_eq!(bc1.block_w, 4);
    assert_eq!(bc1.block_h, 4);
}

fn caps(format: native::WGPUTextureFormat) -> yawgpu_core::FormatCaps {
    let features = [
        yawgpu_core::Feature::TextureCompressionBc,
        yawgpu_core::Feature::Depth32FloatStencil8,
        yawgpu_core::Feature::TextureFormatsTier1,
        yawgpu_core::Feature::TextureFormatsTier2,
    ]
    .into_iter()
    .collect();
    yawgpu_core::TextureFormat::from(format)
        .caps(&features)
        .expect("format is populated")
}

unsafe fn assert_texture_error(test: &ValidationTest, descriptor: native::WGPUTextureDescriptor) {
    let mut texture = std::ptr::null();
    assert_device_error!({
        texture = create_texture(test.device(), descriptor);
    });
    assert!(!texture.is_null());
    yawgpu::wgpuTextureRelease(texture);
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    yawgpu::wgpuDeviceCreateTexture(device, &descriptor)
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
