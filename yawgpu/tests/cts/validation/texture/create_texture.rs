// CTS compressed-format success subcases are intentionally deferred here:
// the Noop adapter does not advertise texture-compression features. The
// transient-attachment cases are yawgpu vendor/tiled-feature coverage and are
// marked N/A in their test bodies.

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn zero_size_and_usage() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(0, 4, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_None,
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn dimension_type_and_format_compatibility() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 1, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                format: native::WGPUTextureFormat_Depth24Plus,
                size: extent(4, 1, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                format: native::WGPUTextureFormat_Depth24Plus,
                size: extent(4, 4, 4),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn dimension_type_and_format_compatibility_additional_formats() {
    let test = ValidationTest::new();
    unsafe {
        for format in [
            native::WGPUTextureFormat_R16Uint,
            native::WGPUTextureFormat_R16Sint,
            native::WGPUTextureFormat_R16Float,
            native::WGPUTextureFormat_RG16Uint,
            native::WGPUTextureFormat_RG16Sint,
            native::WGPUTextureFormat_RG16Float,
            native::WGPUTextureFormat_RGB10A2Uint,
            native::WGPUTextureFormat_RGB10A2Unorm,
            native::WGPUTextureFormat_Depth24PlusStencil8,
        ] {
            assert_texture_ok(
                &test,
                native::WGPUTextureDescriptor {
                    format,
                    ..texture_descriptor()
                },
            );
        }

        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                size: extent(4, 4, 4),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn dimension_type_and_format_compatibility_tier1_formats() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureFormatsTier1]);
    unsafe {
        for format in [
            native::WGPUTextureFormat_R16Unorm,
            native::WGPUTextureFormat_R16Snorm,
            native::WGPUTextureFormat_RG16Unorm,
            native::WGPUTextureFormat_RG16Snorm,
        ] {
            assert_texture_ok(
                &test,
                native::WGPUTextureDescriptor {
                    format,
                    ..texture_descriptor()
                },
            );
        }
    }
}

#[test]
fn mip_level_count_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(&test, texture_descriptor());
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                mipLevelCount: 0,
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn mip_level_count_bound_check() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(7, 3, 1),
                mipLevelCount: 3,
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(7, 3, 1),
                mipLevelCount: 4,
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn mip_level_count_bound_check_bigger_than_integer_bit_width() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                mipLevelCount: u32::MAX,
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn sample_count_various_sample_count_with_all_formats() {
    let test = ValidationTest::new();
    unsafe {
        for (sample_count, valid) in [(1, true), (4, true), (0, false), (2, false), (8, false)] {
            let descriptor = native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                sampleCount: sample_count,
                ..texture_descriptor()
            };
            if valid {
                assert_texture_ok(&test, descriptor);
            } else {
                assert_texture_error(&test, descriptor);
            }
        }
    }
}

#[test]
fn sample_count_format_specific_multisample_capability() {
    let tier1 = ValidationTest::with_features(&[native::WGPUFeatureName_TextureFormatsTier1]);
    unsafe {
        for format in [
            native::WGPUTextureFormat_R8Snorm,
            native::WGPUTextureFormat_RG8Snorm,
            native::WGPUTextureFormat_RGBA8Snorm,
        ] {
            assert_texture_ok(
                &tier1,
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_RenderAttachment,
                    format,
                    sampleCount: 4,
                    ..texture_descriptor()
                },
            );
        }
    }

    let test = ValidationTest::new();
    unsafe {
        for format in [
            native::WGPUTextureFormat_R32Uint,
            native::WGPUTextureFormat_R32Sint,
        ] {
            assert_texture_error(
                &test,
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_RenderAttachment,
                    format,
                    sampleCount: 4,
                    ..texture_descriptor()
                },
            );
        }
    }
}

#[test]
fn sample_count_valid_sample_count_with_other_parameter_varies() {
    let test = ValidationTest::new();
    unsafe {
        for descriptor in [
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                sampleCount: 4,
                mipLevelCount: 2,
                ..texture_descriptor()
            },
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_StorageBinding,
                sampleCount: 4,
                ..texture_descriptor()
            },
            native::WGPUTextureDescriptor {
                sampleCount: 4,
                ..texture_descriptor()
            },
        ] {
            assert_texture_error(&test, descriptor);
        }
    }
}

#[test]
fn sample_count_1d_2d_array_3d() {
    let test = ValidationTest::new();
    unsafe {
        for descriptor in [
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 1, 1),
                sampleCount: 4,
                ..texture_descriptor()
            },
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                size: extent(4, 4, 2),
                sampleCount: 4,
                ..texture_descriptor()
            },
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, 4),
                sampleCount: 4,
                ..texture_descriptor()
            },
        ] {
            assert_texture_error(&test, descriptor);
        }
    }
}

#[test]
fn texture_size_default_value_and_smallest_size_uncompressed_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(1, 1, 1),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_default_value_and_smallest_size_compressed_format() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBC]);
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_BC1RGBAUnorm,
                size: extent(4, 4, 1),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_1d_texture() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(1, 1, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(1, 2, 1),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_2d_texture_uncompressed_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(1, 1, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                size: extent(1, 0, 1),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_2d_texture_compressed_format() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBC]);
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_BC1RGBAUnorm,
                size: extent(4, 4, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_BC1RGBAUnorm,
                size: extent(5, 4, 1),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_3d_texture_uncompressed_format() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(1, 1, 1),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(1, 1, 0),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_3d_texture_compressed_format() {
    let test =
        ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBCSliced3D]);
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                format: native::WGPUTextureFormat_BC1RGBAUnorm,
                size: extent(4, 4, 4),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                format: native::WGPUTextureFormat_BC1RGBAUnorm,
                size: extent(4, 5, 4),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_size_3d_etc2_texture_is_invalid() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionETC2]);
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                format: native::WGPUTextureFormat_ETC2RGBA8Unorm,
                size: extent(4, 4, 4),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_usage() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_Depth24Plus,
                ..texture_descriptor()
            },
        );
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_RGBA8Unorm,
                ..texture_descriptor()
            },
        );
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, 4),
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment,
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 1, 1),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn texture_usage_rgba8_snorm_storage_binding_requires_tier1() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_RGBA8Snorm,
                ..texture_descriptor()
            },
        );
    }

    let tier1 = ValidationTest::with_features(&[native::WGPUFeatureName_TextureFormatsTier1]);
    unsafe {
        assert_texture_ok(
            &tier1,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_StorageBinding,
                format: native::WGPUTextureFormat_RGBA8Snorm,
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn depth_or_array_layers_and_mip_level_count_for_transient_attachments() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_TransientAttachment,
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_TransientAttachment,
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, 4),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn usage() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_None,
                ..texture_descriptor()
            },
        );
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: 1 << 40,
                ..texture_descriptor()
            },
        );
        for usage in [
            native::WGPUTextureUsage_TransientAttachment,
            native::WGPUTextureUsage_TransientAttachment | native::WGPUTextureUsage_CopySrc,
            native::WGPUTextureUsage_TransientAttachment | native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureUsage_TransientAttachment | native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureUsage_TransientAttachment | native::WGPUTextureUsage_StorageBinding,
        ] {
            assert_texture_error(
                &test,
                native::WGPUTextureDescriptor {
                    usage,
                    ..texture_descriptor()
                },
            );
        }
    }
}

#[test]
fn new_usages() {
    let test = ValidationTest::new();
    unsafe {
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_TextureBinding | (1 << 40),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn view_formats() {
    let test = ValidationTest::new();
    unsafe {
        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor()
            },
        );
        let invalid = [native::WGPUTextureFormat_Undefined];
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                viewFormatCount: invalid.len(),
                viewFormats: invalid.as_ptr(),
                ..texture_descriptor()
            },
        );
        let incompatible = [native::WGPUTextureFormat_R8Unorm];
        assert_texture_error(
            &test,
            native::WGPUTextureDescriptor {
                viewFormatCount: incompatible.len(),
                viewFormats: incompatible.as_ptr(),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn view_formats_compressed_srgb_pairs_are_compatible() {
    let bc = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBC]);
    unsafe {
        let view_formats = [native::WGPUTextureFormat_BC3RGBAUnormSrgb];
        assert_texture_ok(
            &bc,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_BC3RGBAUnorm,
                size: extent(4, 4, 1),
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor()
            },
        );
    }

    let etc2 = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionETC2]);
    unsafe {
        let view_formats = [native::WGPUTextureFormat_ETC2RGBA8UnormSrgb];
        assert_texture_ok(
            &etc2,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_ETC2RGBA8Unorm,
                size: extent(4, 4, 1),
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor()
            },
        );
    }

    let astc = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionASTC]);
    unsafe {
        let view_formats = [native::WGPUTextureFormat_ASTC4x4UnormSrgb];
        assert_texture_ok(
            &astc,
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_ASTC4x4Unorm,
                size: extent(4, 4, 1),
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor()
            },
        );
    }
}

#[test]
fn transient_view_formats() {
    // N/A: TRANSIENT_ATTACHMENT is yawgpu's vendor tiled feature, not part of
    // the default Noop CTS port.
}

unsafe fn assert_texture_ok(test: &ValidationTest, descriptor: native::WGPUTextureDescriptor) {
    test.expect_no_validation_error(|| {
        let texture = yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor);
        assert!(!texture.is_null());
        yawgpu::wgpuTextureRelease(texture);
    });
}

unsafe fn assert_texture_error(test: &ValidationTest, descriptor: native::WGPUTextureDescriptor) {
    let mut texture = std::ptr::null();
    assert_device_error!({
        texture = yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor);
    });
    assert!(!texture.is_null());
    yawgpu::wgpuTextureRelease(texture);
}

fn texture_descriptor() -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
