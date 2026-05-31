use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn format() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), texture_descriptor_2d(1, 1));
        assert_view_ok(
            &test,
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_RGBA8Unorm),
        );
        assert_view_error(
            &test,
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_RGBA8UnormSrgb),
        );
        yawgpu::wgpuTextureRelease(texture);

        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_RGBA8UnormSrgb),
        );
        assert_view_error(
            &test,
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_R8Unorm),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn dimension() {
    let test = ValidationTest::new();
    unsafe {
        let texture_1d = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_1D,
                size: extent(4, 1, 1),
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            texture_1d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_1D, 1),
        );
        assert_view_error(
            &test,
            texture_1d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_2D, 1),
        );
        yawgpu::wgpuTextureRelease(texture_1d);

        let texture_2d = create_texture(test.device(), texture_descriptor_2d(1, 12));
        for (dimension, layers, valid) in [
            (native::WGPUTextureViewDimension_2D, 1, true),
            (native::WGPUTextureViewDimension_2DArray, 2, true),
            (native::WGPUTextureViewDimension_Cube, 6, true),
            (native::WGPUTextureViewDimension_CubeArray, 12, true),
            (native::WGPUTextureViewDimension_3D, 1, false),
            (native::WGPUTextureViewDimension_Cube, 5, false),
            (native::WGPUTextureViewDimension_CubeArray, 7, false),
        ] {
            let descriptor = view_descriptor_with_dimension(dimension, layers);
            if valid {
                assert_view_ok(&test, texture_2d, descriptor);
            } else {
                assert_view_error(&test, texture_2d, descriptor);
            }
        }
        assert_view_ok(
            &test,
            texture_2d,
            native::WGPUTextureViewDescriptor {
                dimension: native::WGPUTextureViewDimension_2D,
                arrayLayerCount: native::WGPU_ARRAY_LAYER_COUNT_UNDEFINED,
                baseArrayLayer: 3,
                ..view_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture_2d);

        let texture_3d = create_texture(test.device(), texture_descriptor_3d());
        assert_view_ok(
            &test,
            texture_3d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_3D, 1),
        );
        assert_view_error(
            &test,
            texture_3d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_2D, 1),
        );
        yawgpu::wgpuTextureRelease(texture_3d);
    }
}

#[test]
fn aspect() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_texture(test.device(), texture_descriptor_2d(1, 1));
        assert_view_ok(
            &test,
            color,
            view_descriptor_with_aspect(native::WGPUTextureAspect_All),
        );
        assert_view_error(
            &test,
            color,
            view_descriptor_with_aspect(native::WGPUTextureAspect_DepthOnly),
        );
        yawgpu::wgpuTextureRelease(color);

        let depth = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24Plus,
                usage: native::WGPUTextureUsage_TextureBinding,
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            depth,
            view_descriptor_with_aspect(native::WGPUTextureAspect_DepthOnly),
        );
        assert_view_error(
            &test,
            depth,
            view_descriptor_with_aspect(native::WGPUTextureAspect_StencilOnly),
        );
        yawgpu::wgpuTextureRelease(depth);

        let depth_stencil = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            depth_stencil,
            view_descriptor_with_aspect(native::WGPUTextureAspect_StencilOnly),
        );
        yawgpu::wgpuTextureRelease(depth_stencil);
    }
}

#[test]
fn array_layers() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), texture_descriptor_2d(1, 4));
        for (base, count, valid) in [
            (0, 1, true),
            (3, 1, true),
            (4, 1, false),
            (3, 2, false),
            (0, 0, false),
            (u32::MAX, 2, false),
        ] {
            let descriptor = native::WGPUTextureViewDescriptor {
                baseArrayLayer: base,
                arrayLayerCount: count,
                dimension: native::WGPUTextureViewDimension_2DArray,
                ..view_descriptor()
            };
            if valid {
                assert_view_ok(&test, texture, descriptor);
            } else {
                assert_view_error(&test, texture, descriptor);
            }
        }
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn mip_levels() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                mipLevelCount: 4,
                ..texture_descriptor_2d(1, 1)
            },
        );
        for (base, count, valid) in [
            (0, 1, true),
            (3, 1, true),
            (4, 1, false),
            (3, 2, false),
            (0, 0, false),
            (u32::MAX, 2, false),
        ] {
            let descriptor = native::WGPUTextureViewDescriptor {
                baseMipLevel: base,
                mipLevelCount: count,
                ..view_descriptor()
            };
            if valid {
                assert_view_ok(&test, texture, descriptor);
            } else {
                assert_view_error(&test, texture, descriptor);
            }
        }
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn cube_faces_square() {
    let test = ValidationTest::new();
    unsafe {
        let square = create_texture(test.device(), texture_descriptor_2d(1, 6));
        assert_view_ok(
            &test,
            square,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_Cube, 6),
        );
        yawgpu::wgpuTextureRelease(square);

        let non_square = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(8, 4, 6),
                ..texture_descriptor_2d(1, 6)
            },
        );
        assert_view_error(
            &test,
            non_square,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_Cube, 6),
        );
        yawgpu::wgpuTextureRelease(non_square);

        let non_square_array = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(8, 4, 12),
                ..texture_descriptor_2d(1, 12)
            },
        );
        assert_view_error(
            &test,
            non_square_array,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_CubeArray, 12),
        );
        yawgpu::wgpuTextureRelease(non_square_array);
    }
}

#[test]
fn texture_state() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), texture_descriptor_2d(1, 1));
        yawgpu::wgpuTextureDestroy(texture);
        assert_view_ok(&test, texture, view_descriptor());
        yawgpu::wgpuTextureRelease(texture);

        let mut invalid = std::ptr::null();
        assert_device_error!({
            invalid = yawgpu::wgpuDeviceCreateTexture(
                test.device(),
                &native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_None,
                    ..texture_descriptor_2d(1, 1)
                },
            );
        });
        assert_view_error(&test, invalid, view_descriptor());
        yawgpu::wgpuTextureRelease(invalid);
    }
}

#[test]
fn texture_view_usage() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_TextureBinding
                    | native::WGPUTextureUsage_RenderAttachment,
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            texture,
            view_descriptor_with_usage(native::WGPUTextureUsage_TextureBinding),
        );
        assert_view_error(
            &test,
            texture,
            view_descriptor_with_usage(native::WGPUTextureUsage_StorageBinding),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn texture_view_usage_of_multiple_usages() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_TextureBinding
                    | native::WGPUTextureUsage_RenderAttachment,
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            texture,
            view_descriptor_with_usage(
                native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        assert_view_error(&test, texture, view_descriptor_with_usage(1 << 40));
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn texture_view_usage_with_view_format() {
    let test = ValidationTest::new();
    unsafe {
        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_TextureBinding
                    | native::WGPUTextureUsage_RenderAttachment,
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor_2d(1, 1)
            },
        );
        assert_view_ok(
            &test,
            texture,
            native::WGPUTextureViewDescriptor {
                format: native::WGPUTextureFormat_RGBA8UnormSrgb,
                usage: native::WGPUTextureUsage_TextureBinding,
                ..view_descriptor()
            },
        );
        assert_view_error(
            &test,
            texture,
            native::WGPUTextureViewDescriptor {
                format: native::WGPUTextureFormat_RGBA8UnormSrgb,
                usage: native::WGPUTextureUsage_StorageBinding,
                ..view_descriptor()
            },
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

unsafe fn assert_view_ok(
    test: &ValidationTest,
    texture: native::WGPUTexture,
    descriptor: native::WGPUTextureViewDescriptor,
) {
    test.expect_no_validation_error(|| {
        let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
    });
}

unsafe fn assert_view_error(
    _test: &ValidationTest,
    texture: native::WGPUTexture,
    descriptor: native::WGPUTextureViewDescriptor,
) {
    let mut view = std::ptr::null();
    assert_device_error!({
        view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    });
    assert!(!view.is_null());
    yawgpu::wgpuTextureViewRelease(view);
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

fn view_descriptor_with_format(
    format: native::WGPUTextureFormat,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        format,
        ..view_descriptor()
    }
}

fn view_descriptor_with_dimension(
    dimension: native::WGPUTextureViewDimension,
    array_layer_count: u32,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        dimension,
        arrayLayerCount: array_layer_count,
        ..view_descriptor()
    }
}

fn view_descriptor_with_aspect(
    aspect: native::WGPUTextureAspect,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        aspect,
        ..view_descriptor()
    }
}

fn view_descriptor_with_usage(
    usage: native::WGPUTextureUsage,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        usage,
        ..view_descriptor()
    }
}

fn view_descriptor() -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format: native::WGPUTextureFormat_Undefined,
        dimension: native::WGPUTextureViewDimension_Undefined,
        baseMipLevel: 0,
        mipLevelCount: native::WGPU_MIP_LEVEL_COUNT_UNDEFINED,
        baseArrayLayer: 0,
        arrayLayerCount: native::WGPU_ARRAY_LAYER_COUNT_UNDEFINED,
        aspect: native::WGPUTextureAspect_Undefined,
        usage: native::WGPUTextureUsage_None,
    }
}

fn texture_descriptor_2d(
    mip_level_count: u32,
    array_layer_count: u32,
) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding,
        dimension: native::WGPUTextureDimension_2D,
        size: extent(8, 8, array_layer_count),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: mip_level_count,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

fn texture_descriptor_3d() -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        dimension: native::WGPUTextureDimension_3D,
        size: extent(8, 8, 4),
        ..texture_descriptor_2d(1, 1)
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
