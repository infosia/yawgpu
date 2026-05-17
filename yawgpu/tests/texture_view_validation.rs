use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn zero_counts_are_invalid_but_undefined_counts_are_inferred() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), texture_descriptor_2d(1, 4));

        assert_view_error(
            texture,
            native::WGPUTextureViewDescriptor {
                mipLevelCount: 0,
                ..default_view_descriptor()
            },
        );
        assert_view_error(
            texture,
            native::WGPUTextureViewDescriptor {
                arrayLayerCount: 0,
                ..default_view_descriptor()
            },
        );

        let view = yawgpu::wgpuTextureCreateView(texture, &default_view_descriptor());
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn mip_and_array_ranges_must_fit() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(8, 8, 4),
                mipLevelCount: 4,
                ..texture_descriptor_2d(1, 4)
            },
        );

        assert_view_error(
            texture,
            native::WGPUTextureViewDescriptor {
                baseMipLevel: 3,
                mipLevelCount: 2,
                ..default_view_descriptor()
            },
        );
        assert_view_error(
            texture,
            native::WGPUTextureViewDescriptor {
                baseArrayLayer: 3,
                arrayLayerCount: 2,
                ..default_view_descriptor()
            },
        );

        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn view_dimension_matrix_is_validated() {
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
        let view = create_view_with_dimension(texture_1d, native::WGPUTextureViewDimension_1D, 1);
        yawgpu::wgpuTextureViewRelease(view);
        assert_view_error(
            texture_1d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_Cube, 1),
        );
        yawgpu::wgpuTextureRelease(texture_1d);

        let texture_2d = create_texture(test.device(), texture_descriptor_2d(1, 12));
        let view = create_view_with_dimension(texture_2d, native::WGPUTextureViewDimension_2D, 1);
        yawgpu::wgpuTextureViewRelease(view);
        let view =
            create_view_with_dimension(texture_2d, native::WGPUTextureViewDimension_2DArray, 2);
        yawgpu::wgpuTextureViewRelease(view);
        assert_view_error(
            texture_2d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_3D, 1),
        );
        assert_view_error(
            texture_2d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_Cube, 5),
        );
        let view = create_view_with_dimension(texture_2d, native::WGPUTextureViewDimension_Cube, 6);
        yawgpu::wgpuTextureViewRelease(view);
        assert_view_error(
            texture_2d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_Cube, 7),
        );
        assert_view_error(
            texture_2d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_CubeArray, 5),
        );
        let view =
            create_view_with_dimension(texture_2d, native::WGPUTextureViewDimension_CubeArray, 12);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture_2d);

        let texture_3d = create_texture(test.device(), texture_descriptor_3d());
        let view = create_view_with_dimension(texture_3d, native::WGPUTextureViewDimension_3D, 1);
        yawgpu::wgpuTextureViewRelease(view);
        assert_view_error(
            texture_3d,
            view_descriptor_with_dimension(native::WGPUTextureViewDimension_Cube, 1),
        );
        yawgpu::wgpuTextureRelease(texture_3d);
    }
}

// Dawn-faithful: a view format is compatible iff it equals the texture's
// format OR is explicitly listed in the texture's `viewFormats`. There is
// NO implicit sRGB-counterpart allowance (Dawn `Texture.cpp`
// `ValidateCanViewTextureAs`).
#[test]
fn view_format_compatibility_allows_same_srgb_pair_and_view_formats_only() {
    let test = ValidationTest::new();
    unsafe {
        // Texture with NO viewFormats: only the identical format is allowed.
        let texture = create_texture(test.device(), texture_descriptor_2d(1, 1));

        // Same format => ok.
        let view = create_view_with_format(texture, native::WGPUTextureFormat_RGBA8Unorm);
        yawgpu::wgpuTextureViewRelease(view);
        // sRGB counterpart, NOT listed in viewFormats => device error.
        assert_view_error(
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_RGBA8UnormSrgb),
        );
        // Unrelated / cross-category formats, not listed => device error.
        assert_view_error(
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_R8Unorm),
        );
        assert_view_error(
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_Depth24Plus),
        );
        yawgpu::wgpuTextureRelease(texture);

        // sRGB counterpart explicitly listed in viewFormats => ok; an
        // unlisted cross-category format is still rejected.
        let srgb_view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                viewFormatCount: srgb_view_formats.len(),
                viewFormats: srgb_view_formats.as_ptr(),
                ..texture_descriptor_2d(1, 1)
            },
        );
        let view = create_view_with_format(texture, native::WGPUTextureFormat_RGBA8UnormSrgb);
        yawgpu::wgpuTextureViewRelease(view);
        assert_view_error(
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_R8Unorm),
        );
        yawgpu::wgpuTextureRelease(texture);

        // Unrelated format listed in viewFormats => that listed format is
        // allowed; an unlisted one is not.
        let view_formats = [native::WGPUTextureFormat_R8Unorm];
        let texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                viewFormatCount: view_formats.len(),
                viewFormats: view_formats.as_ptr(),
                ..texture_descriptor_2d(1, 1)
            },
        );
        let view = create_view_with_format(texture, native::WGPUTextureFormat_R8Unorm);
        yawgpu::wgpuTextureViewRelease(view);
        assert_view_error(
            texture,
            view_descriptor_with_format(native::WGPUTextureFormat_RG8Unorm),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn aspect_must_match_view_format_aspects() {
    let test = ValidationTest::new();
    unsafe {
        let color_texture = create_texture(test.device(), texture_descriptor_2d(1, 1));
        assert_view_error(
            color_texture,
            view_descriptor_with_aspect(native::WGPUTextureAspect_DepthOnly),
        );
        assert_view_error(
            color_texture,
            view_descriptor_with_aspect(native::WGPUTextureAspect_StencilOnly),
        );
        let view = yawgpu::wgpuTextureCreateView(color_texture, &default_view_descriptor());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(color_texture);

        let depth_texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24Plus,
                ..texture_descriptor_2d(1, 1)
            },
        );
        let view = create_view_with_aspect(depth_texture, native::WGPUTextureAspect_DepthOnly);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(depth_texture);

        let depth_stencil_texture = create_texture(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                ..texture_descriptor_2d(1, 1)
            },
        );
        let view =
            create_view_with_aspect(depth_stencil_texture, native::WGPUTextureAspect_StencilOnly);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(depth_stencil_texture);
    }
}

#[test]
fn default_descriptor_release_and_error_texture_paths_are_safe() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), texture_descriptor_2d(1, 2));
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);

        let mut error_texture = std::ptr::null();
        assert_device_error!({
            error_texture = create_texture(
                test.device(),
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_None,
                    ..texture_descriptor_2d(1, 1)
                },
            );
        });
        let mut error_view = std::ptr::null();
        assert_device_error!({
            error_view = yawgpu::wgpuTextureCreateView(error_texture, std::ptr::null());
        });
        assert!(!error_view.is_null());
        yawgpu::wgpuTextureViewRelease(error_view);
        yawgpu::wgpuTextureRelease(error_texture);
    }
}

unsafe fn assert_view_error(
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

unsafe fn create_view_with_dimension(
    texture: native::WGPUTexture,
    dimension: native::WGPUTextureViewDimension,
    array_layer_count: u32,
) -> native::WGPUTextureView {
    let descriptor = view_descriptor_with_dimension(dimension, array_layer_count);
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

unsafe fn create_view_with_format(
    texture: native::WGPUTexture,
    format: native::WGPUTextureFormat,
) -> native::WGPUTextureView {
    let descriptor = view_descriptor_with_format(format);
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

unsafe fn create_view_with_aspect(
    texture: native::WGPUTexture,
    aspect: native::WGPUTextureAspect,
) -> native::WGPUTextureView {
    let descriptor = view_descriptor_with_aspect(aspect);
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

fn view_descriptor_with_dimension(
    dimension: native::WGPUTextureViewDimension,
    array_layer_count: u32,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        dimension,
        arrayLayerCount: array_layer_count,
        ..default_view_descriptor()
    }
}

fn view_descriptor_with_format(
    format: native::WGPUTextureFormat,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        format,
        ..default_view_descriptor()
    }
}

fn view_descriptor_with_aspect(
    aspect: native::WGPUTextureAspect,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        aspect,
        ..default_view_descriptor()
    }
}

fn default_view_descriptor() -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
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
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
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
