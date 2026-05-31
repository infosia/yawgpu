//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/copyTextureToTexture.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_encoder, create_error_texture, create_texture, expect_command_buffer, extent, origin,
    texture_descriptor, texture_info, CommandExpectation,
};

#[test]
fn copy_with_invalid_or_destroyed_texture() {
    let test = ValidationTest::new();
    unsafe {
        for src_state in ["valid", "invalid", "destroyed"] {
            for dst_state in ["valid", "invalid", "destroyed"] {
                let src = texture_with_state(&test, src_state, native::WGPUTextureUsage_CopySrc);
                let dst = texture_with_state(&test, dst_state, native::WGPUTextureUsage_CopyDst);
                let expectation = if src_state == "invalid" || dst_state == "invalid" {
                    CommandExpectation::FinishError
                } else if src_state == "destroyed" || dst_state == "destroyed" {
                    CommandExpectation::SubmitError
                } else {
                    CommandExpectation::Success
                };
                expect_copy_texture_to_texture(
                    &test,
                    texture_info(src, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    extent(1, 1, 1),
                    expectation,
                );
                yawgpu::wgpuTextureRelease(dst);
                yawgpu::wgpuTextureRelease(src);
            }
        }
    }
}

#[test]
fn texture_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for (src_mismatched, dst_mismatched) in [(false, false), (true, false), (false, true)] {
            let src = create_copy_texture(
                if src_mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            );
            let dst = create_copy_texture(
                if dst_mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            );
            expect_copy_texture_to_texture(
                &test,
                texture_info(src, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                extent(1, 1, 1),
                if src_mismatched || dst_mismatched {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
            );
            yawgpu::wgpuTextureRelease(dst);
            yawgpu::wgpuTextureRelease(src);
        }
    }
}

#[test]
fn mipmap_level() {
    let test = ValidationTest::new();
    unsafe {
        for dimension in [
            native::WGPUTextureDimension_1D,
            native::WGPUTextureDimension_2D,
            native::WGPUTextureDimension_3D,
        ] {
            for (src_levels, dst_levels, src_level, dst_level) in [
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (1, 1, 0, 1),
                (3, 3, 0, 0),
                (3, 3, 2, 0),
                (3, 3, 3, 0),
                (3, 3, 0, 2),
                (3, 3, 0, 3),
            ] {
                if dimension == native::WGPUTextureDimension_1D
                    && (src_levels != 1 || dst_levels != 1)
                {
                    continue;
                }
                let size = texture_size_for_dimension(dimension, 32, 4, 4);
                let src = create_copy_texture(
                    test.device(),
                    native::WGPUTextureUsage_CopySrc,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    dimension,
                    size,
                    src_levels,
                    1,
                );
                let dst = create_copy_texture(
                    test.device(),
                    native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    dimension,
                    size,
                    dst_levels,
                    1,
                );
                expect_copy_texture_to_texture(
                    &test,
                    texture_info(
                        src,
                        src_level,
                        origin(0, 0, 0),
                        native::WGPUTextureAspect_All,
                    ),
                    texture_info(
                        dst,
                        dst_level,
                        origin(0, 0, 0),
                        native::WGPUTextureAspect_All,
                    ),
                    extent(1, 1, 1),
                    if src_level < src_levels && dst_level < dst_levels {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuTextureRelease(dst);
                yawgpu::wgpuTextureRelease(src);
            }
        }
    }
}

#[test]
fn texture_usage() {
    let test = ValidationTest::new();
    unsafe {
        for src_usage in texture_usages() {
            for dst_usage in texture_usages() {
                let src = create_copy_texture(
                    test.device(),
                    src_usage,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    1,
                );
                let dst = create_copy_texture(
                    test.device(),
                    dst_usage,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    1,
                );
                expect_copy_texture_to_texture(
                    &test,
                    texture_info(src, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    extent(1, 1, 1),
                    if src_usage == native::WGPUTextureUsage_CopySrc
                        && dst_usage == native::WGPUTextureUsage_CopyDst
                    {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuTextureRelease(dst);
                yawgpu::wgpuTextureRelease(src);
            }
        }
    }
}

#[test]
fn sample_count() {
    let test = ValidationTest::new();
    unsafe {
        for src_sample_count in [1, 4] {
            for dst_sample_count in [1, 4] {
                let src = create_copy_texture(
                    test.device(),
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_RenderAttachment,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    src_sample_count,
                );
                let dst = create_copy_texture(
                    test.device(),
                    native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_RenderAttachment,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    dst_sample_count,
                );
                expect_copy_texture_to_texture(
                    &test,
                    texture_info(src, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    extent(4, 4, 1),
                    if src_sample_count == dst_sample_count {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuTextureRelease(dst);
                yawgpu::wgpuTextureRelease(src);
            }
        }
    }
}

#[test]
fn multisampled_copy_restrictions() {
    let test = ValidationTest::new();
    unsafe {
        for (src_x, src_y, dst_x, dst_y, width, height, success) in [
            (0, 0, 0, 0, 32, 16, true),
            (1, 0, 0, 0, 31, 16, false),
            (0, 1, 0, 0, 32, 15, false),
            (0, 0, 1, 0, 31, 16, false),
            (0, 0, 0, 0, 16, 8, false),
        ] {
            let src = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(32, 16, 1),
                1,
                4,
            );
            let dst = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(32, 16, 1),
                1,
                4,
            );
            expect_copy_texture_to_texture(
                &test,
                texture_info(
                    src,
                    0,
                    origin(src_x, src_y, 0),
                    native::WGPUTextureAspect_All,
                ),
                texture_info(
                    dst,
                    0,
                    origin(dst_x, dst_y, 0),
                    native::WGPUTextureAspect_All,
                ),
                extent(width, height, 1),
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuTextureRelease(dst);
            yawgpu::wgpuTextureRelease(src);
        }
    }
}

#[test]
fn texture_format_compatibility() {
    let test = ValidationTest::new();
    unsafe {
        for (src_format, dst_format, success) in [
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureFormat_RGBA8Unorm,
                true,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureFormat_RGBA8UnormSrgb,
                true,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureFormat_BGRA8Unorm,
                false,
            ),
        ] {
            let src = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopySrc,
                src_format,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            );
            let dst = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopyDst,
                dst_format,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            );
            expect_copy_texture_to_texture(
                &test,
                texture_info(src, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                extent(4, 4, 1),
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuTextureRelease(dst);
            yawgpu::wgpuTextureRelease(src);
        }
    }
}

#[test]
fn depth_stencil_copy_restrictions() {
    let test = ValidationTest::new();
    unsafe {
        for (origin_x, origin_y, width, height, success) in [
            (0, 0, 32, 32, true),
            (1, 0, 31, 32, false),
            (0, 1, 32, 31, false),
            (0, 0, 31, 32, false),
            (0, 0, 32, 31, false),
        ] {
            let src = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_Depth32Float,
                native::WGPUTextureDimension_2D,
                extent(32, 32, 1),
                1,
                1,
            );
            let dst = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_Depth32Float,
                native::WGPUTextureDimension_2D,
                extent(32, 32, 1),
                1,
                1,
            );
            expect_copy_texture_to_texture(
                &test,
                texture_info(
                    src,
                    0,
                    origin(origin_x, origin_y, 0),
                    native::WGPUTextureAspect_All,
                ),
                texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                extent(width, height, 1),
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuTextureRelease(dst);
            yawgpu::wgpuTextureRelease(src);
        }
    }
}

#[test]
fn copy_ranges() {
    let test = ValidationTest::new();
    unsafe {
        for dimension in [
            native::WGPUTextureDimension_1D,
            native::WGPUTextureDimension_2D,
            native::WGPUTextureDimension_3D,
        ] {
            let size = texture_size_for_dimension(dimension, 16, 8, 3);
            let mut cases = vec![
                (origin(0, 0, 0), size, true),
                (origin(1, 0, 0), size, false),
            ];
            if dimension != native::WGPUTextureDimension_1D {
                cases.push((origin(0, 1, 0), size, false));
                cases.push((origin(0, 0, 1), size, false));
            }
            for (copy_origin, copy_size, success) in cases {
                let src = create_copy_texture(
                    test.device(),
                    native::WGPUTextureUsage_CopySrc,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    dimension,
                    size,
                    1,
                    1,
                );
                let dst = create_copy_texture(
                    test.device(),
                    native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    dimension,
                    size,
                    1,
                    1,
                );
                expect_copy_texture_to_texture(
                    &test,
                    texture_info(src, 0, copy_origin, native::WGPUTextureAspect_All),
                    texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
                    copy_size,
                    if success {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuTextureRelease(dst);
                yawgpu::wgpuTextureRelease(src);
            }
        }
    }
}

#[test]
fn copy_within_same_texture() {
    let test = ValidationTest::new();
    unsafe {
        for (src_z, dst_z, depth, success) in [
            (0, 2, 1, true),
            (0, 2, 2, true),
            (0, 2, 3, false),
            (2, 0, 3, false),
            (4, 4, 1, false),
        ] {
            let texture = create_copy_texture(
                test.device(),
                native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 7),
                1,
                1,
            );
            expect_copy_texture_to_texture(
                &test,
                texture_info(
                    texture,
                    0,
                    origin(0, 0, src_z),
                    native::WGPUTextureAspect_All,
                ),
                texture_info(
                    texture,
                    0,
                    origin(0, 0, dst_z),
                    native::WGPUTextureAspect_All,
                ),
                extent(16, 16, depth),
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
fn copy_aspects() {
    let test = ValidationTest::new();
    unsafe {
        for (format, valid_aspects) in [
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                &[native::WGPUTextureAspect_All][..],
            ),
            (
                native::WGPUTextureFormat_Depth32Float,
                &[
                    native::WGPUTextureAspect_All,
                    native::WGPUTextureAspect_DepthOnly,
                ][..],
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                &[
                    native::WGPUTextureAspect_All,
                    native::WGPUTextureAspect_StencilOnly,
                ][..],
            ),
        ] {
            for source_aspect in [
                native::WGPUTextureAspect_All,
                native::WGPUTextureAspect_DepthOnly,
                native::WGPUTextureAspect_StencilOnly,
            ] {
                for destination_aspect in [
                    native::WGPUTextureAspect_All,
                    native::WGPUTextureAspect_DepthOnly,
                    native::WGPUTextureAspect_StencilOnly,
                ] {
                    let src = create_copy_texture(
                        test.device(),
                        native::WGPUTextureUsage_CopySrc,
                        format,
                        native::WGPUTextureDimension_2D,
                        extent(16, 8, 1),
                        1,
                        1,
                    );
                    let dst = create_copy_texture(
                        test.device(),
                        native::WGPUTextureUsage_CopyDst,
                        format,
                        native::WGPUTextureDimension_2D,
                        extent(16, 8, 1),
                        1,
                        1,
                    );
                    let success = valid_aspects.contains(&source_aspect)
                        && valid_aspects.contains(&destination_aspect);
                    expect_copy_texture_to_texture(
                        &test,
                        texture_info(src, 0, origin(0, 0, 0), source_aspect),
                        texture_info(dst, 0, origin(0, 0, 0), destination_aspect),
                        extent(16, 8, 1),
                        if success {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                    );
                    yawgpu::wgpuTextureRelease(dst);
                    yawgpu::wgpuTextureRelease(src);
                }
            }
        }
    }
}

#[test]
#[ignore = "Noop does not advertise texture-compression features; CTS expects compressed texture copy range and block-alignment validation when supported"]
fn copy_ranges_with_compressed_texture_formats() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBC]);
    unsafe {
        let src = create_copy_texture(
            test.device(),
            native::WGPUTextureUsage_CopySrc,
            native::WGPUTextureFormat_BC1RGBAUnorm,
            native::WGPUTextureDimension_2D,
            extent(16, 16, 1),
            1,
            1,
        );
        let dst = create_copy_texture(
            test.device(),
            native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_BC1RGBAUnorm,
            native::WGPUTextureDimension_2D,
            extent(16, 16, 1),
            1,
            1,
        );
        expect_copy_texture_to_texture(
            &test,
            texture_info(src, 0, origin(4, 4, 0), native::WGPUTextureAspect_All),
            texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
            extent(12, 12, 1),
            CommandExpectation::Success,
        );
        expect_copy_texture_to_texture(
            &test,
            texture_info(src, 0, origin(1, 0, 0), native::WGPUTextureAspect_All),
            texture_info(dst, 0, origin(0, 0, 0), native::WGPUTextureAspect_All),
            extent(15, 16, 1),
            CommandExpectation::FinishError,
        );
        yawgpu::wgpuTextureRelease(dst);
        yawgpu::wgpuTextureRelease(src);
    }
}

unsafe fn expect_copy_texture_to_texture(
    test: &ValidationTest,
    source: native::WGPUTexelCopyTextureInfo,
    destination: native::WGPUTexelCopyTextureInfo,
    copy_size: native::WGPUExtent3D,
    expectation: CommandExpectation,
) {
    let encoder = create_encoder(test.device());
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &copy_size);
    expect_command_buffer(test, encoder, expectation);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn texture_with_state(
    test: &ValidationTest,
    state: &str,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let texture = match state {
        "valid" | "destroyed" => create_copy_texture(
            test.device(),
            usage,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureDimension_2D,
            extent(4, 4, 1),
            1,
            1,
        ),
        "invalid" => create_error_texture(test),
        _ => unreachable!(),
    };
    if state == "destroyed" {
        yawgpu::wgpuTextureDestroy(texture);
    }
    texture
}

unsafe fn create_copy_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    dimension: native::WGPUTextureDimension,
    size: native::WGPUExtent3D,
    mip_level_count: u32,
    sample_count: u32,
) -> native::WGPUTexture {
    create_texture(
        device,
        texture_descriptor(
            usage,
            format,
            dimension,
            size,
            mip_level_count,
            sample_count,
        ),
    )
}

fn texture_size_for_dimension(
    dimension: native::WGPUTextureDimension,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
) -> native::WGPUExtent3D {
    if dimension == native::WGPUTextureDimension_1D {
        extent(width, 1, 1)
    } else {
        extent(width, height, depth_or_array_layers)
    }
}

fn texture_usages() -> [native::WGPUTextureUsage; 4] {
    [
        native::WGPUTextureUsage_CopySrc,
        native::WGPUTextureUsage_CopyDst,
        native::WGPUTextureUsage_TextureBinding,
        native::WGPUTextureUsage_RenderAttachment,
    ]
}
