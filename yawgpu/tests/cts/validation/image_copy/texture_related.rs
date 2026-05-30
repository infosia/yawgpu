//! CTS port of `webgpu/api/validation/image_copy/texture_related.spec.ts`.
//!
//! Feature-gated compressed-format matrix subcases are deferred on Noop where
//! the required texture compression feature is not advertised.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn valid() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for state in ["valid", "destroyed", "invalid"] {
                let texture = if state == "invalid" {
                    create_error_texture(&test)
                } else {
                    create_texture(
                        test.device(),
                        texture_descriptor(
                            native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                            native::WGPUTextureFormat_RGBA8Unorm,
                            native::WGPUTextureDimension_2D,
                            extent(4, 4, 1),
                            1,
                            1,
                        ),
                    )
                };
                if state == "destroyed" {
                    yawgpu::wgpuTextureDestroy(texture);
                }
                let buffer = create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let mut params = CopyParams::new(texture, buffer, extent(0, 0, 0), 16);
                if state == "destroyed" && *method != CopyMethod::WriteTexture {
                    params.submit = true;
                }
                expect_copy(&test, *method, params, state == "valid");
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn texture_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        for method in ALL_METHODS {
            for mismatched in [false, true] {
                let device = if mismatched { other } else { test.device() };
                let texture = create_texture(
                    device,
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(4, 4, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(0, 0, 0), 16),
                    !mismatched,
                );
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn usage() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for usage in [
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
            ] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        usage,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(4, 4, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let success = match method {
                    CopyMethod::CopyT2B => usage & native::WGPUTextureUsage_CopySrc != 0,
                    CopyMethod::WriteTexture | CopyMethod::CopyB2T => {
                        usage & native::WGPUTextureUsage_CopyDst != 0
                    }
                };
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(0, 0, 0), 16),
                    success,
                );
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn sample_count() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for sample_count in [1, 4] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc
                            | native::WGPUTextureUsage_CopyDst
                            | native::WGPUTextureUsage_RenderAttachment,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(4, 4, 1),
                        1,
                        sample_count,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(0, 0, 0), 16),
                    sample_count == 1,
                );
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn mip_level() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for mip_level in [0, 1, 3] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(32, 32, 1),
                        3,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let mut params = CopyParams::new(texture, buffer, extent(0, 0, 0), 16);
                params.mip_level = mip_level;
                expect_copy(&test, *method, params, mip_level < 3);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn format() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            let color = create_texture(
                test.device(),
                texture_descriptor(
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    1,
                ),
            );
            let buffer = create_buffer(
                test.device(),
                1024,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            );
            let mut params = CopyParams::new(color, buffer, extent(3, 4, 1), 1024);
            params.layout = layout(0, 256, 4);
            expect_copy(&test, *method, params, true);
            yawgpu::wgpuTextureRelease(color);

            let depth = create_texture(
                test.device(),
                texture_descriptor(
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_Depth16Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    1,
                ),
            );
            let mut params = CopyParams::new(depth, buffer, extent(3, 4, 1), 1024);
            params.layout = layout(0, 256, 4);
            params.aspect = native::WGPUTextureAspect_DepthOnly;
            expect_copy(&test, *method, params, false);
            yawgpu::wgpuTextureRelease(depth);
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn origin_alignment() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBC]);
    unsafe {
        for method in ALL_METHODS {
            for origin_x in [0, 1, 4] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_BC1RGBAUnorm,
                        native::WGPUTextureDimension_2D,
                        extent(8, 4, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    1024,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let mut params = CopyParams::new(texture, buffer, extent(0, 0, 0), 1024);
                params.origin = origin(origin_x, 0, 0);
                expect_copy(&test, *method, params, origin_x % 4 == 0);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn size_alignment() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TextureCompressionBC]);
    unsafe {
        for method in ALL_METHODS {
            for width in [0, 1, 4] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_BC1RGBAUnorm,
                        native::WGPUTextureDimension_2D,
                        extent(8, 4, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    1024,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let mut params = CopyParams::new(texture, buffer, extent(width, 4, 1), 1024);
                params.layout = layout(0, 256, 1);
                expect_copy(&test, *method, params, width % 4 == 0);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn copy_rectangle() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for (origin_x, width, success) in [(7, 7, true), (8, 8, false)] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(14, 4, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    1024,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let mut params = CopyParams::new(texture, buffer, extent(width, 1, 1), 1024);
                params.origin = origin(origin_x, 0, 0);
                params.layout = layout(0, 256, 1);
                expect_copy(&test, *method, params, success);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}
