//! CTS port of `webgpu/api/validation/image_copy/buffer_texture_copies.spec.ts`.
//!
//! The `depth32float-stencil8` feature-gated subcases are deferred on Noop
//! where the optional feature is not advertised.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn depth_stencil_format_copy_usage_and_aspect() {
    let test = ValidationTest::new();
    unsafe {
        for (format, aspect, b2t, t2b, write) in [
            (
                native::WGPUTextureFormat_Depth16Unorm,
                native::WGPUTextureAspect_All,
                true,
                true,
                true,
            ),
            (
                native::WGPUTextureFormat_Depth16Unorm,
                native::WGPUTextureAspect_StencilOnly,
                false,
                false,
                false,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUTextureAspect_StencilOnly,
                true,
                true,
                true,
            ),
        ] {
            let texture = depth_texture(test.device(), format, extent(1, 1, 1));
            let buffer = copy_buffer(test.device(), 32);
            let mut params = CopyParams::new(texture, buffer, extent(1, 1, 1), 32);
            params.aspect = aspect;
            let expected = [
                (CopyMethod::CopyB2T, b2t),
                (CopyMethod::CopyT2B, t2b),
                (CopyMethod::WriteTexture, write),
            ];
            for (method, success) in expected {
                expect_copy(&test, method, params, success);
            }
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
fn depth_stencil_format_copy_buffer_size() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            let texture = depth_texture(
                test.device(),
                native::WGPUTextureFormat_Depth16Unorm,
                extent(4, 4, 1),
            );
            let enough = copy_buffer(test.device(), 776);
            let small = copy_buffer(test.device(), 772);
            let mut params = CopyParams::new(texture, enough, extent(4, 4, 1), 776);
            params.layout = layout(0, 256, 4);
            params.aspect = native::WGPUTextureAspect_DepthOnly;
            expect_copy(&test, *method, params, true);
            params.buffer = small;
            params.data_size = 772;
            expect_copy(&test, *method, params, false);
            yawgpu::wgpuBufferRelease(small);
            yawgpu::wgpuBufferRelease(enough);
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
fn depth_stencil_format_copy_buffer_offset() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for offset in [1, 4] {
                let texture = depth_texture(
                    test.device(),
                    native::WGPUTextureFormat_Depth16Unorm,
                    extent(4, 4, 1),
                );
                let buffer = copy_buffer(test.device(), 1024);
                let mut params = CopyParams::new(texture, buffer, extent(4, 4, 1), 1024);
                params.layout = layout(offset, 256, 4);
                params.aspect = native::WGPUTextureAspect_DepthOnly;
                let success = *method == CopyMethod::WriteTexture || offset % 4 == 0;
                expect_copy(&test, *method, params, success);
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
        for method in BUFFER_TEXTURE_METHODS {
            for sample_count in [1, 4] {
                let usage = native::WGPUTextureUsage_CopySrc
                    | native::WGPUTextureUsage_CopyDst
                    | if sample_count > 1 {
                        native::WGPUTextureUsage_RenderAttachment
                    } else {
                        native::WGPUTextureUsage_None
                    };
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        usage,
                        native::WGPUTextureFormat_BGRA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(16, 1, 1),
                        1,
                        sample_count,
                    ),
                );
                let buffer = copy_buffer(test.device(), 64);
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(16, 1, 1), 64),
                    sample_count == 1,
                );
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn texture_buffer_usages() {
    let test = ValidationTest::new();
    unsafe {
        for method in BUFFER_TEXTURE_METHODS {
            for (texture_usage, buffer_usage) in [
                (
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                ),
                (
                    native::WGPUTextureUsage_CopySrc,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                ),
                (
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUBufferUsage_CopySrc,
                ),
            ] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        texture_usage,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(16, 16, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(test.device(), 32, buffer_usage);
                let success = match method {
                    CopyMethod::CopyB2T => {
                        texture_usage & native::WGPUTextureUsage_CopyDst != 0
                            && buffer_usage & native::WGPUBufferUsage_CopySrc != 0
                    }
                    CopyMethod::CopyT2B => {
                        texture_usage & native::WGPUTextureUsage_CopySrc != 0
                            && buffer_usage & native::WGPUBufferUsage_CopyDst != 0
                    }
                    CopyMethod::WriteTexture => unreachable!(),
                };
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(1, 1, 1), 32),
                    success,
                );
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        for method in BUFFER_TEXTURE_METHODS {
            for (buf_mismatched, tex_mismatched) in [(false, false), (true, false), (false, true)] {
                let buffer = copy_buffer(if buf_mismatched { other } else { test.device() }, 32);
                let texture = create_texture(
                    if tex_mismatched { other } else { test.device() },
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(1, 1, 1),
                        1,
                        1,
                    ),
                );
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(1, 1, 1), 32),
                    !buf_mismatched && !tex_mismatched,
                );
                yawgpu::wgpuTextureRelease(texture);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn offset_and_bytes_per_row() {
    let test = ValidationTest::new();
    unsafe {
        for method in BUFFER_TEXTURE_METHODS {
            for (offset, bytes_per_row, origin_x, width, success) in [
                (4, 256, 1, 4, true),
                (2, 256, 0, 4, false),
                (0, 128, 0, 4, false),
                (0, 256, 15, 4, true),
                (0, 256, 0, 5, true),
            ] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        if *method == CopyMethod::CopyB2T {
                            native::WGPUTextureUsage_CopyDst
                        } else {
                            native::WGPUTextureUsage_CopySrc
                        },
                        native::WGPUTextureFormat_R8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(20, 2, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    512,
                    if *method == CopyMethod::CopyB2T {
                        native::WGPUBufferUsage_CopySrc
                    } else {
                        native::WGPUBufferUsage_CopyDst
                    },
                );
                let mut params = CopyParams::new(texture, buffer, extent(width, 2, 1), 512);
                params.origin = origin(origin_x, 0, 0);
                params.layout = layout(offset, bytes_per_row, 2);
                expect_copy(&test, *method, params, success);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

unsafe fn depth_texture(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    size: native::WGPUExtent3D,
) -> native::WGPUTexture {
    create_texture(
        device,
        texture_descriptor(
            native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
            format,
            native::WGPUTextureDimension_2D,
            size,
            1,
            1,
        ),
    )
}

unsafe fn copy_buffer(device: native::WGPUDevice, size: u64) -> native::WGPUBuffer {
    create_buffer(
        device,
        size,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    )
}
