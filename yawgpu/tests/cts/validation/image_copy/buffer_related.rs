//! CTS port of `webgpu/api/validation/image_copy/buffer_related.spec.ts`.
//!
//! Feature-gated compressed-format matrix subcases are deferred on Noop where
//! the required texture compression feature is not advertised.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for method in BUFFER_TEXTURE_METHODS {
            let texture = create_texture(
                test.device(),
                texture_descriptor(
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(2, 2, 1),
                    1,
                    1,
                ),
            );

            let valid = create_buffer(
                test.device(),
                16,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            );
            expect_copy(
                &test,
                *method,
                CopyParams::new(texture, valid, extent(0, 0, 0), 16),
                true,
            );
            yawgpu::wgpuBufferRelease(valid);

            let destroyed = create_buffer(
                test.device(),
                16,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            );
            yawgpu::wgpuBufferDestroy(destroyed);
            let mut params = CopyParams::new(texture, destroyed, extent(0, 0, 0), 16);
            params.submit = true;
            expect_copy(&test, *method, params, false);
            yawgpu::wgpuBufferRelease(destroyed);

            let invalid = create_error_buffer(&test, native::WGPUBufferUsage_CopySrc);
            expect_copy(
                &test,
                *method,
                CopyParams::new(texture, invalid, extent(0, 0, 0), 16),
                false,
            );
            yawgpu::wgpuBufferRelease(invalid);
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
fn buffer_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        for method in BUFFER_TEXTURE_METHODS {
            for mismatched in [false, true] {
                let device = if mismatched { other } else { test.device() };
                let buffer = create_buffer(
                    device,
                    16,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(2, 2, 1),
                        1,
                        1,
                    ),
                );
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(0, 0, 0), 16),
                    !mismatched,
                );
                yawgpu::wgpuTextureRelease(texture);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn usage() {
    let test = ValidationTest::new();
    unsafe {
        for method in BUFFER_TEXTURE_METHODS {
            for usage in [
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_Uniform,
                native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_Uniform,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            ] {
                let buffer = create_buffer(test.device(), 16, usage);
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(2, 2, 1),
                        1,
                        1,
                    ),
                );
                let success = match method {
                    CopyMethod::CopyB2T => usage & native::WGPUBufferUsage_CopySrc != 0,
                    CopyMethod::CopyT2B => usage & native::WGPUBufferUsage_CopyDst != 0,
                    CopyMethod::WriteTexture => unreachable!(),
                };
                expect_copy(
                    &test,
                    *method,
                    CopyParams::new(texture, buffer, extent(0, 0, 0), 16),
                    success,
                );
                yawgpu::wgpuTextureRelease(texture);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn bytes_per_row_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for (bytes_per_row, copy_height, success) in [
                (native::WGPU_COPY_STRIDE_UNDEFINED, 1, true),
                (255, 2, *method == CopyMethod::WriteTexture),
                (256, 2, true),
                (257, 2, *method == CopyMethod::WriteTexture),
            ] {
                let texture = create_texture(
                    test.device(),
                    texture_descriptor(
                        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureDimension_2D,
                        extent(1, copy_height, 1),
                        1,
                        1,
                    ),
                );
                let buffer = create_buffer(
                    test.device(),
                    1024,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let mut params = CopyParams::new(texture, buffer, extent(1, copy_height, 1), 1024);
                params.layout = native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: bytes_per_row,
                    rowsPerImage: copy_height,
                };
                expect_copy(&test, *method, params, success);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}
