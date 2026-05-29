//! CTS port of `webgpu/api/validation/image_copy/layout_related.spec.ts`.
//!
//! Feature-gated compressed-format matrix subcases are deferred on Noop where
//! the required texture compression feature is not advertised.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn bound_on_rows_per_image() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for (rows_per_image, copy_height, copy_depth, success) in [
                (native::WGPU_COPY_STRIDE_UNDEFINED, 1, 1, true),
                (native::WGPU_COPY_STRIDE_UNDEFINED, 1, 2, false),
                (1, 2, 1, false),
                (2, 2, 1, true),
            ] {
                let texture = texture(test.device(), extent(4, 4, 2));
                let buffer = buffer(test.device(), 4096);
                let mut params =
                    CopyParams::new(texture, buffer, extent(4, copy_height, copy_depth), 4096);
                params.layout = native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: 256,
                    rowsPerImage: rows_per_image,
                };
                expect_copy(&test, *method, params, success);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn copy_end_overflows_u64() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for (depth, success) in [(1, true), (16, false)] {
                let texture = texture(test.device(), extent(1, 1, depth));
                let buffer = buffer(test.device(), 10_000);
                let mut params = CopyParams::new(texture, buffer, extent(1, 1, depth), 10_000);
                params.layout = layout(0, 1 << 31, 1 << 31);
                expect_copy(&test, *method, params, success);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn required_bytes_in_copy() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            let texture = create_texture(
                test.device(),
                texture_descriptor(
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_3D,
                    extent(5, 4, 3),
                    1,
                    1,
                ),
            );
            let full_buffer = buffer(test.device(), 2836);
            let mut params = CopyParams::new(texture, full_buffer, extent(5, 4, 3), 2836);
            params.layout = layout(0, 256, 4);
            expect_copy(&test, *method, params, true);

            let small = buffer(test.device(), 2835);
            params.buffer = small;
            params.data_size = 2835;
            expect_copy(&test, *method, params, false);

            yawgpu::wgpuBufferRelease(small);
            yawgpu::wgpuBufferRelease(full_buffer);
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
fn rows_per_image_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for rows_per_image in [1, 2, 3] {
                let texture = texture(test.device(), extent(4, 1, 1));
                let buffer = buffer(test.device(), 1024);
                let mut params = CopyParams::new(texture, buffer, extent(4, 1, 1), 1024);
                params.layout = layout(0, 256, rows_per_image);
                expect_copy(&test, *method, params, true);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for offset in [0, 1, 4, 8] {
                let texture = texture(test.device(), extent(4, 1, 1));
                let buffer = buffer(test.device(), 1024);
                let mut params = CopyParams::new(texture, buffer, extent(4, 1, 1), 1024);
                params.layout = layout(offset, 256, 1);
                let success = *method == CopyMethod::WriteTexture || offset % 4 == 0;
                expect_copy(&test, *method, params, success);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn bound_on_bytes_per_row() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for (bytes_per_row, copy_height, success) in [
                (native::WGPU_COPY_STRIDE_UNDEFINED, 1, true),
                (native::WGPU_COPY_STRIDE_UNDEFINED, 2, false),
                (8, 1, false),
                (16, 1, *method == CopyMethod::WriteTexture),
                (256, 2, true),
            ] {
                let texture = texture(test.device(), extent(4, copy_height, 1));
                let buffer = buffer(test.device(), 1024);
                let mut params = CopyParams::new(texture, buffer, extent(4, copy_height, 1), 1024);
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

#[test]
fn bound_on_offset() {
    let test = ValidationTest::new();
    unsafe {
        for method in ALL_METHODS {
            for (offset_blocks, data_blocks) in [(0, 0), (1, 0), (2, 1), (2, 2)] {
                let texture = texture(test.device(), extent(4, 4, 1));
                let data_size = data_blocks * 4;
                let buffer = buffer(test.device(), data_size);
                let mut params = CopyParams::new(texture, buffer, extent(0, 0, 0), data_size);
                params.layout = layout_undefined(offset_blocks * 4);
                expect_copy(&test, *method, params, offset_blocks <= data_blocks);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

unsafe fn texture(device: native::WGPUDevice, size: native::WGPUExtent3D) -> native::WGPUTexture {
    create_texture(
        device,
        texture_descriptor(
            native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureDimension_2D,
            size,
            1,
            1,
        ),
    )
}

unsafe fn buffer(device: native::WGPUDevice, size: u64) -> native::WGPUBuffer {
    create_buffer(
        device,
        size,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    )
}
