//! CTS port of `webgpu/api/validation/queue/writeTexture.spec.ts`.

use yawgpu::native;
use yawgpu_test::{assert_device_error, expect_no_validation_error, ValidationTest};

use super::common::*;

#[test]
fn texture_state() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for state in ["valid", "invalid", "destroyed"] {
            let texture = match state {
                "invalid" => create_error_texture(&test),
                _ => create_texture(test.device(), native::WGPUTextureUsage_CopyDst),
            };
            if state == "destroyed" {
                yawgpu::wgpuTextureDestroy(texture);
            }
            if state == "valid" {
                expect_no_validation_error(|| queue_write_texture(q, texture, 4));
            } else {
                assert_device_error!({
                    queue_write_texture(q, texture, 4);
                });
            }
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn usages() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for usage in [
            native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureUsage_StorageBinding,
            native::WGPUTextureUsage_StorageBinding | native::WGPUTextureUsage_CopySrc,
            native::WGPUTextureUsage_StorageBinding | native::WGPUTextureUsage_CopyDst,
        ] {
            let texture = create_texture(test.device(), usage);
            let success = usage & native::WGPUTextureUsage_CopyDst != 0;
            if success {
                expect_no_validation_error(|| queue_write_texture(q, texture, 4));
            } else {
                assert_device_error!({
                    queue_write_texture(q, texture, 4);
                });
            }
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn sample_count() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for sample_count in [1, 4] {
            let texture = create_texture_with_descriptor(
                test.device(),
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_CopyDst
                        | native::WGPUTextureUsage_RenderAttachment,
                    sampleCount: sample_count,
                    format: native::WGPUTextureFormat_BGRA8Unorm,
                    ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
                },
            );
            if sample_count == 1 {
                expect_no_validation_error(|| queue_write_texture(q, texture, 4));
            } else {
                assert_device_error!({
                    queue_write_texture(q, texture, 4);
                });
            }
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn texture_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let q = queue(test.device());

        for mismatched in [false, true] {
            let device = if mismatched { other } else { test.device() };
            let texture = create_texture_with_descriptor(
                device,
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_CopyDst
                        | native::WGPUTextureUsage_RenderAttachment,
                    format: native::WGPUTextureFormat_BGRA8Unorm,
                    ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
                },
            );
            if mismatched {
                assert_device_error!({
                    queue_write_texture(q, texture, 4);
                });
            } else {
                expect_no_validation_error(|| queue_write_texture(q, texture, 4));
            }
            yawgpu::wgpuTextureRelease(texture);
        }

        yawgpu::wgpuQueueRelease(q);
        yawgpu::wgpuDeviceRelease(other);
    }
}
