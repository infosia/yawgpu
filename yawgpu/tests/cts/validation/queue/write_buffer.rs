//! CTS port of `webgpu/api/validation/queue/writeBuffer.spec.ts`.

use yawgpu::native;
use yawgpu_test::{assert_device_error, expect_no_validation_error, ValidationTest};

use super::common::*;

#[test]
fn buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for state in ["valid", "invalid", "destroyed"] {
            let buffer = match state {
                "invalid" => create_error_buffer(&test),
                _ => create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false),
            };
            if state == "destroyed" {
                yawgpu::wgpuBufferDestroy(buffer);
            }
            if state == "valid" {
                expect_no_validation_error(|| queue_write_buffer(q, buffer, 0, 16));
            } else {
                assert_device_error!({
                    queue_write_buffer(q, buffer, 0, 16);
                });
            }
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn ranges() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);

        for (offset, size, success) in [
            (0, 4, true),
            (0, 16, true),
            (16, 0, true),
            (20, 0, false),
            (3, 4, false),
            (0, 2, false),
            (12, 8, false),
            (u64::MAX - 3, 8, false),
        ] {
            if success {
                expect_no_validation_error(|| queue_write_buffer(q, buffer, offset, size));
            } else {
                assert_device_error!({
                    queue_write_buffer(q, buffer, offset, size);
                });
            }
        }

        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn usages() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for (usage, success) in [
            (native::WGPUBufferUsage_CopyDst, true),
            (native::WGPUBufferUsage_Storage, false),
            (
                native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
                false,
            ),
            (
                native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst,
                true,
            ),
        ] {
            let buffer = create_buffer(test.device(), 16, usage, false);
            if success {
                expect_no_validation_error(|| queue_write_buffer(q, buffer, 0, 16));
            } else {
                assert_device_error!({
                    queue_write_buffer(q, buffer, 0, 16);
                });
            }
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn buffer_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let q = queue(test.device());

        for mismatched in [false, true] {
            let device = if mismatched { other } else { test.device() };
            let buffer = create_buffer(device, 16, native::WGPUBufferUsage_CopyDst, false);
            if mismatched {
                assert_device_error!({
                    queue_write_buffer(q, buffer, 0, 16);
                });
            } else {
                expect_no_validation_error(|| queue_write_buffer(q, buffer, 0, 16));
            }
            yawgpu::wgpuBufferRelease(buffer);
        }

        yawgpu::wgpuQueueRelease(q);
        yawgpu::wgpuDeviceRelease(other);
    }
}
