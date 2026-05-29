//! CTS port of `webgpu/api/validation/queue/submit.spec.ts`.

use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn command_buffer_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let q = queue(test.device());

        for (cb0_mismatched, cb1_mismatched) in [(false, false), (true, false), (false, true)] {
            let cb0 = create_empty_command_buffer(
                &test,
                if cb0_mismatched { other } else { test.device() },
                true,
            );
            let cb1 = create_empty_command_buffer(
                &test,
                if cb1_mismatched { other } else { test.device() },
                true,
            );
            expect_submit(&test, q, &[cb0, cb1], !(cb0_mismatched || cb1_mismatched));
            yawgpu::wgpuCommandBufferRelease(cb0);
            yawgpu::wgpuCommandBufferRelease(cb1);
        }

        yawgpu::wgpuQueueRelease(q);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn command_buffer_duplicate_buffers() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        let cb = create_empty_command_buffer(&test, test.device(), true);

        expect_submit(&test, q, &[cb, cb], false);

        yawgpu::wgpuCommandBufferRelease(cb);
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn command_buffer_submit_invalidates() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        let cb = create_empty_command_buffer(&test, test.device(), true);

        expect_submit(&test, q, &[cb], true);
        expect_submit(&test, q, &[cb], false);

        yawgpu::wgpuCommandBufferRelease(cb);
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn command_buffer_invalid_submit_invalidates() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());

        let cb1 = create_empty_command_buffer(&test, test.device(), true);
        let cb1_invalid = create_empty_command_buffer(&test, test.device(), false);
        expect_submit(&test, q, &[cb1, cb1_invalid], false);
        expect_submit(&test, q, &[cb1], false);

        let cb2 = create_empty_command_buffer(&test, test.device(), true);
        let cb2_invalid = create_empty_command_buffer(&test, test.device(), false);
        expect_submit(&test, q, &[cb2_invalid, cb2], false);
        expect_submit(&test, q, &[cb2], false);

        yawgpu::wgpuCommandBufferRelease(cb2_invalid);
        yawgpu::wgpuCommandBufferRelease(cb2);
        yawgpu::wgpuCommandBufferRelease(cb1_invalid);
        yawgpu::wgpuCommandBufferRelease(cb1);
        yawgpu::wgpuQueueRelease(q);
    }
}
