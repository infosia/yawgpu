//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/clearBuffer.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_encoder, create_error_buffer, expect_command_buffer, CommandExpectation,
};

#[test]
#[ignore = "core reports destroyed buffers at command-buffer finish; CTS expects clearBuffer recorded before submit with a destroyed buffer to fail at queue submit"]
fn buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for state in ["valid", "invalid", "destroyed"] {
            let buffer = match state {
                "valid" | "destroyed" => {
                    create_buffer(test.device(), 8, native::WGPUBufferUsage_CopyDst)
                }
                "invalid" => create_error_buffer(&test),
                _ => unreachable!(),
            };
            if state == "destroyed" {
                yawgpu::wgpuBufferDestroy(buffer);
            }
            expect_clear_buffer(
                &test,
                buffer,
                0,
                8,
                match state {
                    "valid" => CommandExpectation::Success,
                    "invalid" => CommandExpectation::FinishError,
                    "destroyed" => CommandExpectation::SubmitError,
                    _ => unreachable!(),
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn buffer_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for mismatched in [false, true] {
            let buffer = create_buffer(
                if mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                8,
                native::WGPUBufferUsage_CopyDst,
            );
            expect_clear_buffer(
                &test,
                buffer,
                0,
                8,
                if mismatched {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn default_args() {
    let test = ValidationTest::new();
    unsafe {
        for (offset, size) in [(0, u64::MAX), (4, u64::MAX), (0, 8)] {
            let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_clear_buffer(&test, buffer, offset, size, CommandExpectation::Success);
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for usage in buffer_usages() {
            let buffer = create_buffer(test.device(), 16, usage);
            expect_clear_buffer(
                &test,
                buffer,
                0,
                16,
                if usage == native::WGPUBufferUsage_CopyDst {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn size_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for (size, success) in [
            (0, true),
            (2, false),
            (4, true),
            (5, false),
            (8, true),
            (20, false),
            (u64::MAX, true),
        ] {
            let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_clear_buffer(
                &test,
                buffer,
                0,
                size,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for (offset, success) in [
            (0, true),
            (2, false),
            (4, true),
            (5, false),
            (8, true),
            (20, false),
        ] {
            let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_clear_buffer(
                &test,
                buffer,
                offset,
                8,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn overflow() {
    let test = ValidationTest::new();
    unsafe {
        for (offset, size) in [
            (0, u64::MAX - 7),
            (16, u64::MAX - 7),
            (u64::MAX - 7, 16),
            (u64::MAX - 7, u64::MAX - 7),
        ] {
            let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_clear_buffer(&test, buffer, offset, size, CommandExpectation::FinishError);
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn out_of_bounds() {
    let test = ValidationTest::new();
    unsafe {
        for (offset, size, success) in [
            (0, 32, true),
            (0, 36, false),
            (32, 0, true),
            (32, 4, false),
            (36, 4, false),
            (36, 0, false),
            (20, 16, false),
            (20, 12, true),
        ] {
            let buffer = create_buffer(test.device(), 32, native::WGPUBufferUsage_CopyDst);
            expect_clear_buffer(
                &test,
                buffer,
                offset,
                size,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

unsafe fn expect_clear_buffer(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
    expectation: CommandExpectation,
) {
    let encoder = create_encoder(test.device());
    yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, offset, size);
    expect_command_buffer(test, encoder, expectation);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

fn buffer_usages() -> [native::WGPUBufferUsage; 7] {
    [
        native::WGPUBufferUsage_MapRead,
        native::WGPUBufferUsage_MapWrite,
        native::WGPUBufferUsage_CopySrc,
        native::WGPUBufferUsage_CopyDst,
        native::WGPUBufferUsage_Index,
        native::WGPUBufferUsage_Vertex,
        native::WGPUBufferUsage_Uniform,
    ]
}
