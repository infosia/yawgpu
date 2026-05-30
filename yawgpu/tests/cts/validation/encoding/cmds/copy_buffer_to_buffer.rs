//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/copyBufferToBuffer.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_encoder, create_error_buffer, expect_command_buffer, CommandExpectation,
};

#[test]
fn buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for src_state in ["valid", "invalid", "destroyed"] {
            for dst_state in ["valid", "invalid", "destroyed"] {
                let src = buffer_with_state(&test, src_state, native::WGPUBufferUsage_CopySrc);
                let dst = buffer_with_state(&test, dst_state, native::WGPUBufferUsage_CopyDst);
                let expectation = if src_state == "invalid" || dst_state == "invalid" {
                    CommandExpectation::FinishError
                } else if src_state == "destroyed" || dst_state == "destroyed" {
                    CommandExpectation::SubmitError
                } else {
                    CommandExpectation::Success
                };
                expect_copy_buffer_to_buffer(&test, src, 0, dst, 0, 8, expectation);
                yawgpu::wgpuBufferRelease(dst);
                yawgpu::wgpuBufferRelease(src);
            }
        }
    }
}

#[test]
fn buffer_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for (src_mismatched, dst_mismatched) in [(false, false), (true, false), (false, true)] {
            let src = create_buffer(
                if src_mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                16,
                native::WGPUBufferUsage_CopySrc,
            );
            let dst = create_buffer(
                if dst_mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                16,
                native::WGPUBufferUsage_CopyDst,
            );
            expect_copy_buffer_to_buffer(
                &test,
                src,
                0,
                dst,
                0,
                8,
                if src_mismatched || dst_mismatched {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
            );
            yawgpu::wgpuBufferRelease(dst);
            yawgpu::wgpuBufferRelease(src);
        }
    }
}

#[test]
fn buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for src_usage in buffer_usages() {
            for dst_usage in buffer_usages() {
                let src = create_buffer(test.device(), 16, src_usage);
                let dst = create_buffer(test.device(), 16, dst_usage);
                expect_copy_buffer_to_buffer(
                    &test,
                    src,
                    0,
                    dst,
                    0,
                    8,
                    if src_usage == native::WGPUBufferUsage_CopySrc
                        && dst_usage == native::WGPUBufferUsage_CopyDst
                    {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuBufferRelease(dst);
                yawgpu::wgpuBufferRelease(src);
            }
        }
    }
}

#[test]
fn copy_size_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for (size, success) in [(0, true), (2, false), (4, true), (5, false), (8, true)] {
            let src = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc);
            let dst = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_copy_buffer_to_buffer(
                &test,
                src,
                0,
                dst,
                0,
                size,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(dst);
            yawgpu::wgpuBufferRelease(src);
        }
    }
}

#[test]
fn copy_offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for (src_offset, dst_offset, success) in [
            (0, 0, true),
            (2, 0, false),
            (4, 0, true),
            (5, 0, false),
            (8, 0, true),
            (0, 2, false),
            (0, 4, true),
            (0, 5, false),
            (0, 8, true),
            (4, 4, true),
        ] {
            let src = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc);
            let dst = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_copy_buffer_to_buffer(
                &test,
                src,
                src_offset,
                dst,
                dst_offset,
                8,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(dst);
            yawgpu::wgpuBufferRelease(src);
        }
    }
}

#[test]
fn copy_overflow() {
    let test = ValidationTest::new();
    unsafe {
        for (src_offset, dst_offset, size) in [
            (0, 0, u64::MAX - 7),
            (16, 0, u64::MAX - 7),
            (0, 16, u64::MAX - 7),
            (u64::MAX - 7, 0, 16),
            (0, u64::MAX - 7, 16),
            (u64::MAX - 7, 0, u64::MAX - 7),
            (0, u64::MAX - 7, u64::MAX - 7),
            (u64::MAX - 7, u64::MAX - 7, u64::MAX - 7),
        ] {
            let src = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc);
            let dst = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
            expect_copy_buffer_to_buffer(
                &test,
                src,
                src_offset,
                dst,
                dst_offset,
                size,
                CommandExpectation::FinishError,
            );
            yawgpu::wgpuBufferRelease(dst);
            yawgpu::wgpuBufferRelease(src);
        }
    }
}

#[test]
fn copy_out_of_bounds() {
    let test = ValidationTest::new();
    unsafe {
        for (src_offset, dst_offset, size, success) in [
            (0, 0, 32, true),
            (0, 0, 36, false),
            (36, 0, 4, false),
            (0, 36, 4, false),
            (36, 0, 0, false),
            (0, 36, 0, false),
            (20, 0, 16, false),
            (20, 0, 12, true),
            (0, 20, 16, false),
            (0, 20, 12, true),
        ] {
            let src = create_buffer(test.device(), 32, native::WGPUBufferUsage_CopySrc);
            let dst = create_buffer(test.device(), 32, native::WGPUBufferUsage_CopyDst);
            expect_copy_buffer_to_buffer(
                &test,
                src,
                src_offset,
                dst,
                dst_offset,
                size,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(dst);
            yawgpu::wgpuBufferRelease(src);
        }
    }
}

#[test]
fn copy_within_same_buffer() {
    let test = ValidationTest::new();
    unsafe {
        for (src_offset, dst_offset, size) in [(0, 8, 4), (8, 0, 4), (0, 4, 8), (4, 0, 8)] {
            let buffer = create_buffer(
                test.device(),
                16,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            );
            expect_copy_buffer_to_buffer(
                &test,
                buffer,
                src_offset,
                buffer,
                dst_offset,
                size,
                CommandExpectation::FinishError,
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

unsafe fn buffer_with_state(
    test: &ValidationTest,
    state: &str,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let buffer = match state {
        "valid" | "destroyed" => create_buffer(test.device(), 16, usage),
        "invalid" => create_error_buffer(test),
        _ => unreachable!(),
    };
    if state == "destroyed" {
        yawgpu::wgpuBufferDestroy(buffer);
    }
    buffer
}

unsafe fn expect_copy_buffer_to_buffer(
    test: &ValidationTest,
    src: native::WGPUBuffer,
    src_offset: u64,
    dst: native::WGPUBuffer,
    dst_offset: u64,
    size: u64,
    expectation: CommandExpectation,
) {
    let encoder = create_encoder(test.device());
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, src, src_offset, dst, dst_offset, size);
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
