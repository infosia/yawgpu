//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/setIndexBuffer.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_error_buffer, expect_render_commands, set_index_buffer,
    CommandExpectation, RenderEncodeType,
};

const ENCODE_TYPES: [RenderEncodeType; 2] =
    [RenderEncodeType::RenderPass, RenderEncodeType::RenderBundle];

#[test]
#[ignore = "core reports destroyed index buffers at finish; CTS expects setIndexBuffer recorded before submit with destroyed buffers to fail at queue submit"]
fn index_buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for state in ["valid", "invalid", "destroyed"] {
                let buffer = match state {
                    "valid" | "destroyed" => {
                        create_buffer(test.device(), 16, native::WGPUBufferUsage_Index)
                    }
                    "invalid" => create_error_buffer(&test),
                    _ => unreachable!(),
                };
                if state == "destroyed" {
                    yawgpu::wgpuBufferDestroy(buffer);
                }
                expect_render_commands(
                    &test,
                    encode_type,
                    match state {
                        "valid" => CommandExpectation::Success,
                        "invalid" => CommandExpectation::FinishError,
                        "destroyed" => CommandExpectation::SubmitError,
                        _ => unreachable!(),
                    },
                    |encoder| {
                        set_index_buffer(
                            &encoder,
                            buffer,
                            native::WGPUIndexFormat_Uint32,
                            0,
                            u64::MAX,
                        );
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn index_buffer_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for mismatched in [false, true] {
                let buffer = create_buffer(
                    if mismatched {
                        foreign.device()
                    } else {
                        test.device()
                    },
                    16,
                    native::WGPUBufferUsage_Index,
                );
                expect_render_commands(
                    &test,
                    encode_type,
                    if mismatched {
                        CommandExpectation::FinishError
                    } else {
                        CommandExpectation::Success
                    },
                    |encoder| {
                        set_index_buffer(
                            &encoder,
                            buffer,
                            native::WGPUIndexFormat_Uint32,
                            0,
                            u64::MAX,
                        );
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn index_buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for usage in [
                native::WGPUBufferUsage_Index,
                native::WGPUBufferUsage_CopyDst,
                native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_Index,
            ] {
                let buffer = create_buffer(test.device(), 16, usage);
                expect_render_commands(
                    &test,
                    encode_type,
                    if usage & native::WGPUBufferUsage_Index != 0 {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_index_buffer(
                            &encoder,
                            buffer,
                            native::WGPUIndexFormat_Uint32,
                            0,
                            u64::MAX,
                        );
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for (format, offsets, alignment) in [
                (native::WGPUIndexFormat_Uint16, [0, 1, 2], 2),
                (native::WGPUIndexFormat_Uint32, [0, 2, 4], 4),
            ] {
                for offset in offsets {
                    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_Index);
                    expect_render_commands(
                        &test,
                        encode_type,
                        if offset % alignment == 0 {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                        |encoder| {
                            set_index_buffer(&encoder, buffer, format, offset, u64::MAX);
                        },
                    );
                    yawgpu::wgpuBufferRelease(buffer);
                }
            }
        }
    }
}

#[test]
fn offset_and_size_oob() {
    let test = ValidationTest::new();
    unsafe {
        let cases = buffer_oob_cases(4, 256);
        for encode_type in ENCODE_TYPES {
            for (offset, size, success) in cases {
                let buffer = create_buffer(test.device(), 256, native::WGPUBufferUsage_Index);
                expect_render_commands(
                    &test,
                    encode_type,
                    if success {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_index_buffer(
                            &encoder,
                            buffer,
                            native::WGPUIndexFormat_Uint32,
                            offset,
                            size,
                        );
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

fn buffer_oob_cases(min_alignment: u64, buffer_size: u64) -> [(u64, u64, bool); 15] {
    [
        (0, 0, true),
        (0, 1, true),
        (0, 4, true),
        (0, 5, true),
        (0, buffer_size, true),
        (0, buffer_size + 4, false),
        (min_alignment, buffer_size, false),
        (min_alignment, buffer_size - min_alignment, true),
        (buffer_size - min_alignment, min_alignment, true),
        (buffer_size, 1, false),
        (0, u64::MAX, true),
        (min_alignment, u64::MAX, true),
        (buffer_size - min_alignment, u64::MAX, true),
        (buffer_size, u64::MAX, true),
        (buffer_size + min_alignment, u64::MAX, false),
    ]
}
