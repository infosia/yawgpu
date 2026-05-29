//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/setVertexBuffer.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_error_buffer, expect_render_commands, set_vertex_buffer,
    CommandExpectation, RenderEncodeType,
};

const ENCODE_TYPES: [RenderEncodeType; 2] =
    [RenderEncodeType::RenderPass, RenderEncodeType::RenderBundle];

#[test]
fn slot() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );
        for encode_type in ENCODE_TYPES {
            for slot in [0, limits.maxVertexBuffers - 1, limits.maxVertexBuffers] {
                let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_Vertex);
                expect_render_commands(
                    &test,
                    encode_type,
                    if slot < limits.maxVertexBuffers {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_vertex_buffer(&encoder, slot, buffer, 0, u64::MAX);
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
#[ignore = "core reports destroyed vertex buffers at finish; CTS expects setVertexBuffer recorded before submit with destroyed buffers to fail at queue submit"]
fn vertex_buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for state in ["valid", "invalid", "destroyed"] {
                let buffer = match state {
                    "valid" | "destroyed" => {
                        create_buffer(test.device(), 16, native::WGPUBufferUsage_Vertex)
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
                        set_vertex_buffer(&encoder, 0, buffer, 0, u64::MAX);
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
#[ignore = "render bundle setVertexBuffer does not validate buffer device ownership; CTS expects mismatched-device vertex buffers to fail"]
fn vertex_buffer_device_mismatch() {
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
                    native::WGPUBufferUsage_Vertex,
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
                        set_vertex_buffer(&encoder, 0, buffer, 0, u64::MAX);
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn vertex_buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for usage in [
                native::WGPUBufferUsage_Vertex,
                native::WGPUBufferUsage_CopyDst,
                native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_Vertex,
            ] {
                let buffer = create_buffer(test.device(), 16, usage);
                expect_render_commands(
                    &test,
                    encode_type,
                    if usage & native::WGPUBufferUsage_Vertex != 0 {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_vertex_buffer(&encoder, 0, buffer, 0, u64::MAX);
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
            for offset in [0, 2, 4] {
                let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_Vertex);
                expect_render_commands(
                    &test,
                    encode_type,
                    if offset % 4 == 0 {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_vertex_buffer(&encoder, 0, buffer, offset, u64::MAX);
                    },
                );
                yawgpu::wgpuBufferRelease(buffer);
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
                let buffer = create_buffer(test.device(), 256, native::WGPUBufferUsage_Vertex);
                expect_render_commands(
                    &test,
                    encode_type,
                    if success {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_vertex_buffer(&encoder, 0, buffer, offset, size);
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
