//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/indirect_draw.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_error_buffer, create_render_pipeline, expect_render_commands,
    set_index_buffer, set_pipeline, CommandExpectation, RenderEncodeType, RenderEncoder,
};

const ENCODE_TYPES: [RenderEncodeType; 2] =
    [RenderEncodeType::RenderPass, RenderEncodeType::RenderBundle];

#[test]
fn indirect_buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for indexed in [false, true] {
                for state in ["valid", "invalid", "destroyed"] {
                    let indirect = match state {
                        "valid" | "destroyed" => {
                            create_buffer(test.device(), 256, native::WGPUBufferUsage_Indirect)
                        }
                        "invalid" => create_error_buffer(&test),
                        _ => unreachable!(),
                    };
                    if state == "destroyed" {
                        yawgpu::wgpuBufferDestroy(indirect);
                    }
                    expect_indirect_draw(
                        &test,
                        encode_type,
                        indexed,
                        indirect,
                        0,
                        match state {
                            "valid" => CommandExpectation::Success,
                            "invalid" => CommandExpectation::FinishError,
                            "destroyed" => CommandExpectation::SubmitError,
                            _ => unreachable!(),
                        },
                    );
                    yawgpu::wgpuBufferRelease(indirect);
                }
            }
        }
    }
}

#[test]
fn indirect_buffer_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for indexed in [false, true] {
                for mismatched in [false, true] {
                    let indirect = create_buffer(
                        if mismatched {
                            foreign.device()
                        } else {
                            test.device()
                        },
                        256,
                        native::WGPUBufferUsage_Indirect,
                    );
                    expect_indirect_draw(
                        &test,
                        encode_type,
                        indexed,
                        indirect,
                        0,
                        if mismatched {
                            CommandExpectation::FinishError
                        } else {
                            CommandExpectation::Success
                        },
                    );
                    yawgpu::wgpuBufferRelease(indirect);
                }
            }
        }
    }
}

#[test]
fn indirect_buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for indexed in [false, true] {
                for usage in [
                    native::WGPUBufferUsage_Indirect,
                    native::WGPUBufferUsage_CopyDst,
                    native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_Indirect,
                ] {
                    let indirect = create_buffer(test.device(), 256, usage);
                    expect_indirect_draw(
                        &test,
                        encode_type,
                        indexed,
                        indirect,
                        0,
                        if usage & native::WGPUBufferUsage_Indirect != 0 {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                    );
                    yawgpu::wgpuBufferRelease(indirect);
                }
            }
        }
    }
}

#[test]
fn indirect_offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for indexed in [false, true] {
                for offset in [0, 2, 4] {
                    let indirect =
                        create_buffer(test.device(), 256, native::WGPUBufferUsage_Indirect);
                    expect_indirect_draw(
                        &test,
                        encode_type,
                        indexed,
                        indirect,
                        offset,
                        if offset % 4 == 0 {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                    );
                    yawgpu::wgpuBufferRelease(indirect);
                }
            }
        }
    }
}

#[test]
fn indirect_offset_oob() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for indexed in [false, true] {
                let params_size = if indexed { 20 } else { 16 };
                for (offset, buffer_size, success) in [
                    (0, 0, false),
                    (0, params_size, true),
                    (0, params_size + 1, true),
                    (0, params_size - 1, false),
                    (0, params_size - 4, false),
                    (4, params_size + 4, true),
                    (4, params_size + 3, false),
                    (2, params_size + 4, false),
                    (3, params_size + 4, false),
                    (5, params_size + 4, false),
                    (params_size, params_size, false),
                    (params_size + 4, params_size, false),
                ] {
                    let indirect =
                        create_buffer(test.device(), buffer_size, native::WGPUBufferUsage_Indirect);
                    expect_indirect_draw(
                        &test,
                        encode_type,
                        indexed,
                        indirect,
                        offset,
                        if success {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                    );
                    yawgpu::wgpuBufferRelease(indirect);
                }
            }
        }
    }
}

unsafe fn expect_indirect_draw(
    test: &ValidationTest,
    encode_type: RenderEncodeType,
    indexed: bool,
    indirect: native::WGPUBuffer,
    offset: u64,
    expectation: CommandExpectation,
) {
    unsafe {
        let pipeline = create_render_pipeline(test);
        let index = create_buffer(test.device(), 16, native::WGPUBufferUsage_Index);
        expect_render_commands(test, encode_type, expectation, |encoder| {
            set_pipeline(&encoder, pipeline);
            if indexed {
                set_index_buffer(&encoder, index, native::WGPUIndexFormat_Uint32, 0, u64::MAX);
                draw_indexed_indirect_at(&encoder, indirect, offset);
            } else {
                draw_indirect_at(&encoder, indirect, offset);
            }
        });
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

unsafe fn draw_indirect_at(encoder: &RenderEncoder, indirect: native::WGPUBuffer, offset: u64) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, indirect, offset);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderDrawIndirect(bundle, indirect, offset);
            }
        }
    }
}

unsafe fn draw_indexed_indirect_at(
    encoder: &RenderEncoder,
    indirect: native::WGPUBuffer,
    offset: u64,
) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderDrawIndexedIndirect(pass, indirect, offset);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderDrawIndexedIndirect(bundle, indirect, offset);
            }
        }
    }
}
