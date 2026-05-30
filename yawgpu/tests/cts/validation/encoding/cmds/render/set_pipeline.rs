//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/setPipeline.spec.ts`.

use yawgpu_test::ValidationTest;

use crate::common::{
    create_error_render_pipeline, create_render_pipeline, create_render_pipeline_with,
    default_primitive, expect_render_commands, set_pipeline, vertex_no_input, CommandExpectation,
    RenderEncodeType,
};

const ENCODE_TYPES: [RenderEncodeType; 2] =
    [RenderEncodeType::RenderPass, RenderEncodeType::RenderBundle];

#[test]
fn invalid_pipeline() {
    let test = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for valid in [true, false] {
                let pipeline = if valid {
                    create_render_pipeline(&test)
                } else {
                    create_error_render_pipeline(&test)
                };
                expect_render_commands(
                    &test,
                    encode_type,
                    if valid {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_pipeline(&encoder, pipeline);
                    },
                );
                yawgpu::wgpuRenderPipelineRelease(pipeline);
            }
        }
    }
}

#[test]
fn pipeline_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for encode_type in ENCODE_TYPES {
            for mismatched in [false, true] {
                let pipeline = create_render_pipeline_with(
                    &test,
                    if mismatched {
                        foreign.device()
                    } else {
                        test.device()
                    },
                    vertex_no_input(),
                    &[],
                    default_primitive(),
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
                        set_pipeline(&encoder, pipeline);
                    },
                );
                yawgpu::wgpuRenderPipelineRelease(pipeline);
            }
        }
    }
}
