//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/index_access.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_render_pipeline_with, draw_indexed, expect_render_commands,
    set_index_buffer, set_pipeline, strip_primitive, vertex_no_input, CommandExpectation,
    RenderEncodeType,
};

#[test]
fn out_of_bounds() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_index_pipeline(&test);
        let index_buffer = create_buffer(test.device(), 6 * 4, native::WGPUBufferUsage_Index);
        let cases: [(u32, u32); 13] = [
            (6, 0),
            (5, 1),
            (1, 5),
            (0, 6),
            (0, 7),
            (7, 0),
            (6, 1),
            (1, 6),
            (6, 10000),
            (10000, 0),
            (0xffff_ffffu32, 0xffff_ffffu32),
            (0xffff_ffffu32, 2),
            (2, 0xffff_ffffu32),
        ];
        for (index_count, first_index) in cases {
            for instance_count in [1, 10000] {
                let success = index_count.checked_add(first_index).is_some_and(|v| v <= 6);
                expect_render_commands(
                    &test,
                    RenderEncodeType::RenderPass,
                    if success {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_pipeline(&encoder, pipeline);
                        set_index_buffer(
                            &encoder,
                            index_buffer,
                            native::WGPUIndexFormat_Uint32,
                            0,
                            u64::MAX,
                        );
                        draw_indexed(&encoder, index_count, instance_count, first_index, 0, 0);
                    },
                );
            }
        }
        yawgpu::wgpuBufferRelease(index_buffer);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn out_of_bounds_zero_sized_index_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_index_pipeline(&test);
        let index_buffer = create_buffer(test.device(), 0, native::WGPUBufferUsage_Index);
        for (index_count, first_index) in [(3, 1), (0, 1), (3, 0), (0, 0)] {
            for instance_count in [1, 10000] {
                let success = index_count + first_index == 0;
                expect_render_commands(
                    &test,
                    RenderEncodeType::RenderPass,
                    if success {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    |encoder| {
                        set_pipeline(&encoder, pipeline);
                        set_index_buffer(
                            &encoder,
                            index_buffer,
                            native::WGPUIndexFormat_Uint32,
                            0,
                            u64::MAX,
                        );
                        draw_indexed(&encoder, index_count, instance_count, first_index, 0, 0);
                    },
                );
            }
        }
        yawgpu::wgpuBufferRelease(index_buffer);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

unsafe fn create_index_pipeline(test: &ValidationTest) -> native::WGPURenderPipeline {
    unsafe {
        create_render_pipeline_with(
            test,
            test.device(),
            vertex_no_input(),
            &[],
            strip_primitive(
                native::WGPUPrimitiveTopology_TriangleStrip,
                native::WGPUIndexFormat_Uint32,
            ),
        )
    }
}
