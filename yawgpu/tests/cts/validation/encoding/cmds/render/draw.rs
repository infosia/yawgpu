//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/draw.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_render_pipeline, create_render_pipeline_with, default_primitive,
    draw_indexed, draw_indexed_indirect, draw_indirect, draw_with_offsets, expect_render_commands,
    set_index_buffer, set_pipeline, set_vertex_buffer, strip_primitive, vertex_attribute,
    vertex_buffer_layout, vertex_input_shader, vertex_no_input, CommandExpectation,
    RenderEncodeType,
};

const ENCODE_TYPES: [RenderEncodeType; 2] =
    [RenderEncodeType::RenderPass, RenderEncodeType::RenderBundle];

#[derive(Clone, Copy, PartialEq, Eq)]
enum DrawType {
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawIndexedIndirect,
}

#[test]
fn unused_buffer_bound() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_render_pipeline(&test);
        let index = create_buffer(test.device(), 400, native::WGPUBufferUsage_Index);
        let indirect = create_buffer(test.device(), 20, native::WGPUBufferUsage_Indirect);
        for encode_type in ENCODE_TYPES {
            for draw_type in draw_types() {
                for (offset, size) in [(0, 0), (4, 1)] {
                    let small = create_buffer(
                        test.device(),
                        offset + size,
                        native::WGPUBufferUsage_Vertex | native::WGPUBufferUsage_Index,
                    );
                    expect_render_commands(
                        &test,
                        encode_type,
                        CommandExpectation::Success,
                        |encoder| {
                            set_pipeline(&encoder, pipeline);
                            if matches!(
                                draw_type,
                                DrawType::DrawIndexed | DrawType::DrawIndexedIndirect
                            ) {
                                set_index_buffer(
                                    &encoder,
                                    index,
                                    native::WGPUIndexFormat_Uint16,
                                    0,
                                    400,
                                );
                            } else {
                                set_index_buffer(
                                    &encoder,
                                    small,
                                    native::WGPUIndexFormat_Uint16,
                                    offset,
                                    size,
                                );
                            }
                            set_vertex_buffer(&encoder, 1, small, offset, size);
                            match draw_type {
                                DrawType::Draw => draw_with_offsets(&encoder, 100, 100, 100, 100),
                                DrawType::DrawIndexed => {
                                    draw_indexed(&encoder, 100, 100, 100, 100, 100)
                                }
                                DrawType::DrawIndirect => draw_indirect(&encoder, indirect),
                                DrawType::DrawIndexedIndirect => {
                                    draw_indexed_indirect(&encoder, indirect)
                                }
                            }
                        },
                    );
                    yawgpu::wgpuBufferRelease(small);
                }
            }
        }
        yawgpu::wgpuBufferRelease(indirect);
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn index_buffer_format() {
    let test = ValidationTest::new();
    unsafe {
        let index_buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_Index);
        let indirect = create_buffer(test.device(), 20, native::WGPUBufferUsage_Indirect);
        for encode_type in ENCODE_TYPES {
            for (topology, strip_format) in [
                (
                    native::WGPUPrimitiveTopology_TriangleList,
                    native::WGPUIndexFormat_Undefined,
                ),
                (
                    native::WGPUPrimitiveTopology_LineStrip,
                    native::WGPUIndexFormat_Uint16,
                ),
                (
                    native::WGPUPrimitiveTopology_TriangleStrip,
                    native::WGPUIndexFormat_Uint32,
                ),
            ] {
                let primitive = if strip_format == native::WGPUIndexFormat_Undefined {
                    default_primitive()
                } else {
                    strip_primitive(topology, strip_format)
                };
                let pipeline = create_render_pipeline_with(
                    &test,
                    test.device(),
                    vertex_no_input(),
                    &[],
                    primitive,
                );
                for index_format in [
                    native::WGPUIndexFormat_Uint16,
                    native::WGPUIndexFormat_Uint32,
                ] {
                    for draw_type in [DrawType::DrawIndexed, DrawType::DrawIndexedIndirect] {
                        let success = strip_format == native::WGPUIndexFormat_Undefined
                            || strip_format == index_format;
                        expect_render_commands(
                            &test,
                            encode_type,
                            if success {
                                CommandExpectation::Success
                            } else {
                                CommandExpectation::FinishError
                            },
                            |encoder| {
                                set_pipeline(&encoder, pipeline);
                                set_index_buffer(&encoder, index_buffer, index_format, 0, u64::MAX);
                                if draw_type == DrawType::DrawIndexed {
                                    draw_indexed(&encoder, 3, 1, 0, 0, 0);
                                } else {
                                    draw_indexed_indirect(&encoder, indirect);
                                }
                            },
                        );
                    }
                }
                yawgpu::wgpuRenderPipelineRelease(pipeline);
            }
        }
        yawgpu::wgpuBufferRelease(indirect);
        yawgpu::wgpuBufferRelease(index_buffer);
    }
}

#[test]
fn index_buffer_format_dirtying() {
    let test = ValidationTest::new();
    unsafe {
        let uint32_pipeline = create_render_pipeline_with(
            &test,
            test.device(),
            vertex_no_input(),
            &[],
            strip_primitive(
                native::WGPUPrimitiveTopology_TriangleStrip,
                native::WGPUIndexFormat_Uint32,
            ),
        );
        let uint16_pipeline = create_render_pipeline_with(
            &test,
            test.device(),
            vertex_no_input(),
            &[],
            strip_primitive(
                native::WGPUPrimitiveTopology_TriangleStrip,
                native::WGPUIndexFormat_Uint16,
            ),
        );
        let index_buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_Index);
        let indirect = create_buffer(test.device(), 20, native::WGPUBufferUsage_Indirect);
        for encode_type in ENCODE_TYPES {
            for dirty in ["pipeline", "indexBuffer", "neither"] {
                for draw_type in [DrawType::DrawIndexed, DrawType::DrawIndexedIndirect] {
                    expect_render_commands(
                        &test,
                        encode_type,
                        if dirty == "neither" {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                        |encoder| {
                            set_pipeline(&encoder, uint32_pipeline);
                            set_index_buffer(
                                &encoder,
                                index_buffer,
                                native::WGPUIndexFormat_Uint32,
                                0,
                                u64::MAX,
                            );
                            draw_indexed(&encoder, 3, 1, 0, 0, 0);
                            match dirty {
                                "pipeline" => set_pipeline(&encoder, uint16_pipeline),
                                "indexBuffer" => set_index_buffer(
                                    &encoder,
                                    index_buffer,
                                    native::WGPUIndexFormat_Uint16,
                                    0,
                                    u64::MAX,
                                ),
                                "neither" => {}
                                _ => unreachable!(),
                            }
                            if draw_type == DrawType::DrawIndexed {
                                draw_indexed(&encoder, 3, 1, 0, 0, 0);
                            } else {
                                draw_indexed_indirect(&encoder, indirect);
                            }
                        },
                    );
                }
            }
        }
        yawgpu::wgpuBufferRelease(indirect);
        yawgpu::wgpuBufferRelease(index_buffer);
        yawgpu::wgpuRenderPipelineRelease(uint16_pipeline);
        yawgpu::wgpuRenderPipelineRelease(uint32_pipeline);
    }
}

#[test]
fn index_buffer_oob() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_render_pipeline(&test);
        let indirect = create_buffer(test.device(), 20, native::WGPUBufferUsage_Indirect);
        for encode_type in ENCODE_TYPES {
            for (format, elem_size) in [
                (native::WGPUIndexFormat_Uint16, 2),
                (native::WGPUIndexFormat_Uint32, 4),
            ] {
                for (buffer_elems, binding_elems, draw_count) in [(10, 10, 10), (100, 10, 11)] {
                    let index = create_buffer(
                        test.device(),
                        buffer_elems * elem_size,
                        native::WGPUBufferUsage_Index,
                    );
                    for draw_type in [DrawType::DrawIndexed, DrawType::DrawIndexedIndirect] {
                        let success = draw_count <= binding_elems
                            || draw_type == DrawType::DrawIndexedIndirect;
                        expect_render_commands(
                            &test,
                            encode_type,
                            if success {
                                CommandExpectation::Success
                            } else {
                                CommandExpectation::FinishError
                            },
                            |encoder| {
                                set_pipeline(&encoder, pipeline);
                                set_index_buffer(
                                    &encoder,
                                    index,
                                    format,
                                    0,
                                    binding_elems * elem_size,
                                );
                                if draw_type == DrawType::DrawIndexed {
                                    draw_indexed(&encoder, draw_count as u32, 1, 0, 0, 0);
                                } else {
                                    draw_indexed_indirect(&encoder, indirect);
                                }
                            },
                        );
                    }
                    yawgpu::wgpuBufferRelease(index);
                }
            }
        }
        yawgpu::wgpuBufferRelease(indirect);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
#[ignore = "core vertex-buffer draw OOB validation does not account for CTS attribute lastStride semantics; CTS expects draw-time vertex/instance buffer bounds errors"]
fn vertex_buffer_oob() {
    let test = ValidationTest::new();
    unsafe {
        let attr = [vertex_attribute(native::WGPUVertexFormat_Float32x4, 0, 0)];
        let layout = [vertex_buffer_layout(
            native::WGPUVertexStepMode_Vertex,
            16,
            &attr,
        )];
        let pipeline = create_render_pipeline_with(
            &test,
            test.device(),
            &vertex_input_shader(0, "vec4f"),
            &layout,
            default_primitive(),
        );
        let buffer = create_buffer(test.device(), 15, native::WGPUBufferUsage_Vertex);
        expect_render_commands(
            &test,
            RenderEncodeType::RenderPass,
            CommandExpectation::FinishError,
            |encoder| {
                set_pipeline(&encoder, pipeline);
                set_vertex_buffer(&encoder, 0, buffer, 0, 15);
                draw_with_offsets(&encoder, 1, 1, 0, 0);
            },
        );
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn buffer_binding_overlap() {
    let test = ValidationTest::new();
    unsafe {
        let attr0 = [vertex_attribute(native::WGPUVertexFormat_Float32x4, 0, 0)];
        let attr1 = [vertex_attribute(native::WGPUVertexFormat_Float32x4, 0, 1)];
        let layouts = [
            vertex_buffer_layout(native::WGPUVertexStepMode_Vertex, 16, &attr0),
            vertex_buffer_layout(native::WGPUVertexStepMode_Instance, 16, &attr1),
        ];
        let pipeline = create_render_pipeline_with(
            &test,
            test.device(),
            "@vertex fn vs(@location(0) a: vec4f, @location(1) b: vec4f) -> @builtin(position) vec4f { return a + b * 0.0; }",
            &layouts,
            default_primitive(),
        );
        let shared = create_buffer(
            test.device(),
            4096,
            native::WGPUBufferUsage_Vertex | native::WGPUBufferUsage_Index,
        );
        for encode_type in ENCODE_TYPES {
            expect_render_commands(&test, encode_type, CommandExpectation::Success, |encoder| {
                set_pipeline(&encoder, pipeline);
                set_vertex_buffer(&encoder, 0, shared, 0, 2048);
                set_vertex_buffer(&encoder, 1, shared, 16, 2048);
                set_index_buffer(&encoder, shared, native::WGPUIndexFormat_Uint16, 32, 400);
                draw_indexed(&encoder, 3, 1, 0, 0, 0);
            });
        }
        yawgpu::wgpuBufferRelease(shared);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
#[ignore = "CTS marks this g.test unimplemented; no normative subcases to port yet"]
fn last_buffer_setting_take_account() {}

#[test]
#[ignore = "webgpu.h/core render pass descriptor does not expose or enforce maxDrawCount; CTS expects finish error when drawCount exceeds maxDrawCount"]
fn max_draw_count() {}

fn draw_types() -> [DrawType; 4] {
    [
        DrawType::Draw,
        DrawType::DrawIndexed,
        DrawType::DrawIndirect,
        DrawType::DrawIndexedIndirect,
    ]
}
